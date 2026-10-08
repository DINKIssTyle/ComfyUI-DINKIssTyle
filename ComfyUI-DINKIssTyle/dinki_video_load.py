"""Folder-based video loader with spatial crop and timestamp-based trimming."""

import asyncio
import hashlib
import math
import os
import threading
from fractions import Fraction
from functools import lru_cache

import folder_paths
from aiohttp import web
from server import PromptServer

from .dinki_image_crop import DINKI_Image_Crop, _crop_bounds
from .dinki_load import _category_directory, _normalize_category
from .dinki_photo_specs import DINKI_photo_specifications

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg", ".ts"}
_PREVIEW_LOCK = threading.Lock()


def _video_path(category, filename):
    if not filename or os.path.basename(filename) != filename or "\\" in filename:
        raise ValueError("Invalid video filename.")
    directory = _category_directory(category)
    path = os.path.realpath(os.path.join(directory, filename))
    root = os.path.realpath(folder_paths.get_input_directory())
    if os.path.commonpath((root, path)) != root:
        raise ValueError("Video must stay inside the ComfyUI input folder.")
    if not os.path.isfile(path) or os.path.splitext(filename)[1].lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Video file not found: {filename}")
    return path


def _video_files(category=""):
    directory = _category_directory(category)
    if not os.path.isdir(directory):
        return []
    files = []
    for filename in os.listdir(directory):
        if filename.startswith("."):
            continue
        try:
            _video_path(category, filename)
            files.append(filename)
        except ValueError:
            pass
    return sorted(files, key=str.casefold)


def _video_categories():
    root = folder_paths.get_input_directory()
    categories = [""]
    for directory, children, _ in os.walk(root):
        children[:] = sorted((name for name in children if not name.startswith(".")), key=str.casefold)
        if directory != root:
            category = os.path.relpath(directory, root).replace(os.sep, "/")
            try:
                if _video_files(category):
                    categories.append(category)
            except ValueError:
                pass
    return sorted(categories, key=str.casefold)


def _identity(path):
    stat = os.stat(path)
    return path, stat.st_mtime_ns, stat.st_size


def _rotation(stream, frame=None):
    # Newer PyAV exposes display-matrix rotation on decoded frames.
    value = getattr(frame, "rotation", None)
    if value is None or value == 0:
        value = -float(stream.metadata.get("rotate", 0))
    return int(round(float(value) / 90)) * 90 % 360


@lru_cache(maxsize=64)
def _probe_cached(path, modified, size):
    import av
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("The selected file contains no video stream.")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        if not rate or rate <= 0:
            raise ValueError("Unable to determine the source FPS.")
        origin = float((stream.start_time or 0) * stream.time_base)
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        first = next(container.decode(stream), None)
        if first is None:
            raise ValueError("The selected video contains no readable frames.")
        if stream.start_time is None and first.time is not None:
            origin = float(first.time)
        if duration is None or duration <= 0:
            # Container duration can include longer audio; inspect frame timestamps
            # when the video stream has no trustworthy duration of its own.
            last = first
            for last in container.decode(stream):
                pass
            last_duration = float(last.duration * last.time_base) if getattr(last, "duration", 0) else 1 / float(rate)
            duration = max(0.0, float(last.time or origin) - origin) + last_duration
        rotation = _rotation(stream, first)
        width, height = first.width, first.height
        if rotation in (90, 270):
            width, height = height, width
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Unable to determine the video duration.")
        return {"width": width, "height": height, "duration": duration,
                "fps": float(rate), "fps_numerator": rate.numerator,
                "fps_denominator": rate.denominator, "origin": origin,
                "rotation": rotation, "has_audio": bool(container.streams.audio)}


def _probe(path):
    return dict(_probe_cached(*_identity(path)))


def _finite(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}.")
    return value


def _window(duration, trim_in, trim_out):
    start = _finite(trim_in, "IN")
    requested_end = _finite(trim_out, "OUT")
    end = duration if requested_end == 0 else min(duration, requested_end)
    if start >= end:
        raise ValueError("IN must be before OUT and inside the video.")
    return start, end


def _frame_count(duration, rate):
    return max(1, math.ceil(duration * float(rate) - 1e-8))


def _check_interrupt():
    from comfy.model_management import throw_exception_if_processing_interrupted
    throw_exception_if_processing_interrupted()


def _sample_frames(path, metadata, start, end, rate, check_interrupt=_check_interrupt):
    """Yield the frame covering each output timestamp, using one-frame lookahead.

    A frame before IN may cover IN. OUT is exclusive. VFR is sampled by PTS,
    never by frame index, so changing FPS does not change playback speed.
    """
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        origin = metadata["origin"]
        container.seek(int((origin + start) / stream.time_base), stream=stream, backward=True)
        decoded = iter(container.decode(stream))
        current = next(decoded, None)
        if current is None:
            raise ValueError("No readable frames in the selected interval.")
        following = next(decoded, None)
        fallback_time = float(current.time) - origin if current.time is not None else start
        for index in range(_frame_count(end - start, rate)):
            check_interrupt()
            target = start + index / float(rate)
            while following is not None:
                timestamp = (float(following.time) - origin if following.time is not None
                             else fallback_time + 1 / metadata["fps"])
                if timestamp > target + 1e-8 or timestamp >= end - 1e-8:
                    break
                current, fallback_time = following, timestamp
                following = next(decoded, None)
                check_interrupt()
            yield current


def _pixels(frame, rotation):
    import numpy as np
    pixels = frame.to_ndarray(format="rgb24")
    if rotation:
        pixels = np.rot90(pixels, k=rotation // 90)
    return pixels.copy()


def _read_audio(path, metadata, start, duration):
    """Decode only the selected audio window into ComfyUI's [1,C,N] format."""
    import av
    import torch
    with av.open(path) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        if not sample_rate:
            raise ValueError("Unable to determine the audio sample rate.")
        channels = len(stream.codec_context.layout.channels)
        waveform = torch.zeros((1, channels, round(duration * sample_rate)), dtype=torch.float32)
        resampler = av.AudioResampler(format="fltp", layout=stream.codec_context.layout.name, rate=sample_rate)
        origin = metadata["origin"]
        container.seek(max(0, int((origin + start - 0.1) / stream.time_base)),
                       stream=stream, backward=True)
        cursor = None

        def copy_audio(frame):
            nonlocal cursor
            if frame.pts is not None:
                cursor = round((float(frame.pts * frame.time_base) - origin - start) * sample_rate)
            elif cursor is None:
                raise ValueError("Audio timestamps are missing; cannot preserve synchronization.")
            values = torch.from_numpy(frame.to_ndarray().copy())
            left, right = max(0, cursor), min(waveform.shape[-1], cursor + values.shape[-1])
            if right > left:
                waveform[0, :, left:right] = values[:, left - cursor:right - cursor]
            cursor += values.shape[-1]

        for frame in container.decode(stream):
            _check_interrupt()
            for output in resampler.resample(frame):
                copy_audio(output)
            if cursor is not None and cursor >= waveform.shape[-1]:
                break
        for output in resampler.resample(None):
            copy_audio(output)
        return {"waveform": waveform, "sample_rate": sample_rate}


def _create_preview(path, metadata):
    """Bounded-memory silent proxy for browser-incompatible input codecs."""
    import av
    key = hashlib.sha256(repr(_identity(path)).encode()).hexdigest()[:24]
    filename = f"DKST_Video_Load_{key}.mp4"
    destination = os.path.join(folder_paths.get_temp_directory(), filename)
    with _PREVIEW_LOCK:
        if not os.path.isfile(destination):
            partial = destination + ".partial"
            try:
                scale = min(1, 720 / max(metadata["width"], metadata["height"]))
                width = max(2, int(metadata["width"] * scale) // 2 * 2)
                height = max(2, int(metadata["height"] * scale) // 2 * 2)
                rate = min(Fraction(metadata["fps_numerator"], metadata["fps_denominator"]), Fraction(24))
                with av.open(partial, "w", format="mp4", options={"movflags": "+faststart"}) as container:
                    stream = container.add_stream("libx264", rate=rate)
                    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
                    stream.options = {"crf": "26", "preset": "ultrafast"}
                    for index, source in enumerate(_sample_frames(
                            path, metadata, 0, metadata["duration"], rate, lambda: None)):
                        frame = av.VideoFrame.from_ndarray(_pixels(source, metadata["rotation"]), format="rgb24")
                        frame = frame.reformat(width=width, height=height, format="yuv420p")
                        frame.pts, frame.time_base = index, 1 / rate
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
                os.replace(partial, destination)
            finally:
                if os.path.exists(partial):
                    os.remove(partial)
    return {"filename": filename, "subfolder": "", "type": "temp"}


@PromptServer.instance.routes.get("/dinki/video-load/categories")
async def video_categories(request):
    return web.json_response({"categories": await asyncio.to_thread(_video_categories)})


@PromptServer.instance.routes.get("/dinki/video-load/files")
async def video_files(request):
    try:
        category = _normalize_category(request.query.get("category", ""))
        return web.json_response({"files": await asyncio.to_thread(_video_files, category)})
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.get("/dinki/video-load/metadata")
async def video_metadata(request):
    try:
        category = _normalize_category(request.query.get("category", ""))
        filename = request.query.get("filename", "")
        path = _video_path(category, filename)
        metadata = await asyncio.to_thread(_probe, path)
        metadata.pop("origin", None)
        metadata["source_key"] = hashlib.sha256(repr(_identity(path)).encode()).hexdigest()[:24]
        metadata["preview"] = {"filename": filename, "subfolder": category, "type": "input"}
        return web.json_response(metadata)
    except Exception as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.post("/dinki/video-load/preview")
async def video_preview(request):
    try:
        data = await request.json()
        path = _video_path(data.get("category", ""), data.get("filename", ""))
        metadata = await asyncio.to_thread(_probe, path)
        preview = await asyncio.to_thread(_create_preview, path, metadata)
        return web.json_response({"preview": preview, "silent": True})
    except Exception as error:
        return web.json_response({"error": str(error)}, status=400)


class DINKI_Video_Load_Crop:
    CATEGORY = "DINKIssTyle/Video"
    RETURN_TYPES = ("VIDEO", "IMAGE", "FLOAT", "INT", "FLOAT")
    RETURN_NAMES = ("video", "images", "fps", "frame_count", "duration")
    FUNCTION = "load_and_crop"
    OUTPUT_NODE = True
    DESCRIPTION = "Load a video from input folders, crop, trim and resample FPS while preserving playback speed. OUT=0 means end of video."

    @classmethod
    def INPUT_TYPES(cls):
        crop = DINKI_Image_Crop.INPUT_TYPES()["required"]
        photo = DINKI_photo_specifications.INPUT_TYPES()["required"]
        return {"required": {
            "category": (_video_categories(),),
            "filename": (_video_files() or [""],),
            **{name: spec for name, spec in crop.items() if name != "image"},
            "resolution_multiple": photo["resolution_multiple"],
            "megapixels": photo["megapixels"],
            "trim_in": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e9, "step": 0.001}),
            "trim_out": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e9, "step": 0.001,
                                     "tooltip": "Exclusive OUT in seconds. 0 = end of video."}),
            "fps_mode": (["Original", "Custom"], {"default": "Original"}),
            "output_fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 240.0, "step": 0.01}),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, category, filename, **kwargs):
        try:
            _video_path(category, filename)
        except ValueError as error:
            return str(error)
        return True

    @classmethod
    def IS_CHANGED(cls, category, filename, **kwargs):
        try:
            return repr(_identity(_video_path(category, filename)))
        except (OSError, ValueError):
            return float("NaN")

    def load_and_crop(self, category, filename, aspect_ratio="Original", custom_width=1,
                      custom_height=1, crop_x=0.0, crop_y=0.0, crop_width=1.0,
                      crop_height=1.0, resolution_multiple=8, megapixels=1.0,
                      trim_in=0.0, trim_out=0.0, fps_mode="Original", output_fps=24.0):
        import torch
        import torch.nn.functional as F
        from comfy_api.latest import InputImpl, Types
        from comfy.utils import ProgressBar

        path = _video_path(category, filename)
        metadata = _probe(path)
        start, end = _window(metadata["duration"], trim_in, trim_out)
        if fps_mode not in ("Original", "Custom"):
            raise ValueError("Unknown FPS mode.")
        rate = (Fraction(metadata["fps_numerator"], metadata["fps_denominator"])
                if fps_mode == "Original" else Fraction(str(_finite(output_fps, "FPS", 0.01))).limit_denominator(1000000))
        if fps_mode == "Custom" and rate > 240:
            raise ValueError("Custom FPS must be <= 240.")
        mp = _finite(megapixels, "Megapixels", 0.1)
        if mp > 64:
            raise ValueError("Megapixels must be <= 64.")
        left, top, crop_w, crop_h = _crop_bounds(
            metadata["width"], metadata["height"], aspect_ratio, custom_width, custom_height,
            crop_x, crop_y, crop_width, crop_height)
        # A shape-only view lets us reuse the existing resolution calculation.
        shape = torch.empty((1, 1, 1, 3)).expand(1, crop_h, crop_w, 3)
        width, height, _ = DINKI_photo_specifications().calculate_resolution(
            mp, "Basic 1:1", False, resolution="Image",
            resolution_multiple=resolution_multiple, image=shape)
        count = _frame_count(end - start, rate)
        images = torch.empty((count, height, width, 3), dtype=torch.float32)
        frames = _sample_frames(path, metadata, start, end, rate)
        progress = ProgressBar(count)
        last_frame = None
        try:
            for index, frame in enumerate(frames):
                if frame is not last_frame:
                    pixels = _pixels(frame, metadata["rotation"])[top:top + crop_h, left:left + crop_w]
                    image = torch.from_numpy(pixels.copy()).float().div_(255).unsqueeze(0)
                    if (crop_w, crop_h) != (width, height):
                        image = F.interpolate(image.permute(0, 3, 1, 2), size=(height, width),
                                              mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1)
                    last_frame = frame
                images[index] = image[0]
                progress.update(1)
        finally:
            frames.close()
        duration = count / float(rate)
        audio = _read_audio(path, metadata, start, min(end - start, duration))
        if audio is not None:
            target_samples = round(duration * audio["sample_rate"])
            audio["waveform"] = F.pad(audio["waveform"], (0, max(0, target_samples - audio["waveform"].shape[-1])))
        video = InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=audio, frame_rate=rate))
        return {"ui": {"resolution": [f"{width} × {height}"], "fps": [float(rate)],
                       "frame_count": [count], "duration": [duration]},
                "result": (video, images, float(rate), count, duration)}
