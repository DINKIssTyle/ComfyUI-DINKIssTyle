"""Encode image batches using the codecs already shipped with ComfyUI's PyAV.

Format selection follows the representative presets in VideoHelperSuite;
no VHS dependency, external FFmpeg executable or package installation is required.
"""
import json
import math
import os
import time
import uuid
from fractions import Fraction

import folder_paths
from . import dinki_video_encoding as encoding

PIXEL_FORMATS = ["auto", "yuv420p", "yuv422p", "yuv444p", "yuv420p10le",
                 "yuv422p10le", "yuv444p10le", "bgra", "rgba64le", "nv12", "p010le"]
FORMATS = {
    "h264-mp4": ("mp4", "mp4", "libx264", "aac", "yuv420p"),
    "h265-mp4": ("mp4", "mp4", "libx265", "aac", "yuv420p"),
    "vp9-webm": ("webm", "webm", "libvpx-vp9", "libopus", "yuv420p"),
    "av1-webm": ("webm", "webm", "libsvtav1", "libopus", "yuv420p"),
    "prores-mov": ("mov", "mov", "prores_ks", "pcm_s16le", "yuv422p10le"),
    "ffv1-mkv": ("mkv", "matroska", "ffv1", "flac", "bgra"),
    "gif": ("gif", None, None, None, "auto"),
    "webp": ("webp", None, None, None, "auto"),
}
FIXED_RATE_FORMATS = {"prores-mov", "ffv1-mkv", "gif", "webp"}


def _pixel_formats(format, encoder="auto"):
    config = FORMATS[format]
    return encoding.pixel_choices(config[2], config[4], encoder)


def available_formats():
    result = []
    for format in FORMATS:
        try:
            if _pixel_formats(format): result.append(format)
        except (ImportError, ValueError):
            continue
    return result


def encoding_inputs(default_pixel="yuv420p", default_bitrate=8.0):
    return {
        "pixel_format": (PIXEL_FORMATS, {"default": default_pixel,
            "tooltip": "Chroma sampling / pixel format (not a color-space transform). Auto chooses a compatible format."}),
        "bitrate_mbps": ("FLOAT", {"default": default_bitrate, "min": 0.0, "max": 1000.0,
            "step": 0.1, "tooltip": "Target video bitrate in Mbps. 0 = automatic quality. Not used for ProRes, FFV1, GIF or WebP."}),
    }


def format_options():
    formats = available_formats()
    return formats, {"default": "h264-mp4" if "h264-mp4" in formats else formats[0],
                     **format_metadata(formats)}


def format_metadata(formats):
    return {"dkst_pixel_formats": {name: _pixel_formats(name) for name in formats},
            "dkst_encoder_pixels": {name: {encoder: _pixel_formats(name, encoder)
                for encoder in encoding.ENCODERS} for name in formats}}


def encoder_input():
    return (encoding.ENCODERS, {"default": "auto", "tooltip":
        "Auto prefers verified hardware. CPU, NVIDIA NVENC, or Apple VideoToolbox can be selected explicitly. Hardware requires an installed compatible encoder and device."})


def _number(value, name, minimum, maximum):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number.")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}.")
    return value


def validate_encoding(format, pixel_format, bitrate_mbps, encoder="cpu"):
    if format not in FORMATS:
        raise ValueError(f"Unknown video format: {format}")
    if encoder not in encoding.ENCODERS:
        raise ValueError(f"Unknown encoder: {encoder}")
    if format in FIXED_RATE_FORMATS and encoder not in ("auto", "cpu"):
        raise ValueError(f"{format} supports CPU encoding only.")
    choices = _pixel_formats(format, encoder)
    if pixel_format not in choices:
        raise ValueError(f"{format} does not support {pixel_format}. Choose {', '.join(choices)}.")
    _number(bitrate_mbps, "Bitrate (Mbps)", 0, 1000)
    return FORMATS[format][4] if pixel_format == "auto" else pixel_format


def _interrupt():
    from comfy.model_management import throw_exception_if_processing_interrupted
    throw_exception_if_processing_interrupted()


def _shape(images):
    import torch
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or min(images.shape[:3]) < 1 or images.shape[-1] not in (3, 4):
        raise ValueError("images must be a nonempty RGB or RGBA IMAGE batch [frames,height,width,channels].")
    return images.shape[2], images.shape[1]


def _frame_pixels(image, depth=8, alpha=False):
    import numpy as np
    import torch
    if not torch.isfinite(image).all():
        raise ValueError("images contain non-finite pixels.")
    image = image.detach().float().cpu().clamp(0, 1)
    if image.shape[-1] == 4 and not alpha:
        image = image[..., :3] * image[..., 3:4]
    maximum = 65535 if depth == 16 else 255
    return (image * maximum).round().numpy().astype(np.uint16 if depth == 16 else np.uint8)


def _audio_data(audio, duration):
    import torch
    if audio is None:
        return None
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2):
        raise ValueError("audio must contain a mono or stereo waveform with shape [1,channels,samples].")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000:
        raise ValueError("audio sample_rate must be an integer from 8000 to 192000.")
    samples = round(duration * sample_rate)
    waveform = waveform[0, :, :samples].detach().float().cpu()
    if not torch.isfinite(waveform).all():
        raise ValueError("audio contains non-finite samples.")
    return waveform, sample_rate, samples


def _dimensions(width, height, pixel_format):
    horizontal = 2 if "420" in pixel_format or "422" in pixel_format or pixel_format in ("nv12", "p010le") else 1
    vertical = 2 if "420" in pixel_format or pixel_format in ("nv12", "p010le") else 1
    return width + (-width % horizontal), height + (-height % vertical)


def _encode(path, images, audio, rate, format, pixel_format, bitrate_mbps, metadata,
            container_override=None, selection=None):
    import av
    import numpy as np
    from comfy.utils import ProgressBar
    config = FORMATS[format]
    selection = selection or encoding.Selection(config[2], pixel_format, bitrate=round(float(bitrate_mbps) * 1_000_000))
    pixel_format = selection.pixel
    source_w, source_h = _shape(images)
    width, height = _dimensions(source_w, source_h, pixel_format)
    container_format = container_override or config[1]
    options = {"movflags": "+faststart+use_metadata_tags"} if container_format in ("mp4", "mov") else {}
    audio_data = _audio_data(audio, len(images) / float(rate))
    if selection.engine == "ffmpeg":
        from .dinki_video_ffmpeg import encode
        return encode(path, images, audio, rate, config, selection, metadata, container_format,
                      (width, height), _frame_pixels, audio_data, _interrupt)
    progress = ProgressBar(len(images))
    with av.open(path, "w", format=container_format, options=options) as container:
        for key, value in (metadata or {}).items():
            container.metadata[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        stream = container.add_stream(selection.codec, rate=rate)
        stream.width, stream.height, stream.pix_fmt = width, height, pixel_format
        stream.codec_context.thread_count = 2
        # Input images are SDR RGB; request BT.709 RGB -> YUV conversion explicitly.
        stream.codec_context.color_primaries = 1
        stream.codec_context.color_trc = 1
        stream.codec_context.colorspace = 1
        stream.codec_context.color_range = 2 if pixel_format in ("bgra", "rgba64le") else 1
        stream.options = {
            "libx264": {"preset": "medium"},
            "libx265": {"preset": "medium", "x265-params": "log-level=error:pools=none"},
            "libvpx-vp9": {"deadline": "good", "cpu-used": "4"},
            "libsvtav1": {"preset": "8"},
            "prores_ks": {"profile": "4" if "444" in pixel_format else "3"},
            "ffv1": {"level": "3"},
        }.get(selection.codec, encoding.hardware_options(selection.device))
        if format not in FIXED_RATE_FORMATS:
            stream.bit_rate = selection.bitrate
            if not bitrate_mbps and selection.device == "cpu":
                stream.options = {**stream.options, "crf": "23" if format != "av1-webm" else "30"}
        if format == "h265-mp4" and container_format in ("mp4", "mov"):
            stream.codec_context.codec_tag = "hvc1"
        if selection.device != "cpu":
            # Open only the hardware context here, before input conversion/audio:
            # Auto retries device initialization failures, never data or I/O errors.
            try:
                stream.codec_context.time_base = 1 / rate
                stream.codec_context.open()
            except av.FFmpegError as error:
                raise encoding.EncoderInitializationError(str(error)) from error
        sound = None
        resampler = None
        if audio_data is not None:
            waveform, sample_rate, total_samples = audio_data
            audio_codec = ("aac" if container_format == "mp4" else "libopus" if container_format == "webm"
                           else config[3])
            supported_rates = av.Codec(audio_codec, "w").audio_rates
            target_rate = 48000 if audio_codec == "libopus" or (
                supported_rates and sample_rate not in supported_rates) else sample_rate
            layout = "mono" if waveform.shape[0] == 1 else "stereo"
            sound = container.add_stream(audio_codec, rate=target_rate)
            sound.layout = layout
            if audio_codec in ("aac", "libopus"):
                sound.bit_rate = 192000
            resampler = av.AudioResampler(format=sound.codec_context.format.name, layout=layout, rate=target_rate)
            audio_cursor = 0

        def mux_audio(until):
            nonlocal audio_cursor
            if sound is None:
                return
            until = min(total_samples, until)
            while audio_cursor < until:
                _interrupt()
                count = min(4096, until - audio_cursor)
                values = np.zeros((waveform.shape[0], count), dtype=np.float32)
                available = min(count, max(0, waveform.shape[-1] - audio_cursor))
                if available:
                    values[:, :available] = waveform[:, audio_cursor:audio_cursor + available].numpy()
                frame = av.AudioFrame.from_ndarray(values, format="fltp", layout=layout)
                frame.sample_rate, frame.pts, frame.time_base = sample_rate, audio_cursor, Fraction(1, sample_rate)
                for converted in resampler.resample(frame):
                    for packet in sound.encode(converted):
                        container.mux(packet)
                audio_cursor += count

        for index, image in enumerate(images):
            _interrupt()
            high_depth = "10" in pixel_format or "16" in pixel_format or "64" in pixel_format or pixel_format == "p010le"
            keep_alpha = pixel_format in ("bgra", "rgba64le")
            pixels = _frame_pixels(image, 16 if high_depth else 8, keep_alpha)
            if (width, height) != (source_w, source_h):
                pixels = np.pad(pixels, ((0, height-source_h), (0, width-source_w), (0, 0)), mode="edge")
            input_format = ("rgba64le" if pixels.shape[-1] == 4 else "rgb48le") if high_depth else (
                "rgba" if pixels.shape[-1] == 4 else "rgb24")
            frame = av.VideoFrame.from_ndarray(pixels, format=input_format)
            frame = frame.reformat(format=pixel_format, src_colorspace="ITU709", dst_colorspace="ITU709")
            frame.pts, frame.time_base = index, 1 / rate
            for packet in stream.encode(frame):
                container.mux(packet)
            mux_audio(round((index + 1) / float(rate) * sample_rate) if sound else 0)
            progress.update(1)
        for packet in stream.encode():
            container.mux(packet)
        if sound is not None:
            mux_audio(total_samples)
            for converted in resampler.resample(None):
                for packet in sound.encode(converted): container.mux(packet)
            for packet in sound.encode(): container.mux(packet)
    return width, height


def _save_animation(path, images, rate, format, metadata):
    from PIL import Image
    from comfy.utils import ProgressBar
    # GIF stores delays in 10ms units; WebP stores delays in 1ms units.
    unit = 10 if format == "gif" else 1
    if float(rate) > 1000 / unit:
        raise ValueError(f"{format.upper()} cannot represent FPS above {1000 / unit:g}.")
    progress = ProgressBar(len(images))
    frames = []
    try:
        for image in images:
            _interrupt()
            frame = Image.fromarray(_frame_pixels(image, alpha=format == "webp"))
            frames.append(frame)
            progress.update(1)
        # Cumulative rounding prevents per-frame rounding from drifting over time.
        ends = [round(index * 1000 / float(rate) / unit) * unit for index in range(len(frames) + 1)]
        delays = [ends[i+1] - ends[i] for i in range(len(frames))]
        options = {"format": format.upper(), "save_all": True, "append_images": frames[1:],
                   "duration": delays, "loop": 0}
        if format == "gif":
            options.update(disposal=2)
            if metadata: options["comment"] = json.dumps(metadata, ensure_ascii=False).encode()
        else:
            options.update(lossless=True, method=4)
        frames[0].save(path, **options)
    finally:
        for frame in frames: frame.close()
    return images.shape[2], images.shape[1]


def _destination(prefix, root, width, height, extension):
    directory, base, counter, subfolder, _ = folder_paths.get_save_image_path(prefix, root, width, height)
    if os.path.commonpath((os.path.realpath(root), os.path.realpath(directory))) != os.path.realpath(root):
        raise ValueError("Video filename prefix must stay inside the selected ComfyUI folder.")
    os.makedirs(directory, exist_ok=True)
    filename = f"{base}_{counter:05}_.{extension}"
    while os.path.exists(os.path.join(directory, filename)):
        counter += 1
        filename = f"{base}_{counter:05}_.{extension}"
    return os.path.join(directory, filename), {"filename": filename, "subfolder": subfolder,
        "type": "output" if root == folder_paths.get_output_directory() else "temp"}


def save_media(images, audio=None, frame_rate=24.0, filename_prefix="DKST_Video", format="h264-mp4",
               pixel_format="auto", bitrate_mbps=8.0, always_save=False, metadata=None,
               container_override=None, encoder="cpu"):
    width, height = _shape(images)
    rate = frame_rate if isinstance(frame_rate, Fraction) else Fraction(str(_number(frame_rate, "Frame rate", .01, 1000))).limit_denominator(1000000)
    _number(rate, "Frame rate", .01, 1000)
    pixel = validate_encoding(format, pixel_format, bitrate_mbps, encoder)
    animated = FORMATS[format][2] is None
    if animated and audio is not None:
        raise ValueError("GIF and WebP cannot contain audio. Disconnect audio or choose a video format.")
    _interrupt()
    # Validate audio before selecting a hardware device, so invalid data never
    # triggers a device fallback. Full pixel checks remain incremental.
    if not animated: _audio_data(audio, len(images) / float(rate))
    selection = None if animated else encoding.select_encoder(
        FORMATS[format][2], pixel, encoder, bitrate_mbps, width, height, rate)
    extension = {"matroska": "mkv"}.get(container_override, container_override) or FORMATS[format][0]
    root = folder_paths.get_output_directory() if always_save else folder_paths.get_temp_directory()
    path, descriptor = _destination(filename_prefix, root, width, height, extension)
    partial = path + f".{uuid.uuid4().hex}.partial"
    encode_started = time.perf_counter()
    try:
        if animated:
            width, height = _save_animation(partial, images, rate, format, metadata)
        else:
            try:
                width, height = _encode(partial, images, audio, rate, format, pixel, bitrate_mbps,
                                        metadata, container_override, selection)
            except encoding.EncoderInitializationError as error:
                if encoder != "auto" or selection.device == "cpu": raise
                if os.path.exists(partial): os.remove(partial)
                selection = encoding.select_encoder(FORMATS[format][2], pixel, "cpu", bitrate_mbps, width, height, rate)
                from dataclasses import replace
                selection = replace(selection, reason=f"Hardware initialization failed: {error}")
                width, height = _encode(partial, images, audio, rate, format, pixel, bitrate_mbps,
                                        metadata, container_override, selection)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial): os.remove(partial)
    preview = descriptor
    info = selection.info() if selection else {"label": f"{format.upper()} · CPU", "encoder": "cpu", "reason": ""}
    info["seconds"] = round(time.perf_counter() - encode_started, 4)
    preview_info = info
    preview_seconds = 0.0
    if not animated and not (format == "h264-mp4" and selection.pixel in ("yuv420p", "nv12") and extension == "mp4"):
        # Reuse the same encoder to keep the preview synchronized with its audio.
        preview_started = time.perf_counter()
        proxy = save_media(images, audio, rate, f"preview_{filename_prefix}", "h264-mp4", "yuv420p", 0, False,
                           encoder="auto" if encoder != "cpu" else "cpu")
        preview_seconds = round(time.perf_counter() - preview_started, 4)
        preview = proxy["ui"]["dkst_video"][0]
        preview_info = proxy["ui"]["encoding"][0]
    return {"ui": {"dkst_video": [descriptor], "dkst_video_preview": [preview],
                   "resolution": [f"{width} × {height}"], "encoding": [info],
                   "preview_encoding": [preview_info], "preview_seconds": [preview_seconds]}, "result": (path,)}


class DINKI_Video_Combine:
    CATEGORY = "DINKIssTyle/Video"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "combine_video"
    OUTPUT_NODE = True
    DESCRIPTION = "Combine IMAGE frames and optional AUDIO into a video or animation using installed PyAV/Pillow codecs."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "DKST_Video"}),
            "format": format_options(),
            "frame_rate": ("FLOAT", {"default": 24.0, "min": .01, "max": 1000.0, "step": .01}),
            **encoding_inputs(),
            "always_save": ("BOOLEAN", {"default": False}),
        }, "optional": {"audio": ("AUDIO",), "encoder": encoder_input()},
           "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    @classmethod
    def VALIDATE_INPUTS(cls, format, pixel_format="yuv420p", bitrate_mbps=8.0, encoder="cpu", **kwargs):
        try:
            validate_encoding(format, pixel_format, bitrate_mbps, encoder)
        except (ValueError, ImportError) as error:
            return str(error)
        return True

    def combine_video(self, images, filename_prefix="DKST_Video", format="h264-mp4", frame_rate=24.0,
                      pixel_format="yuv420p", bitrate_mbps=8.0, always_save=False, audio=None,
                      prompt=None, extra_pnginfo=None, encoder="cpu"):
        from comfy.cli_args import args
        metadata = None
        if not args.disable_metadata:
            metadata = dict(extra_pnginfo or {})
            if prompt is not None: metadata["prompt"] = prompt
        return save_media(images, audio, frame_rate, filename_prefix, format,
                          pixel_format, bitrate_mbps, always_save, metadata, encoder=encoder)
