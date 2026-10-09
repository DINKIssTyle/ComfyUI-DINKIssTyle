"""Hardware encoder discovery and isolated, bounded initialization probes.

This file can run as a subprocess without importing ComfyUI or torch.
"""
import importlib.util
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, replace
from functools import lru_cache
from fractions import Fraction

ENCODERS = ["auto", "cpu", "nvenc", "videotoolbox"]
HARDWARE = {
    "libx264": {"nvenc": "h264_nvenc", "videotoolbox": "h264_videotoolbox"},
    "libx265": {"nvenc": "hevc_nvenc", "videotoolbox": "hevc_videotoolbox"},
    "libsvtav1": {"nvenc": "av1_nvenc"},
}
PIXELS = ["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le",
          "yuv444p10le", "bgra", "rgba64le", "nv12", "p010le"]
LABELS = {"cpu": "CPU", "nvenc": "NVENC", "videotoolbox": "VideoToolbox"}
# 128px is below the NVENC minimum on newer NVIDIA GPUs. Use the same
# compatible canvas in both engines; this does not resize the user's video.
PROBE_WIDTH = PROBE_HEIGHT = 256
_probe_cache = {}
_probe_lock = threading.Lock()
logger = logging.getLogger(__name__)


class EncoderInitializationError(RuntimeError):
    """A device could not start encoding; Auto may choose software instead."""


@dataclass(frozen=True)
class Selection:
    codec: str
    pixel: str
    device: str = "cpu"
    engine: str = "pyav"
    executable: str = ""
    reason: str = ""
    bitrate: int = 0

    @property
    def label(self):
        name = {"libx264": "H.264", "h264_nvenc": "H.264", "h264_videotoolbox": "H.264",
                "libx265": "HEVC", "hevc_nvenc": "HEVC", "hevc_videotoolbox": "HEVC",
                "libsvtav1": "AV1", "av1_nvenc": "AV1", "libvpx-vp9": "VP9",
                "prores_ks": "ProRes", "ffv1": "FFV1"}.get(self.codec, self.codec)
        return f"{name} · {LABELS[self.device]}"

    def info(self):
        return {"label": self.label, "codec": self.codec, "encoder": self.device,
                "engine": self.engine, "pixel_format": self.pixel,
                "bitrate_mbps": self.bitrate / 1_000_000, "reason": self.reason}


def hardware_options(device):
    if device == "videotoolbox":
        return {"allow_sw": "0"}
    if device == "nvenc":
        return {"preset": "p4", "rc": "vbr"}
    return {}


@lru_cache(maxsize=1)
def ffmpeg_runtime():
    """Find an existing executable, never install or download one."""
    candidates = [os.environ.get("DKST_FFMPEG_PATH"), os.environ.get("VHS_FORCE_FFMPEG_PATH"),
                  shutil.which("ffmpeg")]
    if importlib.util.find_spec("imageio_ffmpeg"):
        try:
            import imageio_ffmpeg
            candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
        except (ImportError, RuntimeError, OSError):
            pass
    for path in dict.fromkeys(candidate for candidate in candidates if candidate):
        try:
            version = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=5, check=True)
            listing = subprocess.run([path, "-hide_banner", "-encoders"], capture_output=True,
                                     text=True, timeout=5, check=True)
            codecs = {match[1] for match in re.finditer(r"^\s*V\S{5}\s+(\S+)", listing.stdout, re.M)}
            return path, version.stdout.splitlines()[0], codecs
        except (OSError, subprocess.SubprocessError, IndexError):
            continue
    return "", "", set()


@lru_cache(maxsize=32)
def registered_pixels(codec, engine="pyav"):
    try:
        if engine == "pyav":
            import av
            return {entry.name for entry in av.Codec(codec, "w").video_formats or []}
        path, _, codecs = ffmpeg_runtime()
        if codec not in codecs:
            return set()
        result = subprocess.run([path, "-hide_banner", "-h", f"encoder={codec}"],
                                capture_output=True, text=True, timeout=5, check=True)
        match = re.search(r"Supported pixel formats:\s*([^\r\n]+)", result.stdout + result.stderr)
        return set(match[1].split()) if match else set()
    except (ImportError, ValueError, OSError, subprocess.SubprocessError):
        return set()


def mapped_pixel(pixel, supported):
    if pixel in supported:
        return pixel
    # Identical bit depth and chroma sampling; only the plane layout changes.
    alternatives = {"yuv420p": "nv12", "nv12": "yuv420p",
                    "yuv420p10le": "p010le", "p010le": "yuv420p10le"}
    alternative = alternatives.get(pixel)
    return alternative if alternative in supported else None


def pixel_choices(cpu_codec, default_pixel, encoder="auto"):
    if cpu_codec is None:
        return ["auto"]
    supported = set()
    if encoder in ("auto", "cpu"):
        supported.update(registered_pixels(cpu_codec))
    for device, codec in HARDWARE.get(cpu_codec, {}).items():
        if encoder not in ("auto", device):
            continue
        for engine in ("pyav", "ffmpeg"):
            supported.update(registered_pixels(codec, engine))
    choices = [value for value in PIXELS if mapped_pixel(value, supported)]
    return ["auto", *choices] if choices else []


def candidates(cpu_codec, pixel, encoder):
    devices = ("videotoolbox", "nvenc") if platform.system() == "Darwin" else ("nvenc",)
    if encoder != "auto":
        devices = (encoder,)
    for device in devices:
        codec = HARDWARE.get(cpu_codec, {}).get(device)
        if codec is None:
            continue
        for engine in ("pyav", "ffmpeg"):
            actual_pixel = mapped_pixel(pixel, registered_pixels(codec, engine))
            if actual_pixel:
                path = ffmpeg_runtime()[0] if engine == "ffmpeg" else ""
                yield Selection(codec, actual_pixel, device, engine, path)


def probe(selection):
    """Cache a real 256px encode+flush by runtime, codec and pixel format."""
    import av
    runtime = ffmpeg_runtime()[1] if selection.engine == "ffmpeg" else (
        av.__version__, repr(av.library_versions))
    key = (selection.engine, selection.executable, runtime, selection.codec, selection.pixel,
           PROBE_WIDTH, PROBE_HEIGHT)
    with _probe_lock:
        if key in _probe_cache:
            return _probe_cache[key]
        payload = {"engine": selection.engine, "executable": selection.executable,
                   "codec": selection.codec, "pixel": selection.pixel, "device": selection.device}
        try:
            result = subprocess.run([sys.executable, os.path.abspath(__file__), "--probe", json.dumps(payload)],
                                    capture_output=True, text=True, timeout=12)
            value = json.loads(result.stdout)
            if result.returncode != 0:
                value = (False, value.get("error", "Encoder probe failed"))
            else:
                value = (True, "")
        except subprocess.TimeoutExpired:
            value = (False, "Hardware encoder initialization timed out")
        except (OSError, ValueError):
            value = (False, "Hardware encoder probe failed")
        _probe_cache[key] = value
        return value


def select_encoder(cpu_codec, pixel, encoder, bitrate_mbps, width, height, rate):
    if encoder not in ENCODERS:
        raise ValueError(f"Unknown encoder: {encoder}")
    requested_bitrate = round(float(bitrate_mbps) * 1_000_000)
    cpu_pixel = mapped_pixel(pixel, registered_pixels(cpu_codec))
    cpu = Selection(cpu_codec, cpu_pixel or pixel, bitrate=requested_bitrate)
    if encoder == "cpu":
        if cpu_pixel is None:
            raise ValueError(f"CPU encoder {cpu_codec} does not support {pixel}.")
        return cpu
    if encoder == "auto" and cpu_codec not in HARDWARE:
        return cpu
    errors = []
    for selection in candidates(cpu_codec, pixel, encoder):
        ok, error = probe(selection)
        if ok:
            # Hardware encoders do not share CPU CRF semantics. Auto bitrate uses
            # a resolution/FPS target and reports the applied value to the UI.
            bitrate = requested_bitrate or round(min(1e9, max(500000, width * height * float(rate) * .15)))
            import av
            runtime = ffmpeg_runtime()[1] if selection.engine == "ffmpeg" else f"PyAV {av.__version__} {av.library_versions}"
            logger.info("DKST video encoder: %s via %s; %s; executable=%s; target=%s Mbps",
                        selection.codec, selection.engine, runtime, selection.executable or "embedded", bitrate / 1e6)
            return replace(selection, bitrate=bitrate)
        errors.append(f"{selection.codec} ({selection.engine}): {error}")
    reason = "; ".join(errors) or f"No usable hardware encoder for {cpu_codec} / {pixel}"
    if encoder != "auto":
        raise EncoderInitializationError(reason)
    if cpu_pixel is None:
        raise EncoderInitializationError(reason + f"; CPU cannot preserve {pixel} either")
    return replace(cpu, reason=reason)


def _probe_main(config):
    import av
    import tempfile
    device = config["device"]
    if config["engine"] == "pyav":
        with tempfile.TemporaryFile() as target:
            with av.open(target, "w", format="matroska") as container:
                stream = container.add_stream(config["codec"], rate=30)
                stream.width, stream.height = PROBE_WIDTH, PROBE_HEIGHT
                stream.pix_fmt = config["pixel"]
                stream.bit_rate = 1_000_000
                stream.options = hardware_options(device)
                for index in range(3):
                    frame = av.VideoFrame(PROBE_WIDTH, PROBE_HEIGHT, config["pixel"])
                    for plane in frame.planes:
                        plane.update(bytes(plane.buffer_size))
                    frame.pts, frame.time_base = index, Fraction(1, 30)
                    for packet in stream.encode(frame): container.mux(packet)
                for packet in stream.encode(): container.mux(packet)
    else:
        args = [config["executable"], "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{PROBE_WIDTH}x{PROBE_HEIGHT}", "-r", "30", "-i", "-", "-frames:v", "3",
                "-c:v", config["codec"], "-pix_fmt", config["pixel"], "-b:v", "1M"]
        for key, value in hardware_options(device).items(): args.extend(["-" + key, value])
        result = subprocess.run(args + ["-f", "null", "-"], input=bytes(PROBE_WIDTH * PROBE_HEIGHT * 3 * 3),
                                capture_output=True, timeout=10)
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", "replace")[-2000:])


if __name__ == "__main__":
    try:
        _probe_main(json.loads(sys.argv[2]))
        print(json.dumps({"ok": True}))
    except Exception as error:
        print(json.dumps({"error": str(error)}))
        sys.exit(1)
