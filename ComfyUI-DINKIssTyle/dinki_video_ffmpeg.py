"""Optional adapter for a hardware encoder in an already installed FFmpeg."""
import json
import os
import subprocess
import tempfile
import threading

from .dinki_video_encoding import EncoderInitializationError, hardware_options


def _run_pipe(args, chunks, interrupt):
    """Keep cancellation responsive even when FFmpeg stops reading stdin."""
    done = threading.Event()
    errors = []
    with tempfile.TemporaryFile() as diagnostics:
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=diagnostics)

        def write():
            try:
                for chunk in chunks:
                    process.stdin.write(chunk)
                process.stdin.close()
            except BaseException as error:
                errors.append(error)
            finally:
                try:
                    if not process.stdin.closed: process.stdin.close()
                except (OSError, ValueError):
                    pass
                done.set()

        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        try:
            while not done.wait(.05):
                interrupt()
                if process.poll() is not None: break
            while process.poll() is None:
                interrupt()
                try:
                    process.wait(timeout=.05)
                except subprocess.TimeoutExpired:
                    continue
            writer.join(timeout=2)
            diagnostics.seek(0)
            message = diagnostics.read().decode("utf-8", "replace")[-4000:]
            if errors and not isinstance(errors[0], (BrokenPipeError, OSError)):
                raise errors[0]
            if process.returncode or errors:
                raise RuntimeError("FFmpeg encoding failed: " + message)
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            writer.join(timeout=2)
            if not process.stdin.closed: process.stdin.close()


def _metadata_file(path, metadata):
    def escape(value):
        return value.replace("\\", "\\\\").replace("\n", "\\\n").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#")
    with open(path, "w", encoding="utf-8") as output:
        output.write(";FFMETADATA1\n")
        for key, value in (metadata or {}).items():
            value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            output.write(escape(key) + "=" + escape(value) + "\n")


def encode(path, images, audio, rate, config, selection, metadata, container_format,
           dimensions, pixels, audio_data, interrupt):
    import numpy as np
    from comfy.utils import ProgressBar
    width, height = dimensions
    source_h, source_w = images.shape[1:3]
    high_depth = "10" in selection.pixel or selection.pixel == "p010le"
    pixel_input = "rgb48le" if high_depth else "rgb24"
    progress = ProgressBar(len(images))
    with tempfile.TemporaryDirectory(prefix="dkst_encode_", dir=os.path.dirname(path)) as workspace:
        # Verify the actual resolution/FPS in a separate process before streaming
        # user frames. This makes initialization failures distinguishable from I/O.
        init = [selection.executable, "-v", "error", "-f", "lavfi", "-i",
                f"color=size={width}x{height}:rate={rate}", "-frames:v", "1", "-c:v", selection.codec,
                "-pix_fmt", selection.pixel, "-b:v", str(selection.bitrate)]
        for key, value in hardware_options(selection.device).items(): init.extend(["-" + key, value])
        try:
            result = subprocess.run(init + ["-f", "null", "-"], capture_output=True, timeout=12)
        except subprocess.TimeoutExpired as error:
            raise EncoderInitializationError("FFmpeg hardware initialization timed out") from error
        if result.returncode:
            raise EncoderInitializationError(result.stderr.decode("utf-8", "replace")[-2000:])
        interrupt()
        metadata_path = os.path.join(workspace, "metadata.txt")
        _metadata_file(metadata_path, metadata)
        silent = os.path.join(workspace, "video.partial")
        args = [selection.executable, "-v", "error", "-nostdin", "-n", "-f", "rawvideo",
                "-pix_fmt", pixel_input, "-color_range", "pc", "-colorspace", "rgb",
                "-color_primaries", "bt709", "-color_trc", "bt709",
                "-s", f"{width}x{height}", "-r", str(rate), "-i", "-",
                "-f", "ffmetadata", "-i", metadata_path, "-map", "0:v", "-map_metadata", "1",
                "-c:v", selection.codec, "-pix_fmt", selection.pixel, "-b:v", str(selection.bitrate),
                "-vf", "scale=out_color_matrix=bt709", "-color_range", "tv", "-colorspace", "bt709",
                "-color_primaries", "bt709", "-color_trc", "bt709"]
        for key, value in hardware_options(selection.device).items(): args.extend(["-" + key, value])
        if selection.codec.startswith("hevc_") and container_format in ("mov", "mp4"):
            args.extend(["-tag:v", "hvc1"])
        if container_format in ("mp4", "mov"):
            args.extend(["-movflags", "+faststart+use_metadata_tags"])
        args.extend(["-f", container_format, silent if audio is not None else path])

        def frames():
            for image in images:
                interrupt()
                frame = pixels(image, 16 if high_depth else 8)
                if (width, height) != (source_w, source_h):
                    frame = np.pad(frame, ((0, height-source_h), (0, width-source_w), (0, 0)), mode="edge")
                yield frame.tobytes()
                progress.update(1)

        _run_pipe(args, frames(), interrupt)
        if audio is not None:
            waveform, sample_rate, total_samples = audio_data
            audio_codec = "aac" if container_format == "mp4" else "libopus" if container_format == "webm" else config[3]
            mux = [selection.executable, "-v", "error", "-nostdin", "-n", "-i", silent,
                   "-f", "f32le", "-ar", str(sample_rate), "-ac", str(waveform.shape[0]), "-i", "-",
                   "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "0", "-c:v", "copy",
                   "-c:a", audio_codec]
            if audio_codec in ("aac", "libopus"):
                mux.extend(["-b:a", "192k", "-ar", "48000"])
            if container_format in ("mp4", "mov"):
                mux.extend(["-movflags", "+faststart+use_metadata_tags"])
            mux.extend(["-f", container_format, path])

            def samples():
                for index in range(0, total_samples, 4096):
                    interrupt()
                    count = min(4096, total_samples-index)
                    values = np.zeros((count, waveform.shape[0]), dtype=np.float32)
                    available = min(count, max(0, waveform.shape[-1]-index))
                    if available:
                        values[:available] = waveform[:, index:index+available].numpy().T
                    yield values.tobytes()

            _run_pipe(mux, samples(), interrupt)
    return width, height
