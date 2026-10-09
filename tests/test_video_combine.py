"""Exercise real PyAV/Pillow encoders without a running ComfyUI instance."""
import importlib
import json
import math
import os
import sys
import tempfile
import time
import types
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

try:
    import av
    import numpy as np
    import torch
    from PIL import Image
except ImportError as error:
    raise unittest.SkipTest("Video combine integration tests need av, torch, numpy and Pillow") from error

ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"


class VideoCombineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = tempfile.TemporaryDirectory()
        cls.input = Path(cls.storage.name) / "input"; cls.input.mkdir()
        cls.temp = Path(cls.storage.name) / "temp"; cls.temp.mkdir()
        cls.output = Path(cls.storage.name) / "output"; cls.output.mkdir()
        package = types.ModuleType("dkst_combine_test"); package.__path__ = [str(ROOT)]
        def save_path(prefix, directory, width, height):
            subfolder, _, name = prefix.replace("\\", "/").rpartition("/")
            return os.path.join(directory, subfolder), name, 1, subfolder, prefix
        cls.args = types.SimpleNamespace(disable_metadata=False)
        cls.patches = patch.dict(sys.modules, {
            "folder_paths": types.SimpleNamespace(get_output_directory=lambda: str(cls.output),
                get_temp_directory=lambda: str(cls.temp), get_save_image_path=save_path),
            "dkst_combine_test": package,
            "comfy": types.ModuleType("comfy"),
            "comfy.cli_args": types.SimpleNamespace(args=cls.args),
            "comfy.utils": types.SimpleNamespace(ProgressBar=lambda count: types.SimpleNamespace(update=lambda n: None)),
            "comfy.model_management": types.SimpleNamespace(throw_exception_if_processing_interrupted=lambda: None),
            "comfy_api": types.ModuleType("comfy_api"),
            "comfy_api.latest": types.SimpleNamespace(Types=types.SimpleNamespace()),
        }); cls.patches.start()
        cls.module = importlib.import_module("dkst_combine_test.dinki_video_combine")
        cls.viewer = importlib.import_module("dkst_combine_test.dinki_viewer")

    @classmethod
    def tearDownClass(cls):
        cls.patches.stop(); cls.storage.cleanup()

    def images(self, width=64, height=64, channels=3):
        images = torch.zeros((6, height, width, channels))
        for i in range(6): images[i, ..., :3] = (i + 1) / 10
        if channels == 4: images[..., 3] = .5
        return images

    def audio(self, seconds=.25):
        t = torch.arange(round(48000 * seconds)) / 48000
        return {"waveform": (.2 * torch.sin(2 * math.pi * 440 * t)).reshape(1, 1, -1), "sample_rate": 48000}

    def test_requested_schema_and_filename_output(self):
        cls = self.module.DINKI_Video_Combine
        schema = cls.INPUT_TYPES()
        self.assertEqual(schema["required"]["images"][0], "IMAGE")
        self.assertEqual(schema["optional"]["audio"][0], "AUDIO")
        self.assertEqual(schema["required"]["filename_prefix"][1]["default"], "DKST_Video")
        self.assertEqual(schema["required"]["pixel_format"][1]["default"], "yuv420p")
        self.assertFalse(schema["required"]["always_save"][1]["default"])
        self.assertEqual(cls.RETURN_TYPES, ("STRING",))
        self.assertEqual(cls.RETURN_NAMES, ("filename",))

    def test_real_video_codecs_audio_dimensions_and_preview(self):
        for format in self.module.available_formats():
            if format in ("gif", "webp"): continue
            with self.subTest(format=format):
                result = self.module.save_media(self.images(), self.audio(), 12, "codec/"+format,
                                                format, "auto", 2, True, {"prompt": {"test": True}})
                path = result["result"][0]
                self.assertTrue(Path(path).is_file())
                self.assertEqual(result["ui"]["dkst_video"][0]["type"], "output")
                self.assertEqual(result["ui"]["resolution"], ["64 × 64"])
                with av.open(path) as container:
                    expected = {"h264-mp4":"h264", "h265-mp4":"hevc", "vp9-webm":"vp9",
                                "av1-webm":"av1", "prores-mov":"prores", "ffv1-mkv":"ffv1"}[format]
                    self.assertEqual(container.streams.video[0].codec_context.codec.canonical_name, expected)
                    self.assertTrue(container.streams.audio)
                    frames = list(container.decode(video=0))
                    self.assertEqual(len(frames), 6)
                    self.assertAlmostEqual(float(container.streams.video[0].average_rate), 12, places=2)
                with av.open(path) as container:
                    samples = list(container.decode(audio=0))
                    self.assertGreater(sum(frame.samples for frame in samples), 20000)
                preview = result["ui"]["dkst_video_preview"][0]
                preview_path = (self.temp if preview["type"] == "temp" else self.output) / preview["subfolder"] / preview["filename"]
                with av.open(str(preview_path)) as container:
                    self.assertEqual(container.streams.video[0].codec_context.name, "h264")
                    self.assertTrue(container.streams.audio)

    def test_fractional_rate_pixel_format_padding_and_metadata(self):
        result = self.module.save_media(self.images(63, 47), None, Fraction(30000, 1001),
                                        "fractional", pixel_format="yuv420p", bitrate_mbps=4,
                                        metadata={"prompt":{"a":1}, "workflow":{"nodes":[]}})
        self.assertEqual(result["ui"]["resolution"], ["64 × 48"])
        self.assertEqual(result["ui"]["dkst_video_preview"], result["ui"]["dkst_video"])
        with av.open(result["result"][0]) as container:
            stream = container.streams.video[0]
            self.assertEqual(stream.average_rate, Fraction(30000, 1001))
            self.assertEqual(stream.format.name, "yuv420p")
            self.assertEqual(json.loads(container.metadata["prompt"]), {"a":1})
            frames = list(container.decode(video=0))
            self.assertEqual(len(frames), 6)
            self.assertEqual(frames[0].to_ndarray(format="rgb24").shape, (48, 64, 3))

    def test_ten_bit_encoding_preserves_more_than_eight_bit_levels(self):
        images = self.images()
        images[:, :, :, 0] = torch.linspace(.2, .21, 64).reshape(1, 1, 64)
        result = self.module.save_media(images, frame_rate=12, filename_prefix="ten_bit",
                                        pixel_format="yuv444p10le", bitrate_mbps=0)
        with av.open(result["result"][0]) as container:
            self.assertEqual(container.streams.video[0].format.name, "yuv444p10le")
            frame = next(container.decode(video=0))
            red = frame.to_ndarray(format="rgb48le")[0, :, 0]
            self.assertGreater(len(np.unique(red)), 4)

    def test_audio_is_trimmed_or_silence_padded_to_video_duration(self):
        for source_samples in (4410, 44100):
            audio = {"waveform": torch.full((1, 2, source_samples), .2), "sample_rate": 44100}
            result = self.module.save_media(self.images(), audio, 12, "audio_length",
                                            "ffv1-mkv", "auto")
            with av.open(result["result"][0]) as container:
                frames = list(container.decode(audio=0))
                resampler = av.AudioResampler(format="fltp", layout="stereo", rate=44100)
                planar = [converted.to_ndarray() for frame in frames for converted in resampler.resample(frame)]
                planar.extend(frame.to_ndarray() for frame in resampler.resample(None))
                samples = np.concatenate(planar, axis=1)
                self.assertEqual(samples.shape, (2, 22050))
                self.assertGreater(np.abs(samples[:, :source_samples]).mean(), 0)
                if source_samples < 22050:
                    self.assertEqual(np.abs(samples[:, source_samples:]).max(), 0)

    def test_lossless_rgb_export_retains_rgba_alpha_and_pixels(self):
        images = self.images(channels=4)
        expected = (images[0].numpy() * 255).round().astype(np.uint8)
        result = self.module.save_media(images, frame_rate=12, filename_prefix="rgba",
                                        format="ffv1-mkv", pixel_format="auto")
        with av.open(result["result"][0]) as container:
            actual = next(container.decode(video=0)).to_ndarray(format="rgba")
            np.testing.assert_array_equal(actual, expected)

    def test_aac_resamples_sample_rates_unsupported_by_encoder(self):
        audio = {"waveform": torch.zeros((1, 1, 10000)), "sample_rate": 10000}
        result = self.module.save_media(self.images(), audio, frame_rate=12, filename_prefix="resample")
        with av.open(result["result"][0]) as container:
            self.assertEqual(container.streams.audio[0].sample_rate, 48000)
            self.assertTrue(list(container.decode(audio=0)))

    def test_animation_rounds_cumulative_timing_and_uses_image_preview(self):
        for format in ("gif", "webp"):
            result = self.module.save_media(self.images(), frame_rate=24,
                filename_prefix="animated", format=format, pixel_format="auto", bitrate_mbps=0)
            self.assertEqual(result["ui"]["dkst_video_preview"], result["ui"]["dkst_video"])
            with Image.open(result["result"][0]) as image:
                self.assertEqual(image.n_frames, 6)
                total = 0
                for i in range(image.n_frames):
                    image.seek(i); image.load(); total += image.info.get("duration", 0)
                self.assertAlmostEqual(total, 250, delta=10)

    def test_unsupported_audio_and_invalid_inputs_do_not_create_files(self):
        for options in [{"format":"gif", "audio":self.audio()}, {"format":"webp", "audio":self.audio()},
                        {"format":"prores-mov", "pixel_format":"yuv420p"}, {"frame_rate":0},
                        {"bitrate_mbps":float("nan")}, {"pixel_format":"bogus"}]:
            with self.subTest(options=options.keys()):
                with self.assertRaises(ValueError):
                    self.module.save_media(self.images(), filename_prefix="invalid", **options)
        self.assertFalse(list(self.temp.glob("invalid*")))

    def test_cancelled_or_invalid_encoding_removes_partial_file(self):
        images = self.images(); images[3,0,0,0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.module.save_media(images, filename_prefix="bad")
        self.assertFalse(list(self.temp.glob("bad*")))
        with patch.object(self.module, "_interrupt", side_effect=RuntimeError("cancelled")):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self.module.save_media(self.images(), filename_prefix="cancelled")
        self.assertFalse(list(self.temp.glob("cancelled*")))

    def test_duplicate_prefix_does_not_overwrite_previous_export(self):
        first = self.module.save_media(self.images(), filename_prefix="repeat")["result"][0]
        original = Path(first).read_bytes()
        second = self.module.save_media(self.images(), filename_prefix="repeat")["result"][0]
        self.assertNotEqual(first, second)
        self.assertEqual(Path(first).read_bytes(), original)

    def test_combine_and_extended_player_use_shared_encoder(self):
        result = self.module.DINKI_Video_Combine().combine_video(self.images(), audio=self.audio(), frame_rate=12)
        self.assertTrue(Path(result["result"][0]).is_file())
        video = types.SimpleNamespace(get_components=lambda: types.SimpleNamespace(
            images=self.images(), audio=self.audio(), frame_rate=Fraction(12)))
        for format in ("h265-mp4", "prores-mov", "ffv1-mkv", "mp4", "mkv", "webm"):
            with self.subTest(format=format):
                result = self.viewer.DINKI_Video_Viewer().preview_video(video, format=format,
                    bitrate_mbps=2, pixel_format="auto")
                self.assertIs(result["result"][0], video)
                descriptor = result["ui"]["dkst_video"][0]
                self.assertTrue((self.temp / descriptor["subfolder"] / descriptor["filename"]).is_file())

    def test_metadata_disable_and_optional_viewer_controls(self):
        schema = self.viewer.DINKI_Video_Viewer.INPUT_TYPES()
        self.assertEqual(schema["optional"]["pixel_format"][1]["default"], "auto")
        self.assertEqual(schema["optional"]["bitrate_mbps"][1]["default"], 0)
        self.args.disable_metadata = True
        try:
            result = self.module.DINKI_Video_Combine().combine_video(self.images(), prompt={"private":"no"})
            with av.open(result["result"][0]) as container:
                self.assertNotIn("prompt", container.metadata)
        finally: self.args.disable_metadata = False

    def test_encoder_default_schema_and_legacy_api_cpu(self):
        for cls in (self.module.DINKI_Video_Combine, self.viewer.DINKI_Video_Viewer):
            self.assertEqual(cls.INPUT_TYPES()["optional"]["encoder"][1]["default"], "auto")
        result = self.module.DINKI_Video_Combine().combine_video(self.images())
        self.assertEqual(result["ui"]["encoding"][0]["encoder"], "cpu")
        self.assertGreaterEqual(result["ui"]["encoding"][0]["seconds"], 0)
        self.assertEqual(result["ui"]["preview_seconds"], [0])

    def test_actual_initialization_failure_retries_only_in_auto(self):
        backend = self.module.encoding
        hardware = backend.Selection("h264_videotoolbox", "yuv420p", "videotoolbox", bitrate=2_000_000)
        original = self.module._encode
        attempts = []
        def encode(*args):
            attempts.append(args[-1].device)
            if args[-1].device != "cpu":
                Path(args[0]).write_bytes(b"partial")
                raise backend.EncoderInitializationError("device busy")
            return original(*args)
        with patch.object(backend, "candidates", side_effect=lambda *args: iter([hardware])), \
             patch.object(backend, "probe", return_value=(True, "")), patch.object(self.module, "_encode", side_effect=encode):
            result = self.module.save_media(self.images(), encoder="auto", filename_prefix="device_retry")
        self.assertEqual(attempts, ["videotoolbox", "cpu"])
        self.assertIn("device busy", result["ui"]["encoding"][0]["reason"])
        self.assertFalse(list(self.temp.glob("device_retry*.partial")))

    def test_io_or_frame_errors_do_not_retry_cpu(self):
        backend = self.module.encoding
        hardware = backend.Selection("h264_videotoolbox", "yuv420p", "videotoolbox")
        for error in (PermissionError("disk"), ValueError("pixels"), RuntimeError("cancelled")):
            with patch.object(backend, "candidates", side_effect=lambda *args: iter([hardware])), \
                 patch.object(backend, "probe", return_value=(True, "")), \
                 patch.object(self.module, "_encode", side_effect=error) as encode:
                with self.assertRaises(type(error)):
                    self.module.save_media(self.images(), encoder="auto", filename_prefix="no_retry")
                self.assertEqual(encode.call_count, 1)
        self.assertFalse(list(self.temp.glob("no_retry*")))

    def test_ffmpeg_pipe_cancellation_stops_a_blocked_encoder(self):
        adapter = importlib.import_module("dkst_combine_test.dinki_video_ffmpeg")
        def cancel(): raise RuntimeError("cancelled")
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            adapter._run_pipe([sys.executable, "-c", "import time; time.sleep(10)"],
                              iter([bytes(10_000_000)]), cancel)
        self.assertLess(time.monotonic()-start, 3)

    def test_ffmpeg_pipe_propagates_input_error_without_waiting_forever(self):
        adapter = importlib.import_module("dkst_combine_test.dinki_video_ffmpeg")
        def bad_frames():
            yield b"frame"
            raise ValueError("invalid pixel data")
        with self.assertRaisesRegex(ValueError, "invalid pixel data"):
            adapter._run_pipe([sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
                              bad_frames(), lambda: None)

    @unittest.skipUnless(os.environ.get("DKST_RUN_HARDWARE_TESTS") == "1", "Hardware tests require an enabled device")
    def test_real_videotoolbox_audio_fractional_fps_and_ten_bit(self):
        backend = self.module.encoding
        if not backend.registered_pixels("h264_videotoolbox"):
            self.skipTest("VideoToolbox not registered")
        for format, pixel in (("h264-mp4", "yuv420p"), ("h265-mp4", "yuv420p"), ("h265-mp4", "yuv420p10le")):
            with self.subTest(format=format, pixel=pixel):
                result = self.module.save_media(self.images(128, 128), self.audio(), Fraction(30000, 1001),
                    "hardware", format, pixel, 2, encoder="videotoolbox", metadata={"prompt":{"hardware":True}})
                info = result["ui"]["encoding"][0]
                self.assertEqual(info["encoder"], "videotoolbox")
                self.assertEqual(info["engine"], "pyav")
                with av.open(result["result"][0]) as container:
                    stream = container.streams.video[0]
                    self.assertEqual(stream.average_rate, Fraction(30000, 1001))
                    self.assertEqual(len(list(container.decode(video=0))), 6)
                    self.assertEqual(stream.format.name, pixel)
                    self.assertEqual(json.loads(container.metadata["prompt"]), {"hardware":True})
                with av.open(result["result"][0]) as container:
                    self.assertTrue(list(container.decode(audio=0)))

    @unittest.skipUnless(os.environ.get("DKST_RUN_HARDWARE_TESTS") == "1", "Hardware tests require an enabled device")
    def test_real_existing_ffmpeg_hardware_adapter(self):
        backend = self.module.encoding
        if not backend.registered_pixels("hevc_videotoolbox", "ffmpeg"):
            self.skipTest("FFmpeg VideoToolbox not registered")
        candidates = backend.candidates
        def cli_only(*args):
            return (item for item in candidates(*args) if item.engine == "ffmpeg")
        with patch.object(backend, "candidates", side_effect=cli_only):
            result = self.module.save_media(self.images(128, 128), self.audio(), Fraction(30000, 1001),
                "ffmpeg_hardware", "h265-mp4", "yuv420p10le", 2, encoder="videotoolbox",
                metadata={"prompt":{"text":"한국어; # = 줄\n바꿈"}})
        info = result["ui"]["encoding"][0]
        self.assertEqual(info["engine"], "ffmpeg")
        self.assertEqual(info["pixel_format"], "p010le")
        with av.open(result["result"][0]) as container:
            self.assertEqual(container.streams.video[0].average_rate, Fraction(30000, 1001))
            self.assertEqual(container.streams.video[0].format.name, "yuv420p10le")
            self.assertEqual(json.loads(container.metadata["prompt"]), {"text":"한국어; # = 줄\n바꿈"})
            self.assertEqual(len(list(container.decode(video=0))), 6)
        with av.open(result["result"][0]) as container:
            self.assertTrue(list(container.decode(audio=0)))

    @unittest.skipUnless(os.environ.get("DKST_RUN_HARDWARE_TESTS") == "1", "Hardware tests require an enabled device")
    def test_real_hardware_benchmark_and_auto_player(self):
        backend = self.module.encoding
        if not backend.registered_pixels("h264_videotoolbox"):
            self.skipTest("VideoToolbox not registered")
        backend.select_encoder("libx264", "yuv420p", "videotoolbox", 8, 640, 360, 24)
        generator = torch.Generator().manual_seed(42)
        images = torch.rand((32, 360, 640, 3), generator=generator)
        report = []
        for encoder in ("cpu", "videotoolbox"):
            start_cpu = time.process_time()
            start = time.monotonic()
            result = self.module.save_media(images, self.audio(), 24, "benchmark_"+encoder,
                                            "h264-mp4", "yuv420p", 8, encoder=encoder)
            report.append({"encoder":encoder, "seconds":round(time.monotonic()-start, 3),
                           "cpu_seconds":round(time.process_time()-start_cpu, 3),
                           "bytes":Path(result["result"][0]).stat().st_size})
            with av.open(result["result"][0]) as container:
                stream = container.streams.video[0]
                self.assertEqual((stream.width, stream.height), (640, 360))
                self.assertEqual(len(list(container.decode(video=0))), 32)
                self.assertEqual(stream.average_rate, 24)
        print("\nHardware benchmark (32 noisy 640x360 frames, 24 FPS, 8 Mbps):", json.dumps(report))
        video = types.SimpleNamespace(get_components=lambda: types.SimpleNamespace(
            images=self.images(128, 128), audio=self.audio(), frame_rate=Fraction(24)))
        result = self.viewer.DINKI_Video_Viewer().preview_video(video, encoder="auto")
        self.assertIs(result["result"][0], video)
        self.assertEqual(result["ui"]["encoding"][0]["encoder"], "videotoolbox")

    @unittest.skipUnless(os.environ.get("DKST_RUN_HARDWARE_TESTS") == "1", "Hardware tests require an enabled device")
    def test_real_nvenc_when_device_is_available(self):
        backend = self.module.encoding
        if not list(backend.candidates("libx264", "yuv420p", "nvenc")):
            self.skipTest("NVENC hardware unavailable on this host")
        for format in ("h264-mp4", "h265-mp4", "av1-webm"):
            if not list(backend.candidates(self.module.FORMATS[format][2], "yuv420p", "nvenc")):
                continue
            result = self.module.save_media(self.images(256, 256), self.audio(), 24,
                                            "nvenc", format, "yuv420p", 2, encoder="nvenc")
            self.assertEqual(result["ui"]["encoding"][0]["encoder"], "nvenc")
            with av.open(result["result"][0]) as container:
                self.assertEqual((container.streams.video[0].width, container.streams.video[0].height), (256, 256))
                self.assertEqual(len(list(container.decode(video=0))), 6)


if __name__ == "__main__": unittest.main()
