"""Real-codec video loader tests; run in an environment with av/torch/numpy/Pillow/aiohttp."""
import asyncio
import importlib
import json
import math
import sys
import tempfile
import types
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

try:
    import av
    import numpy as np
    import torch
    import PIL
    import aiohttp
except ImportError as error:
    raise unittest.SkipTest("Video integration tests need av, torch, numpy, Pillow and aiohttp") from error

ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"


class VideoLoadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = tempfile.TemporaryDirectory()
        cls.input = Path(cls.storage.name) / "input"
        cls.temp = Path(cls.storage.name) / "temp"
        cls.input.mkdir(); cls.temp.mkdir()
        routes = types.SimpleNamespace(get=lambda path: lambda fn: fn, post=lambda path: lambda fn: fn)
        package = types.ModuleType("dkst_video_test")
        package.__path__ = [str(ROOT)]
        latest = types.SimpleNamespace(
            Types=types.SimpleNamespace(VideoComponents=lambda **kw: types.SimpleNamespace(**kw)),
            InputImpl=types.SimpleNamespace(VideoFromComponents=lambda components: components))
        cls.modules = patch.dict(sys.modules, {
            "folder_paths": types.SimpleNamespace(get_input_directory=lambda: str(cls.input),
                                                  get_temp_directory=lambda: str(cls.temp)),
            "server": types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes))),
            "dkst_video_test": package,
            "comfy": types.ModuleType("comfy"),
            "comfy.utils": types.SimpleNamespace(ProgressBar=lambda count: types.SimpleNamespace(update=lambda n: None)),
            "comfy.model_management": types.SimpleNamespace(throw_exception_if_processing_interrupted=lambda: None),
            "comfy_api": types.ModuleType("comfy_api"), "comfy_api.latest": latest,
        })
        cls.modules.start()
        cls.module = importlib.import_module("dkst_video_test.dinki_video_load")
        cls.write_video(cls.input / "source.mp4", Fraction(10), [i / 10 for i in range(20)], audio=True)
        (cls.input / "clips").mkdir()
        cls.write_video(cls.input / "clips" / "variable.mkv", Fraction(10), [0, .1, .4, .5, .8, 1.0])
        cls.write_video(cls.input / "fractional.mp4", Fraction(30000, 1001), [i * 1001 / 30000 for i in range(30)])
        cls.write_video(cls.input / "rotated.mp4", Fraction(10), [0, .1], rotation=90)

    @classmethod
    def tearDownClass(cls):
        cls.modules.stop()
        cls.storage.cleanup()

    @staticmethod
    def write_video(path, rate, times, audio=False, rotation=0):
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264" if path.suffix == ".mp4" else "ffv1", rate=rate)
            stream.width, stream.height = 64, 48
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, 30000)
            if rotation:
                stream.set_display_rotation(rotation)
            if audio:
                sound = container.add_stream("aac", rate=48000)
                sound.layout = "mono"
            for index, time in enumerate(times):
                pixels = np.full((48, 64, 3), min(index * 10, 250), dtype=np.uint8)
                # Right half has a distinct color to verify crop geometry.
                pixels[:, 32:, 0] = 240
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts, frame.time_base = round(time * 30000), Fraction(1, 30000)
                for packet in stream.encode(frame): container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
            if audio:
                for offset in range(0, 96000, 1024):
                    t = np.arange(offset, min(offset + 1024, 96000)) / 48000
                    data = (0.3 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)[None, :]
                    frame = av.AudioFrame.from_ndarray(data, format="fltp", layout="mono")
                    frame.sample_rate, frame.pts, frame.time_base = 48000, offset, Fraction(1, 48000)
                    for packet in sound.encode(frame): container.mux(packet)
                for packet in sound.encode(): container.mux(packet)

    def load(self, **kwargs):
        return self.module.DINKI_Video_Load_Crop().load_and_crop(
            "", "source.mp4", megapixels=0.1, **kwargs)["result"]

    def test_categories_files_and_path_containment(self):
        self.assertEqual(self.module._video_categories(), ["", "clips"])
        self.assertEqual(self.module._video_files("clips"), ["variable.mkv"])
        for category, name in [("../", "source.mp4"), ("", "../source.mp4"), ("", "missing.mov")]:
            with self.assertRaises(ValueError): self.module._video_path(category, name)
        outside = self.input.parent / "outside.mp4"
        outside.write_bytes(b"fake")
        link = self.input / "escape.mp4"
        link.symlink_to(outside)
        try:
            self.assertNotIn("escape.mp4", self.module._video_files())
            with self.assertRaises(ValueError): self.module._video_path("", "escape.mp4")
        finally: link.unlink()

    def test_trim_downsample_and_audio_are_same_duration(self):
        video, images, fps, count, duration = self.load(trim_in=.5, trim_out=1.5,
                                                       fps_mode="Custom", output_fps=5)
        self.assertEqual((fps, count, duration), (5, 5, 1))
        self.assertEqual(images.shape[0], 5)
        self.assertIs(video.images, images)
        self.assertEqual(video.frame_rate, Fraction(5))
        self.assertEqual(video.audio["waveform"].shape, (1, 1, 48000))
        # Input index 5 is held at t=.5; index 13 covers t=1.3.
        self.assertAlmostEqual(float(images[0, :, :, 1].mean()), 50 / 255, delta=.015)
        self.assertAlmostEqual(float(images[-1, :, :, 1].mean()), 130 / 255, delta=.015)
        audio = video.audio["waveform"][0, 0].numpy()
        ideal = .3 * np.sin(2 * math.pi * 440 * (.5 + np.arange(48000) / 48000))
        self.assertGreater(np.corrcoef(audio[1000:-1000], ideal[1000:-1000])[0, 1], .99)

    def test_upsample_and_frame_covering_in(self):
        _, images, fps, count, duration = self.load(trim_in=.55, trim_out=.75,
                                                   fps_mode="Custom", output_fps=20)
        self.assertEqual((fps, count), (20, 4))
        self.assertAlmostEqual(duration, .2)
        self.assertAlmostEqual(float(images[0, :, :, 1].mean()), 50 / 255, delta=.015)
        self.assertAlmostEqual(float(images[-1, :, :, 1].mean()), 70 / 255, delta=.015)

    def test_crop_megapixels_and_multiple(self):
        _, images, _, _, _ = self.load(trim_out=.1, aspect_ratio="1:1",
                                      crop_x=.5, crop_width=.5, crop_height=2/3,
                                      resolution_multiple=32)
        self.assertEqual(images.shape[1:3], (320, 320))
        self.assertGreater(float(images[..., 0].mean()), .9)

    def test_original_fractional_fps_and_out_zero(self):
        result = self.module.DINKI_Video_Load_Crop().load_and_crop(
            "", "fractional.mp4", megapixels=.1)
        video, images, fps, count, duration = result["result"]
        self.assertEqual(video.frame_rate, Fraction(30000, 1001))
        self.assertEqual(count, 30)
        self.assertAlmostEqual(duration, 1.001)
        self.assertIsNone(video.audio)

    def test_vfr_uses_timestamps(self):
        path = str(self.input / "clips" / "variable.mkv")
        meta = self.module._probe(path)
        frames = list(self.module._sample_frames(path, meta, 0, 1, Fraction(10)))
        values = [int(frame.to_ndarray(format="rgb24")[0, 0, 1]) for frame in frames]
        self.assertEqual(values, [0, 10, 10, 10, 20, 30, 30, 30, 40, 40])

    def test_rotation_matches_display_geometry_and_pixels(self):
        path = str(self.input / "rotated.mp4")
        meta = self.module._probe(path)
        self.assertEqual((meta["width"], meta["height"]), (48, 64))
        with av.open(path) as container:
            frame = next(container.decode(video=0))
            expected = np.rot90(frame.to_ndarray(format="rgb24"), k=round(frame.rotation / 90))
            np.testing.assert_array_equal(self.module._pixels(frame, meta["rotation"]), expected)

    def test_invalid_parameters_and_exclusive_out(self):
        for values in [{"trim_in": 2}, {"trim_in": 1, "trim_out": 1}, {"trim_in": float("nan")},
                       {"fps_mode": "Custom", "output_fps": 0}, {"fps_mode": "Custom", "output_fps": 241},
                       {"megapixels": float("inf")}, {"crop_width": 0}]:
            options = {"megapixels": .1, **values}
            with self.assertRaises(ValueError):
                self.module.DINKI_Video_Load_Crop().load_and_crop("", "source.mp4", **options)
        _, images, _, count, _ = self.load(trim_out=.2)
        self.assertEqual(count, 2)
        self.assertLess(float(images[-1, :, :, 1].mean()), .06)

    def test_proxy_is_playable_and_cached(self):
        path = str(self.input / "clips" / "variable.mkv")
        meta = self.module._probe(path)
        descriptor = self.module._create_preview(path, meta)
        output = self.temp / descriptor["filename"]
        stamp = output.stat().st_mtime_ns
        with av.open(str(output)) as container:
            self.assertEqual(container.streams.video[0].codec_context.name, "h264")
            self.assertGreater(len(list(container.decode(video=0))), 0)
        self.assertEqual(self.module._create_preview(path, meta), descriptor)
        self.assertEqual(output.stat().st_mtime_ns, stamp)

    def test_metadata_route_does_not_expose_absolute_path(self):
        request = types.SimpleNamespace(query={"category": "", "filename": "source.mp4"})
        response = asyncio.run(self.module.video_metadata(request))
        data = json.loads(response.text)
        self.assertEqual(response.status, 200)
        self.assertTrue(data["has_audio"])
        self.assertNotIn(str(self.input), response.text)
        self.assertNotIn("origin", data)

    def test_processing_can_be_interrupted(self):
        path = str(self.input / "source.mp4")
        def interrupt(): raise RuntimeError("cancelled")
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            list(self.module._sample_frames(path, self.module._probe(path), 0, 1, Fraction(10), interrupt))


if __name__ == "__main__": unittest.main()
