"""Selection, device failure and probe caching do not require a GPU."""
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

PATH = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle" / "dinki_video_encoding.py"
spec = importlib.util.spec_from_file_location("dkst_encoding_tests", PATH)
encoding = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = encoding
spec.loader.exec_module(encoding)


class EncoderSelectionTests(unittest.TestCase):
    def setUp(self):
        encoding._probe_cache.clear()
        modules = patch.dict(sys.modules, {"av": SimpleNamespace(__version__="test", library_versions={})})
        modules.start()
        self.addCleanup(modules.stop)

    def test_auto_selects_verified_hardware_and_reports_auto_bitrate(self):
        hardware = encoding.Selection("hevc_nvenc", "p010le", "nvenc")
        with patch.object(encoding, "registered_pixels", return_value={"yuv420p10le"}), \
             patch.object(encoding, "candidates", return_value=iter([hardware])), \
             patch.object(encoding, "probe", return_value=(True, "")):
            selected = encoding.select_encoder("libx265", "yuv420p10le", "auto", 0, 1920, 1080, 30)
        self.assertEqual(selected.device, "nvenc")
        self.assertEqual(selected.pixel, "p010le")
        self.assertGreater(selected.bitrate, 0)
        self.assertEqual(selected.label, "HEVC · NVENC")

    def test_auto_failure_preserves_ten_bit_and_reports_reason(self):
        hardware = encoding.Selection("hevc_nvenc", "p010le", "nvenc")
        with patch.object(encoding, "registered_pixels", return_value={"yuv420p10le"}), \
             patch.object(encoding, "candidates", return_value=iter([hardware])), \
             patch.object(encoding, "probe", return_value=(False, "No device")):
            selected = encoding.select_encoder("libx265", "p010le", "auto", 8, 1920, 1080, 30)
        self.assertEqual(selected.device, "cpu")
        self.assertEqual(selected.pixel, "yuv420p10le")
        self.assertIn("No device", selected.reason)

    def test_explicit_device_failure_never_silently_uses_cpu(self):
        with patch.object(encoding, "registered_pixels", return_value={"yuv420p"}), \
             patch.object(encoding, "candidates", return_value=iter([])):
            with self.assertRaises(encoding.EncoderInitializationError):
                encoding.select_encoder("libx264", "yuv420p", "nvenc", 8, 64, 64, 24)

    def test_auto_does_not_quantize_unsupported_pixel_formats(self):
        def registered(codec, engine="pyav"):
            return {"yuv444p10le"} if codec == "libx264" else {"yuv420p"}
        with patch.object(encoding, "registered_pixels", side_effect=registered), \
             patch.object(encoding, "probe") as probe:
            selected = encoding.select_encoder("libx264", "yuv444p10le", "auto", 8, 64, 64, 24)
        self.assertEqual(selected.pixel, "yuv444p10le")
        self.assertEqual(selected.device, "cpu")
        probe.assert_not_called()

    def test_video_without_hardware_support_uses_cpu_without_warning(self):
        with patch.object(encoding, "registered_pixels", return_value={"bgra"}):
            selected = encoding.select_encoder("ffv1", "bgra", "auto", 0, 64, 64, 24)
        self.assertEqual(selected.reason, "")

    def test_probe_is_real_subprocess_bounded_and_cached(self):
        mock_av = SimpleNamespace(__version__="test", library_versions={"avcodec": (1, 2)})
        selection = encoding.Selection("h264_nvenc", "yuv420p", "nvenc")
        with patch.dict(sys.modules, {"av": mock_av}), patch.object(encoding.subprocess, "run",
            return_value=SimpleNamespace(stdout=json.dumps({"ok": True}), returncode=0)) as run:
            self.assertEqual(encoding.probe(selection), (True, ""))
            self.assertEqual(encoding.probe(selection), (True, ""))
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs["timeout"], 12)
            self.assertIn("--probe", run.call_args.args[0])

    def test_probe_timeout_reports_device_failure(self):
        mock_av = SimpleNamespace(__version__="test", library_versions={})
        with patch.dict(sys.modules, {"av": mock_av}), patch.object(encoding.subprocess, "run",
            side_effect=subprocess.TimeoutExpired("probe", 12)):
            ok, reason = encoding.probe(encoding.Selection("h264_nvenc", "yuv420p", "nvenc"))
        self.assertFalse(ok)
        self.assertIn("timed out", reason)

    def test_nvenc_probe_does_not_reject_a_device_due_to_undersized_test_frames(self):
        # Reproduce the reported driver rejection in both real probe paths.
        # The output video can be large even when a tiny discovery frame fails.
        for engine in ("pyav", "ffmpeg"):
            for mode in ("auto", "nvenc"):
                with self.subTest(engine=engine, mode=mode):
                    encoding._probe_cache.clear()
                    encoded_sizes = []

                    def verify_size(width, height):
                        if min(width, height) < 145:
                            raise RuntimeError("Frame Dimension less than the minimum supported value")
                        encoded_sizes.append((width, height))

                    stream = MagicMock()
                    def encode(frame=None):
                        if frame is not None:
                            verify_size(stream.width, stream.height)
                            self.assertEqual((frame.width, frame.height), (stream.width, stream.height))
                        return []
                    stream.encode.side_effect = encode
                    container = MagicMock()
                    container.__enter__.return_value = container
                    container.add_stream.return_value = stream
                    mock_av = SimpleNamespace(__version__="test", library_versions={},
                        open=lambda *args, **kwargs: container,
                        VideoFrame=lambda w, h, pixel: SimpleNamespace(width=w, height=h, planes=[]))

                    def run(args, **kwargs):
                        if "--probe" in args:
                            try:
                                encoding._probe_main(json.loads(args[-1]))
                                return SimpleNamespace(stdout=json.dumps({"ok": True}), returncode=0)
                            except RuntimeError as error:
                                return SimpleNamespace(stdout=json.dumps({"error": str(error)}), returncode=1)
                        width, height = map(int, args[args.index("-s") + 1].split("x"))
                        verify_size(width, height)
                        self.assertEqual(len(kwargs["input"]), width * height * 3 * 3)
                        return SimpleNamespace(returncode=0)

                    hardware = encoding.Selection("h264_nvenc", "yuv420p", "nvenc", engine,
                                                  "ffmpeg-test" if engine == "ffmpeg" else "")
                    with patch.dict(sys.modules, {"av": mock_av}), \
                         patch.object(encoding, "registered_pixels", return_value={"yuv420p"}), \
                         patch.object(encoding, "ffmpeg_runtime", return_value=("ffmpeg-test", "test", {"h264_nvenc"})), \
                         patch.object(encoding, "candidates", return_value=iter([hardware])), \
                         patch.object(encoding.subprocess, "run", side_effect=run):
                        selected = encoding.select_encoder("libx264", "yuv420p", mode, 23.5, 2272, 1280, 24)
                    self.assertEqual(selected.device, "nvenc")
                    self.assertEqual(selected.engine, engine)
                    self.assertEqual(selected.reason, "")
                    self.assertTrue(encoded_sizes)

    def test_plane_layout_aliases_preserve_sampling_and_depth(self):
        self.assertEqual(encoding.mapped_pixel("yuv420p10le", {"p010le"}), "p010le")
        self.assertIsNone(encoding.mapped_pixel("yuv444p10le", {"p010le"}))
        self.assertIsNone(encoding.mapped_pixel("yuv420p10le", {"nv12"}))


if __name__ == "__main__": unittest.main()
