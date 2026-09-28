"""Synthetic audio checks for the VoxCPM2 ComfyUI integration."""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle_VoxCPM2" / "nodes.py"
SPEC = importlib.util.spec_from_file_location("dinki_voxcpm2_nodes", MODULE_PATH)
nodes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nodes)


class FakeVox:
    def __init__(self):
        self.tts_model = type("TTS", (), {"sample_rate": 48000})()
        self.calls = []

    def generate(self, **kwargs):
        import numpy as np
        import soundfile as sf

        self.calls.append(kwargs)
        path = kwargs.get("prompt_wav_path") or kwargs.get("reference_wav_path")
        if path:
            audio, rate = sf.read(path)
            assert rate == 24000 and len(audio) == 2400
        return np.zeros(480, dtype="float32")


class FakeWhisper:
    def __init__(self):
        self.audio = None

    def transcribe(self, audio, **kwargs):
        self.audio = audio
        return {"text": "  reference transcript  "}


class VoxCPM2NodeTests(unittest.TestCase):
    def test_voxcpm_load_disables_compile_warmup(self):
        fake_module = types.ModuleType("voxcpm")
        fake_module.VoxCPM = type("VoxCPM", (), {"from_pretrained": Mock(return_value=object())})
        nodes._vox_cache.update(path=None, model=None)
        with patch.dict(sys.modules, {"voxcpm": fake_module}), \
             patch.object(nodes, "_check_vox"):
            model = nodes._load_vox("local-model")
        fake_module.VoxCPM.from_pretrained.assert_called_once_with(
            str(Path("local-model").resolve()), load_denoiser=False, optimize=False,
        )
        self.assertIs(model, nodes._vox_cache["model"])

    def test_tts_without_reference_uses_advanced_values_and_comfy_audio_shape(self):
        model = FakeVox()
        with patch.object(nodes, "_load_vox", return_value=model):
            result, = nodes.TTSCloning().run("vox", "  widget text  ", cfg=2.3,
                                             inference_steps=12)
        self.assertEqual(model.calls[0], {
            "text": "widget text", "cfg_value": 2.3, "inference_timesteps": 12,
        })
        self.assertEqual(tuple(result["waveform"].shape), (1, 1, 480))
        self.assertEqual(result["sample_rate"], 48000)
        inputs = nodes.TTSCloning.INPUT_TYPES()
        self.assertTrue(inputs["required"]["cfg"][1]["advanced"])
        self.assertTrue(inputs["required"]["inference_steps"][1]["advanced"])
        self.assertNotIn("whisper_model_path", inputs["required"])
        self.assertTrue(inputs["required"]["text"][1]["multiline"])
        self.assertNotIn("text_input", inputs["optional"])

    def test_clone_modes_and_temporary_reference_cleanup(self):
        import torch

        audio = {"waveform": torch.zeros(1, 2, 2400), "sample_rate": 24000}
        model = FakeVox()
        with patch.object(nodes, "_load_vox", return_value=model):
            nodes.TTSCloning().run("vox", "hello", reference_audio=audio)
            nodes.TTSCloning().run("vox", "hello", reference_audio=audio,
                                   reference_transcript="spoken words")
        self.assertIn("reference_wav_path", model.calls[0])
        self.assertNotIn("prompt_text", model.calls[0])
        self.assertEqual(model.calls[1]["prompt_text"], "spoken words")
        self.assertIn("prompt_wav_path", model.calls[1])
        for call in model.calls:
            path = call.get("prompt_wav_path") or call.get("reference_wav_path")
            self.assertFalse(os.path.exists(path))

    def test_reference_audio_reads_and_regenerates_sidecar_transcript(self):
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as directory, patch.object(nodes, "VOICE_ROOT", Path(directory)):
            sf.write(Path(directory) / "voice.wav", np.zeros(48000, dtype="float32"), 48000)
            sidecar = Path(directory) / "voice.txt"
            sidecar.write_text("existing text", encoding="utf-8")
            with patch.dict(sys.modules, {"comfy.audio": None}), \
                 patch.object(nodes, "_transcribe", side_effect=["first pass", "second pass"]):
                loaded = nodes.ReferenceAudio().run("whisper", "voice.wav")
                first = nodes.ReferenceAudio().run("whisper", "voice.wav", transcribe_action=True,
                                                   request_id="abc")
                second = nodes.ReferenceAudio().run("whisper", "voice.wav", transcribe_action=True)
            self.assertEqual(loaded["result"][1], "existing text")
            self.assertEqual(first["ui"], {"transcript": ["first pass"], "request_id": ["abc"]})
            self.assertEqual(second["result"][1], "second pass")
            self.assertEqual(sidecar.read_text(encoding="utf-8"), "second pass")
            self.assertEqual(tuple(loaded["result"][0]["waveform"].shape), (1, 1, 48000))
            self.assertEqual(loaded["result"][0]["sample_rate"], 48000)
            sidecar.unlink()
            with patch.dict(sys.modules, {"comfy.audio": None}):
                self.assertEqual(nodes.ReferenceAudio().run("whisper", "voice.wav")["result"][1], "")
            self.assertEqual(nodes._voice_names(), ["voice.wav"])
            with self.assertRaises(ValueError):
                nodes._voice_path("../outside.wav")

    def test_reference_audio_uses_pyav_for_other_codecs(self):
        import numpy as np
        import soundfile as sf

        class FakeFrame:
            def to_ndarray(self):
                return np.zeros((2, 160), dtype="float32")

        class FakeResampler:
            def __init__(self, **kwargs):
                self.options = kwargs

            def resample(self, frame):
                return [FakeFrame()] if frame is not None else []

        class FakeContainer:
            streams = types.SimpleNamespace(audio=[types.SimpleNamespace(
                codec_context=types.SimpleNamespace(sample_rate=16000), layout="stereo")])

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def decode(self, **kwargs):
                self.decode_options = kwargs
                return [object()]

        container = FakeContainer()
        fake_av = types.ModuleType("av")
        fake_av.open = Mock(return_value=container)
        fake_av.AudioResampler = FakeResampler
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(nodes, "VOICE_ROOT", Path(directory)), \
             patch.object(sf, "read", side_effect=RuntimeError("unsupported codec")), \
             patch.dict(sys.modules, {"av": fake_av, "comfy.audio": None}):
            (Path(directory) / "voice.m4a").write_bytes(b"fixture")
            audio = nodes._load_voice_audio("voice.m4a")
        self.assertEqual(tuple(audio["waveform"].shape), (1, 2, 160))
        self.assertEqual(audio["sample_rate"], 16000)
        self.assertEqual(container.decode_options, {"audio": 0})

    def test_whisper_transcription_resamples_reference_audio(self):
        import torch

        whisper = FakeWhisper()
        audio = {"waveform": torch.zeros(1, 1, 48000), "sample_rate": 48000}
        with patch.object(nodes, "_load_whisper", return_value=whisper):
            transcript = nodes._transcribe(audio, "whisper-model")
        self.assertEqual(transcript, "reference transcript")
        self.assertEqual(len(whisper.audio), 16000)

    def test_model_manager_outputs_selected_paths_without_implicit_download(self):
        with patch.object(nodes, "_download_vox") as download_vox, \
             patch.object(nodes, "_download_whisper") as download_whisper:
            output = nodes.Downloader().run("VoxCPM2", "small")
            self.assertTrue(output["result"][0].endswith("/model/VoxCPM2"))
            self.assertTrue(output["result"][1].endswith("/model/Whisper/small.pt"))
            download_vox.assert_not_called()
            download_whisper.assert_not_called()


if __name__ == "__main__":
    unittest.main()
