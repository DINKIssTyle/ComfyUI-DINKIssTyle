"""Synthetic audio checks for the VoxCPM2 ComfyUI integration."""

import importlib.util
import os
import sys
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

    def test_synthesis_uses_connected_text_and_comfy_audio_shape(self):
        model = FakeVox()
        with patch.object(nodes, "_load_vox", return_value=model):
            result, = nodes.TTS().run("vox", "whisper", "widget text", 2.3, 12,
                                      text_input="connected text")
        self.assertEqual(model.calls[0], {
            "text": "connected text", "cfg_value": 2.3, "inference_timesteps": 12,
        })
        self.assertEqual(tuple(result["waveform"].shape), (1, 1, 480))
        self.assertEqual(result["sample_rate"], 48000)

    def test_clone_modes_and_temporary_reference_cleanup(self):
        import torch

        audio = {"waveform": torch.zeros(1, 2, 2400), "sample_rate": 24000}
        model = FakeVox()
        with patch.object(nodes, "_load_vox", return_value=model):
            nodes.Cloning().run("vox", "whisper", audio, "hello", "")
            nodes.Cloning().run("vox", "whisper", audio, "hello", "spoken words")
        self.assertIn("reference_wav_path", model.calls[0])
        self.assertNotIn("prompt_text", model.calls[0])
        self.assertEqual(model.calls[1]["prompt_text"], "spoken words")
        self.assertIn("prompt_wav_path", model.calls[1])
        for call in model.calls:
            path = call.get("prompt_wav_path") or call.get("reference_wav_path")
            self.assertFalse(os.path.exists(path))

    def test_transcribe_button_resamples_and_skips_voxcpm(self):
        import torch

        audio = {"waveform": torch.zeros(1, 1, 48000), "sample_rate": 48000}
        whisper = FakeWhisper()
        with patch.object(nodes, "_load_whisper", return_value=whisper), \
             patch.object(nodes, "_load_vox", side_effect=AssertionError("VoxCPM loaded")):
            output = nodes.Cloning().run("vox", "whisper", audio, "", "",
                                            transcribe_only=True, request_id="abc")
        self.assertEqual(len(whisper.audio), 16000)
        self.assertEqual(output["ui"], {"transcript": ["reference transcript"], "request_id": ["abc"]})
        self.assertIs(output["result"][0], audio)

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
