"""Real INT8 safetensors and embedding/linear numerics; no full GPU model."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file, load_file

VENDOR = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle/qwen21_trainer/vendor"
sys.path.insert(0, str(VENDOR / "src"))
from fizgig.qwen_image21.text_weights import decode_text_weights
from fizgig.qwen_image21.embedder import Qwen21TextEncoder


def marker(**params):
    return torch.tensor(list(json.dumps({"format": "int8_tensorwise", **params}).encode()), dtype=torch.uint8)


class TestTextWeights(unittest.TestCase):
    def test_rotation_scale_embedding_and_linear_outputs(self):
        h = torch.tensor([[1., 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]) / 2
        codes = torch.tensor([[1, -2, 3, -4, 5, -6, 7, -8], [-3, 1, 2, 4, -2, 7, 5, -1]], dtype=torch.int8)
        scale = torch.tensor([[0.00012345], [0.375]])
        expected = ((codes.float() * scale).reshape(2, 2, 4) @ h).reshape(2, 8)
        for conf in ({"convrot": True, "convrot_groupsize": 4},
                     {"params": {"convrot": True, "convrot_groupsize": 4}}):
            for name in ("model.embed_tokens", "model.layers.0.self_attn.q_proj"):
                state = {name + ".weight": codes.clone(), name + ".weight_scale": scale.clone(),
                         name + ".comfy_quant": marker(**conf)}
                dense, count = decode_text_weights(state, torch.float32)
                self.assertEqual(count, 1)
                self.assertEqual(list(dense), [name + ".weight"])
                torch.testing.assert_close(dense[name + ".weight"], expected)
                inputs = torch.arange(24).reshape(3, 8).float() / 24
                torch.testing.assert_close(nn.functional.linear(inputs, dense[name + ".weight"]), inputs @ expected.T)
                torch.testing.assert_close(nn.functional.embedding(torch.tensor([1, 0]), dense[name + ".weight"]), expected[[1, 0]])

    def test_scalar_scale_bf16_and_plain_checkpoints(self):
        state = {"layer.weight": torch.tensor([[2, -3], [4, -5]], dtype=torch.int8),
                 "layer.weight_scale": torch.tensor(0.125), "layer.comfy_quant": marker(),
                 "norm.weight": torch.ones(2), "position_ids": torch.arange(2),
                 "layer.comfy_attention.config": torch.tensor([123, 125], dtype=torch.uint8)}
        result, count = decode_text_weights(state)
        self.assertEqual(count, 1)
        self.assertEqual(result["layer.weight"].dtype, torch.bfloat16)
        self.assertEqual(result["norm.weight"].dtype, torch.bfloat16)
        self.assertEqual(result["position_ids"].dtype, torch.int64)
        torch.testing.assert_close(result["layer.weight"], torch.tensor([[.25, -.375], [.5, -.625]], dtype=torch.bfloat16))
        self.assertNotIn("layer.comfy_attention.config", result)
        plain, count = decode_text_weights({"layer.weight": torch.ones(2, 2)})
        self.assertEqual(count, 0)
        self.assertEqual(plain["layer.weight"].dtype, torch.bfloat16)

    def test_corrupt_unsupported_or_orphan_quantization_is_rejected(self):
        base = {"layer.weight": torch.ones(2, 8, dtype=torch.int8),
                "layer.weight_scale": torch.ones(2, 1), "layer.comfy_quant": marker()}
        for change in ({"layer.comfy_quant": marker(format="nvfp4")},
                       {"layer.comfy_quant": marker(convrot=True, convrot_groupsize=8)},
                       {"layer.comfy_quant": marker(convrot=True, convrot_groupsize=16)},
                       {"layer.comfy_quant": torch.tensor([1.])},
                       {"layer.comfy_quant": torch.tensor([123], dtype=torch.uint8)},
                       {"layer.weight": torch.ones(2, 8)},
                       {"layer.weight_scale": torch.ones(3)},
                       {"layer.weight_scale": torch.tensor(float("nan"))}):
            with self.subTest(change=change.keys()), self.assertRaisesRegex(ValueError, "Qwen3-VL quantized checkpoint"):
                decode_text_weights({**base, **change})
        for state in ({"layer.weight_scale": torch.ones(1)}, {"layer.weight": torch.ones(2, 8, dtype=torch.int8)}):
            with self.assertRaisesRegex(ValueError, "undecoded"):
                decode_text_weights(state)

    def test_encoder_constructor_loads_real_quantized_file_without_casting_markers(self):
        class TinyHF(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.language_model = nn.Module()
                self.model.language_model.embed_tokens = nn.Embedding(4, 8)
                self.model.language_model.layers = nn.ModuleList([nn.Module()])
                self.model.language_model.layers[0].self_attn = nn.Module()
                self.model.language_model.layers[0].self_attn.q_proj = nn.Linear(8, 8, bias=False)
                self.model.visual = nn.Linear(2, 2)
                self.lm_head = nn.Linear(8, 4, bias=False)
            @classmethod
            def _from_config(cls, config):
                return cls()
        tokenizer = lambda *args, **kwargs: types.SimpleNamespace(input_ids=[1, 2])
        transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda _: tokenizer),
            Qwen3VLConfig=types.SimpleNamespace(from_dict=lambda config: config),
            Qwen3VLForConditionalGeneration=TinyHF)
        state = TinyHF().state_dict()
        for name in ("model.language_model.embed_tokens", "model.language_model.layers.0.self_attn.q_proj"):
            shape = state[name + ".weight"].shape
            state[name + ".weight"] = torch.arange(shape.numel()).reshape(shape).to(torch.int8)
            state[name + ".weight_scale"] = torch.full((shape[0], 1), 0.01)
            state[name + ".comfy_quant"] = marker(convrot=True, convrot_groupsize=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder.safetensors"
            save_file(state, str(path))
            reference, _ = decode_text_weights(load_file(str(path)), torch.float32)
            with patch.dict(sys.modules, {"transformers": transformers,
                    "accelerate": types.SimpleNamespace(init_empty_weights=nullcontext)}), \
                    patch("fizgig.qwen_image21.embedder._text_encoder_config", return_value={}):
                encoder = Qwen21TextEncoder(str(path), tokenizer_dir="local", device="cpu", dtype=torch.float32)
            loaded = encoder.model.model.language_model
            torch.testing.assert_close(loaded.embed_tokens.weight, reference["model.language_model.embed_tokens.weight"])
            torch.testing.assert_close(loaded.layers[0].self_attn.q_proj.weight, reference["model.language_model.layers.0.self_attn.q_proj.weight"])
            self.assertIsNone(encoder.model.model.visual)
            self.assertIsNone(encoder.model.lm_head)
            self.assertTrue(all(not p.requires_grad for p in encoder.model.parameters()))


if __name__ == "__main__":
    unittest.main()
