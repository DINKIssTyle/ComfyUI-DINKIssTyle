"""VAE layout conversion and real CPU encode/decode, without CUDA weights."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

VENDOR = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle/lora_trainer/vendor"
sys.path.insert(0, str(VENDOR / "src"))
from fizgig.qwen_image21.vae import AutoencoderKLQwenImage21, load_qwen21_vae
from fizgig.qwen_image21.vae_weights import normalize_vae_weights


class TestVAELayouts(unittest.TestCase):
    def test_comfy_layout_residual_attention_sampling_and_projection_keys(self):
        # Representative native names from ComfyUI's Wan2.2 image VAE.
        pairs = {
            "encoder.conv1.weight": "encoder.conv_in.weight",
            "encoder.downsamples.0.downsamples.0.residual.0.gamma": "encoder.down_blocks.0.resnets.0.norm1.gamma",
            "encoder.downsamples.1.downsamples.0.shortcut.weight": "encoder.down_blocks.1.resnets.0.conv_shortcut.weight",
            "encoder.downsamples.0.downsamples.1.residual.2.weight": "encoder.down_blocks.0.resnets.1.conv1.weight",
            "encoder.downsamples.0.downsamples.2.resample.1.weight": "encoder.down_blocks.0.downsampler.resample.1.weight",
            "encoder.downsamples.1.downsamples.2.time_conv.weight": "encoder.down_blocks.1.downsampler.time_conv.weight",
            "encoder.middle.0.residual.3.gamma": "encoder.mid_block.resnets.0.norm2.gamma",
            "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
            "encoder.middle.2.residual.6.bias": "encoder.mid_block.resnets.1.conv2.bias",
            "encoder.head.0.gamma": "encoder.norm_out.gamma",
            "encoder.head.2.weight": "encoder.conv_out.weight",
            "conv1.weight": "quant_conv.weight",
            "conv2.weight": "post_quant_conv.weight",
            "decoder.conv1.weight": "decoder.conv_in.weight",
            "decoder.middle.1.norm.gamma": "decoder.mid_block.attentions.0.norm.gamma",
            "decoder.upsamples.0.upsamples.2.residual.2.weight": "decoder.up_blocks.0.resnets.2.conv1.weight",
            "decoder.upsamples.0.upsamples.3.resample.1.bias": "decoder.up_blocks.0.upsampler.resample.1.bias",
            "decoder.upsamples.1.upsamples.3.time_conv.weight": "decoder.up_blocks.1.upsampler.time_conv.weight",
            "decoder.head.2.bias": "decoder.conv_out.bias",
        }
        expected = {target: torch.randn(2, 2, 3, 3) if target.endswith("weight") else torch.randn(2)
                    for target in pairs.values()}
        native = {source: expected[target].unsqueeze(2) if target.endswith("weight") and ".resample." not in target
                  and ".attentions." not in target else expected[target] for source, target in pairs.items()}
        converted, layout = normalize_vae_weights(native, expected)
        self.assertEqual(layout, "ComfyUI")
        self.assertEqual(converted.keys(), expected.keys())
        for key in expected:
            self.assertTrue(torch.equal(converted[key], expected[key]), key)

    def test_real_safetensors_loader_temporal_axis_and_encode_decode(self):
        cfg = dict(base_dim=4, decoder_base_dim=4, z_dim=64, dim_mult=[1, 2, 4, 8, 8],
                   temperal_downsample=[False, True, True, True], num_res_blocks=2,
                   in_channels=4, out_channels=4, is_residual=True)
        torch.manual_seed(3)
        original = AutoencoderKLQwenImage21(**cfg).eval()
        state = original.state_dict()
        # Diffusers key layout can also contain Conv3d singleton temporal axes.
        state = {key: value.unsqueeze(2) if value.ndim == 4 and key.endswith("weight") else value
                 for key, value in state.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vae.safetensors"
            save_file(state, str(path))
            from builtins import open as real_open
            def config_open(name, *args, **kwargs):
                if str(name).endswith("vae_config.json"):
                    import io
                    return io.StringIO(json.dumps(cfg))
                return real_open(name, *args, **kwargs)
            with patch("builtins.open", side_effect=config_open):
                loaded = load_qwen21_vae(str(path), device="cpu", dtype=torch.float32)
        photo = torch.randn(1, 4, 1, 32, 32)
        with torch.no_grad():
            expected = original.encode(photo).latent_dist.mode()
            actual = loaded.encode(photo).latent_dist.mode()
            self.assertEqual(tuple(actual.shape), (1, 64, 1, 2, 2))
            self.assertTrue(torch.equal(actual, expected))
            decoded = loaded.decode(actual).sample
            self.assertEqual(decoded.shape, photo.shape)
            self.assertTrue(torch.isfinite(decoded).all())

    def test_incompatible_or_partial_checkpoints_fail_with_short_diagnostics(self):
        expected = {"encoder.conv_in.weight": torch.zeros(2, 4, 3, 3)}
        invalid = (
            {},
            {"encoder.conv1.weight": torch.zeros(2, 4, 3, 3, 3)},  # temporal video kernel
            {"encoder.conv1.weight": torch.zeros(2, 3, 1, 3, 3)},  # RGB wrong architecture
            {"encoder.conv1.weight": torch.zeros(2, 4, 1, 3, 3), "unknown": torch.zeros(1)},
        )
        for state in invalid:
            with self.assertRaisesRegex(ValueError, "Incompatible Qwen Image 2.1 VAE"):
                normalize_vae_weights(state, expected)

    def test_vendor_patch_and_manifest_are_in_sync(self):
        root = VENDOR.parents[2]
        for module in ("vae_weights", "text_weights"):
            self.assertEqual((root / f"scripts/patches/qwen21_{module}.py").read_bytes(),
                             (VENDOR / f"src/fizgig/qwen_image21/{module}.py").read_bytes())
        import hashlib
        manifest = json.loads((VENDOR / "UPSTREAM.json").read_text())
        for key, digest in manifest["files"].items():
            self.assertEqual(hashlib.sha256((VENDOR / "src" / key).read_bytes()).hexdigest(), digest, key)


if __name__ == "__main__":
    unittest.main()
