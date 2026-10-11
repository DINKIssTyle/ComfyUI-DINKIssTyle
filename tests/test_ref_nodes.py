"""Unit tests for DKST PS Image Ref nodes."""

import os
import shutil
import tempfile
import unittest

import torch

# Create dummy folder_paths if comfy is not running
import folder_paths

from ComfyUI_DINKIssTyle.dinki_ref_core import (
    blur_latent,
    calculate_token_count,
    load_ref_safetensors,
    optimize_latent_2d,
    pool_latent_2d,
    save_ref_safetensors,
)
from ComfyUI_DINKIssTyle.dinki_ref import (
    DINKI_Apply_Image_Ref,
    DINKI_Create_Image_Ref,
    DINKI_Inspect_Image_Ref,
    DINKI_Load_Image_Ref,
)


class DummyVAE:
    def encode(self, pixels):
        # pixels: [B, H, W, 3] -> return [B, 4, H // 8, W // 8]
        b, h, w, _ = pixels.shape
        return torch.randn((b, 4, max(2, h // 8), max(2, w // 8)), dtype=torch.float32)

    def decode(self, latent):
        # latent: [B, 4, H, W] -> return [B, H * 8, W * 8, 3]
        b, _, h, w = latent.shape
        return torch.zeros((b, h * 8, w * 8, 3), dtype=torch.float32)


class DummyModel:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        new_m = DummyModel()
        new_m.model_options = {k: v for k, v in self.model_options.items()}
        return new_m


class TestImageRefNodes(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pool_and_optimize(self):
        # Full latent [1, 4, 32, 32]
        z_full = torch.randn(1, 4, 32, 32)
        # Pool to 4x4
        z_pooled = pool_latent_2d(z_full, 4, 4)
        self.assertEqual(z_pooled.shape, (1, 4, 4, 4))

        # Optimize for 10 steps
        z_refined = optimize_latent_2d(z_pooled, z_full, steps=10)
        self.assertEqual(z_refined.shape, (1, 4, 4, 4))
        self.assertFalse(torch.isnan(z_refined).any())

    def test_blur_latent(self):
        z = torch.randn(1, 4, 16, 16)
        blurred = blur_latent(z, factor=4)
        self.assertEqual(blurred.shape, z.shape)
        self.assertFalse(torch.isnan(blurred).any())

    def test_safetensors_io(self):
        latent = torch.randn(1, 4, 8, 8)
        meta = {"mode": "Compressed", "test_key": 123}
        filepath = os.path.join(self.temp_dir, "test_ref.safetensors")

        saved = save_ref_safetensors(filepath, latent, meta)
        self.assertTrue(os.path.isfile(saved))

        loaded_latent, loaded_meta = load_ref_safetensors(saved)
        self.assertEqual(loaded_latent.shape, latent.shape)
        self.assertEqual(loaded_meta.get("test_key"), 123)
        self.assertTrue(torch.allclose(loaded_latent, latent, atol=1e-5))

    def test_creator_and_inspect_nodes(self):
        creator = DINKI_Create_Image_Ref()
        vae = DummyVAE()
        dummy_images = torch.rand(2, 64, 64, 3)

        image_ref, info = creator.create_ref(
            vae=vae,
            ref_name="unit_test_ref",
            image_gallery="[]",
            mode="Compressed (training)",
            refinement_steps=5,
            pool_grid=4,
            save_to_disk=False,
            images=dummy_images,
        )

        self.assertIn("latent", image_ref)
        self.assertEqual(image_ref["latent"].shape, (2, 4, 4, 4))
        self.assertIn("unit_test_ref", info)

        # Test Inspect Node
        inspector = DINKI_Inspect_Image_Ref()
        preview, tokens, inspect_info = inspector.inspect_ref(image_ref, vae=vae)
        self.assertEqual(preview.shape, (2, 32, 32, 3))
        self.assertGreater(tokens, 0)
        self.assertIn("unit_test_ref", inspect_info)

    def test_apply_node(self):
        applier = DINKI_Apply_Image_Ref()
        model = DummyModel()
        image_ref = {
            "name": "test",
            "latent": torch.randn(1, 4, 4, 4),
            "meta": {},
        }
        (patched_model,) = applier.apply_ref(model, image_ref, strength=0.8)
        opts = patched_model.model_options.get("transformer_options", {})
        self.assertIn("ref_latents", opts)
        self.assertEqual(len(opts["ref_latents"]), 1)


if __name__ == "__main__":
    unittest.main()
