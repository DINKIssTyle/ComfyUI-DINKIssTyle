"""Unit tests for the retained DKST PS Image Ref Creator."""

import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors import safe_open

# Import the real Creator without requiring a full ComfyUI installation.
NODE_DIR = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"
package = types.ModuleType("dkst_ref_tests")
package.__path__ = [str(NODE_DIR)]
folder_paths = types.ModuleType("folder_paths")
with patch.dict(sys.modules, {"dkst_ref_tests": package, "folder_paths": folder_paths}):
    core = importlib.import_module("dkst_ref_tests.dinki_ref_core")
    nodes = importlib.import_module("dkst_ref_tests.dinki_ref")

calculate_token_count = core.calculate_token_count
optimize_latent_2d = core.optimize_latent_2d
pool_latent_2d = core.pool_latent_2d
DINKI_Create_Image_Ref = nodes.DINKI_Create_Image_Ref


class DummyVAE:
    def encode(self, pixels):
        b, h, w, _ = pixels.shape
        return torch.randn(b, 4, max(2, h // 8), max(2, w // 8))


class DummyH3VAE:
    def __init__(self):
        self.encode_batches = []

    def encode(self, pixels):
        self.encode_batches.append(pixels.shape[0])
        # A video VAE treats its IMAGE batch as a time sequence.
        return pixels.mean().expand(1, 24, pixels.shape[0], 4, 4).clone()


class TestImageRefCreator(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.temp_dir = temp_dir.name
        folder_paths.get_input_directory = lambda: self.temp_dir
        folder_paths.get_temp_directory = lambda: self.temp_dir
        folder_paths.get_output_directory = lambda: self.temp_dir
        folder_paths.get_annotated_filepath = lambda name: os.path.join(self.temp_dir, name)

    def test_creator_schema_and_storage_directory(self):
        self.assertEqual(core.REF_FOLDER_NAME, "ref")
        self.assertEqual(core.get_ref_dir(), os.path.join(self.temp_dir, "ref"))
        required = DINKI_Create_Image_Ref.INPUT_TYPES()["required"]
        self.assertEqual(required["image_gallery"], ("DKST_IMAGE_REF_GRID", {"default": "[]"}))
        self.assertEqual(required["mode"][1]["default"], "Full Reference (encode)")

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


    def test_refinement_inside_comfy_inference_mode(self):
        with torch.inference_mode():
            full = torch.randn(1, 4, 16, 16)
            pooled = pool_latent_2d(full, 4, 4)
            self.assertTrue(full.is_inference())
            self.assertTrue(pooled.is_inference())
            refined = optimize_latent_2d(pooled, full, steps=5)
            self.assertEqual(refined.shape, pooled.shape)
            self.assertTrue(torch.isfinite(refined).all())
            self.assertFalse(refined.requires_grad)
            self.assertTrue(torch.is_inference_mode_enabled())
            image_ref, _ = DINKI_Create_Image_Ref().create_ref(
                DummyVAE(), "inference", "[]", "Compressed (training)", 5, 4, False,
                images=torch.rand(2, 64, 64, 3))
            self.assertEqual(image_ref["latent"].shape, (2, 4, 4, 4))


    def test_temporal_compression_keeps_channel_and_frame_order(self):
        latent = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2, 1, 1).expand(2, 3, 2, 4, 4)
        outputs = iter(latent.split(1))
        vae = types.SimpleNamespace(encode=lambda pixels: next(outputs))
        image_ref, _ = DINKI_Create_Image_Ref().create_ref(
            vae, "temporal", "[]", "Compressed (training)", 0, 2, False,
            images=torch.zeros(2, 32, 32, 3))
        expected = latent[:, :, :, :2, :2]
        self.assertTrue(torch.equal(image_ref["latent"], expected))


    def test_h3_creator_encodes_still_images_individually(self):
        vae = DummyH3VAE()
        images = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1, 1).expand(8, 64, 64, 3) / 8
        created, _ = DINKI_Create_Image_Ref().create_ref(
            vae, "h3_person", "[]", "Full Reference (encode)", 0, 4, False, images=images)
        self.assertEqual(vae.encode_batches, [1] * 8)
        self.assertEqual(tuple(created["latent"].shape), (8, 24, 1, 4, 4))
        self.assertEqual(created["meta"]["reference_layout"], "independent_images")
        for i, latent in enumerate(created["latent"]):
            self.assertAlmostEqual(latent.mean().item(), i / 8)

    def test_model_specific_token_estimates(self):
        self.assertEqual(calculate_token_count(torch.zeros(8, 64, 102, 68)), 55488)
        self.assertEqual(calculate_token_count(torch.zeros(8, 128, 102, 68)), 55488)
        self.assertEqual(calculate_token_count(torch.zeros(2, 24, 1, 5, 3)), 12)
        self.assertEqual(calculate_token_count(torch.zeros(8, 64, 102, 68), (2, 2)), 13872)

    def test_saved_creator_tensor_and_metadata_roundtrip(self):
        for mode in ("Full Reference (encode)", "Compressed (training)"):
            with self.subTest(mode=mode):
                created, info = DINKI_Create_Image_Ref().create_ref(
                    DummyVAE(), "roundtrip", "[]", mode, 0, 4, True,
                    images=torch.rand(2, 64, 64, 3))
                filepath = os.path.join(self.temp_dir, "ref", "roundtrip.safetensors")
                self.assertIn(filepath, info)
                with safe_open(filepath, framework="pt") as saved:
                    self.assertTrue(torch.equal(created["latent"], saved.get_tensor("latent")))
                    self.assertEqual(created["meta"], json.loads(saved.metadata()[core.META_KEY]))
                    self.assertEqual(saved.metadata()["kind"], "image_ref")


if __name__ == "__main__":
    unittest.main()
