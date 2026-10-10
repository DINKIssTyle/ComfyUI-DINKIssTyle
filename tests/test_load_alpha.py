import base64
import importlib.util
import io
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"


class LoadAlphaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.temp.name)
        pixels = np.array([[[179, 0, 230, 0], [255, 51, 26, 255]]], dtype=np.uint8)
        Image.fromarray(pixels, mode="RGBA").save(self.input_dir / "transparent.png")
        Image.new("RGB", (2, 1), (40, 80, 120)).save(self.input_dir / "opaque.png")

        folder_paths = MagicMock()
        folder_paths.get_input_directory.return_value = str(self.input_dir)
        folder_paths.get_temp_directory.return_value = str(self.input_dir)
        server = MagicMock()
        server.PromptServer.instance.routes.get.side_effect = lambda path: lambda fn: fn
        server.PromptServer.instance.routes.post.side_effect = lambda path: lambda fn: fn

        package_name = "dkst_load_alpha_test"
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT)]
        with patch.dict(sys.modules, {
            package_name: package,
            "folder_paths": folder_paths,
            "server": server,
            "aiohttp": MagicMock(),
        }):
            loaded = {}
            for name in ("dinki_image_crop", "dinki_load", "dinki_photo_specs", "dinki_load_crop"):
                spec = importlib.util.spec_from_file_location(
                    f"{package_name}.{name}", ROOT / f"{name}.py"
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                loaded[name] = module
        self.loader = loaded["dinki_load"].DINKI_Image_Load()
        self.cropper = loaded["dinki_load_crop"].DINKI_Image_Load_Crop()
        self.photo = loaded["dinki_photo_specs"].DINKI_photo_specifications
        self.render_expanded = loaded["dinki_load_crop"]._render_expanded_canvas

    def tearDown(self):
        self.temp.cleanup()

    def test_transparent_png_keeps_rgba_and_separate_mask_outputs(self):
        image, mask, alpha = self.loader.load_image("", "transparent.png")["result"]
        self.assertEqual(image.shape, (1, 1, 2, 4))
        self.assertEqual(float(image[0, 0, 0, 3]), 0.0)
        self.assertEqual(float(image[0, 0, 1, 3]), 1.0)
        self.assertTrue(torch.equal(image[..., 3], alpha))
        self.assertTrue(torch.equal(mask, 1.0 - alpha))

    def test_opaque_image_keeps_rgb_output(self):
        image, mask, alpha = self.loader.load_image("", "opaque.png")["result"]
        self.assertEqual(image.shape, (1, 1, 2, 3))
        self.assertTrue(torch.all(mask == 0))
        self.assertTrue(torch.all(alpha == 1))

    def test_crop_resizes_rgba_without_purple_edge_bleed(self):
        with patch.object(self.photo, "calculate_resolution", return_value=(4, 2, "")):
            result = self.cropper.load_and_crop("", "transparent.png")
        image, mask, alpha = result["result"]
        self.assertEqual(image.shape, (1, 2, 4, 4))
        self.assertTrue(torch.equal(image[..., 3], alpha))
        self.assertTrue(torch.allclose(mask + alpha, torch.ones_like(alpha)))
        self.assertGreater(float(alpha[0, 0, 1]), 0.0)
        self.assertLess(float(alpha[0, 0, 1]), 1.0)
        self.assertLess(float(image[0, 0, 1, 2]), 0.15)

        preview_uri = result["ui"]["source_preview"][0]
        self.assertTrue(preview_uri.startswith("data:image/png;base64,"))
        with Image.open(io.BytesIO(base64.b64decode(preview_uri.split(",", 1)[1]))) as preview:
            self.assertEqual(preview.mode, "RGBA")
            self.assertEqual(preview.getpixel((0, 0))[3], 0)

    def test_loaded_transparency_survives_grid(self):
        image, _, _ = self.loader.load_image("", "transparent.png")["result"]
        grid = runpy.run_path(str(ROOT / "dinki_grid.py"))["DINKI_Grid"]()
        output, = grid.generate_grid(
            cols=1, rows=1, frame_thickness=0, bg_color_hex="#000000",
            resize_method="No Resize (Top-Left)", limit_output=False,
            max_output_width=512, max_output_height=512, image_1=image,
        )
        self.assertEqual(output.shape, (1, 1, 2, 4))
        self.assertEqual(float(output[0, 0, 0, 3]), 0.0)
        self.assertEqual(float(output[0, 0, 1, 3]), 1.0)

    def test_expand_preserves_full_portrait_with_padding_mask_and_rgb_channels(self):
        pixels = np.full((36, 18, 3), (40, 80, 120), dtype=np.uint8)
        pixels[0] = (255, 0, 0)
        pixels[-1] = (0, 0, 255)
        Image.fromarray(pixels).save(self.input_dir / "portrait.png")
        with patch.object(self.photo, "calculate_resolution", return_value=(64, 36, "")):
            output = self.cropper.load_and_crop("", "portrait.png", aspect_ratio="16:9", crop_mode="Expand")
        image, mask, alpha = output["result"]
        self.assertEqual(output["ui"]["crop_rect"], [[-23, 0, 64, 36]])
        self.assertEqual(image.shape, (1, 36, 64, 3))
        self.assertTrue(torch.allclose(image[0, :, 23:41], torch.from_numpy(pixels.astype(np.float32) / 255), atol=1e-6))
        self.assertTrue(torch.all(alpha[:, :, :22] == 0))
        self.assertTrue(torch.all(alpha[:, :, 42:] == 0))
        self.assertTrue(torch.all(mask[:, :, :22] == 1))
        self.assertTrue(torch.all(image[:, :, :22] == 0))
        self.assertTrue(torch.allclose(alpha[:, :, 23:41], torch.ones_like(alpha[:, :, 23:41]), atol=1e-6))
        self.assertTrue(torch.equal(mask, 1 - alpha))

    def test_expand_virtual_canvas_controls_generation_size(self):
        Image.new("RGB", (18, 36), (40, 80, 120)).save(self.input_dir / "portrait.png")
        output = self.cropper.load_and_crop("", "portrait.png", aspect_ratio="16:9",
                                           crop_mode="Expand", megapixels=0.1, resolution_multiple=32)
        image, mask, alpha = output["result"]
        self.assertEqual(image.shape, (1, 256, 416, 3))
        self.assertEqual(mask.shape, (1, 256, 416))
        self.assertEqual(alpha.shape, mask.shape)
        self.assertEqual(output["ui"]["resolution"], ["416 × 256"])
        self.assertTrue(torch.all(mask[:, :, :100] == 1))
        self.assertTrue(torch.allclose(alpha[:, :, 160:250], torch.ones_like(alpha[:, :, 160:250]), atol=1e-6))

    def test_expand_premultiplied_alpha_has_no_hidden_color_bleed(self):
        source = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
        source[..., :3] = torch.tensor([0.7, 0.0, 0.9])
        source[:, 1:3, 1:3] = torch.tensor([1.0, 0.2, 0.1, 1.0])
        image, mask, alpha = self.render_expanded(source, source[..., 3], (-2, -2, 8, 8), 16, 16)
        self.assertTrue(torch.equal(image[..., 3], alpha))
        self.assertTrue(torch.equal(mask, 1 - alpha))
        visible = alpha > 1e-6
        self.assertTrue(torch.any((alpha > 0) & (alpha < 1)))
        self.assertTrue(torch.allclose(image[..., 2][visible], torch.full_like(image[..., 2][visible], 0.1), atol=1e-6))
        self.assertTrue(torch.all(image[..., :3][alpha == 0] == 0))
        self.assertEqual(float(alpha[0, 0, 0]), 0)

    def test_expand_samples_partial_source_and_image_batches(self):
        source = torch.arange(2 * 4 * 4 * 3, dtype=torch.float32).reshape(2, 4, 4, 3) / 100
        alphas = torch.ones((2, 4, 4), dtype=torch.float32)
        image, mask, alpha = self.render_expanded(source, alphas, (-2, 2, 4, 4), 4, 4)
        self.assertEqual(image.shape, (2, 4, 4, 3))
        self.assertTrue(torch.allclose(image[:, :2, 2:], source[:, 2:, :2], atol=1e-6))
        self.assertTrue(torch.all(image[:, 2:] == 0))
        self.assertTrue(torch.all(alpha[:, :, :2] == 0))
        self.assertTrue(torch.all(mask[:, 2:] == 1))

    def test_expand_completely_outside_source_returns_empty_canvas(self):
        source = torch.ones((1, 4, 4, 4), dtype=torch.float32)
        for bounds in ((10, 10, 8, 8), (-20, -20, 8, 8)):
            with self.subTest(bounds=bounds):
                image, mask, alpha = self.render_expanded(source, source[..., 3], bounds, 16, 16)
                self.assertTrue(torch.all(image == 0))
                self.assertTrue(torch.all(alpha == 0))
                self.assertTrue(torch.all(mask == 1))

    def test_expand_large_virtual_canvas_and_tiny_crop_bound_intermediate_sizes(self):
        source = torch.ones((1, 40, 40, 3), dtype=torch.float32)
        alphas = torch.ones((1, 40, 40), dtype=torch.float32)
        import torch.nn.functional as functional
        interpolate = functional.interpolate
        sample = functional.grid_sample
        with patch.object(functional, "interpolate", wraps=interpolate) as resized, \
                patch.object(functional, "grid_sample", wraps=sample) as sampled:
            image, mask, alpha = self.render_expanded(source, alphas, (-200000, -200000, 400000, 400000), 64, 64)
            self.assertEqual(image.shape, (1, 64, 64, 3))
            self.assertEqual(resized.call_args.kwargs["size"], (1, 1))
            self.assertEqual(sampled.call_args.args[0].shape[2:], (1, 1))
            self.assertEqual(sampled.call_args.args[1].shape, (1, 64, 64, 2))
            self.assertGreater(float(alpha.sum()), 0)
        with patch.object(functional, "interpolate", wraps=interpolate) as resized, \
                patch.object(functional, "grid_sample", wraps=sample) as sampled:
            image, _, alpha = self.render_expanded(source, alphas, (20, 20, 1, 1), 16, 16)
            resized.assert_not_called()
            self.assertEqual(sampled.call_args.args[0].shape[2:], (1, 1))
            self.assertEqual(image.shape, (1, 16, 16, 3))
            self.assertTrue(torch.allclose(image, torch.ones_like(image)))
            self.assertTrue(torch.allclose(alpha, torch.ones_like(alpha)))

    def test_expand_downsampling_preserves_antialiasing_and_fractional_source_alpha(self):
        source = torch.zeros((2, 40, 40, 4), dtype=torch.float32)
        checker = (torch.arange(40)[:, None] + torch.arange(40)[None, :]) % 2
        source[..., :3] = checker[None, ..., None]
        source[0, ..., 3] = 0.25
        source[1, ..., 3] = 0.75
        image, mask, alpha = self.render_expanded(source, source[..., 3], (-20, -20, 80, 80), 8, 8)
        center = image[:, 3:5, 3:5, :3]
        self.assertTrue(torch.allclose(center, torch.full_like(center, 0.5), atol=0.01))
        self.assertTrue(torch.allclose(alpha[0, 2:6, 2:6], torch.full((4, 4), 0.25)))
        self.assertTrue(torch.allclose(alpha[1, 2:6, 2:6], torch.full((4, 4), 0.75)))
        self.assertTrue(torch.equal(mask, 1 - alpha))


if __name__ == "__main__":
    unittest.main()
