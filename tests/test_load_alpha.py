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


if __name__ == "__main__":
    unittest.main()
