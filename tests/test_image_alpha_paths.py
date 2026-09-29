import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch


ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"


class ImageAlphaPathsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = runpy.run_path(str(ROOT / "dinki_grid.py"))["DINKI_Grid"]()
        cls.batch = runpy.run_path(str(ROOT / "dinki_batchImages.py"))["DINKI_BatchImages"]()

        comfy = MagicMock()
        nodes = MagicMock(MAX_RESOLUTION=16384)
        with patch.dict(sys.modules, {
            "folder_paths": MagicMock(),
            "comfy": comfy,
            "comfy.sd": comfy.sd,
            "comfy.utils": comfy.utils,
            "comfy.model_management": comfy.model_management,
            "nodes": nodes,
        }):
            ps = runpy.run_path(str(ROOT / "dinki_ps.py"))
        cls.pad = ps["DINKI_Resize_And_Pad"]()
        cls.unpad = ps["DINKI_Remove_Pad_From_Image"]()

        cls.temp = tempfile.TemporaryDirectory()
        folder_paths = MagicMock()
        folder_paths.get_input_directory.return_value = cls.temp.name
        server = MagicMock()
        server.PromptServer.instance.routes.post.side_effect = lambda path: lambda fn: fn
        with patch.dict(sys.modules, {
            "folder_paths": folder_paths,
            "server": server,
            "aiohttp": MagicMock(),
        }):
            color = runpy.run_path(str(ROOT / "dinki_color.py"))
        cls.lut = color["DINKI_Color_Lut"]()
        cls.deband = color["DINKI_Deband"]()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def rgba_image():
        image = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
        image[..., 0] = 0.7
        image[..., 2] = 0.9
        image[..., 3] = 0.0
        image[:, 1:3, 1:3, :3] = torch.tensor([1.0, 0.2, 0.1])
        image[:, 1:3, 1:3, 3] = 1.0
        return image

    def test_grid_retains_transparency_and_promotes_mixed_rgb_cells(self):
        rgba = self.rgba_image()
        rgb = torch.full((1, 4, 4, 3), 0.4)
        output, = self.grid.generate_grid(
            cols=2, rows=1, frame_thickness=0, bg_color_hex="#000000",
            resize_method="No Resize (Top-Left)", limit_output=False,
            max_output_width=512, max_output_height=512,
            image_1=rgba, image_2=rgb,
        )
        self.assertEqual(output.shape, (1, 4, 8, 4))
        self.assertEqual(float(output[0, 0, 0, 3]), 0.0)
        self.assertEqual(float(output[0, 1, 1, 3]), 1.0)
        self.assertEqual(float(output[0, 0, 4, 3]), 1.0)

    def test_grid_resize_modes_keep_transparent_corners(self):
        rgba = self.rgba_image()
        reference = torch.zeros((1, 8, 8, 3))
        for method in ("Keep Ratio (Fit)", "Keep Ratio (Crop)", "Stretch"):
            with self.subTest(method=method):
                output, = self.grid.generate_grid(
                    cols=1, rows=1, frame_thickness=0, bg_color_hex="#000000",
                    resize_method=method, limit_output=True,
                    max_output_width=4, max_output_height=4,
                    reference_image=2, image_1=rgba, image_2=reference,
                )
                self.assertEqual(output.shape, (1, 4, 4, 4))
                self.assertLess(float(output[0, 0, 0, 3]), 0.02)
                self.assertGreater(float(output[0, 1, 1, 3]), 0.5)

    def test_resize_and_remove_pad_keep_source_alpha(self):
        rgba = self.rgba_image()[:, :2]
        padded, info = self.pad.process(rgba, 8, 8, "nearest", True)
        self.assertEqual(padded.shape[-1], 4)
        self.assertEqual(float(padded[0, 0, 0, 3]), 1.0)
        self.assertEqual(float(padded[0, 2, 0, 3]), 0.0)
        restored, = self.unpad.process(padded, info, True)
        self.assertEqual(restored.shape[-1], 4)
        self.assertEqual(float(restored[0, 0, 0, 3]), 0.0)

    def test_batch_promotes_rgb_when_combined_with_rgba(self):
        rgba = self.rgba_image()
        rgb = torch.full((1, 4, 4, 3), 0.4)
        output, = self.batch.run(True, image1=rgb, image2=rgba)
        self.assertEqual(output.shape, (2, 4, 4, 4))
        self.assertTrue(torch.all(output[0, ..., 3] == 1))
        self.assertTrue(torch.equal(output[1, ..., 3], rgba[0, ..., 3]))

    def test_lut_and_deband_preserve_alpha(self):
        rgba = self.rgba_image()
        lookup = torch.zeros((1, 3, 2, 2, 2))
        lookup[:, 0] = 1.0
        self.lut.get_lut_tensor = lambda name: lookup
        colored, = self.lut.apply_lut(rgba, "test.cube", 1.0)
        self.assertEqual(colored.shape, rgba.shape)
        self.assertTrue(torch.equal(colored[..., 3], rgba[..., 3]))
        self.assertTrue(torch.allclose(colored[..., 0], torch.ones_like(colored[..., 0])))

        debanded, = self.deband.apply_deband(rgba, True, 20.0, 1, 4.0, 1)
        self.assertEqual(debanded.shape, rgba.shape)
        self.assertTrue(torch.equal(debanded[..., 3], rgba[..., 3]))


if __name__ == "__main__":
    unittest.main()
