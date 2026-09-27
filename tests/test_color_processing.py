import io
import asyncio
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"


class ColorProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.xmp_dir = Path(cls.temp.name) / "adobe_xmp"
        cls.xmp_dir.mkdir()
        folder_paths = MagicMock()
        folder_paths.get_input_directory.return_value = cls.temp.name
        folder_paths.get_full_path.side_effect = lambda kind, name: str(cls.xmp_dir / name)
        folder_paths.get_folder_paths.return_value = [str(cls.xmp_dir)]
        folder_paths.get_filename_list.return_value = []
        server = MagicMock()
        server.PromptServer.instance.routes.post.side_effect = lambda path: lambda function: function
        with patch.dict(sys.modules, {"folder_paths": folder_paths, "server": server}):
            module = runpy.run_path(str(ROOT / "dinki_color.py"))
        cls.auto = module["DINKI_Auto_Adjustment"]()
        cls.xmp = module["DINKI_adobe_xmp"]()
        cls.preview = module["DINKI_Adobe_XMP_Preview"]()
        cls.curve_lut = staticmethod(module["_calculate_pchip_lut"])
        cls.preview_route = staticmethod(module["preview_xmp_route"])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def write_xmp(self, name, settings):
        path = self.xmp_dir / name
        path.write_text(
            f'<rdf:RDF xmlns:rdf="{RDF}" xmlns:crs="{CRS}">'
            f'<rdf:Description {settings}/></rdf:RDF>', encoding="utf-8",
        )
        return path

    def auto_apply(self, image, tone=False, contrast=False, color=False, skin=False):
        return self.auto.apply(image, tone, contrast, color, skin, 0.1, 1.0)[0]

    def test_black_and_flat_images_remain_finite(self):
        black = torch.zeros((1, 8, 8, 3))
        output = self.auto_apply(black, tone=True, contrast=True, color=True, skin=True)
        self.assertTrue(torch.equal(output, black))
        gray = torch.full((1, 8, 8, 3), 0.35)
        contrast = self.auto_apply(gray, contrast=True)
        self.assertTrue(torch.allclose(contrast, gray, atol=1e-6))
        self.assertTrue(torch.isfinite(self.auto_apply(gray, tone=True)).all())

    def test_auto_color_and_skin_skip_unsupported_scenes(self):
        blue = torch.tensor([0.0, 0.0, 1.0]).reshape(1, 1, 1, 3).expand(1, 16, 16, 3).clone()
        self.assertTrue(torch.allclose(self.auto_apply(blue, skin=True), blue, atol=1e-6))
        self.assertTrue(torch.allclose(self.auto_apply(blue, color=True), blue, atol=1e-6))
        cast = torch.tensor([0.6, 0.5, 0.4]).reshape(1, 1, 1, 3).expand(1, 16, 16, 3).clone()
        adjusted = self.auto_apply(cast, color=True)
        self.assertLess(float(adjusted[..., 0].mean() - adjusted[..., 2].mean()), 0.2)

    def test_auto_adjust_preserves_alpha(self):
        rgba = torch.rand((2, 9, 7, 4))
        result = self.auto_apply(rgba, tone=True, contrast=True, color=True, skin=True)
        self.assertEqual(result.shape, rgba.shape)
        self.assertTrue(torch.equal(result[..., 3], rgba[..., 3]))
        self.assertTrue(torch.isfinite(result).all())

    def test_xmp_rgba_and_grain_are_repeatable(self):
        self.write_xmp("complex.xmp", '''crs:Exposure2012="0.5"
            crs:HueAdjustmentOrange="15" crs:GrainAmount="50"
            crs:GrainSize="60" crs:PostCropVignetteAmount="-20"
            crs:PostCropVignetteRoundness="30"''')
        rgba = torch.rand((2, 3, 5, 4))
        first = self.xmp.apply_preset(rgba, "complex.xmp", 0.75, 123)[0]
        again = self.xmp.apply_preset(rgba, "complex.xmp", 0.75, 123)[0]
        other = self.xmp.apply_preset(rgba, "complex.xmp", 0.75, 124)[0]
        self.assertEqual(first.shape, rgba.shape)
        self.assertTrue(torch.equal(first[..., 3], rgba[..., 3]))
        self.assertTrue(torch.equal(first, again))
        self.assertFalse(torch.equal(first[..., :3], other[..., :3]))
        self.assertTrue(torch.isfinite(first).all())

    def test_xmp_exposure_uses_linear_light_and_center_pixel_has_no_vignette(self):
        self.write_xmp("exposure.xmp", 'crs:Exposure2012="1"')
        midgray = torch.full((1, 1, 1, 3), 0.5)
        exposed = self.xmp.apply_preset(midgray, "exposure.xmp", 1.0)[0]
        expected = 1.055 * (2.0 * ((0.5 + 0.055) / 1.055) ** 2.4) ** (1.0 / 2.4) - 0.055
        self.assertAlmostEqual(float(exposed[0, 0, 0, 0]), expected, places=5)
        self.write_xmp("vignette.xmp", 'crs:PostCropVignetteAmount="-80"')
        vignetted = self.xmp.apply_preset(midgray, "vignette.xmp", 1.0)[0]
        self.assertTrue(torch.allclose(vignetted, midgray))

    def test_curve_lut_rejects_duplicate_x(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.curve_lut([(0, 0), (0, 255)])
        self.assertTrue(np.isfinite(self.curve_lut([(0, 0), (255, 255)])).all())

    def test_preview_tokens_keep_input_images_separate(self):
        red = torch.tensor([1.0, 0.0, 0.0]).reshape(1, 1, 1, 3).expand(1, 4, 4, 3).clone()
        blue = torch.tensor([0.0, 0.0, 1.0]).reshape(1, 1, 1, 3).expand(1, 4, 4, 3).clone()
        red_token = self.preview.apply_preset_preview(red, "-- None --", 1.0)["ui"]["preview_token"][0]
        blue_token = self.preview.apply_preset_preview(blue, "-- None --", 1.0)["ui"]["preview_token"][0]
        red_image = Image.open(io.BytesIO(self.preview.process_preview(red_token, "-- None --", 1.0)))
        blue_image = Image.open(io.BytesIO(self.preview.process_preview(blue_token, "-- None --", 1.0)))
        self.assertEqual(red_image.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(blue_image.getpixel((0, 0)), (0, 0, 255))
        self.assertIsNone(self.preview.process_preview("unknown", "-- None --", 1.0))

    def test_preview_route_rejects_missing_and_expired_tokens(self):
        request = MagicMock()
        request.json.side_effect = lambda: self._json({"xmp_file": "-- None --", "strength": 1.0})
        self.assertEqual(asyncio.run(self.preview_route(request)).status, 400)
        request.json.side_effect = lambda: self._json({
            "preview_token": "unknown", "xmp_file": "-- None --", "strength": 1.0,
        })
        self.assertEqual(asyncio.run(self.preview_route(request)).status, 404)

    @staticmethod
    async def _json(value):
        return value


if __name__ == "__main__":
    unittest.main()
