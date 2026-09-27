import asyncio
import base64
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import torch
from PIL import Image


SOURCE = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle" / "dinki_photo_studio.py"


class PhotoStudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        server = MagicMock()
        server.PromptServer.instance.routes.get.side_effect = lambda path: lambda function: function
        server.PromptServer.instance.routes.post.side_effect = lambda path: lambda function: function
        server.PromptServer.instance.user_manager.get_request_user_filepath.side_effect = (
            lambda request, file, create_dir=True: str(Path(cls.temp.name) / request.user / file))
        with patch.dict(sys.modules, {"server": server}):
            module = runpy.run_path(str(SOURCE))
        cls.module = module
        cls.node = module["DINKI_Photo_Studio"]()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def apply(self, image, depth=None, **changes):
        settings = self.module["_defaults"]()
        settings.update(changes)
        return self.node.apply(image, depth_image=depth, **settings)[0]

    def test_neutral_and_inactive_are_exact_bypasses(self):
        image = torch.rand((2, 13, 11, 4))
        self.assertIs(self.apply(image), image)
        settings = self.module["_defaults"]()
        settings["light_exposure"] = 1
        self.assertIs(self.node.apply(image, active=False, **settings)[0], image)

    def test_controls_modify_image_and_preserve_alpha(self):
        torch.manual_seed(4)
        image = torch.rand((1, 24, 28, 4)) * 0.8 + 0.1
        cases = {
            "light_exposure": 1.0, "light_contrast": 60.0,
            "light_highlights": 60.0, "light_shadows": 60.0,
            "light_whites": 60.0, "light_blacks": 60.0,
            "color_temperature": 70.0, "color_tint": 70.0,
            "color_vibrance": 70.0, "color_saturation": 70.0,
            "effects_texture": 70.0, "effects_clarity": 70.0,
            "effects_dehaze": 70.0, "effects_vignette": -70.0,
            "effects_grain": 70.0, "effects_glow": 70.0,
            "detail_sharpening": 70.0, "detail_noise_reduction": 70.0,
            "detail_color_noise_reduction": 70.0,
            "optics_distortion": 70.0, "optics_vignette": 70.0,
        }
        for name, value in cases.items():
            with self.subTest(name=name):
                changes = {name: value}
                if name in ("color_temperature", "color_tint"):
                    changes["color_white_balance"] = "Custom"
                result = self.apply(image, **changes)
                self.assertEqual(result.shape, image.shape)
                if name != "optics_distortion":
                    self.assertTrue(torch.equal(result[..., 3], image[..., 3]))
                self.assertTrue(torch.isfinite(result).all())
                self.assertGreater(float((result[..., :3] - image[..., :3]).abs().max()), 1e-5)

    def test_blur_requires_valid_depth_and_supports_batch_broadcast(self):
        image = torch.rand((2, 32, 32, 3))
        with self.assertRaisesRegex(ValueError, "depth_image"):
            self.apply(image, lens_blur_apply=True)
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            self.apply(image, torch.rand((1, 16, 32, 3)), lens_blur_apply=True)
        with self.assertRaisesRegex(ValueError, "batch size"):
            self.apply(image, torch.rand((3, 32, 32, 3)), lens_blur_apply=True)
        depth = torch.ones((1, 16, 16, 3))
        output = self.apply(image, depth, lens_blur_apply=True, lens_blur_focus=0.0)
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_depth_blur_sharp_focus_and_grain_seed(self):
        checker = ((torch.arange(64)[:, None] + torch.arange(64)[None, :]) % 2).float()
        image = checker[None, :, :, None].expand(1, 64, 64, 3).clone()
        depth = torch.zeros((1, 64, 64, 1))
        depth[:, 16:48, 16:48] = 1
        blurred = self.apply(image, depth, lens_blur_apply=True,
                             lens_blur_focus=1, lens_blur_amount=90)
        self.assertGreater(float((blurred - image).abs().sum()), 1)
        subject_change = (blurred[:, 24:40, 24:40] - image[:, 24:40, 24:40]).abs().mean()
        background_change = (blurred[:, :12, :12] - image[:, :12, :12]).abs().mean()
        self.assertLess(float(subject_change), float(background_change))
        grain_a = self.apply(image, effects_grain=50, grain_seed=42)
        grain_b = self.apply(image, effects_grain=50, grain_seed=42)
        self.assertTrue(torch.equal(grain_a, grain_b))

    def test_low_resolution_depth_follows_image_silhouette(self):
        rgb = torch.zeros((1, 3, 64, 64))
        rgb[:, 0, :, :32] = 1
        rgb[:, 2, :, 32:] = 1
        depth = torch.zeros((1, 1, 16, 16))
        depth[:, :, :, :8] = 1
        refined = self.module["_joint_upsample_depth"](depth, rgb)
        self.assertGreater(float(refined[0, 0, 32, 31]), 0.8)
        self.assertLess(float(refined[0, 0, 32, 32]), 0.2)

    def test_in_focus_silhouette_does_not_bleed_into_background(self):
        image = torch.zeros((1, 64, 64, 3))
        image[:, :, :32, 0] = 0.8
        image[:, :, 32:, 2] = 0.8
        depth = torch.zeros((1, 16, 16, 1))
        depth[:, :, :8] = 1
        result = self.apply(image, depth, lens_blur_apply=True,
                            lens_blur_focus=1, lens_blur_amount=90,
                            lens_blur_bokeh=1,
                            lens_blur_depth_blur_radius=0)
        self.assertGreater(float(result[0, 32, 31, 0]), 0.79)
        self.assertLess(float(result[0, 32, 32:40, 0].max()), 0.01)
        self.assertGreater(float(result[0, 32, 32, 2]), 0.79)

    def test_blurred_foreground_overlaps_focused_background_without_a_dark_halo(self):
        image = torch.zeros((1, 64, 64, 3))
        image[:, :, :32, 0] = 0.8
        image[:, :, 32:, 2] = 0.8
        depth = torch.zeros((1, 16, 16, 1))
        depth[:, :, :8] = 1
        result = self.apply(image, depth, lens_blur_apply=True,
                            lens_blur_focus=0, lens_blur_amount=50,
                            lens_blur_bokeh=4,
                            lens_blur_depth_blur_radius=0)
        self.assertGreater(float(result[0, 32, 31, 0]), 0.75)
        self.assertLess(float(result[0, 32, 31, 2]), 0.05)
        self.assertGreater(float(result[0, 32, 32, 0]), 0.2)
        self.assertGreater(float(result[0, 32, 32, 2]), 0.5)

    def test_blur_handles_a_narrow_depth_range(self):
        image = torch.zeros((1, 64, 64, 3))
        image[:, :, :32, 0] = 0.8
        image[:, :, 32:, 2] = 0.8
        depth = torch.full((1, 64, 64, 1), 0.4)
        depth[:, :, :32] = 0.58
        result = self.apply(image, depth, lens_blur_apply=True,
                            lens_blur_focus=0.4, lens_blur_amount=80,
                            lens_blur_bokeh=2)
        self.assertGreater(float(result[0, 32, 16, 0]), 0.2)
        self.assertGreater(float(result[0, 32, 48, 2]), 0.5)

    def test_blur_keeps_a_small_visible_background(self):
        image = torch.full((1, 64, 64, 3), 0.8)
        image[:, :2, :2] = 0.2
        depth = torch.ones((1, 64, 64, 1))
        depth[:, :2, :2] = 0
        result = self.apply(image, depth, lens_blur_apply=True,
                            lens_blur_focus=0, lens_blur_amount=80,
                            lens_blur_bokeh=2)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreater(float(result.mean()), 0.1)

    def test_smooth_depth_still_blurs_away_from_focus(self):
        checker = ((torch.arange(128)[:, None] + torch.arange(128)[None, :]) % 2).float()
        image = checker[None, :, :, None].expand(1, 128, 128, 3).clone()
        depth = torch.linspace(0, 1, 128).view(1, 1, 128, 1).expand(1, 128, 128, 1)
        result = self.apply(image, depth, lens_blur_apply=True,
                            lens_blur_focus=0.5, lens_blur_amount=70,
                            lens_blur_bokeh=2)
        focus_change = (result[:, :, 60:68] - image[:, :, 60:68]).abs().mean()
        far_change = (result[:, :, :16] - image[:, :, :16]).abs().mean()
        self.assertLess(float(focus_change), 0.01)
        self.assertGreater(float(far_change), 0.2)

    def test_bokeh_boost_brightens_isolated_lights_not_bright_planes(self):
        point = torch.zeros((1, 129, 129, 3))
        point[:, 63:66, 63:66] = 1
        depth = torch.zeros((1, 129, 129, 1))
        controls = {"lens_blur_apply": True, "lens_blur_focus": 1,
                    "lens_blur_amount": 50, "lens_blur_bokeh": 4}
        normal = self.apply(point, depth, **controls)
        boosted = self.apply(point, depth, **controls, lens_blur_bokeh_boost=100)
        self.assertGreater(float(boosted.sum()), float(normal.sum()) * 1.25)
        bright_plane = torch.full((1, 32, 32, 3), 0.9)
        normal_plane = self.apply(bright_plane, depth[:, :32, :32], **controls)
        boosted_plane = self.apply(bright_plane, depth[:, :32, :32], **controls,
                                   lens_blur_bokeh_boost=100)
        self.assertTrue(torch.allclose(normal_plane, boosted_plane, atol=1e-5))

    def test_aperture_blades_change_bokeh_shape_and_f_number_changes_spread(self):
        required = self.node.INPUT_TYPES()["required"]
        self.assertEqual(required["lens_blur_bokeh"][1]["min"], 0.1)
        self.assertEqual(required["lens_blur_bokeh"][1]["max"], 30.0)
        self.assertEqual(required["lens_blur_aperture_blades"],
                         ("INT", {"default": 9, "min": 5, "max": 18, "step": 1,
                                  "display": "slider", "tooltip":
                                  self.module["TOOLTIPS"]["lens_blur_aperture_blades"]}))
        point = torch.zeros((1, 129, 129, 3))
        point[:, 64, 64] = 1
        depth = torch.zeros((1, 129, 129, 1))
        five = self.apply(point, depth, lens_blur_apply=True, lens_blur_focus=1,
                          lens_blur_amount=35, lens_blur_bokeh=4,
                          lens_blur_aperture_blades=5)
        nine = self.apply(point, depth, lens_blur_apply=True, lens_blur_focus=1,
                          lens_blur_amount=35, lens_blur_bokeh=4,
                          lens_blur_aperture_blades=9)
        self.assertGreater(float((five - nine).abs().max()), 1e-4)
        wide = self.apply(point, depth, lens_blur_apply=True, lens_blur_focus=1,
                          lens_blur_amount=35, lens_blur_bokeh=0.1)
        narrow = self.apply(point, depth, lens_blur_apply=True, lens_blur_focus=1,
                            lens_blur_amount=35, lens_blur_bokeh=30)
        self.assertLess(float(wide[:, 64, 64].mean()), float(narrow[:, 64, 64].mean()))
        self.assertGreater(int((wide[..., 0] > 1e-6).sum()),
                           int((narrow[..., 0] > 1e-6).sum()))

    def test_stopped_down_aperture_adds_blade_dependent_rays_to_point_lights(self):
        point = torch.zeros((1, 129, 129, 3))
        point[:, 64, 64] = 1
        depth = torch.zeros((1, 129, 129, 1))
        common = {"lens_blur_apply": True, "lens_blur_focus": 0,
                  "lens_blur_amount": 50, "lens_blur_bokeh": 30}
        five = self.apply(point, depth, **common, lens_blur_aperture_blades=5)
        six = self.apply(point, depth, **common, lens_blur_aperture_blades=6)
        self.assertGreater(float((five - point).abs().max()), 1e-3)
        self.assertGreater(float((five - six).abs().max()), 1e-3)
        wide = self.apply(point, depth, **{**common, "lens_blur_bokeh": 4},
                          lens_blur_aperture_blades=5)
        self.assertTrue(torch.equal(wide, point))
        self.assertEqual(self.module["_starburst_kernel"](8, 5, "cpu", torch.float32).shape,
                         (1, 1, 17, 17))

    def test_auto_white_balance_reduces_simple_cast(self):
        image = torch.tensor([0.55, 0.48, 0.43]).reshape(1, 1, 1, 3).expand(1, 24, 24, 3).clone()
        corrected = self.apply(image, color_white_balance="Auto")
        before = float(image[..., 0].mean() - image[..., 2].mean())
        after = float(corrected[..., 0].mean() - corrected[..., 2].mean())
        self.assertLess(after, before)

    def test_auto_light_suggestion_does_not_change_execution_settings(self):
        required = self.node.INPUT_TYPES()["required"]
        self.assertNotIn("auto_light_request", required)
        dark = torch.linspace(0.03, 0.4, 32).reshape(1, 1, 32, 1).expand(1, 32, 32, 3).clone()
        bright = torch.linspace(0.6, 0.97, 32).reshape(1, 1, 32, 1).expand(1, 32, 32, 3).clone()
        dark_auto = self.module["_auto_light_controls"](dark)
        bright_auto = self.module["_auto_light_controls"](bright)
        self.assertGreater(dark_auto["light_exposure"], 0)
        self.assertLess(bright_auto["light_exposure"], 0)
        for name in dark_auto:
            self.assertIn(name, required)
            self.assertGreaterEqual(dark_auto[name], required[name][1]["min"])
            self.assertLessEqual(dark_auto[name], required[name][1]["max"])
        self.assertEqual(self.module["_auto_light_controls"](torch.zeros_like(dark)),
                         {name: 0 for name in dark_auto})
        settings = self.module["_defaults"]()
        settings["light_exposure"] = -1.0
        response = self.node.execute(dark, **settings)
        self.assertEqual(response["ui"]["auto_light_suggestion"][0], dark_auto)
        expected = self.apply(dark, **settings)
        self.assertTrue(torch.allclose(response["result"][0], expected))
        self.assertIsNone(response["ui"]["depth_preview"][0])
        self.assertEqual(len(response["result"]), 1)

    def test_depth_preview_uses_input_depth_values_even_when_blur_is_off(self):
        image = torch.zeros((1, 2, 2, 3))
        depth = torch.tensor([0.0, 0.25, 0.5, 1.0]).reshape(1, 2, 2, 1)
        response = self.node.execute(image, active=False, depth_image=depth,
                                     lens_blur_depth_blur_radius=0)
        self.assertIs(response["result"][0], image)
        uri = response["ui"]["depth_preview"][0]
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        preview = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        self.assertEqual(preview.size, (2, 2))
        self.assertEqual(list(preview.tobytes()), [0, 64, 128, 255])
        large = torch.rand((1, 400, 200, 1))
        uri = self.module["_depth_preview"](large)
        preview = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        self.assertEqual(preview.size, (160, 320))
        image = torch.rand((1, 16, 16, 3))
        depth = torch.rand((1, 16, 16, 1))
        settings = self.module["_defaults"]()
        settings["lens_blur_apply"] = True
        response = self.node.execute(image, depth_image=depth, **settings)
        self.assertTrue(response["ui"]["depth_preview"][0])
        self.assertTrue(torch.equal(response["result"][0],
                                    self.node.apply(image, depth_image=depth, **settings)[0]))

    def test_depth_boundary_smoothing_matches_preview_and_can_be_disabled(self):
        required = self.node.INPUT_TYPES()["required"]
        self.assertEqual(required["lens_blur_depth_blur_radius"][1]["default"], 5)
        self.assertEqual(required["lens_blur_depth_sigma"][1]["default"], 2.0)
        image = torch.zeros((1, 32, 32, 3))
        image[:, :, :16, 0] = 0.8
        image[:, :, 16:, 2] = 0.8
        depth = torch.zeros((1, 32, 32, 1))
        depth[:, :, :16] = 1
        rgb = image.permute(0, 3, 1, 2)
        raw = self.module["_prepare_depth"](depth, rgb, True, 0, 2.0)
        smooth = self.module["_prepare_depth"](depth, rgb, True, 5, 2.0)
        self.assertTrue(torch.equal(raw, depth.permute(0, 3, 1, 2)))
        self.assertLess(float(smooth[0, 0, 16, 15]), 1)
        self.assertGreater(float(smooth[0, 0, 16, 16]), 0)
        response = self.node.execute(image, active=False, depth_image=depth)
        uri = response["ui"]["depth_preview"][0]
        preview = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        self.assertAlmostEqual(preview.getpixel((15, 16)),
                               round(float(smooth[0, 0, 16, 15]) * 255), delta=1)
        controls = {"lens_blur_apply": True, "lens_blur_focus": 1,
                    "lens_blur_amount": 60, "lens_blur_bokeh": 4}
        sharp_depth = self.apply(image, depth, **controls, lens_blur_depth_blur_radius=0)
        soft_depth = self.apply(image, depth, **controls)
        self.assertGreater(float((sharp_depth - soft_depth).abs().max()), 0.01)

    def test_camera_raw_style_slider_ranges_and_optics_vignette_direction(self):
        required = self.node.INPUT_TYPES()["required"]
        self.assertEqual(required["color_temperature"][1]["min"], -100)
        self.assertEqual(required["color_temperature"][1]["max"], 100)
        self.assertIn("Not Kelvin", required["color_temperature"][1]["tooltip"])
        self.assertEqual(required["light_exposure"][1]["display"], "slider")
        self.assertEqual(required["optics_vignette"][1]["min"], -100)
        self.assertEqual(required["grain_seed"][1]["display"], "number")
        image = torch.full((1, 25, 25, 3), 0.5)
        darkened = self.apply(image, optics_vignette=-50)
        self.assertLess(float(darkened[0, 0, 0, 0]), float(darkened[0, 12, 12, 0]))

    def test_preset_round_trip_and_validation(self):
        save = self.module["_save_preset"]
        read = self.module["_read_presets"]
        path_a = Path(self.temp.name) / "a" / "presets.json"
        path_b = Path(self.temp.name) / "b" / "presets.json"
        settings = self.module["_defaults"]()
        settings["light_exposure"] = 0.7
        save(path_a, "Portrait", settings, False)
        self.assertEqual(read(path_a)["Portrait"]["light_exposure"], 0.7)
        self.assertEqual(read(path_b), {})
        with self.assertRaises(FileExistsError):
            save(path_a, "Portrait", settings, False)
        with self.assertRaises(ValueError):
            save(path_a, "../escape", settings, False)
        with self.assertRaises(ValueError):
            save(path_a, "Bad", {**settings, "light_exposure": float("nan")}, False)
        with self.assertRaises(ValueError):
            save(path_a, "Bad", {"light_exposure": 1}, False)
        self.assertEqual(json.loads(path_a.read_text())["version"], 1)

    def test_preset_without_aperture_blades_uses_nine_blade_default(self):
        path = Path(self.temp.name) / "old" / "presets.json"
        path.parent.mkdir(parents=True)
        old_settings = self.module["_defaults"]()
        old_settings.pop("lens_blur_aperture_blades")
        old_settings.pop("lens_blur_depth_blur_radius")
        old_settings.pop("lens_blur_depth_sigma")
        path.write_text(json.dumps({"version": 1, "presets": {"Old": old_settings}}))
        restored = self.module["_read_presets"](path)
        self.assertEqual(restored["Old"]["lens_blur_aperture_blades"], 9)
        self.assertEqual(restored["Old"]["lens_blur_depth_blur_radius"], 5)
        self.assertEqual(restored["Old"]["lens_blur_depth_sigma"], 2.0)
        restored["Old"]["lens_blur_aperture_blades"] = 11
        self.module["_save_preset"](path, "New", restored["Old"], False)
        self.assertEqual(self.module["_read_presets"](path)["New"]["lens_blur_aperture_blades"], 11)

    def test_preset_routes_use_the_current_user(self):
        settings = self.module["_defaults"]()
        body = json.dumps({"name": "Soft", "settings": settings, "overwrite": False}).encode()
        request = MagicMock()
        request.user = "route-user"
        request.content_length = len(body)
        request.content.read = AsyncMock(return_value=body)
        posted = asyncio.run(self.module["photo_studio_save_preset"](request))
        self.assertEqual(posted.status, 200)
        listed = asyncio.run(self.module["photo_studio_presets"](request))
        self.assertIn("Soft", json.loads(listed.text)["presets"])
        second = MagicMock()
        second.user = "another-user"
        second_list = asyncio.run(self.module["photo_studio_presets"](second))
        self.assertEqual(json.loads(second_list.text)["presets"], {})
        conflict = asyncio.run(self.module["photo_studio_save_preset"](request))
        self.assertEqual(conflict.status, 409)


if __name__ == "__main__":
    unittest.main()
