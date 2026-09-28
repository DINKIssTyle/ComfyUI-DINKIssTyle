import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace


NODE_FILE = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle/dinki_photo_specs.py"
PhotoSpecs = runpy.run_path(str(NODE_FILE))["DINKI_photo_specifications"]


class PhotoSpecsTests(unittest.TestCase):
    def setUp(self):
        self.node = PhotoSpecs()

    def test_input_defaults_keep_custom_workflows_on_multiple_eight(self):
        inputs = PhotoSpecs.INPUT_TYPES()
        self.assertEqual(
            list(inputs["required"]),
            ["resolution", "resolution_multiple", "megapixels", "aspect_ratio", "orientation"],
        )
        self.assertEqual(inputs["required"]["resolution"][1]["default"], "Custom")
        self.assertEqual(inputs["required"]["resolution_multiple"],
                         ("INT", {"default": 8, "min": 4, "max": 128, "step": 4}))
        self.assertEqual(inputs["required"]["megapixels"],
                         ("FLOAT", {"default": 1.0, "min": 0.1, "max": 64.0,
                                    "step": 0.01, "round": 0.01}))
        self.assertEqual(inputs["optional"]["image"], ("IMAGE",))

    def test_ai_generation_megapixel_presets(self):
        for preset, side in (("0.25MP", 512), ("0.56MP", 768),
                             ("1MP", 1024), ("1.68MP", 1328), ("4MP", 2048)):
            with self.subTest(preset=preset):
                width, height, info = self.node.calculate_resolution(
                    preset, "Basic 1:1", False)
                self.assertEqual((width, height), (side, side))
                self.assertIn(preset, info)

    def test_numeric_megapixels_support_hundredths_and_multiple_of_four(self):
        for megapixels in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98):
            with self.subTest(megapixels=megapixels):
                width, height, _ = self.node.calculate_resolution(
                    megapixels, "Basic 1:1", False)
                self.assertGreater(width, 0)
                self.assertEqual(width, height)
        width, height, info = self.node.calculate_resolution(
            0.98, "Basic 1:1", False, resolution_multiple=12)
        self.assertEqual((width, height), (1008, 1008))
        self.assertIn("0.98MP, multiple 12", info)

    def test_existing_custom_result_and_old_call_signature(self):
        self.assertEqual(
            self.node.calculate_resolution("1MP", "Basic 1:1", False),
            (1024, 1024, "1024x1024 (Basic 1:1, 1MP)"),
        )
        self.assertEqual(
            self.node.calculate_resolution("1MP", "Photo 4:6", "Landscape")[:2],
            (1256, 840),
        )

    def test_image_mode_uses_source_ratio_and_direction(self):
        image = SimpleNamespace(shape=(2, 900, 1600, 3))
        width, height, info = self.node.calculate_resolution(
            "1MP", "ignored in Image mode", "ignored", resolution="Image",
            resolution_multiple="32", image=image,
        )
        self.assertEqual((width, height), (1376, 768))
        self.assertIn("Image 1600x900, 16:9, 1MP, multiple 32", info)

    def test_selected_multiple_applies_to_both_dimensions(self):
        for multiple in (4, 8, 12, 16, 32, 128):
            with self.subTest(multiple=multiple):
                width, height, _ = self.node.calculate_resolution(
                    "2MP", "Photo 3:4", False,
                    resolution_multiple=str(multiple),
                )
                self.assertEqual(width % multiple, 0)
                self.assertEqual(height % multiple, 0)
                self.assertLess(width, height)

    def test_image_mode_requires_valid_image(self):
        with self.assertRaisesRegex(ValueError, "connect an image"):
            self.node.calculate_resolution("1MP", "Basic 1:1", False, resolution="Image")
        with self.assertRaisesRegex(ValueError, "shape"):
            self.node.calculate_resolution(
                "1MP", "Basic 1:1", False, resolution="Image",
                image=SimpleNamespace(shape=(900, 1600, 3)),
            )


if __name__ == "__main__":
    unittest.main()
