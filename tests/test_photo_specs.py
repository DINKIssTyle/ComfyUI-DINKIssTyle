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
        self.assertEqual(inputs["required"]["resolution_multiple"][1]["default"], "8")
        self.assertEqual(inputs["optional"]["image"], ("IMAGE",))

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
        for multiple in (4, 8, 16, 32):
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
