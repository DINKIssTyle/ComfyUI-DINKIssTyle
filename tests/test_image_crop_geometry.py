"""Exercise crop geometry without requiring the ComfyUI/PyTorch runtime."""

import ast
import math
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle" / "dinki_image_crop.py"
tree = ast.parse(SOURCE.read_text())
parts = [node for node in tree.body if
         isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
         target.id == "RATIOS" for target in node.targets) or
         isinstance(node, ast.FunctionDef) and node.name in ("_ratio_parts", "_crop_bounds") or
         isinstance(node, ast.ClassDef) and node.name == "DINKI_Image_Crop"]
namespace = {"math": math}
exec(compile(ast.Module(body=parts, type_ignores=[]), str(SOURCE), "exec"), namespace)
crop_bounds = namespace["_crop_bounds"]


class CropGeometryTests(unittest.TestCase):
    def test_preview_node_is_always_reexecuted(self):
        node_type = namespace["DINKI_Image_Crop"]
        self.assertTrue(node_type.OUTPUT_NODE)
        changed = node_type.IS_CHANGED()
        self.assertNotEqual(changed, changed)

    def test_original_is_lossless(self):
        self.assertEqual(crop_bounds(403, 301, "Original", 1, 1, 0, 0, 1, 1),
                         (0, 0, 403, 301))

    def test_preset_center_crop_has_exact_ratio(self):
        left, top, width, height = crop_bounds(400, 300, "1:1", 1, 1, 0, 0, 1, 1)
        self.assertEqual((left, top, width, height), (50, 0, 300, 300))
        self.assertEqual(crop_bounds(400, 300, "16:9", 1, 1, 0, 0, 1, 1),
                         (0, 38, 400, 225))

    def test_custom_ratio_and_normalized_position(self):
        self.assertEqual(crop_bounds(400, 300, "Custom", 4, 5,
                                     0.5, 0.2, 0.5, 0.8),
                         (204, 60, 192, 240))

    def test_pixel_rounding_does_not_jump_down_a_ratio_step(self):
        self.assertEqual(crop_bounds(400, 300, "4:3", 1, 1,
                                     0.1, 0.1, 0.5025, 0.5033333333)[2:],
                         (200, 150))

    def test_coprime_source_and_tiny_crop_remain_valid(self):
        left, top, width, height = crop_bounds(403, 301, "Original", 1, 1,
                                               0.2, 0.1, 0.4, 0.4)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        self.assertLessEqual(left + width, 403)
        self.assertLessEqual(top + height, 301)

    def test_invalid_geometry_rejected(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                crop_bounds(400, 300, "Original", 1, 1, value, 0, 1, 1)
        with self.assertRaises(ValueError):
            crop_bounds(400, 300, "Custom", 0, 5, 0, 0, 1, 1)


if __name__ == "__main__":
    unittest.main()
