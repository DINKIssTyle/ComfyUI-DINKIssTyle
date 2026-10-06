import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class ExecutionBlocker:
    def __init__(self, message):
        self.message = message


graph_utils = types.ModuleType("comfy_execution.graph_utils")
graph_utils.ExecutionBlocker = ExecutionBlocker
with patch.dict(sys.modules, {
    "comfy_execution": types.ModuleType("comfy_execution"),
    "comfy_execution.graph_utils": graph_utils,
}):
    node = runpy.run_path(str(Path(__file__).resolve().parents[1]
                             / "ComfyUI-DINKIssTyle" / "dinki_image_size.py"))["DINKI_GetImageSize"]


class ImageSizeTests(unittest.TestCase):
    def test_image_is_optional_and_outputs_match_core_node(self):
        self.assertEqual(node.INPUT_TYPES(), {"required": {}, "optional": {"image": ("IMAGE",)}})
        self.assertEqual(node.RETURN_NAMES, ("width", "height", "batch_size"))
        self.assertEqual(node.RETURN_TYPES, ("INT", "INT", "INT"))

    def test_rgb_and_rgba_batches_report_dimensions(self):
        for channels in (3, 4):
            with self.subTest(channels=channels):
                image = types.SimpleNamespace(shape=(2, 768, 1024, channels))
                self.assertEqual(node().get_size(image), (1024, 768, 2))

    def test_missing_blocked_and_empty_images_block_all_outputs(self):
        for image in (None, ExecutionBlocker(None),
                      types.SimpleNamespace(shape=(0, 768, 1024, 3)),
                      types.SimpleNamespace(shape=(1, 0, 1024, 3)),
                      types.SimpleNamespace(shape=(1, 768, 0, 3))):
            with self.subTest(image=image):
                outputs = node().get_size(image)
                self.assertEqual(len(outputs), 3)
                self.assertTrue(all(isinstance(value, ExecutionBlocker) and value.message is None
                                    for value in outputs))
        self.assertTrue(all(isinstance(value, ExecutionBlocker) for value in node().get_size()))


if __name__ == "__main__":
    unittest.main()
