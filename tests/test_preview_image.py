import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"


class PreviewImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder_paths = MagicMock()
        folder_paths.get_output_directory.return_value = "/output"
        folder_paths.get_temp_directory.return_value = "/temp"
        with patch.dict(sys.modules, {"folder_paths": folder_paths, "numpy": MagicMock()}):
            module = runpy.run_path(str(ROOT / "dinki_preview.py"))
        cls.folder_paths = folder_paths
        cls.node_class = module["DINKI_Preview_Image"]

    def test_declares_requested_inputs_and_defaults(self):
        required = self.node_class.INPUT_TYPES()["required"]
        self.assertEqual(required["filename_prefix"][1]["default"], "DKST_Image ")
        self.assertEqual(required["format"][0], ["png", "exr", "avif", "webp"])
        self.assertEqual(required["bit_depth"][0], ["8bit", "16bit"])
        self.assertEqual(required["input_color_space"][0], ["sRGB"])
        self.assertFalse(required["always_save"][1]["default"])

    def test_is_image_output_node(self):
        self.assertEqual(self.node_class.RETURN_TYPES, ("IMAGE",))
        self.assertTrue(self.node_class.OUTPUT_NODE)

    def test_display_name_registration(self):
        with patch.dict(sys.modules, {"folder_paths": MagicMock(), "numpy": MagicMock()}):
            module = runpy.run_path(str(ROOT / "dinki_preview.py"))
        self.assertEqual(
            module["NODE_DISPLAY_NAME_MAPPINGS"]["DINKI_Preview_Image"],
            "DKST Preview (Image)",
        )

    def test_uses_temp_or_output_directory_and_reports_resolution(self):
        class FakeImages(list):
            shape = (1, 720, 1280, 3)

        images = FakeImages([object()])
        for always_save, expected_directory, expected_type in (
            (False, "/temp", "temp"),
            (True, "/output", "output"),
        ):
            with self.subTest(always_save=always_save):
                self.folder_paths.get_save_image_path.reset_mock()
                self.folder_paths.get_save_image_path.return_value = (
                    "/target",
                    "DKST_Image",
                    1,
                    "",
                    "DKST_Image",
                )
                node = self.node_class()
                with (
                    patch.object(node, "_tensor_to_numpy", return_value=object()),
                    patch.object(node, "_save_image"),
                ):
                    result = node.preview_image(images, always_save=always_save)

                call = self.folder_paths.get_save_image_path.call_args
                self.assertEqual(call.args[1], expected_directory)
                self.assertEqual(result["ui"]["images"][0]["type"], expected_type)
                self.assertEqual(result["ui"]["resolution"], ["1280 × 720"])
                self.assertIs(result["result"][0], images)


if __name__ == "__main__":
    unittest.main()
