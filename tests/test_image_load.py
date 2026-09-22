import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"


class ImageLoadTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.temp_directory.name)
        (self.input_dir / "root.png").touch()
        (self.input_dir / "ignore.txt").touch()
        (self.input_dir / "portraits" / "studio").mkdir(parents=True)
        (self.input_dir / "portraits" / "studio" / "person.webp").touch()

        folder_paths = MagicMock()
        folder_paths.get_input_directory.return_value = str(self.input_dir)
        self.temp_dir = self.input_dir / "temp"
        self.temp_dir.mkdir()
        folder_paths.get_temp_directory.return_value = str(self.temp_dir)
        routes = MagicMock()
        routes.get.side_effect = lambda _path: lambda function: function
        routes.post.side_effect = lambda _path: lambda function: function
        prompt_server = MagicMock()
        prompt_server.instance.routes = routes
        server = MagicMock()
        server.PromptServer = prompt_server
        aiohttp = MagicMock()
        aiohttp.web = MagicMock()

        with patch.dict(
            sys.modules,
            {"folder_paths": folder_paths, "server": server, "aiohttp": aiohttp},
        ):
            self.module = runpy.run_path(str(ROOT / "dinki_load.py"))

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_lists_root_files_and_nested_image_categories(self):
        node = self.module["DINKI_Image_Load"]
        required = node.INPUT_TYPES()["required"]
        self.assertEqual(required["category"][0], ["", "portraits/studio"])
        self.assertEqual(required["filename"][0], ["root.png"])

    def test_category_controls_file_list(self):
        self.assertEqual(
            self.module["_image_files"]("portraits/studio"),
            ["person.webp"],
        )

    def test_rejects_path_traversal_and_non_images(self):
        image_path = self.module["_image_path"]
        with self.assertRaises(ValueError):
            image_path("../outside", "file.png")
        with self.assertRaises(ValueError):
            image_path("", "../root.png")
        with self.assertRaises(ValueError):
            image_path("", "ignore.txt")

    def test_declares_image_mask_and_alpha_outputs(self):
        node = self.module["DINKI_Image_Load"]
        self.assertEqual(node.RETURN_TYPES, ("IMAGE", "MASK", "MASK"))
        self.assertEqual(node.RETURN_NAMES, ("IMAGE", "MASK", "ALPHA"))

    def test_backend_source_reports_resolution(self):
        source = (ROOT / "dinki_load.py").read_text()
        self.assertIn('"resolution": [f"{width} × {height}"]', source)

    def test_frontend_uses_nodes_2_dom_widget(self):
        source = (ROOT / "js" / "dinki_nodes.js").read_text()
        self.assertIn('node.addDOMWidget(', source)
        self.assertIn('"dkst_image_preview"', source)
        self.assertIn('getMinHeight: () => 120', source)
        self.assertIn('getMaxHeight: () => 320', source)
        self.assertIn('getHeight: () => 240', source)

    def test_removes_only_managed_paste_files_on_next_load(self):
        managed = self.temp_dir / "DKST_Paste_old.png"
        unrelated = self.temp_dir / "other.png"
        managed.touch()
        unrelated.touch()
        self.module["_cleanup_pasted_temp_images"]()
        self.assertFalse(managed.exists())
        self.assertTrue(unrelated.exists())

    def test_deletes_only_an_explicit_managed_temp_image(self):
        managed = self.temp_dir / "DKST_Paste_current.png"
        unrelated = self.temp_dir / "other.png"
        managed.touch()
        unrelated.touch()
        self.module["_delete_pasted_temp_image"](managed.name)
        self.assertFalse(managed.exists())
        self.assertTrue(unrelated.exists())
        with self.assertRaises(ValueError):
            self.module["_delete_pasted_temp_image"](unrelated.name)


if __name__ == "__main__":
    unittest.main()
