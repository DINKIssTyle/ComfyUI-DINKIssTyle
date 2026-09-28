import csv
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


NODE_FILE = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle/dinki_prompt_csv_library.py"


class PromptCsvLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        routes = MagicMock()
        routes.get.side_effect = lambda _path: lambda function: function
        server = MagicMock()
        server.PromptServer.instance.routes = routes
        aiohttp = MagicMock()
        with patch.dict(sys.modules, {"server": server, "aiohttp": aiohttp}):
            cls.module = runpy.run_path(str(NODE_FILE))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name)
        with (path / "Cinema_Prompt.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows([
                ("Category", "Technique Name", "Prompt it Code"),
                ("Camera Movement", "", ""),
                ("Camera Movement", "SLOW ZOOM IN", "Slow zoom, with a fixed camera."),
                ("Camera Movement", "DOLLY IN", "Forward camera move."),
                ("Lighting", "DAYLIGHT", "Soft daylight."),
                ("Lighting", "EMPTY", ""),
            ])
        self.module["_csv_path"].__globals__["CSV_DIRECTORY"] = path

    def tearDown(self):
        self.directory.cleanup()

    def test_cinema_file_has_expected_sections_and_skips_headers(self):
        sections = self.module["_csv_sections"]("Cinema_Prompt.csv")
        self.assertEqual(len(sections), 2)
        self.assertEqual(sum(map(len, sections.values())), 3)
        self.assertIn("SLOW ZOOM IN", sections["Camera Movement"])
        self.assertEqual(sections["Camera Movement"]["SLOW ZOOM IN"],
                         "Slow zoom, with a fixed camera.")
        self.assertNotIn("EMPTY", sections["Lighting"])
        self.assertNotIn("Category", sections)

    def test_input_types_use_cinema_by_default_and_have_string_port(self):
        node = self.module["DINKI_PromptCsvLibrary"]
        inputs = node.INPUT_TYPES()
        self.assertIn("Cinema_Prompt.csv", inputs["required"]["csv_file"][0])
        self.assertEqual(inputs["required"]["csv_file"][1]["default"], "Cinema_Prompt.csv")
        self.assertEqual(inputs["required"]["prompt"][1]["multiline"], True)
        self.assertEqual(inputs["optional"]["text_input"], ("STRING", {"forceInput": True}))
        self.assertEqual(node.RETURN_TYPES, ("STRING",))

    def test_rejects_paths_outside_csv_folder(self):
        csv_path = self.module["_csv_path"]
        for filename in ("../Cinema_Prompt.csv", "/tmp/other.csv", "..\\other.csv", "other.txt"):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                csv_path(filename)

    def test_output_uses_current_edited_prompt_and_optional_prefix(self):
        node = self.module["DINKI_PromptCsvLibrary"]()
        args = ("Cinema_Prompt.csv", "Camera Movement", "SLOW ZOOM IN", "edited text")
        self.assertEqual(node.compose_prompt(*args), ("edited text",))
        self.assertEqual(node.compose_prompt(*args, text_input="prefix"), ("prefix, edited text",))
        self.assertEqual(node.compose_prompt(*args[:-1], "", text_input="prefix"), ("prefix",))


if __name__ == "__main__":
    unittest.main()
