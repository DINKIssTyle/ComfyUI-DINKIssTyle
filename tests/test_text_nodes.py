import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"
NODES = runpy.run_path(str(ROOT / "dinki_text.py"))
Multiline = NODES["DINKI_Text_Multiline"]
Concatenate = NODES["DINKI_Text_Concatenate"]


class TextNodeTests(unittest.TestCase):
    def test_multiline_text_is_returned_unchanged(self):
        text = "first line\n  second line"
        self.assertEqual(Multiline().output_text(text), (text,))
        self.assertTrue(Multiline.INPUT_TYPES()["required"]["text"][1]["multiline"])

    def test_concatenate_has_ten_text_inputs(self):
        optional = Concatenate.INPUT_TYPES()["optional"]
        self.assertEqual(list(optional), [f"text_{letter}" for letter in "abcdefghij"])
        self.assertTrue(all(config[1]["forceInput"] for config in optional.values()))

    def test_concatenate_cleans_edges_and_omits_empty_inputs(self):
        result = Concatenate().concatenate(
            delimiter=" | ",
            clean_whitespace=True,
            text_a="  alpha  ",
            text_b="   ",
            text_c="beta\n",
        )
        self.assertEqual(result, ("alpha | beta",))

    def test_concatenate_can_preserve_whitespace(self):
        result = Concatenate().concatenate(
            delimiter=",",
            clean_whitespace=False,
            text_a=" alpha ",
            text_b="beta",
        )
        self.assertEqual(result, (" alpha ,beta",))


if __name__ == "__main__":
    unittest.main()
