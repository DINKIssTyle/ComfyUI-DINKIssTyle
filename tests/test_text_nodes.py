import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"
NODES = runpy.run_path(str(ROOT / "dinki_text.py"))
Multiline = NODES["DINKI_Text_Multiline"]
Concatenate = NODES["DINKI_Text_Concatenate"]
Split = NODES["DINKI_Text_Split"]


class TextNodeTests(unittest.TestCase):
    def test_split_literal_phrase_and_trim(self):
        result = Split().split_text("  첫째 [다음] 둘째 [다음] 셋째  ", "[다음]")
        self.assertEqual(result, ("첫째", "둘째", "셋째") + ("",) * 7)

    def test_split_preserves_empty_section_positions(self):
        self.assertEqual(
            Split().split_text(",alpha,,beta,"),
            ("", "alpha", "", "beta") + ("",) * 6,
        )

    def test_split_retains_overflow_in_last_output(self):
        text = "|".join(str(i) for i in range(1, 13))
        self.assertEqual(
            Split().split_text(text, "|"),
            tuple(str(i) for i in range(1, 10)) + ("10|11|12",),
        )

    def test_split_empty_delimiter_and_missing_delimiter(self):
        for delimiter in ("", "absent"):
            self.assertEqual(Split().split_text("original", delimiter), ("original",) + ("",) * 9)
        self.assertEqual(Split().split_text(""), ("",) * 10)

    def test_split_newlines_and_preserves_whitespace_when_disabled(self):
        self.assertEqual(
            Split().split_text(" first \n second ", "\n", False),
            (" first ", " second ") + ("",) * 8,
        )

    def test_split_is_case_sensitive_and_does_not_interpret_regex(self):
        self.assertEqual(Split().split_text("a.*b.*C", ".*"), ("a", "b", "C") + ("",) * 7)
        self.assertEqual(Split().split_text("aXbxc", "x"), ("aXb", "c") + ("",) * 8)

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
