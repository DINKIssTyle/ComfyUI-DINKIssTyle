import runpy
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"
NODES = runpy.run_path(str(ROOT / "dinki_text.py"))
Multiline = NODES["DINKI_Text_Multiline"]
MultilineCache = NODES["DINKI_Text_Multiline_Cache"]
Note = NODES["DINKI_Text_Note"]
Concatenate = NODES["DINKI_Text_Concatenate"]
Split = NODES["DINKI_Text_Split"]


class TextNodeTests(unittest.TestCase):
    def test_multiline_cache_has_a_separate_raw_incoming_socket_and_saved_text(self):
        schema = MultilineCache.INPUT_TYPES()
        self.assertTrue(schema["required"]["text"][1]["multiline"])
        self.assertFalse(schema["required"]["text"][1]["dynamicPrompts"])
        incoming_type, options = schema["optional"]["text_in"]
        self.assertEqual(incoming_type, "STRING")
        self.assertTrue(options["forceInput"])
        self.assertTrue(options["rawLink"])
        self.assertFalse(options.get("lazy", False), "enabled sources must execute before reading their result")
        self.assertEqual(schema["hidden"], {"execution_list": "EXECUTION_LIST", "unique_id": "UNIQUE_ID"})
        self.assertEqual(MultilineCache.RETURN_TYPES, ("STRING",))
        self.assertEqual(MultilineCache.RETURN_NAMES, ("text",))

    def test_multiline_cache_new_text_replaces_old_text_unchanged(self):
        text = "첫 번째 줄\n  second line {literal|prompt}  "
        result = MultilineCache().output_text("previous", text)
        self.assertEqual(result, {"ui": {"dkst_cached_text": [text]}, "result": (text,)})

    def test_multiline_cache_absent_empty_and_blocked_input_preserve_the_widget(self):
        saved = "cached\n  text  "
        for value in (None, "", object(), [], [None, "", object()], 42):
            with self.subTest(value=value):
                self.assertEqual(MultilineCache().output_text(saved, value),
                                 {"ui": {}, "result": (saved,)})
        self.assertEqual(MultilineCache().output_text(saved)["result"], (saved,))
        self.assertEqual(MultilineCache().output_text()["result"], ("",))

    def test_multiline_cache_reads_the_requested_raw_link_output(self):
        calls = []
        entry = types.SimpleNamespace(outputs=[["wrong socket"], ["new\ntext"]])
        execution_list = types.SimpleNamespace(get_cache=lambda source, target:
            calls.append((source, target)) or entry)
        result = MultilineCache().output_text("old", ["producer", 1], execution_list, "cache-node")
        self.assertEqual(calls, [("producer", "cache-node")])
        self.assertEqual(result["result"], ("new\ntext",))

    def test_multiline_cache_raw_blocked_or_missing_outputs_do_not_erase_text(self):
        for entry in (None, types.SimpleNamespace(outputs=None), types.SimpleNamespace(outputs=[]),
                      types.SimpleNamespace(outputs=[[None, "", object()]])):
            execution_list = types.SimpleNamespace(get_cache=lambda source, target: entry)
            with self.subTest(entry=entry):
                result = MultilineCache().output_text("saved", ["source", 0], execution_list, "target")
                self.assertEqual(result, {"ui": {}, "result": ("saved",)})
        self.assertEqual(MultilineCache().output_text("saved", ["source", 0])["result"], ("saved",))

    def test_multiline_cache_rejects_invalid_link_indexes_and_missing_context(self):
        execution_list = types.SimpleNamespace(get_cache=lambda source, target:
            types.SimpleNamespace(outputs=[["new"]]))
        for link, context, node_id in ((["source", -1], execution_list, "target"),
                                      (["source", 2], execution_list, "target"),
                                      (["source", 0], execution_list, None),
                                      (["source", 0], None, "target")):
            with self.subTest(link=link, node_id=node_id):
                self.assertEqual(MultilineCache().output_text("saved", link, context, node_id)["result"], ("saved",))

    def test_multiline_cache_keeps_last_nonempty_text_in_a_source_list(self):
        execution_list = types.SimpleNamespace(get_cache=lambda source, target:
            types.SimpleNamespace(outputs=[["first", None, "last\nline", "", object()]]))
        result = MultilineCache().output_text("saved", ["source", 0], execution_list, "target")
        self.assertEqual(result["result"], ("last\nline",))
        self.assertEqual(result["ui"]["dkst_cached_text"], ["last\nline"])
        self.assertEqual(MultilineCache().output_text("saved", " \n ")["result"], (" \n ",))

    def test_multiline_cache_survives_recreation_without_sharing_node_state(self):
        first = MultilineCache().output_text("", "received")
        saved_widget = first["ui"]["dkst_cached_text"][0]
        self.assertEqual(MultilineCache().output_text(saved_widget)["result"], ("received",))
        self.assertEqual(MultilineCache().output_text()["result"], ("",))
        self.assertEqual(MultilineCache().output_text("manual replacement")["result"], ("manual replacement",))
        self.assertEqual(MultilineCache().output_text("")["result"], ("",))

    def test_note_is_text_only_and_has_no_output(self):
        self.assertTrue(Note.INPUT_TYPES()["required"]["text"][1]["multiline"])
        self.assertEqual(Note.CATEGORY, "DINKIssTyle/Text")
        self.assertEqual(Note.RETURN_TYPES, ())
        self.assertEqual(Note().note("Saved note"), ())

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
