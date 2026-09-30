import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


NODE_FILE = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle" / "dinki_switch.py"


class ExecutionBlocker:
    def __init__(self, message):
        self.message = message


graph_utils = types.ModuleType("comfy_execution.graph_utils")
graph_utils.ExecutionBlocker = ExecutionBlocker
graph_utils.is_link = lambda value: (isinstance(value, list) and len(value) == 2
                                     and isinstance(value[0], str) and isinstance(value[1], int))
with patch.dict(sys.modules, {"comfy_execution": types.ModuleType("comfy_execution"),
                              "comfy_execution.graph_utils": graph_utils}):
    nodes = runpy.run_path(str(NODE_FILE))
    Switch = nodes["DINKI_IfElseSwitch"]
    ImageSwitch = nodes["DINKI_IfElseImageSwitch"]
    Branch = nodes["DINKI_IfElseBranch"]


class FakeExecutionList:
    def __init__(self, value):
        self.value = value

    def get_cache(self, source_id, target_id):
        assert (source_id, target_id) == ("source", "switch")
        return types.SimpleNamespace(outputs=[self.value])


class IfElseSwitchTests(unittest.TestCase):
    def test_socket_schema(self):
        schema = Switch.INPUT_TYPES()
        self.assertEqual(schema["required"], {})
        self.assertEqual(
            list(schema["optional"])[:-3],
            [f"on_{branch}_{i}" for i in range(1, 11) for branch in ("false", "true")],
        )
        self.assertEqual(list(schema["optional"])[-3:], ["switch", "default_switch", "empty_as_none"])
        self.assertTrue(all(config[1]["lazy"] for name, config in schema["optional"].items()
                            if name not in ("switch", "default_switch", "empty_as_none")))
        self.assertTrue(schema["optional"]["switch"][1]["forceInput"])
        self.assertTrue(schema["optional"]["empty_as_none"][1]["advanced"])
        self.assertEqual(Switch.RETURN_NAMES, tuple(f"output_{i}" for i in range(1, 11)) + ("switch",))
        self.assertEqual(Switch.RETURN_TYPES[-1], "BOOLEAN")
        self.assertEqual(len(Switch.RETURN_TYPES), 11)

    def test_branch_schema_has_four_pairs_and_four_outputs(self):
        schema = Branch.INPUT_TYPES()
        self.assertEqual(schema["required"], {})
        self.assertEqual(
            list(schema["optional"]),
            [f"on_{branch}_{i}" for i in range(1, 5) for branch in ("false", "true")]
            + ["switch", "empty_as_none"],
        )
        self.assertTrue(all(config[1]["lazy"] for name, config in schema["optional"].items()
                            if name not in ("empty_as_none", "switch")))
        self.assertTrue(schema["optional"]["switch"][1]["forceInput"])
        self.assertEqual(Branch.RETURN_NAMES, tuple(f"output_{i}" for i in range(1, 5)) + ("switch",))
        self.assertEqual(Branch.RETURN_TYPES[-1], "BOOLEAN")
        self.assertEqual(len(Branch.RETURN_TYPES), 5)

    def test_image_switch_schema_keeps_ten_pairs_and_raw_image_trigger(self):
        schema = ImageSwitch.INPUT_TYPES()
        self.assertEqual(schema["required"], {})
        self.assertEqual(
            list(schema["optional"]),
            [f"on_{branch}_{i}" for i in range(1, 11) for branch in ("false", "true")]
            + ["image", "empty_as_none"],
        )
        self.assertEqual(schema["optional"]["image"],
                         ("IMAGE", {"forceInput": True, "rawLink": True}))
        self.assertEqual(schema["hidden"],
                         {"execution_list": "EXECUTION_LIST", "unique_id": "UNIQUE_ID"})
        self.assertEqual(ImageSwitch.RETURN_NAMES, Switch.RETURN_NAMES)

    def test_selected_branch_preserves_each_value_and_missing_positions(self):
        image = object()
        values = {
            "on_false_1": image,
            "on_true_1": "text",
            "on_false_5": 0,
            "on_true_5": False,
            "on_false_10": {"samples": image},
            "on_true_10": [1, 2],
        }
        false_result = Switch().select(False, **values)
        true_result = Switch().select(True, **values)
        self.assertIs(false_result[0], image)
        self.assertEqual(false_result[4], 0)
        self.assertEqual(false_result[9], {"samples": image})
        self.assertEqual(true_result[0], "text")
        self.assertIs(true_result[4], False)
        self.assertEqual(true_result[9], [1, 2])
        self.assertIs(false_result[10], False)
        self.assertIs(true_result[10], True)
        for result in (false_result, true_result):
            for index in (1, 2, 3, 5, 6, 7, 8):
                self.assertIsInstance(result[index], ExecutionBlocker)
                self.assertIsNone(result[index].message)

    def test_unconnected_selected_output_is_silently_blocked(self):
        result = Switch().select(False, on_true_1="unused", on_false_2="ready")
        self.assertIsInstance(result[0], ExecutionBlocker)
        self.assertIsNone(result[0].message)
        self.assertEqual(result[1], "ready")
        self.assertIs(result[10], False)

    def test_connected_none_is_not_treated_as_disconnected(self):
        result = Switch().select(False, on_false_1=None)
        self.assertIsNone(result[0])
        self.assertIsInstance(result[1], ExecutionBlocker)

    def test_switch_socket_overrides_widget_and_drives_lazy_branch(self):
        image = object()
        self.assertEqual(Switch.check_lazy_status(default_switch=False, switch=True,
                                                  on_false_1=None, on_true_2=None),
                         ["on_true_2"])
        result = Switch().select(default_switch=False, switch=True, on_false_1="unused",
                                 on_true_2=image)
        self.assertIs(result[1], image)
        self.assertIs(result[10], True)
        self.assertEqual(Switch().select(default_switch=True, on_true_1="ready")[0], "ready")

    def test_selected_empty_outputs_can_feed_optional_inputs_as_none(self):
        result = Switch().select(False, on_false_1="latent", on_true_2="image",
                                 empty_as_none="2, 4")
        self.assertEqual(result[0], "latent")
        self.assertIsNone(result[1])
        self.assertIsNone(result[3])
        self.assertIsInstance(result[2], ExecutionBlocker)
        self.assertIs(result[10], False)
        true_result = Switch().select(True, on_true_2="image", empty_as_none="2")
        self.assertEqual(true_result[1], "image")

    def test_branch_can_use_same_empty_output_policy(self):
        result = Branch().select(False, on_false_1="latent", empty_as_none="2")
        self.assertEqual(result[0], "latent")
        self.assertIsNone(result[1])
        self.assertIsInstance(result[2], ExecutionBlocker)
        self.assertIs(result[4], False)

    def test_lazy_status_requests_only_connected_selected_inputs(self):
        values = {
            "on_false_1": None,
            "on_false_3": "ready",
            "on_false_10": None,
            "on_true_2": None,
        }
        self.assertEqual(Switch.check_lazy_status(False, **values), ["on_false_1", "on_false_10"])
        self.assertEqual(Switch.check_lazy_status(True, **values), ["on_true_2"])

    def test_branch_uses_switch_status_from_upstream_node(self):
        upstream = Switch().select(False)
        self.assertTrue(all(isinstance(value, ExecutionBlocker) for value in upstream[:10]))
        self.assertIs(upstream[10], False)

        values = {"on_false_1": "fallback", "on_true_1": "processed", "on_true_4": 42}
        self.assertEqual(Branch.check_lazy_status(upstream[10], on_true_1=None, on_false_1=None),
                         ["on_false_1"])
        branch_output = Branch().select(upstream[10], **values)
        self.assertEqual(branch_output[0], "fallback")
        self.assertTrue(all(isinstance(value, ExecutionBlocker) for value in branch_output[1:4]))
        self.assertIs(branch_output[4], False)

        true_output = Branch().select(Switch().select(True)[10], **values)
        self.assertEqual(true_output[0], "processed")
        self.assertEqual(true_output[3], 42)
        self.assertIs(true_output[4], True)

    def test_branch_state_output_drives_a_following_switch(self):
        for state in (False, True):
            branch = Branch().select(state, on_false_1="fallback", on_true_1="processed")
            self.assertIs(branch[4], state)
            self.assertEqual(
                Switch.check_lazy_status(switch=branch[4], on_false_1=None, on_true_1=None),
                ["on_true_1" if state else "on_false_1"],
            )
            downstream = Switch().select(switch=branch[4], on_false_1="off", on_true_1="on")
            self.assertEqual(downstream[0], "on" if state else "off")
            self.assertIs(downstream[10], state)

    def test_branch_defaults_to_false_when_switch_is_unconnected(self):
        self.assertEqual(Branch.check_lazy_status(on_false_1=None, on_true_1=None),
                         ["on_false_1"])
        self.assertEqual(Branch().select(on_false_1="fallback", on_true_1="unused")[0],
                         "fallback")
        self.assertIs(Branch().select()[4], False)

    def test_image_switch_selects_only_branch_matching_actual_image(self):
        link = ["source", 0]
        image = object()
        ready = FakeExecutionList([image])
        missing = FakeExecutionList([None])
        values = {"on_false_1": "empty latent", "on_true_1": "reference latent"}
        self.assertEqual(ImageSwitch.check_lazy_status(link, ready, "switch",
                                                       on_false_2=None, on_true_2=None),
                         ["on_true_2"])
        self.assertEqual(ImageSwitch.check_lazy_status(link, missing, "switch",
                                                       on_false_2=None, on_true_2=None),
                         ["on_false_2"])
        self.assertEqual(ImageSwitch().select(link, ready, "switch", **values)[0],
                         "reference latent")
        self.assertIs(ImageSwitch().select(link, ready, "switch", **values)[10], True)
        self.assertEqual(ImageSwitch().select(link, missing, "switch", **values)[0],
                         "empty latent")
        self.assertIs(ImageSwitch().select(link, missing, "switch", **values)[10], False)

    def test_image_switch_treats_blocked_or_empty_image_as_false(self):
        link = ["source", 0]
        class EmptyImage:
            def numel(self):
                return 0

        for value in ([ExecutionBlocker(None)], [None], [], [EmptyImage()]):
            result = ImageSwitch().select(link, FakeExecutionList(value), "switch",
                                          on_false_1="fallback", empty_as_none="2")
            self.assertEqual(result[0], "fallback")
            self.assertIsNone(result[1])
            self.assertIs(result[10], False)
        self.assertIs(ImageSwitch().select(on_false_1="fallback")[10], False)


if __name__ == "__main__":
    unittest.main()
