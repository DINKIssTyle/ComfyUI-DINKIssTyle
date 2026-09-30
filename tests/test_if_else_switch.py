import runpy
import unittest
from pathlib import Path


NODE_FILE = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle" / "dinki_switch.py"
Switch = runpy.run_path(str(NODE_FILE))["DINKI_IfElseSwitch"]


class IfElseSwitchTests(unittest.TestCase):
    def test_socket_schema(self):
        schema = Switch.INPUT_TYPES()
        self.assertEqual(list(schema["required"]), ["switch"])
        self.assertEqual(
            list(schema["optional"]),
            [f"on_{branch}_{i}" for i in range(1, 11) for branch in ("false", "true")],
        )
        self.assertTrue(all(config[1]["lazy"] for config in schema["optional"].values()))
        self.assertEqual(Switch.RETURN_NAMES, tuple(f"output_{i}" for i in range(1, 11)))
        self.assertEqual(len(Switch.RETURN_TYPES), 10)

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
        self.assertEqual(
            Switch().select(False, **values),
            (image, None, None, None, 0, None, None, None, None, {"samples": image}),
        )
        self.assertEqual(
            Switch().select(True, **values),
            ("text", None, None, None, False, None, None, None, None, [1, 2]),
        )

    def test_lazy_status_requests_only_connected_selected_inputs(self):
        values = {
            "on_false_1": None,
            "on_false_3": "ready",
            "on_false_10": None,
            "on_true_2": None,
        }
        self.assertEqual(Switch.check_lazy_status(False, **values), ["on_false_1", "on_false_10"])
        self.assertEqual(Switch.check_lazy_status(True, **values), ["on_true_2"])


if __name__ == "__main__":
    unittest.main()
