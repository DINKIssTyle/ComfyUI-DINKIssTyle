"""Fit Node frontend-only schema and loader registration, without ComfyUI imports."""
import ast
from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"
FitNode = runpy.run_path(str(ROOT / "dinki_tool.py"))["DINKI_Fit_Node"]


class FitNodeTests(unittest.TestCase):
    def test_view_control_needs_no_execution_or_output_sockets(self):
        inputs = FitNode.INPUT_TYPES()["required"]
        defaults = {name: definition[1]["default"] for name, definition in inputs.items()}
        self.assertEqual(defaults, {"shortcut_key": "", "target_node_id": "", "smoothness": 0.2})
        self.assertEqual(inputs["target_node_id"][0], "STRING")
        self.assertEqual(FitNode.RETURN_TYPES, ())
        self.assertEqual(FitNode.CATEGORY, "DINKIssTyle/Util")
        self.assertFalse(getattr(FitNode, "OUTPUT_NODE", False))
        self.assertEqual(getattr(FitNode(), FitNode.FUNCTION)(**defaults), ())

    def test_node_loader_and_display_name_register_fit_node(self):
        tree = ast.parse((ROOT / "__init__.py").read_text())
        assignments = {node.targets[0].id: node.value for node in tree.body
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        modules = dict(ast.literal_eval(assignments["NODE_MODULES"]))
        self.assertIn("DINKI_Fit_Node", modules["dinki_tool"])
        names = ast.literal_eval(next(node.value for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "NODE_DISPLAY_NAME_MAPPINGS" and isinstance(node.value, ast.Dict)))
        self.assertEqual(names["DINKI_Fit_Node"], "DKST Util (Fit Node)")


if __name__ == "__main__":
    unittest.main()
