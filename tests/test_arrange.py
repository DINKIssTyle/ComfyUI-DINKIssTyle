import ast
import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"
Arrange = runpy.run_path(str(ROOT / "dinki_tool.py"))["DINKI_Arrange"]


class ArrangeTests(unittest.TestCase):
    def test_arrange_is_a_frontend_utility_without_sockets_or_queue_side_effects(self):
        self.assertEqual(Arrange.INPUT_TYPES(), {"required": {}})
        self.assertEqual(Arrange.RETURN_TYPES, ())
        self.assertEqual(Arrange.CATEGORY, "DINKIssTyle/Util")
        self.assertEqual(getattr(Arrange(), Arrange.FUNCTION)(), ())
        self.assertFalse(getattr(Arrange, "OUTPUT_NODE", False))

    def test_node_is_imported_and_registered_with_the_requested_display_name(self):
        tree = ast.parse((ROOT / "__init__.py").read_text())
        self.assertTrue(any(isinstance(node, ast.ImportFrom) and node.module == "dinki_tool"
                            and any(alias.name == "DINKI_Arrange" for alias in node.names)
                            for node in tree.body))
        mappings = {node.targets[0].id: node.value for node in tree.body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS")}
        for name, mapping in mappings.items():
            values = {key.value: value for key, value in zip(mapping.keys, mapping.values)}
            expected = values["DINKI_Arrange"]
            if name == "NODE_CLASS_MAPPINGS":
                self.assertEqual(expected.id, "DINKI_Arrange")
            else:
                self.assertEqual(expected.value, "DKST Util (Arrange)")


if __name__ == "__main__":
    unittest.main()
