import json
import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


NODE_FILE = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle/dinki_multi_lora.py"


class FakeLoader:
    instances = []

    def __init__(self):
        self.calls = []
        self.instances.append(self)

    def load_lora_model_only(self, model, name, strength):
        self.calls.append((model, name, strength))
        return (model + [(name, strength)],)


class MultiLoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder_paths = ModuleType("folder_paths")
        folder_paths.get_filename_list = lambda kind: ["a.safetensors", "b.safetensors"]
        nodes = ModuleType("nodes")
        nodes.LoraLoaderModelOnly = FakeLoader
        with patch.dict(sys.modules, {"folder_paths": folder_paths, "nodes": nodes}):
            cls.node_class = runpy.run_path(str(NODE_FILE))["DINKI_Multi_LoRA_Loader"]

    def setUp(self):
        FakeLoader.instances.clear()
        self.node = self.node_class()

    def test_schema_has_only_model_input_and_output(self):
        inputs = self.node_class.INPUT_TYPES()["required"]
        self.assertEqual(list(inputs), ["model", "lora_stack"])
        self.assertEqual(inputs["lora_stack"][1]["lora_names"],
                         ["None", "a.safetensors", "b.safetensors"])
        self.assertEqual(self.node_class.RETURN_TYPES, ("MODEL",))

    def test_applies_enabled_rows_in_order_and_bypasses_others(self):
        stack = [
            {"name": "a.safetensors", "enabled": True, "strength_model": 0.6},
            {"name": "b.safetensors", "enabled": False, "strength_model": 1},
            {"name": "None", "enabled": True, "strength_model": 1},
            {"name": "b.safetensors", "enabled": True, "strength_model": 0},
            {"name": "b.safetensors", "enabled": True, "strength_model": -0.4},
        ]
        self.assertEqual(self.node.load_loras([], json.dumps(stack)),
                         ([("a.safetensors", 0.6), ("b.safetensors", -0.4)],))
        self.assertEqual(len(FakeLoader.instances), 2)

    def test_empty_stack_passes_model_through(self):
        model = object()
        self.assertIs(self.node.load_loras(model, "[]")[0], model)
        self.assertEqual(FakeLoader.instances, [])

    def test_reuses_loader_for_repeated_lora(self):
        stack = json.dumps([{"name": "a.safetensors"}, {"name": "a.safetensors"}])
        self.assertEqual(len(self.node.load_loras([], stack)[0]), 2)
        self.assertEqual(len(FakeLoader.instances), 1)

    def test_invalid_active_strength_fails_before_loading(self):
        stack = json.dumps([{"name": "a.safetensors"},
                            {"name": "b.safetensors", "strength_model": "nan"}])
        with self.assertRaisesRegex(ValueError, "strength_model"):
            self.node.load_loras([], stack)
        self.assertEqual(FakeLoader.instances, [])


if __name__ == "__main__":
    unittest.main()
