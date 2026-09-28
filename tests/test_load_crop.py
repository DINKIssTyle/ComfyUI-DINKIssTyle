"""Check Load & Crop composition without requiring a running ComfyUI host."""

import ast
import importlib.util
import math
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"


class FakeTensor:
    def __init__(self, shape, origin=(0, 0)):
        self.shape = shape
        self.origin = origin

    def __getitem__(self, selection):
        batch, rows, columns, *channels = selection
        height = rows.stop - rows.start
        width = columns.stop - columns.start
        return FakeTensor((self.shape[0], height, width, *self.shape[3:]),
                          (self.origin[0] + columns.start, self.origin[1] + rows.start))

    def permute(self, *order):
        return FakeTensor(tuple(self.shape[index] for index in order), self.origin)

    def unsqueeze(self, axis):
        shape = list(self.shape)
        shape.insert(axis, 1)
        return FakeTensor(tuple(shape), self.origin)

    def squeeze(self, axis):
        shape = list(self.shape)
        assert shape.pop(axis) == 1
        return FakeTensor(tuple(shape), self.origin)

    def __rsub__(self, value):
        assert value == 1.0
        return FakeTensor(self.shape, self.origin)


def fake_interpolate(tensor, size, mode, align_corners, antialias):
    assert (mode, align_corners, antialias) == ("bilinear", False, True)
    return FakeTensor((tensor.shape[0], tensor.shape[1], *size), tensor.origin)


class FakeLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"category": ([""],), "filename": (["source.png"],)},
                "optional": {"source_type": (["input", "temp"],)}}

    @classmethod
    def VALIDATE_INPUTS(cls, category, filename, source_type="input"):
        return filename.endswith(".png")

    def load_image(self, category, filename, source_type="input"):
        size = (400, 300) if filename == "source.png" else (300, 400)
        width, height = size
        return {"result": (FakeTensor((2, height, width, 3)),
                           FakeTensor((2, height, width)),
                           FakeTensor((2, height, width)))}


def load_node_type():
    source = ROOT / "dinki_image_crop.py"
    tree = ast.parse(source.read_text())
    parts = [node for node in tree.body if
             isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
             target.id == "RATIOS" for target in node.targets) or
             isinstance(node, ast.FunctionDef) and node.name in ("_ratio_parts", "_crop_bounds") or
             isinstance(node, ast.ClassDef) and node.name == "DINKI_Image_Crop"]
    crop_namespace = {"math": math}
    exec(compile(ast.Module(body=parts, type_ignores=[]), str(source), "exec"), crop_namespace)
    crop_module = types.ModuleType("dkst_test_pkg.dinki_image_crop")
    crop_module.DINKI_Image_Crop = crop_namespace["DINKI_Image_Crop"]
    crop_module._crop_bounds = crop_namespace["_crop_bounds"]
    crop_module._source_preview = lambda image: f"preview:{image.shape[2]}x{image.shape[1]}"
    load_module = types.ModuleType("dkst_test_pkg.dinki_load")
    load_module.DINKI_Image_Load = FakeLoad
    photo_module = types.ModuleType("dkst_test_pkg.dinki_photo_specs")
    photo_module.DINKI_photo_specifications = runpy.run_path(
        str(ROOT / "dinki_photo_specs.py"))["DINKI_photo_specifications"]
    package = types.ModuleType("dkst_test_pkg")
    package.__path__ = []
    torch_module = types.ModuleType("torch")
    torch_module.__path__ = []
    torch_nn = types.ModuleType("torch.nn")
    torch_nn.__path__ = []
    torch_functional = types.ModuleType("torch.nn.functional")
    torch_functional.interpolate = fake_interpolate
    torch_nn.functional = torch_functional
    torch_module.nn = torch_nn
    modules = {"dkst_test_pkg": package,
               "dkst_test_pkg.dinki_image_crop": crop_module,
               "dkst_test_pkg.dinki_load": load_module,
               "dkst_test_pkg.dinki_photo_specs": photo_module,
               "torch": torch_module, "torch.nn": torch_nn,
               "torch.nn.functional": torch_functional}
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            "dkst_test_pkg.dinki_load_crop", ROOT / "dinki_load_crop.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.DINKI_Image_Load_Crop


LoadCrop = load_node_type()


class LoadCropTests(unittest.TestCase):
    def test_controls_and_outputs(self):
        inputs = LoadCrop.INPUT_TYPES()
        required = inputs["required"]
        self.assertEqual(list(required)[:3], ["category", "filename", "aspect_ratio"])
        self.assertIn("resolution_multiple", required)
        self.assertIn("megapixels", required)
        self.assertTrue(inputs["optional"]["source_type"][1]["socketless"])
        self.assertEqual(LoadCrop.RETURN_TYPES, ("IMAGE", "MASK", "MASK"))
        self.assertEqual(LoadCrop.RETURN_NAMES, ("image", "mask", "alpha"))
        self.assertTrue(LoadCrop.OUTPUT_NODE)
        changed = LoadCrop.IS_CHANGED()
        self.assertNotEqual(changed, changed)

    def test_crops_and_resizes_image_mask_alpha_to_one_megapixel(self):
        result = LoadCrop().load_and_crop("", "source.png", aspect_ratio="1:1")
        image, mask, alpha = result["result"]
        self.assertEqual(image.shape, (2, 1024, 1024, 3))
        self.assertEqual(mask.shape, (2, 1024, 1024))
        self.assertEqual(alpha.shape, (2, 1024, 1024))
        self.assertEqual(image.origin, (50, 0))
        self.assertEqual(mask.origin, image.origin)
        self.assertEqual(alpha.origin, image.origin)
        self.assertEqual(result["ui"]["source_preview"], ["preview:400x300"])
        self.assertEqual(result["ui"]["crop_rect"], [[50, 0, 300, 300]])
        self.assertEqual(result["ui"]["resolution"], ["1024 × 1024"])

    def test_new_source_recomputes_preview_crop_and_portrait_target(self):
        result = LoadCrop().load_and_crop("", "portrait.png", aspect_ratio="4:5",
                                          resolution_multiple="32", megapixels="2MP")
        image, mask, alpha = result["result"]
        self.assertEqual(mask.shape[1:], image.shape[1:3])
        self.assertEqual(alpha.shape[1:], image.shape[1:3])
        height, width = image.shape[1:3]
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)
        self.assertLess(width, height)
        self.assertEqual(result["ui"]["source_preview"], ["preview:300x400"])
        self.assertEqual(result["ui"]["resolution"], [f"{width} × {height}"])

    def test_fractional_megapixels_resize_all_outputs_together(self):
        self.assertIn("0.56MP", LoadCrop.INPUT_TYPES()["required"]["megapixels"][0])
        result = LoadCrop().load_and_crop("", "source.png", aspect_ratio="1:1",
                                          megapixels="0.56MP")
        image, mask, alpha = result["result"]
        self.assertEqual(image.shape, (2, 768, 768, 3))
        self.assertEqual(mask.shape, (2, 768, 768))
        self.assertEqual(alpha.shape, (2, 768, 768))
        self.assertEqual(result["ui"]["resolution"], ["768 × 768"])

    def test_validation_keeps_loader_file_rules(self):
        self.assertTrue(LoadCrop.VALIDATE_INPUTS("", "valid.png"))
        self.assertFalse(LoadCrop.VALIDATE_INPUTS("", "invalid.txt"))


if __name__ == "__main__":
    unittest.main()
