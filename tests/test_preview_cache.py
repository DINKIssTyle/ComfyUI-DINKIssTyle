import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"


class Tensor:
    """CPU tensor adapter: persistence and PNG tests use real NumPy/Pillow."""
    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape, self.ndim = self.data.shape, self.data.ndim

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class ExecutionBlocker:
    def __init__(self, message):
        self.message = message


def is_link(value):
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], str) and isinstance(value[1], int))


class ExecutionList:
    def __init__(self, outputs=None):
        self.entry = None if outputs is None else types.SimpleNamespace(outputs=outputs)
        self.requested = []

    def make_input_strong_link(self, target, name):
        self.requested.append((target, name))

    def get_cache(self, source, target):
        return self.entry


class PreviewCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        folder_paths = types.SimpleNamespace(
            get_output_directory=lambda: self.temp.name, get_temp_directory=lambda: self.temp.name)
        graph_utils = types.SimpleNamespace(is_link=is_link, ExecutionBlocker=ExecutionBlocker)
        torch = types.SimpleNamespace(from_numpy=Tensor,
                                      cat=lambda values, dim: Tensor(np.concatenate([v.data for v in values], axis=dim)))
        package = types.ModuleType("dkst_preview_cache_test")
        package.__path__ = [str(ROOT)]
        self.modules = patch.dict(sys.modules, {
            "folder_paths": folder_paths, "torch": torch,
            "comfy_execution.graph_utils": graph_utils, "dkst_preview_cache_test": package,
        })
        self.modules.start()
        spec = importlib.util.spec_from_file_location("dkst_preview_cache_test.dinki_preview_cache",
                                                     ROOT / "dinki_preview_cache.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.node_class = module.DINKI_Preview_Image_Cache
        self.node = self.node_class()
        self.image = Tensor(np.linspace(-0.05, 1.05, 2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3))

    def tearDown(self):
        self.modules.stop()
        self.temp.cleanup()

    def capture(self, images=None):
        result = self.node.preview_cached_image(images=images or self.image)
        return result["ui"]["dkst_cache_file"][0], result

    def test_default_capture_once_has_lazy_raw_image_and_persisted_reference(self):
        schema = self.node_class.INPUT_TYPES()
        self.assertEqual(schema["required"]["cache_mode"][1]["default"], "Capture once")
        self.assertEqual(schema["optional"]["images"], ("IMAGE", {"lazy": True, "rawLink": True}))
        self.assertTrue(schema["optional"]["cache_file"][1]["forceWidget"])
        self.assertEqual(self.node.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(self.node.RETURN_NAMES, ("images",))
        self.assertTrue(self.node.OUTPUT_NODE)

    def test_capture_saves_exact_float_batch_and_usable_previews(self):
        filename, result = self.capture()
        np.testing.assert_array_equal(result["result"][0].data, self.image.data)
        self.assertEqual(result["ui"]["resolution"], ["4 × 3"])
        path = Path(self.node._cache_path(filename))
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), self.image.data)
        self.assertEqual(len(result["ui"]["dkst_images"]), 2)
        for item in result["ui"]["dkst_images"]:
            self.assertEqual(item["type"], "output")
            with Image.open(self.directory / item["subfolder"] / item["filename"]) as image:
                self.assertEqual(image.size, (4, 3))

    def test_saved_reference_survives_recreation_and_does_not_share_node_ids(self):
        filename, _ = self.capture()
        restored = self.node_class().preview_cached_image(cache_file=filename)
        np.testing.assert_array_equal(restored["result"][0].data, self.image.data)
        with self.assertRaisesRegex(ValueError, "no saved image"):
            self.node_class().preview_cached_image()

    def test_capture_once_ignores_bypassed_original_image(self):
        filename, _ = self.capture()
        original = Tensor(np.zeros_like(self.image.data))
        execution = ExecutionList([[original]])
        self.node.check_lazy_status(cache_file=filename, images=["original", 0],
                                    execution_list=execution, unique_id="cache")
        result = self.node.preview_cached_image(cache_file=filename, images=["original", 0],
                                                execution_list=execution, unique_id="cache")
        self.assertEqual(execution.requested, [])
        np.testing.assert_array_equal(result["result"][0].data, self.image.data)
        self.assertNotIn("dkst_cache_file", result["ui"], "fallback must not overwrite a newer widget reference")

    def test_first_capture_and_refresh_explicitly_promote_raw_lazy_dependency(self):
        filename, _ = self.capture()
        for mode, file in [("Capture once", ""), ("Refresh", filename)]:
            execution = ExecutionList()
            self.assertEqual(self.node.check_lazy_status(mode, file, ["depth", 0], execution, "cache"), [])
            self.assertEqual(execution.requested, [("cache", "images")])
        execution = ExecutionList()
        self.node.check_lazy_status("Use cache", "", ["depth", 0], execution, "cache")
        self.assertEqual(execution.requested, [])

    def test_refresh_captures_selected_output_and_leaves_previous_snapshot_immutable(self):
        filename, _ = self.capture()
        next_image = Tensor(np.ones_like(self.image.data) * 0.1234567)
        execution = ExecutionList([[self.image], [next_image]])
        result = self.node.preview_cached_image("Refresh", ["depth", 1], filename, execution, "cache")
        next_filename = result["ui"]["dkst_cache_file"][0]
        self.assertNotEqual(next_filename, filename)
        np.testing.assert_array_equal(result["result"][0].data, next_image.data)
        np.testing.assert_array_equal(np.load(self.node._cache_path(filename)), self.image.data)

    def test_missing_empty_and_blocked_refresh_keep_previous_image(self):
        filename, _ = self.capture()
        for outputs in [None, [], [[None]], [[ExecutionBlocker(None)]],
                        [[Tensor(np.empty((0, 3, 4, 3), dtype=np.float32))]]]:
            result = self.node.preview_cached_image("Refresh", ["depth", 0], filename,
                                                    ExecutionList(outputs), "cache")
            np.testing.assert_array_equal(result["result"][0].data, self.image.data)
            self.assertNotIn("dkst_cache_file", result["ui"])

    def test_raw_list_output_preserves_all_image_batches(self):
        execution = ExecutionList([[self.image, ExecutionBlocker(None), self.image]])
        result = self.node.preview_cached_image(images=["source", 0], execution_list=execution, unique_id="cache")
        np.testing.assert_array_equal(result["result"][0].data, np.concatenate([self.image.data] * 2))
        self.assertEqual(len(result["ui"]["dkst_images"]), 4)

    def test_cached_tensor_mutation_cannot_change_saved_pixels(self):
        filename, result = self.capture()
        result["result"][0].data[:] = 0
        restored = self.node.preview_cached_image(cache_file=filename)
        np.testing.assert_array_equal(restored["result"][0].data, self.image.data)

    def test_rgba_and_gray_cache_previews(self):
        for channels in [1, 4]:
            image = Tensor(np.ones((1, 2, 3, channels), dtype=np.float32) * 0.5)
            _, result = self.capture(image)
            np.testing.assert_array_equal(result["result"][0].data, image.data)

    def test_deleted_preview_is_regenerated_from_lossless_snapshot(self):
        filename, first = self.capture()
        item = first["ui"]["dkst_images"][0]
        preview = self.directory / item["subfolder"] / item["filename"]
        preview.unlink()
        self.node.preview_cached_image("Use cache", cache_file=filename)
        self.assertTrue(preview.is_file())

    def test_deleted_cache_is_recaptured_only_in_capture_mode(self):
        filename, _ = self.capture()
        Path(self.node._cache_path(filename)).unlink()
        self.assertEqual(self.node.IS_CHANGED(cache_file=filename), "missing")
        with self.assertRaisesRegex(ValueError, "no saved image"):
            self.node.preview_cached_image("Use cache", images=self.image, cache_file=filename)
        execution = ExecutionList([[self.image]])
        self.node.check_lazy_status("Capture once", filename, ["depth", 0], execution, "cache")
        self.assertEqual(execution.requested, [("cache", "images")])
        result = self.node.preview_cached_image(images=self.image, cache_file=filename)
        self.assertNotEqual(result["ui"]["dkst_cache_file"][0], filename)

    def test_no_image_fails_with_actionable_message_instead_of_wrong_placeholder(self):
        for mode in ["Capture once", "Refresh", "Use cache"]:
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "Connect an active image source"):
                self.node.preview_cached_image(mode)

    def test_corrupt_cache_can_be_replaced_with_refresh(self):
        filename, _ = self.capture()
        Path(self.node._cache_path(filename)).write_bytes(b"invalid snapshot")
        with self.assertRaisesRegex(ValueError, "Use Refresh"):
            self.node.preview_cached_image(cache_file=filename)
        result = self.node.preview_cached_image("Refresh", self.image, filename)
        self.assertNotEqual(result["ui"]["dkst_cache_file"][0], filename)

    def test_capture_failure_does_not_publish_a_partial_snapshot(self):
        with patch.object(self.node, "_save_image", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.capture()
        self.assertEqual(list((self.directory / self.node.CACHE_FOLDER).iterdir()), [])

    def test_cache_reference_cannot_escape_cache_folder(self):
        for filename in ["../secret.npy", "/tmp/secret.npy", "a.npy", "A" * 32 + ".npy"]:
            with self.assertRaises(ValueError):
                self.node._cache_path(filename)
        filename = "a" * 32 + ".npy"
        folder = self.directory / self.node.CACHE_FOLDER
        folder.mkdir()
        (folder / filename).symlink_to(self.directory / "outside.npy")
        with self.assertRaisesRegex(ValueError, "inside"):
            self.node._cache_path(filename)


if __name__ == "__main__":
    unittest.main()
