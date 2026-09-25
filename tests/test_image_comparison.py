import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image


NODE_FILE = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle/dinki_image_comparison.py"


class Tensor:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape

    def __getitem__(self, index):
        return Tensor(self.array[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class ImageComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder_paths = MagicMock()
        with patch.dict(sys.modules, {"folder_paths": cls.folder_paths}):
            cls.node_class = runpy.run_path(str(NODE_FILE))["DINKI_Image_Comparison"]

    def test_larger_image_dimensions_are_used_for_both_inputs(self):
        first = Tensor(np.full((1, 2, 2, 3), 0.25, dtype=np.float32))
        second = Tensor(np.full((1, 4, 4, 3), 0.75, dtype=np.float32))
        aligned_first, aligned_second, difference = self.node_class._prepare(first, second)
        self.assertEqual(aligned_first.shape, (4, 4, 3))
        self.assertEqual(aligned_second.shape, (4, 4, 3))
        np.testing.assert_allclose(difference, 0.5, atol=1e-6)

        first = Tensor(np.full((1, 3, 5, 3), 0.25, dtype=np.float32))
        second = Tensor(np.full((1, 2, 4, 3), 0.75, dtype=np.float32))
        aligned_first, aligned_second, _ = self.node_class._prepare(first, second)
        self.assertEqual(aligned_first.shape, (3, 5, 3))
        self.assertEqual(aligned_second.shape, (3, 5, 3))

    def test_difference_is_absolute_per_channel_and_alpha_uses_black_background(self):
        first = Tensor(np.array([[[[1.0, 0.2, 0.0, 0.5]]]], dtype=np.float32))
        second = Tensor(np.array([[[[0.2, 0.4, 0.8]]]], dtype=np.float32))
        _, _, difference = self.node_class._prepare(first, second)
        np.testing.assert_allclose(difference[0, 0], [0.3, 0.3, 0.8], atol=1e-6)

    def test_output_saves_three_aligned_previews(self):
        with tempfile.TemporaryDirectory() as folder:
            self.folder_paths.get_temp_directory.return_value = folder
            self.folder_paths.get_save_image_path.return_value = (
                folder, "compare", 1, "", "compare"
            )
            first = Tensor(np.zeros((1, 2, 2, 3), dtype=np.float32))
            second = Tensor(np.ones((1, 4, 4, 3), dtype=np.float32))
            result = self.node_class().compare(first, second, "Difference")
            images = result["ui"]["dkst_comparison"]
            self.assertEqual(len(images), 3)
            self.assertEqual(result["ui"]["resolution"], ["4 × 4"])
            self.assertTrue(all(item["type"] == "temp" for item in images))
            with Image.open(Path(folder) / images[2]["filename"]) as preview:
                self.assertEqual(preview.size, (4, 4))
                self.assertEqual(preview.getpixel((0, 0)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
