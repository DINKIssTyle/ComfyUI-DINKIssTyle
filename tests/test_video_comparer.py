import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


NODE_FILE = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle/dinki_comparer.py"


class Tensor:
    def __init__(self, pixels):
        self.pixels = pixels
        self.shape = pixels.shape

    def __getitem__(self, index):
        return Tensor(self.pixels[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.pixels


class VideoComparerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder_paths = MagicMock()
        cls.imageio = MagicMock()
        with patch.dict(sys.modules, {
            "torch": MagicMock(),
            "folder_paths": cls.folder_paths,
            "imageio": cls.imageio,
        }):
            cls.node_class = runpy.run_path(str(NODE_FILE))["DINKI_Image_Comparer_MOV"]

    def test_rgb_and_rgba_inputs_produce_rgb_sweep_frames(self):
        rgb = np.zeros((1, 64, 64, 3), dtype=np.float32)
        rgb[..., 0] = 1.0
        rgba = np.zeros((1, 64, 64, 4), dtype=np.float32)
        rgba[..., 1] = 1.0
        rgba[..., 3] = 0.5

        with tempfile.TemporaryDirectory() as folder:
            self.folder_paths.get_output_directory.return_value = folder
            self.folder_paths.get_temp_directory.return_value = folder
            self.folder_paths.get_save_image_path.return_value = (folder, "compare", 1, "", "compare")
            for first, second in ((rgb, rgba), (rgba, rgb)):
                with self.subTest(first_channels=first.shape[-1]):
                    self.imageio.mimsave.reset_mock()
                    result = self.node_class().compare_images(
                        Tensor(first), Tensor(second), 0, 0, "nearest",
                        0.5, 0.5, 2, "gif", 90, 0, False, "compare",
                    )
                    frames = self.imageio.mimsave.call_args.args[1]
                    self.assertEqual(len(frames), 4)
                    self.assertTrue(all(frame.shape == (64, 64, 3) for frame in frames))
                    self.assertEqual(tuple(frames[0][0, 0]),
                                     (255, 0, 0) if first.shape[-1] == 3 else (0, 128, 0))
                    self.assertEqual(result["result"][0], str(Path(folder) / "compare_00001_.gif"))


if __name__ == "__main__":
    unittest.main()
