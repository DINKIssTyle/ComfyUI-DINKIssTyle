import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class GridTests(unittest.TestCase):
    def setUp(self):
        # Exercise grid sizing and placement independently of tensor codecs.
        self.pil = MagicMock()
        self.torch = MagicMock()
        with patch.dict(sys.modules, {
            "torch": self.torch, "numpy": MagicMock(), "PIL": self.pil,
        }):
            module = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                      / "ComfyUI-DINKIssTyle/dinki_grid.py"))
        self.node = module["DINKI_Grid"]()
        self.node.tensor_to_pil = lambda image: image
        self.node.pil_to_tensor = lambda image: image
        self.canvas = MagicMock()
        self.pil.Image.new.return_value = self.canvas

    def image(self, width, height):
        image = MagicMock()
        image.size = (width, height)
        def resize(size, method):
            image.resize.return_value.width, image.resize.return_value.height = size
            return image.resize.return_value
        image.resize.side_effect = resize
        return image

    def generate(self, **kwargs):
        options = dict(cols=2, rows=1, frame_thickness=0, bg_color_hex="#000000",
                       resize_method="Stretch", limit_output=False,
                       max_output_width=512, max_output_height=512)
        options.update(kwargs)
        return self.node.generate_grid(**options)

    def test_reference_uses_slot_number_not_connected_image_position(self):
        first, tenth = self.image(100, 80), self.image(320, 240)
        self.generate(image_2=first, image_10=tenth, reference_image=10, frame_thickness=10)
        self.pil.Image.new.assert_called_once_with("RGB", (640, 240), (0, 0, 0))
        self.assertEqual(first.resize.call_args.args[0], (300, 220))
        self.assertEqual(tenth.resize.call_args.args[0], (300, 220))
        self.assertEqual(self.canvas.paste.call_args_list[0].args,
                         (first.resize.return_value, (10, 10)))
        self.assertEqual(self.canvas.paste.call_args_list[1].args,
                         (tenth.resize.return_value, (330, 10)))

    def test_default_and_empty_reference_preserve_first_connected_size(self):
        for extra in ({}, {"reference_image": 7}):
            with self.subTest(extra=extra):
                self.pil.Image.new.reset_mock()
                self.generate(image_3=self.image(100, 80), image_10=self.image(320, 240), **extra)
                self.pil.Image.new.assert_called_once_with("RGB", (200, 80), (0, 0, 0))

    def test_reference_can_be_outside_visible_grid_cells(self):
        self.generate(cols=1, image_1=self.image(100, 80), image_10=self.image(320, 240),
                      reference_image=10)
        self.pil.Image.new.assert_called_once_with("RGB", (320, 240), (0, 0, 0))
        self.assertEqual(self.canvas.paste.call_count, 1)

    def test_size_limit_applies_after_reference_size(self):
        self.canvas.width, self.canvas.height = 1280, 480
        self.generate(image_10=self.image(640, 480), reference_image=10, limit_output=True)
        self.assertEqual(self.canvas.resize.call_args.args[0], (512, 192))

    def test_no_images_preserves_blank_output(self):
        self.generate(reference_image=10)
        self.torch.zeros.assert_called_once_with((1, 512, 512, 3))
        self.pil.Image.new.assert_not_called()


if __name__ == "__main__":
    unittest.main()
