import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


NODE_FILE = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle/dinki_overlay.py"


class Tensor:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return Tensor(self.array[index])

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def unsqueeze(self, dimension):
        return Tensor(np.expand_dims(self.array, dimension))


class OverlayAlphaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch = types.SimpleNamespace(
            from_numpy=Tensor,
            cat=lambda tensors, dim=0: Tensor(
                np.concatenate([tensor.array for tensor in tensors], axis=dim)
            ),
        )
        with patch.dict(sys.modules, {"torch": torch}):
            cls.node_class = runpy.run_path(str(NODE_FILE))["DINKI_Overlay"]

    def apply_without_overlays(self, pixels, **overrides):
        options = dict(
            image=Tensor(pixels[None]),
            text_content="",
            enable_text=False,
            enable_overlay_image=False,
            font_name="Default",
            text_position="Center",
            text_align="Center",
            text_wrap_percent=0,
            line_spacing_multiplier=1.1,
            text_margin_percent=0,
            text_size_percent=5,
            text_color_hex="#FFFFFF",
            text_opacity=100,
            enable_stroke=False,
            stroke_color_hex="#000000",
            stroke_width=1,
            enable_shadow=False,
            shadow_color_hex="#000000",
            shadow_offset_x=0,
            shadow_offset_y=0,
            shadow_spread=0,
            shadow_opacity=100,
            overlay_position="Center",
            overlay_margin_percent=0,
            overlay_size_percent=15,
            overlay_opacity=100,
        )
        options.update(overrides)
        result, = self.node_class().apply_overlay(**options)
        return result.array[0]

    def test_transparent_base_retains_alpha_instead_of_exposing_hidden_color(self):
        pixels = np.array(
            [[[0.7, 0.0, 0.9, 0.0], [1.0, 0.2, 0.1, 1.0]],
             [[0.7, 0.0, 0.9, 0.5], [1.0, 0.2, 0.1, 1.0]]],
            dtype=np.float32,
        )
        output = self.apply_without_overlays(pixels)
        self.assertEqual(output.shape, pixels.shape)
        np.testing.assert_allclose(output[..., 3], pixels[..., 3], atol=1 / 255)
        np.testing.assert_allclose(output[0, 1, :3], pixels[0, 1, :3], atol=1 / 255)

    def test_opaque_rgb_base_keeps_three_channels(self):
        pixels = np.full((2, 2, 3), 0.4, dtype=np.float32)
        output = self.apply_without_overlays(pixels)
        self.assertEqual(output.shape, pixels.shape)
        np.testing.assert_allclose(output, pixels, atol=1 / 255)

    def test_overlay_mask_uses_inverse_alpha_and_does_not_square_opacity(self):
        base = np.zeros((2, 2, 3), dtype=np.float32)
        overlay = np.zeros((2, 2, 3), dtype=np.float32)
        overlay[..., 0] = 1.0
        mask = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        output = self.apply_without_overlays(
            base, enable_overlay_image=True,
            overlay_image=Tensor(overlay[None]), overlay_mask=Tensor(mask[None]),
            overlay_position="Top-Left", overlay_size_percent=100,
        )
        np.testing.assert_allclose(output[0, 0], [0.0, 0.0, 0.0], atol=1 / 255)
        np.testing.assert_allclose(output[0, 1], [1.0, 0.0, 0.0], atol=1 / 255)
        np.testing.assert_allclose(output[1, 0], [0.5, 0.0, 0.0], atol=2 / 255)

    def test_overlay_mask_combines_with_existing_overlay_alpha(self):
        base = np.zeros((2, 2, 3), dtype=np.float32)
        overlay = np.zeros((2, 2, 4), dtype=np.float32)
        overlay[..., 0] = 1.0
        overlay[..., 3] = 0.5
        mask = np.full((2, 2), 0.25, dtype=np.float32)
        output = self.apply_without_overlays(
            base, enable_overlay_image=True,
            overlay_image=Tensor(overlay[None]), overlay_mask=Tensor(mask[None]),
            overlay_position="Top-Left", overlay_size_percent=100,
        )
        np.testing.assert_allclose(output[0, 0], [0.375, 0.0, 0.0], atol=2 / 255)

    def test_matching_loader_mask_is_not_applied_twice(self):
        base = np.zeros((2, 2, 3), dtype=np.float32)
        overlay = np.zeros((2, 2, 4), dtype=np.float32)
        overlay[..., 0] = 1.0
        overlay[..., 3] = 0.5
        mask = np.full((2, 2), 0.5, dtype=np.float32)
        output = self.apply_without_overlays(
            base, enable_overlay_image=True,
            overlay_image=Tensor(overlay[None]), overlay_mask=Tensor(mask[None]),
            overlay_position="Top-Left", overlay_size_percent=100,
        )
        np.testing.assert_allclose(output[0, 0], [0.5, 0.0, 0.0], atol=2 / 255)


if __name__ == "__main__":
    unittest.main()
