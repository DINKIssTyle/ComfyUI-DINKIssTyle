import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


NODE_FILE = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle/dinki_tiled_upscale.py"
with patch.dict(sys.modules, {"torch": SimpleNamespace(from_numpy=lambda array: array)}):
    NODES = runpy.run_path(str(NODE_FILE))

Split = NODES["DINKI_TileSplit"]
Stitch = NODES["DINKI_TileStitch"]


class TiledUpscaleTests(unittest.TestCase):
    def test_qwen_receives_individual_images_and_matching_seeds(self):
        image = np.zeros((1, 80, 130, 3), dtype=np.float32)
        tiles, seeds, layout, resolution, count = Split().split(
            image, tile_size=64, overlap=16, upscale_factor=2.0, seed=2**64 - 2
        )
        self.assertEqual(count, 6)
        self.assertEqual(resolution, 128)
        self.assertEqual(seeds, [2**64 - 2, 2**64 - 1, 0, 1, 2, 3])
        self.assertTrue(all(tile.shape == (1, resolution, resolution, 3) for tile in tiles))
        self.assertEqual(layout["positions"], [(0, 0), (33, 0), (66, 0),
                                               (0, 16), (33, 16), (66, 16)])
        self.assertEqual(Split.OUTPUT_IS_LIST, (True, True, False, False, False))
        self.assertTrue(Stitch.INPUT_IS_LIST)

    def test_odd_sized_image_round_trips_through_overlapping_upscaled_tiles(self):
        image = np.random.default_rng(12).random((1, 79, 133, 3), dtype=np.float32)
        tiles, _, layout, _, _ = Split().split(image, 64, 16, 2.0, 10)
        stitched, = Stitch().stitch(tiles, [layout])
        expected = NODES["_resize_image"](image[0], 266, 158)[None]
        self.assertEqual(stitched.shape, (1, 158, 266, 3))
        np.testing.assert_allclose(stitched, expected, atol=2e-6)

    def test_neighboring_tiles_share_identical_pixels_before_qwen_edit(self):
        image = np.random.default_rng(4).random((1, 64, 96, 3), dtype=np.float32)
        tiles, _, layout, resolution, _ = Split().split(image, 64, 16, 2.0, 0)
        offset = NODES["_scaled_starts"](layout["x_starts"], 96, 64, resolution)[0][1]
        np.testing.assert_array_equal(tiles[0][0, :, offset:], tiles[1][0, :, :resolution - offset])

    def test_small_input_is_edge_padded_and_cropped_to_its_original_extent(self):
        image = np.random.default_rng(2).random((1, 19, 27, 3), dtype=np.float32)
        tiles, _, layout, _, count = Split().split(image, 64, 16, 2.0, 0)
        self.assertEqual(count, 1)
        self.assertEqual(tiles[0].shape, (1, 128, 128, 3))
        expected = NODES["_resize_image"](image[0], 54, 38)[None]
        np.testing.assert_array_equal(tiles[0][0, -1, -1], expected[0, -1, -1])
        stitched, = Stitch().stitch(tiles, [layout])
        np.testing.assert_allclose(stitched, expected, atol=2e-6)

    def test_overlap_blends_without_a_hard_step(self):
        image = np.zeros((1, 64, 96, 3), dtype=np.float32)
        _, _, layout, _, count = Split().split(image, 64, 16, 1.0, 0)
        self.assertEqual(count, 2)
        black = np.zeros((1, 64, 64, 3), dtype=np.float32)
        white = np.ones_like(black)
        stitched, = Stitch().stitch([black, white], [layout])
        profile = stitched[0, 32, :, 0]
        self.assertEqual(profile[20], 0.0)
        self.assertEqual(profile[75], 1.0)
        self.assertTrue(np.all(np.diff(profile) >= -1e-6))
        self.assertTrue(0.0 < profile[48] < 1.0)

    def test_large_actual_overlap_blends_only_near_the_seam(self):
        # A 75px image needs two 64px tiles, so they overlap by 53px even
        # though the requested overlap is only 8px.
        image = np.zeros((1, 64, 75, 3), dtype=np.float32)
        _, _, layout, _, count = Split().split(image, 64, 8, 1.0, 0, blend_width=8)
        self.assertEqual(count, 2)
        black = np.zeros((1, 64, 64, 3), dtype=np.float32)
        white = np.ones_like(black)
        stitched, = Stitch().stitch([black, white], [layout])
        profile = stitched[0, 32, :, 0]
        self.assertEqual(profile[25], 0.0)
        self.assertEqual(profile[50], 0.0)
        self.assertTrue(0.0 < profile[60] < 1.0)
        self.assertEqual(profile[70], 1.0)

    def test_portrait_sized_input_keeps_face_center_from_one_tile(self):
        x_starts = NODES["_axis_starts"](1200, 1024, 128)
        y_starts = NODES["_axis_starts"](1500, 1024, 128)
        self.assertEqual((x_starts, y_starts), ([0, 176], [0, 476]))
        horizontal = NODES["_axis_weights"]([x * 2 for x in x_starts], 2048, 64, 256)
        vertical = NODES["_axis_weights"]([y * 2 for y in y_starts], 2048, 64, 256)
        self.assertEqual((horizontal[0][1200], horizontal[1][1200 - 352]), (1.0, 0.0))
        self.assertEqual((vertical[0][1500], vertical[1][1500 - 952]), (1.0, 0.0))

    def test_wrong_qwen_resolution_or_missing_tile_fails(self):
        image = np.zeros((1, 64, 96, 3), dtype=np.float32)
        _, _, layout, _, _ = Split().split(image, 64, 16, 2.0, 0)
        with self.assertRaisesRegex(ValueError, "Expected 2 processed tiles"):
            Stitch().stitch([np.zeros((1, 128, 128, 3), dtype=np.float32)], [layout])
        with self.assertRaisesRegex(ValueError, "qwen_resolution"):
            Stitch().stitch([np.zeros((1, 64, 64, 3), dtype=np.float32)] * 2, [layout])

    def test_split_requires_one_source_image(self):
        image = np.zeros((2, 64, 96, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            Split().split(image, 64, 16, 2.0, 0)

    def test_rgba_overlap_uses_premultiplied_alpha(self):
        image = np.zeros((1, 64, 96, 4), dtype=np.float32)
        _, _, layout, _, _ = Split().split(image, 64, 16, 1.0, 0)
        transparent_red = np.zeros((1, 64, 64, 4), dtype=np.float32)
        transparent_red[..., 0] = 1.0
        opaque_blue = np.zeros_like(transparent_red)
        opaque_blue[..., 2:] = 1.0
        stitched, = Stitch().stitch([transparent_red, opaque_blue], [layout])
        np.testing.assert_allclose(stitched[0, 32, 48, :3], [0.0, 0.0, 1.0], atol=1e-6)
        self.assertTrue(0.0 < stitched[0, 32, 48, 3] < 1.0)


if __name__ == "__main__":
    unittest.main()
