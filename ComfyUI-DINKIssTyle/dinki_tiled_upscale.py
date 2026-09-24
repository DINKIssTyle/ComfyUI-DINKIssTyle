"""Tile-list helpers for image-edit upscaling workflows such as Qwen Image 2.1."""

import math

import numpy as np
import torch
from PIL import Image


MAX_SEED = 0xFFFFFFFFFFFFFFFF


def _numpy_image(value):
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()


def _axis_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    count = math.ceil((length - tile_size) / (tile_size - overlap)) + 1
    distance = length - tile_size
    # Distribute the final, usually shorter stride over the whole axis.
    return [i * distance // (count - 1) for i in range(count)]


def _scaled_starts(starts, source_length, tile_size, target):
    scale = target / tile_size
    output_length = int(round(source_length * scale))
    scaled = [int(round(start * scale)) for start in starts]
    if source_length >= tile_size:
        scaled[-1] = output_length - target
    return scaled, output_length


def _resize_image(image, width, height):
    """Resize the entire source before cropping, so shared pixels stay aligned."""
    if image.shape[:2] == (height, width):
        return image.astype(np.float32, copy=False)

    def resize_channel(channel):
        return np.asarray(
            Image.fromarray(np.asarray(channel, dtype=np.float32)).resize(
                (width, height), Image.Resampling.LANCZOS
            ), dtype=np.float32
        )

    if image.shape[-1] == 4:
        alpha = np.clip(image[:, :, 3], 0.0, 1.0)
        resized_alpha = np.clip(resize_channel(alpha), 0.0, 1.0)
        premultiplied = [resize_channel(image[:, :, c] * alpha) for c in range(3)]
        color = np.stack(premultiplied, axis=-1)
        np.divide(color, resized_alpha[:, :, None], out=color,
                  where=resized_alpha[:, :, None] > 1e-8)
        color[resized_alpha <= 1e-8] = 0.0
        resized = np.concatenate((color, resized_alpha[:, :, None]), axis=-1)
    else:
        resized = np.stack([resize_channel(image[:, :, c]) for c in range(3)], axis=-1)
    return np.clip(resized, 0.0, 1.0)


def _axis_weights(starts, tile_size, blend_width, nominal_overlap):
    """Assign each tile its region, feathering only near adjacent tile seams."""
    weights = [np.ones(tile_size, dtype=np.float32) for _ in starts]
    for i in range(len(starts) - 1):
        overlap_start = starts[i + 1]
        overlap_end = starts[i] + tile_size
        actual_overlap = overlap_end - overlap_start
        if actual_overlap <= 0:
            continue

        width = min(blend_width, actual_overlap)
        # Keep the seam near the preceding tile's edge when the final tiles
        # overlap much more than requested. This preserves one generated
        # version of the subject over most of the shared area.
        seam = overlap_end - min(nominal_overlap, actual_overlap) // 2
        blend_start = max(overlap_start, min(seam - width // 2, overlap_end - width))
        blend_end = blend_start + width
        left_start = blend_start - starts[i]
        left_end = blend_end - starts[i]
        right_start = blend_start - starts[i + 1]
        right_end = blend_end - starts[i + 1]

        weights[i][left_end:] = 0.0
        weights[i + 1][:right_start] = 0.0
        if width:
            ramp = (1.0 - np.cos(np.pi * (np.arange(width, dtype=np.float32) + 0.5) / width)) * 0.5
            weights[i][left_start:left_end] *= 1.0 - ramp
            weights[i + 1][right_start:right_end] *= ramp
    return weights


class DINKI_TileSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 32}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.25}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
            },
            "optional": {
                "blend_width": ("INT", {"default": 32, "min": 0, "max": 512, "step": 16}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "TILE_LAYOUT", "INT", "INT")
    RETURN_NAMES = ("tiles", "tile_seeds", "tile_layout", "qwen_resolution", "tile_count")
    OUTPUT_IS_LIST = (True, True, False, False, False)
    FUNCTION = "split"
    CATEGORY = "DINKIssTyle/PS"

    def split(self, image, tile_size, overlap, upscale_factor, seed, blend_width=32):
        if tile_size < 1 or overlap < 0 or overlap >= tile_size:
            raise ValueError("Tile overlap must be smaller than tile_size.")
        if blend_width < 0:
            raise ValueError("Blend width must be nonnegative.")
        if not 0 <= seed <= MAX_SEED:
            raise ValueError("Seed must be an unsigned 64-bit integer.")

        source = _numpy_image(image)
        if source.ndim != 4 or source.shape[0] != 1 or source.shape[-1] not in (3, 4):
            raise ValueError("Tile Split expects exactly one RGB or RGBA IMAGE [1,H,W,C].")
        _, height, width, _ = source.shape
        if width < 1 or height < 1:
            raise ValueError("Input image must have nonzero width and height.")

        target = int(round(tile_size * upscale_factor / 32)) * 32
        if target < tile_size or target > 4096:
            raise ValueError("Qwen output tile resolution must be between tile_size and 4096.")
        x_starts = _axis_starts(width, tile_size, overlap)
        y_starts = _axis_starts(height, tile_size, overlap)
        count = len(x_starts) * len(y_starts)
        if count > 4096:
            raise ValueError(f"Tile count {count} is too large; increase tile_size or reduce overlap.")

        scaled_x, output_width = _scaled_starts(x_starts, width, tile_size, target)
        scaled_y, output_height = _scaled_starts(y_starts, height, tile_size, target)
        enlarged = _resize_image(source[0], output_width, output_height)

        tiles = []
        positions = []
        for row, y in enumerate(y_starts):
            for column, x in enumerate(x_starts):
                sx, sy = scaled_x[column], scaled_y[row]
                patch = enlarged[sy:min(sy + target, output_height),
                                 sx:min(sx + target, output_width), :]
                pad_h = target - patch.shape[0]
                pad_w = target - patch.shape[1]
                if pad_h or pad_w:
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
                tiles.append(torch.from_numpy(patch[None].copy()))
                positions.append((x, y))

        layout = {
            "version": 1,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "target_tile_size": target,
            "overlap": overlap,
            "blend_width": min(blend_width, overlap),
            "x_starts": x_starts,
            "y_starts": y_starts,
            "positions": positions,
        }
        seeds = [(seed + i) & MAX_SEED for i in range(count)]
        return (tiles, seeds, layout, target, count)


class DINKI_TileStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_tiles": ("IMAGE",),
                "tile_layout": ("TILE_LAYOUT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    INPUT_IS_LIST = True
    FUNCTION = "stitch"
    CATEGORY = "DINKIssTyle/PS"

    def stitch(self, processed_tiles, tile_layout):
        if len(tile_layout) != 1 or not isinstance(tile_layout[0], dict):
            raise ValueError("Connect tile_layout directly from the matching Tile Split node.")
        layout = tile_layout[0]
        if layout.get("version") != 1:
            raise ValueError("Unsupported tile layout version.")

        x_starts = layout["x_starts"]
        y_starts = layout["y_starts"]
        expected_count = len(x_starts) * len(y_starts)
        if len(processed_tiles) != expected_count:
            raise ValueError(f"Expected {expected_count} processed tiles, got {len(processed_tiles)}.")

        tile_size = layout["tile_size"]
        target = layout["target_tile_size"]
        scale = target / tile_size
        width = int(round(layout["width"] * scale))
        height = int(round(layout["height"] * scale))
        scaled_x, _ = _scaled_starts(x_starts, layout["width"], tile_size, target)
        scaled_y, _ = _scaled_starts(y_starts, layout["height"], tile_size, target)

        channels = None
        for index, tile in enumerate(processed_tiles):
            shape = tuple(tile.shape)
            if len(shape) != 4 or shape[0] != 1 or shape[1:3] != (target, target):
                raise ValueError(
                    f"Processed tile {index + 1} must have shape [1,{target},{target},C]. "
                    "Connect qwen_resolution to Qwen's resolution input, or to both width and height "
                    "with custom_size enabled. Keep the tile order."
                )
            if shape[-1] not in (3, 4) or (channels is not None and shape[-1] != channels):
                raise ValueError("Processed tiles must all be RGB or all be RGBA.")
            channels = shape[-1]

        blend_width = int(round(layout.get("blend_width", layout.get("overlap", 0)) * scale))
        nominal_overlap = int(round(layout.get("overlap", blend_width) * scale))
        x_weights = _axis_weights(scaled_x, target, blend_width, nominal_overlap)
        y_weights = _axis_weights(scaled_y, target, blend_width, nominal_overlap)

        accumulation = np.zeros((height, width, channels), dtype=np.float32)
        weight_sum = np.zeros((height, width, 1), dtype=np.float32)
        for index, image in enumerate(processed_tiles):
            row, column = divmod(index, len(scaled_x))
            x, y = scaled_x[column], scaled_y[row]
            crop_width = min(target, width - x)
            crop_height = min(target, height - y)
            weight = y_weights[row][:crop_height, None] * x_weights[column][None, :crop_width]
            weight = weight[:, :, None]
            pixels = _numpy_image(image)[0, :crop_height, :crop_width].astype(np.float32, copy=False)
            if channels == 4:
                accumulation[y:y + crop_height, x:x + crop_width, :3] += pixels[:, :, :3] * pixels[:, :, 3:4] * weight
                accumulation[y:y + crop_height, x:x + crop_width, 3:4] += pixels[:, :, 3:4] * weight
            else:
                accumulation[y:y + crop_height, x:x + crop_width] += pixels * weight
            weight_sum[y:y + crop_height, x:x + crop_width] += weight

        if np.any(weight_sum <= 0):
            raise ValueError("Tile layout left uncovered pixels in the stitched image.")
        accumulation /= weight_sum
        if channels == 4:
            np.divide(accumulation[:, :, :3], accumulation[:, :, 3:4],
                      out=accumulation[:, :, :3], where=accumulation[:, :, 3:4] > 1e-8)
        np.clip(accumulation, 0.0, 1.0, out=accumulation)
        return (torch.from_numpy(accumulation[None]),)
