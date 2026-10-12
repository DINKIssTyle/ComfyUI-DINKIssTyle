"""Interactive aspect-ratio crop for ComfyUI IMAGE tensors."""

import base64
import io
import math

import torch
import torch.nn.functional as F
from PIL import Image


RATIOS = {
    "1:1": (1, 1),
    "4:5": (4, 5),
    "5:4": (5, 4),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "21:9": (21, 9),
}


def _ratio_parts(aspect_ratio, custom_width, custom_height, width, height):
    if aspect_ratio == "Original":
        a, b = width, height
    elif aspect_ratio == "Custom":
        if type(custom_width) is not int or type(custom_height) is not int or \
                not 1 <= custom_width <= 10000 or not 1 <= custom_height <= 10000:
            raise ValueError("Custom aspect ratio values must be integers from 1 to 10000")
        a, b = custom_width, custom_height
    else:
        if aspect_ratio not in RATIOS:
            raise ValueError("Unknown crop aspect ratio")
        a, b = RATIOS[aspect_ratio]
    common = math.gcd(a, b)
    return a // common, b // common


def _crop_bounds(width, height, aspect_ratio, custom_width, custom_height,
                 crop_x, crop_y, crop_width, crop_height, crop_mode="Crop"):
    a, b = _ratio_parts(aspect_ratio, custom_width, custom_height, width, height)
    if crop_mode not in ("Crop", "Expand"):
        raise ValueError("Unknown crop mode")
    for name, value in (("crop_x", crop_x), ("crop_y", crop_y),
                        ("crop_width", crop_width), ("crop_height", crop_height)):
        lower, upper = (0, 1) if crop_mode == "Crop" else \
            ((-10000, 10000) if name in ("crop_x", "crop_y") else (0, 10000))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{name} must be a finite value from {lower} to {upper}")
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("Crop width and height must be greater than zero")

    if crop_mode == "Expand":
        if (crop_x, crop_y, crop_width, crop_height) == (0, 0, 1, 1):
            # API callers with untouched defaults get the same full-image fit
            # as selecting Expand in the interactive UI.
            scale = math.ceil(max(width / a, height / b))
            out_width, out_height = a * scale, b * scale
            return (round((width - out_width) / 2),
                    round((height - out_height) / 2), out_width, out_height)
        scale = min(crop_width * width / a, crop_height * height / b)
        out_width, out_height = max(1, round(a * scale)), max(1, round(b * scale))
        center_x = (crop_x + crop_width / 2) * width
        center_y = (crop_y + crop_height / 2) * height
        return (round(center_x - out_width / 2),
                round(center_y - out_height / 2), out_width, out_height)

    requested_width = max(1, min(width, round(crop_width * width)))
    requested_height = max(1, min(height, round(crop_height * height)))
    # The UI keeps a continuous ratio, while pixel rounding can put one side
    # just below the next integer multiple. Choose the closest valid multiple.
    multiple = min(width // a, height // b,
                   round(min(crop_width * width / a, crop_height * height / b)))
    if multiple:
        out_width, out_height = multiple * a, multiple * b
    else:
        ratio = a / b
        out_width = requested_width
        out_height = max(1, round(out_width / ratio))
        if out_height > requested_height:
            out_height = requested_height
            out_width = max(1, round(out_height * ratio))
        out_width = min(width, out_width)
        out_height = min(height, out_height)

    center_x = (crop_x + crop_width / 2) * width
    center_y = (crop_y + crop_height / 2) * height
    left = max(0, min(width - out_width, round(center_x - out_width / 2)))
    top = max(0, min(height - out_height, round(center_y - out_height / 2)))
    return left, top, out_width, out_height


def _source_preview(image):
    """Encode only the first input image, bounded for websocket UI output."""
    preview = image[:1].detach().float().clamp(0, 1)
    height, width = preview.shape[1:3]
    if max(height, width) > 512:
        scale = 512 / max(height, width)
        preview = F.interpolate(preview.permute(0, 3, 1, 2),
                                size=(max(1, round(height * scale)),
                                      max(1, round(width * scale))),
                                mode="area").permute(0, 2, 3, 1)
    pixels = (preview[0] * 255).round().to(torch.uint8).cpu().numpy()
    stream = io.BytesIO()
    if pixels.shape[-1] == 4:
        Image.fromarray(pixels, "RGBA").save(stream, format="PNG", compress_level=3)
        mime = "image/png"
    else:
        Image.fromarray(pixels, "RGB").save(stream, format="JPEG", quality=82)
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(stream.getvalue()).decode("ascii")


class DINKI_Image_Crop:
    CATEGORY = "DINKIssTyle/Image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "crop"
    # The node is also its own preview target, even with no downstream output.
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The UI preview must be regenerated for each queued run. Upstream
        # image sources can change while their graph inputs stay identical.
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "aspect_ratio": (["Original", *RATIOS, "Custom"], {"default": "Original"}),
            "custom_width": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1}),
            "custom_height": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1}),
            "crop_x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                                 "display": "number"}),
            "crop_y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                                 "display": "number"}),
            "crop_width": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1.0,
                                     "step": 0.001, "display": "number"}),
            "crop_height": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1.0,
                                      "step": 0.001, "display": "number"}),
        }}

    def crop(self, image, aspect_ratio="Original", custom_width=1, custom_height=1,
             crop_x=0.0, crop_y=0.0, crop_width=1.0, crop_height=1.0):
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or \
                image.shape[0] < 1 or image.shape[1] < 1 or image.shape[2] < 1 or \
                image.shape[-1] not in (3, 4):
            raise ValueError("image must be a nonempty BHWC RGB or RGBA IMAGE")
        if not torch.isfinite(image).all():
            raise ValueError("image contains non-finite values")
        height, width = image.shape[1:3]
        left, top, out_width, out_height = _crop_bounds(
            width, height, aspect_ratio, custom_width, custom_height,
            crop_x, crop_y, crop_width, crop_height)
        result = image if (left, top, out_width, out_height) == (0, 0, width, height) else \
            image[:, top:top + out_height, left:left + out_width, :]
        return {"ui": {"source_preview": [_source_preview(image)],
                       "source_size": [[width, height]],
                       "crop_rect": [[left, top, out_width, out_height]]},
                "result": (result,)}
