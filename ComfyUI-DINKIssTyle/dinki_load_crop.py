"""Load an image, interactively crop it, and resize the result."""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .dinki_image_crop import DINKI_Image_Crop, _crop_bounds, _source_preview
from .dinki_load import DINKI_Image_Load
from .dinki_photo_specs import DINKI_photo_specifications


def _render_expanded_canvas(images, alphas, bounds, width, height):
    """Sample a virtual crop directly at output size, with transparent padding.

    Only downsample the visible source before sampling, so neither a very large
    virtual canvas nor a tiny crop creates an oversized intermediate tensor.
    Premultiplied alpha keeps hidden RGB from bleeding into transparent edges.
    """
    left, top, canvas_width, canvas_height = bounds
    source_height, source_width = images.shape[1:3]
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(source_width, left + canvas_width), min(source_height, top + canvas_height)
    result = images.new_zeros((images.shape[0], height, width, 4))
    if x1 > x0 and y1 > y0:
        alpha = alphas[:, y0:y1, x0:x1].unsqueeze(-1)
        source = torch.cat((images[:, y0:y1, x0:x1, :3] * alpha, alpha), dim=-1)
        source = source.permute(0, 3, 1, 2)
        filtered_size = (max(1, min(y1 - y0, math.ceil((y1 - y0) * height / canvas_height))),
                         max(1, min(x1 - x0, math.ceil((x1 - x0) * width / canvas_width))))
        if filtered_size != source.shape[2:]:
            source = F.interpolate(source, size=filtered_size, mode="bilinear",
                                   align_corners=False, antialias=True)
        pixel_width, pixel_height = canvas_width / width, canvas_height / height
        edges_x = left + torch.arange(width, device=images.device, dtype=images.dtype) * pixel_width
        coverage_x = ((source_width - edges_x) / pixel_width).clamp(0, 1) \
            - (-edges_x / pixel_width).clamp(0, 1)
        xs = (edges_x - x0 + pixel_width / 2) * (2.0 / (x1 - x0)) - 1.0
        # Bound the sampling grid's temporary memory even for large outputs.
        for row in range(0, height, 128):
            end = min(height, row + 128)
            edges_y = top + torch.arange(row, end, device=images.device, dtype=images.dtype) * pixel_height
            coverage_y = ((source_height - edges_y) / pixel_height).clamp(0, 1) \
                - (-edges_y / pixel_height).clamp(0, 1)
            ys = (edges_y - y0 + pixel_height / 2) * (2.0 / (y1 - y0)) - 1.0
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(images.shape[0], -1, -1, -1)
            sampled = F.grid_sample(
                source, grid, mode="bilinear", padding_mode="border", align_corners=False,
            ).permute(0, 2, 3, 1)
            # Source coverage supplies transparent borders without fading an
            # opaque image's first/last pixel when magnifying a small crop.
            coverage = coverage_y[:, None] * coverage_x[None, :]
            result[:, row:end] = sampled * coverage[None, ..., None]
    alpha = result[..., 3].clamp(0, 1)
    if images.shape[-1] == 4:
        rgb = torch.where(alpha.unsqueeze(-1) > 1e-8,
                          result[..., :3] / alpha.unsqueeze(-1).clamp_min(1e-8), 0.0)
        image = torch.cat((rgb, alpha.unsqueeze(-1)), dim=-1)
    else:
        image = result[..., :3]
    return image, 1.0 - alpha, alpha


class DINKI_Image_Load_Crop(DINKI_Image_Load):
    CATEGORY = "DINKIssTyle/Image"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "alpha")
    FUNCTION = "load_and_crop"
    OUTPUT_NODE = True
    DESCRIPTION = "Load, crop or expand an image to a generation canvas. Expand preserves the full image when fitting a ratio and marks padding with mask=1, alpha=0."

    @classmethod
    def INPUT_TYPES(cls):
        loader = DINKI_Image_Load.INPUT_TYPES()
        crop = DINKI_Image_Crop.INPUT_TYPES()["required"]
        photo = DINKI_photo_specifications.INPUT_TYPES()["required"]
        required = dict(loader["required"])
        required.update({name: spec for name, spec in crop.items() if name != "image"})
        # Expand coordinates remain relative to the original image. Crop mode
        # still validates/clamps to the former 0..1 bounds.
        for name in ("crop_x", "crop_y", "crop_width", "crop_height"):
            options = dict(required[name][1])
            options.update(min=-10000.0 if name in ("crop_x", "crop_y") else 0.000001,
                           max=10000.0)
            required[name] = ("FLOAT", options)
        required["resolution_multiple"] = photo["resolution_multiple"]
        required["megapixels"] = photo["megapixels"]
        source_spec = loader["optional"]["source_type"]
        source_choices = source_spec[0]
        source_options = source_spec[1] if len(source_spec) > 1 else {}
        # Internal temp/input selector: keep it optional for existing API
        # prompts, but do not create an unused socket below the crop canvas.
        optional = {"source_type": (source_choices, {**source_options, "socketless": True}),
                    "crop_mode": (["Crop", "Expand"], {"default": "Crop", "socketless": True})}
        return {"required": required, "optional": optional}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Re-read the file and regenerate the preview on every queued run.
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, category, filename, source_type="input", **kwargs):
        return DINKI_Image_Load.VALIDATE_INPUTS(category, filename, source_type)

    def load_and_crop(self, category, filename, aspect_ratio="Original",
                      custom_width=1, custom_height=1, crop_x=0.0, crop_y=0.0,
                      crop_width=1.0, crop_height=1.0, resolution_multiple=8,
                      megapixels=1.0, source_type="input", crop_mode="Crop"):
        loaded = self.load_image(category, filename, source_type)["result"]
        images, masks, alphas = loaded
        source_height, source_width = images.shape[1:3]
        left, top, out_width, out_height = _crop_bounds(
            source_width, source_height, aspect_ratio, custom_width, custom_height,
            crop_x, crop_y, crop_width, crop_height, crop_mode)
        if crop_mode == "Expand":
            # Photo Specs needs only shape to calculate the output resolution.
            # Do not allocate a tensor at the virtual canvas's source size.
            canvas = SimpleNamespace(shape=(images.shape[0], out_height, out_width, images.shape[-1]))
            width, height, _ = DINKI_photo_specifications().calculate_resolution(
                megapixels=megapixels, aspect_ratio="Basic 1:1", orientation=False,
                resolution="Image", resolution_multiple=resolution_multiple, image=canvas)
            cropped_images, cropped_masks, cropped_alphas = _render_expanded_canvas(
                images, alphas, (left, top, out_width, out_height), width, height)
            return self._output(images, source_width, source_height,
                                (left, top, out_width, out_height), width, height,
                                cropped_images, cropped_masks, cropped_alphas)
        cropped_images = images[:, top:top + out_height, left:left + out_width, :]
        cropped_masks = masks[:, top:top + out_height, left:left + out_width]
        cropped_alphas = alphas[:, top:top + out_height, left:left + out_width]

        width, height, _ = DINKI_photo_specifications().calculate_resolution(
            megapixels=megapixels, aspect_ratio="Basic 1:1", orientation=False,
            resolution="Image", resolution_multiple=resolution_multiple,
            image=cropped_images)
        if (width, height) != (out_width, out_height):
            if cropped_images.shape[-1] == 4:
                premultiplied = torch.cat((
                    cropped_images[..., :3] * cropped_alphas.unsqueeze(-1),
                    cropped_alphas.unsqueeze(-1),
                ), dim=-1)
                resized = F.interpolate(
                    premultiplied.permute(0, 3, 1, 2), size=(height, width),
                    mode="bilinear", align_corners=False, antialias=True,
                ).permute(0, 2, 3, 1)
                cropped_alphas = resized[..., 3]
                resized_rgb = torch.where(
                    cropped_alphas.unsqueeze(-1) > 1e-8,
                    resized[..., :3] / cropped_alphas.unsqueeze(-1).clamp_min(1e-8),
                    0.0,
                )
                cropped_images = torch.cat((resized_rgb, cropped_alphas.unsqueeze(-1)), dim=-1)
            else:
                cropped_images = F.interpolate(
                    cropped_images.permute(0, 3, 1, 2), size=(height, width),
                    mode="bilinear", align_corners=False, antialias=True,
                ).permute(0, 2, 3, 1)
                cropped_alphas = F.interpolate(
                    cropped_alphas.unsqueeze(1), size=(height, width),
                    mode="bilinear", align_corners=False, antialias=True,
                ).squeeze(1)
            cropped_masks = 1.0 - cropped_alphas
        return self._output(images, source_width, source_height,
                            (left, top, out_width, out_height), width, height,
                            cropped_images, cropped_masks, cropped_alphas)

    @staticmethod
    def _output(images, source_width, source_height, bounds, width, height,
                cropped_images, cropped_masks, cropped_alphas):
        return {
            "ui": {
                "source_preview": [_source_preview(images)],
                "source_size": [[source_width, source_height]],
                "crop_rect": [list(bounds)],
                "resolution": [f"{width} × {height}"],
            },
            "result": (cropped_images, cropped_masks, cropped_alphas),
        }
