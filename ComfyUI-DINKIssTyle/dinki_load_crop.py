"""Load an image, interactively crop it, and resize the result."""

import torch
import torch.nn.functional as F

from .dinki_image_crop import DINKI_Image_Crop, _crop_bounds, _source_preview
from .dinki_load import DINKI_Image_Load
from .dinki_photo_specs import DINKI_photo_specifications


class DINKI_Image_Load_Crop(DINKI_Image_Load):
    CATEGORY = "DINKIssTyle/Image"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "alpha")
    FUNCTION = "load_and_crop"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        loader = DINKI_Image_Load.INPUT_TYPES()
        crop = DINKI_Image_Crop.INPUT_TYPES()["required"]
        photo = DINKI_photo_specifications.INPUT_TYPES()["required"]
        required = dict(loader["required"])
        required.update({name: spec for name, spec in crop.items() if name != "image"})
        required["resolution_multiple"] = photo["resolution_multiple"]
        required["megapixels"] = photo["megapixels"]
        source_spec = loader["optional"]["source_type"]
        source_choices = source_spec[0]
        source_options = source_spec[1] if len(source_spec) > 1 else {}
        # Internal temp/input selector: keep it optional for existing API
        # prompts, but do not create an unused socket below the crop canvas.
        optional = {"source_type": (source_choices, {**source_options, "socketless": True})}
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
                      megapixels=1.0, source_type="input"):
        loaded = self.load_image(category, filename, source_type)["result"]
        images, masks, alphas = loaded
        source_height, source_width = images.shape[1:3]
        left, top, out_width, out_height = _crop_bounds(
            source_width, source_height, aspect_ratio, custom_width, custom_height,
            crop_x, crop_y, crop_width, crop_height)
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
        return {
            "ui": {
                "source_preview": [_source_preview(images)],
                "source_size": [[source_width, source_height]],
                "crop_rect": [[left, top, out_width, out_height]],
                "resolution": [f"{width} × {height}"],
            },
            "result": (cropped_images, cropped_masks, cropped_alphas),
        }
