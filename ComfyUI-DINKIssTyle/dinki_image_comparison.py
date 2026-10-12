"""Interactive image comparison preview for ComfyUI."""

import os

import folder_paths
import numpy as np
from PIL import Image


class DINKI_Image_Comparison:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image_1": ("IMAGE",),
            "image_2": ("IMAGE",),
            "mode": (["Slide", "Difference"],),
        }}

    RETURN_TYPES = ()
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "DINKIssTyle/Image"

    @staticmethod
    def _first_rgb(images):
        if len(images.shape) != 4 or images.shape[0] < 1:
            raise ValueError("Image comparison expects a nonempty IMAGE batch.")
        image = images[0].detach().cpu().numpy()
        if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
            raise ValueError("Image comparison expects grayscale, RGB, or RGBA images.")
        image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        image = np.clip(image, 0.0, 1.0)
        if image.shape[2] == 1:
            return np.repeat(image, 3, axis=2)
        if image.shape[2] == 4:
            return image[:, :, :3] * image[:, :, 3:4]
        return image

    @staticmethod
    def _fit_contain(image, width, height):
        if image.shape[:2] == (height, width):
            return image
        source_height, source_width = image.shape[:2]
        if source_width * height >= width * source_height:
            scaled_width = width
            scaled_height = max(1, source_height * width // source_width)
        else:
            scaled_width = max(1, source_width * height // source_height)
            scaled_height = height
        if (scaled_height, scaled_width) == (source_height, source_width):
            scaled = image
        else:
            channels = [
                np.asarray(Image.fromarray(image[:, :, channel]).resize(
                    (scaled_width, scaled_height), Image.Resampling.LANCZOS), dtype=np.float32)
                for channel in range(3)
            ]
            scaled = np.clip(np.stack(channels, axis=2), 0.0, 1.0)
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        left = (width - scaled_width) // 2
        top = (height - scaled_height) // 2
        canvas[top:top + scaled_height, left:left + scaled_width] = scaled
        return canvas

    @classmethod
    def _prepare(cls, image_1, image_2):
        first = cls._first_rgb(image_1)
        second = cls._first_rgb(image_2)
        first_size = first.shape[0] * first.shape[1]
        second_size = second.shape[0] * second.shape[1]
        height, width = (first if first_size >= second_size else second).shape[:2]
        first = cls._fit_contain(first, width, height)
        second = cls._fit_contain(second, width, height)
        return first, second, np.abs(first - second)

    def compare(self, image_1, image_2, mode="Slide"):
        if mode not in ("Slide", "Difference"):
            raise ValueError("Comparison mode must be Slide or Difference.")
        first, second, difference = self._prepare(image_1, image_2)
        height, width = first.shape[:2]
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            "DKST_Comparison", folder_paths.get_temp_directory(), width, height
        )
        descriptors = []
        for index, pixels in enumerate((first, second, difference)):
            saved_name = f"{filename}_{counter + index:05}_.png"
            path = os.path.join(folder, saved_name)
            Image.fromarray(np.rint(pixels * 255).astype(np.uint8), "RGB").save(path)
            descriptors.append({"filename": saved_name, "subfolder": subfolder, "type": "temp"})
        return {"ui": {
            "dkst_comparison": descriptors,
            "resolution": [f"{width} × {height}"],
        }}
