# ComfyUI/custom_nodes/ComfyUI-DINKIssTyle/dinki_resize_pad.py

import torch
import numpy as np
from PIL import Image

# === 공통 헬퍼 ===

def tensor_to_pil(tensors) -> list[Image.Image]:
    # tensors: torch.Tensor (B,H,W,C) or np.ndarray (B,H,W,C)
    if isinstance(tensors, np.ndarray):
        arr = tensors
    else:
        arr = tensors.detach().cpu().numpy()
    imgs = []
    for tensor in arr:
        img = (np.clip(tensor, 0.0, 1.0) * 255.0).astype(np.uint8)  # (H,W,C)
        imgs.append(Image.fromarray(img))
    return imgs


def pil_to_tensor(pil_images: list[Image.Image]) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(
                np.array(pil_image).astype(np.float32) / 255.0
            )
            for pil_image in pil_images
        ]
    )


# === 노드들 ===

class DINKI_Resize_And_Pad:
    UPSCALE_METHODS = ["lanczos", "bicubic", "area", "nearest"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "target_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "upscale_method": (cls.UPSCALE_METHODS,),
                # 🔄 INT → BOOLEAN (토글 스위치)
                "resize_and_pad": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "PAD_INFO")
    RETURN_NAMES = ("output_image", "pad_info")
    FUNCTION = "process"
    CATEGORY = "DINKIssTyle/Image"

    def process(self, input_image: torch.Tensor, target_size: int, upscale_method: str, resize_and_pad: bool):
        if not resize_and_pad:
            pad_info_out = (0, 0, 0, 0, 1)
            return (input_image, pad_info_out)

        pad_color = (255, 255, 255)
        pil_images = tensor_to_pil(input_image)
        processed_pil_images, pad_info_out = [], None
        resampling_filter = {
            "lanczos": Image.Resampling.LANCZOS,
            "bicubic": Image.Resampling.BICUBIC,
            "area": Image.Resampling.BOX,
            "nearest": Image.Resampling.NEAREST,
        }[upscale_method]

        for pil_image in pil_images:
            orig_width, orig_height = pil_image.size
            ratio = min(target_size / orig_width, target_size / orig_height)
            new_width, new_height = int(orig_width * ratio), int(orig_height * ratio)

            resized_image = pil_image.resize((new_width, new_height), resample=resampling_filter)
            padded_image = Image.new("RGB", (target_size, target_size), pad_color)

            pad_left = (target_size - new_width) // 2
            pad_top = (target_size - new_height) // 2
            padded_image.paste(resized_image, (pad_left, pad_top))
            processed_pil_images.append(padded_image)

            if pad_info_out is None:
                pad_right = target_size - new_width - pad_left
                pad_bottom = target_size - new_height - pad_top
                pad_info_out = (pad_left, pad_top, pad_right, pad_bottom, target_size)

        return (pil_to_tensor(processed_pil_images), pad_info_out)


class DINKI_Remove_Pad_From_Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "pad_info": ("PAD_INFO",),
                "remove_pad": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # DINKI_Upscale_Latent_By에서 나온 latent_scale
                "latent_scale": ("FLOAT", {"default": 0.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "DINKIssTyle/Image"

    def process(self, input_image: torch.Tensor, pad_info: any, remove_pad: bool, latent_scale: float = 0.0):
        if not remove_pad:
            return (input_image,)

        # pad_info 자체가 None 이면 그냥 원본 반환 (혹은 에러를 내도 됨)
        if pad_info is None:
            print("[DINKI_Remove_Pad_From_Image] pad_info is None, bypassing.")
            return (input_image,)

        pad_info_tuple = pad_info[0] if isinstance(pad_info, list) else pad_info

        # 여기서도 None / 잘못된 타입 방어
        if (
            pad_info_tuple is None or
            not isinstance(pad_info_tuple, (tuple, list)) or
            len(pad_info_tuple) < 5
        ):
            print(f"[DINKI_Remove_Pad_From_Image] Invalid pad_info: {pad_info_tuple}, bypassing.")
            return (input_image,)

        # 튜플/리스트 앞 5개만 사용 (left, top, right, bottom, original_size)
        left, top, right, bottom, original_size = pad_info_tuple[:5]

        pil_images = tensor_to_pil(input_image)
        cropped_images = []

        for pil_image in pil_images:
            final_width, final_height = pil_image.size

            # 1) 이미지에서 직접 계산한 스케일 (기존 방식)
            scale_from_image = final_width / float(original_size)

            # 2) 기본값은 이미지 기반 스케일
            scale_factor = scale_from_image

            # 3) latent_scale이 들어왔고, 둘이 크게 다르지 않을 때만 latent_scale 사용
            if latent_scale is not None and latent_scale > 0.0:
                tolerance = 0.1  # 10% 이내면 OK
                diff = abs(scale_from_image - float(latent_scale))

                if diff <= tolerance * scale_from_image:
                    scale_factor = float(latent_scale)
                else:
                    scale_factor = scale_from_image

            scaled_left   = int(left   * scale_factor)
            scaled_top    = int(top    * scale_factor)
            scaled_right  = int(right  * scale_factor)
            scaled_bottom = int(bottom * scale_factor)

            crop_box = (
                scaled_left,
                scaled_top,
                final_width  - scaled_right,
                final_height - scaled_bottom,
            )
            cropped_images.append(pil_image.crop(crop_box))

        return (pil_to_tensor(cropped_images),)
