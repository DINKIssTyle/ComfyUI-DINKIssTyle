import os
import torch
import numpy as np
import folder_paths
import comfy.sd
import comfy.utils
import nodes
import torch.nn.functional as F
from PIL import Image
from typing import Type, List, Tuple, Union # Type hinting 추가
from nodes import MAX_RESOLUTION
from comfy.utils import common_upscale 
import comfy.model_management

# ============================================================================
# 상수 정의 (Missing definition fixed)
# ============================================================================
NONE_LABEL = "None"

def tensor_to_pil(tensors) -> List[Image.Image]:
    if isinstance(tensors, np.ndarray):
        arr = tensors
    else:
        arr = tensors.detach().cpu().numpy()
    imgs = []
    for tensor in arr:
        img = (np.clip(tensor, 0.0, 1.0) * 255.0).astype(np.uint8)
        imgs.append(Image.fromarray(img))
    return imgs


def pil_to_tensor(pil_images: List[Image.Image]) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(
                np.array(pil_image).astype(np.float32) / 255.0
            )
            for pil_image in pil_images
        ]
    )

# ============================================================================
# 1. DINKI Toggle UNet Loader
# ============================================================================

class DINKI_ToggleUNetLoader:
    @classmethod
    def INPUT_TYPES(cls):
        safelist = folder_paths.get_filename_list("diffusion_models")
        safetensor_unets = [NONE_LABEL] + safelist if safelist else [NONE_LABEL]

        gguf_list = folder_paths.get_filename_list("unet_gguf")
        # GGUF 폴더가 없을 경우를 대비해 빈 리스트 처리
        if gguf_list is None: 
            gguf_list = []
        gguf_unets = [NONE_LABEL] + gguf_list if gguf_list else [NONE_LABEL]

        return {
            "required": {
                "use_gguf": ("BOOLEAN", {
                    "default": False,
                    "label_on": "GGUF",
                    "label_off": "safetensors",
                }),
                "safetensors_unet": (safetensor_unets, {"default": NONE_LABEL}),
                "gguf_unet": (gguf_unets, {"default": NONE_LABEL}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_unet"
    CATEGORY = "DINKIssTyle/PS"
    TITLE = "DINKI UNet Loader (safetensors / GGUF)"

    def _get_gguf_loader_class(self) -> Type:
        node_map = getattr(nodes, "NODE_CLASS_MAPPINGS", {})
        if "UnetLoaderGGUFAdvanced" in node_map:
            return node_map["UnetLoaderGGUFAdvanced"]
        if "UnetLoaderGGUF" in node_map:
            return node_map["UnetLoaderGGUF"]
        raise RuntimeError("ComfyUI-GGUF custom node not found.")

    def load_unet(self, use_gguf: bool, safetensors_unet: str, gguf_unet: str):
        if use_gguf:
            if gguf_unet == NONE_LABEL:
                raise ValueError("No GGUF UNet selected.")
            loader_class = self._get_gguf_loader_class()
            loader = loader_class()
            return loader.load_unet(gguf_unet)
        else:
            if safetensors_unet == NONE_LABEL:
                raise ValueError("No safetensors UNet selected.")
            loader = nodes.UNETLoader()
            return loader.load_unet(safetensors_unet, "default")


# ============================================================================
# 2. Resize and Pad
# ============================================================================

class DINKI_Resize_And_Pad:
    UPSCALE_METHODS = ["lanczos", "bicubic", "area", "nearest"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "target_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "resolution_multiple": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
                "upscale_method": (cls.UPSCALE_METHODS,),
                "resize_and_pad": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "PAD_INFO")
    RETURN_NAMES = ("output_image", "pad_info")
    FUNCTION = "process"
    CATEGORY = "DINKIssTyle/PS"

    def process(self, input_image: torch.Tensor, target_size: int, resolution_multiple: int, upscale_method: str, resize_and_pad: bool):
        if not resize_and_pad:
            # Bypass info: (left, top, right, bottom, original_size)
            # original_size는 복원이 불필요하므로 1로 설정하여 scale 계산 시 0나눗셈 방지
            pad_info_out = (0, 0, 0, 0, 1)
            return (input_image, pad_info_out)

        remainder = target_size % resolution_multiple
        if remainder != 0:
            if remainder >= resolution_multiple / 2:
                target_size = target_size + (resolution_multiple - remainder)
            else:
                target_size = target_size - remainder
        
        target_size = max(target_size, resolution_multiple)
        pad_color = (0, 0, 0) # 보통 패딩은 검정색을 선호하지만, 필요시 (255,255,255) 변경 가능

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


# ============================================================================
# 3. Remove Pad
# ============================================================================

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
                "latent_scale": ("FLOAT", {"default": 0.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "DINKIssTyle/PS"

    def process(self, input_image: torch.Tensor, pad_info: any, remove_pad: bool, latent_scale: float = 0.0):
        if not remove_pad:
            return (input_image,)

        # 안전한 pad_info 추출
        pad_info_tuple = pad_info
        if isinstance(pad_info, list) and len(pad_info) > 0:
            pad_info_tuple = pad_info[0]
            
        if not isinstance(pad_info_tuple, (tuple, list)) or len(pad_info_tuple) < 5:
            print(f"[🅳INKIssTyle - Remove_Pad] Invalid pad_info: {pad_info}, bypassing.")
            return (input_image,)

        left, top, right, bottom, original_size = pad_info_tuple[:5]
        
        # Bypass 모드에서 생성된 pad_info(0,0,0,0,1)인 경우 처리
        if left == 0 and top == 0 and right == 0 and bottom == 0:
             return (input_image,)

        pil_images = tensor_to_pil(input_image)
        cropped_images = []

        for pil_image in pil_images:
            final_width, final_height = pil_image.size
            scale_from_image = final_width / float(original_size)
            scale_factor = scale_from_image

            if latent_scale is not None and latent_scale > 0.0:
                tolerance = 0.1
                diff = abs(scale_from_image - float(latent_scale))
                if diff <= tolerance * scale_from_image:
                    scale_factor = float(latent_scale)

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


# ============================================================================
# 4. Upscale Latent By (Bypassable)
# ============================================================================

def _resize_any(mask, width, height, method: str):
    if mask is None:
        return None
    return common_upscale(mask, width, height, method, "disabled")

class DINKI_Upscale_Latent_By:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_method": ([
                    "nearest-exact", "bilinear", "area", "bicubic", "bislerp",
                ], {"default": "nearest-exact"}),
                "scale_by": ("FLOAT", {"default": 1.50, "min": 0.01, "max": 8.0, "step": 0.01}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "snap_to_multiple": ("INT", {"default": 8, "min": 1, "max": 64}),
            },
        }

    RETURN_TYPES = ("LATENT", "FLOAT")
    RETURN_NAMES = ("samples", "latent_scale")
    FUNCTION = "apply"
    CATEGORY = "DINKIssTyle/PS"

    def apply(self, samples, upscale_method, scale_by, enabled, snap_to_multiple=8):
        if not enabled or abs(scale_by - 1.0) < 1e-6:
            return (samples, 1.0)

        s = samples.copy()
        x = s["samples"]
        
        # 마지막 두 차원(H, W) 기준
        height = x.shape[-2]
        width  = x.shape[-1]

        new_w = max(1, int(round(width  * float(scale_by))))
        new_h = max(1, int(round(height * float(scale_by))))

        if snap_to_multiple > 1:
            m = int(snap_to_multiple)
            new_w = (new_w + m - 1) // m * m
            new_h = (new_h + m - 1) // m * m

        s["samples"] = common_upscale(x, new_w, new_h, upscale_method, "disabled")

        if "noise_mask" in s and s["noise_mask"] is not None:
            s["noise_mask"] = _resize_any(s["noise_mask"], new_w, new_h, upscale_method)

        effective_scale = new_w / float(width)
        return (s, float(effective_scale))


# ============================================================================
# 5. Mask Weighted Mix
# ============================================================================

class DINKI_Mask_Weighted_Mix:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {"required": {}, "optional": {}}
        for i in range(1, 6):
            inputs["optional"][f"mask_{i}"] = ("MASK",)
            inputs["optional"][f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
        return inputs

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mixed_mask",)
    FUNCTION = "mix_masks"
    CATEGORY = "DINKIssTyle/PS"

    def mix_masks(self, **kwargs):
        final_mask = None

        for i in range(1, 6):
            mask = kwargs.get(f"mask_{i}")
            strength = kwargs.get(f"strength_{i}", 1.0)

            if mask is not None:
                weighted_mask = mask * strength

                if final_mask is None:
                    final_mask = weighted_mask
                else:
                    if final_mask.shape[-2:] != weighted_mask.shape[-2:]:
                        # ComfyUI Mask: (Batch, H, W) -> unsqueeze(1) -> (Batch, 1, H, W)
                        wm_dim = weighted_mask.unsqueeze(1)
                        target_h, target_w = final_mask.shape[-2], final_mask.shape[-1]
                        
                        # align_corners=False 가 기본값으로 적절
                        wm_resized = F.interpolate(wm_dim, size=(target_h, target_w), mode="bilinear", align_corners=False)
                        weighted_mask = wm_resized.squeeze(1)

                    final_mask = torch.max(final_mask, weighted_mask)

        if final_mask is None:
            final_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")

        return (final_mask,)


# ============================================================================
# 6. Empty or Image Latent
# ============================================================================

class DINKI_Empty_Or_Image_Latent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # [추가] 작동 모드 선택 (Auto: 자동 감지, Bypass: 이미지 무시하고 빈 잠재 생성)
                "mode": (["Auto", "Bypass"], {"default": "Auto"}),
                
                "vae": ("VAE",),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "tooltip": "Used only if no image is input or Bypass mode"}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "tooltip": "Used only if no image is input or Bypass mode"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "Used only if no image is input or Bypass mode"}),
                "denoise": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise strength for img2img. Ignored (set to 1.0) if no image input."}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT", "FLOAT")
    RETURN_NAMES = ("LATENT", "denoise")
    FUNCTION = "process"
    CATEGORY = "DINKIssTyle/PS"

    def process(self, mode, vae, width, height, batch_size, denoise, image=None):
        # 1. [Auto 모드]이고 [이미지]가 연결되어 있는 경우 -> img2img (VAE Encode)
        if mode == "Auto" and image is not None:
            # 입력된 이미지를 VAE로 인코딩
            pixels = image
            pixels = pixels.to(comfy.model_management.get_torch_device())
            t = vae.encode(pixels[:,:,:,:3])
            
            # 사용자가 설정한 denoise 값 그대로 출력
            return ({"samples": t}, denoise)

        # 2. [Bypass 모드]이거나 [이미지]가 없는 경우 -> txt2img (Empty Latent)
        else:
            latent_width = width // 8
            latent_height = height // 8
            latent = torch.zeros([batch_size, 4, latent_height, latent_width], device=comfy.model_management.intermediate_device())
            
            # 이미지가 없거나 무시되었으므로 denoise 1.0 강제 출력
            return ({"samples": latent}, 1.0)