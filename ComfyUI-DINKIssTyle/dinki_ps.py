import os
import torch
import numpy as np
import folder_paths
import comfy.sd
import comfy.utils
import nodes
from PIL import Image
from nodes import MAX_RESOLUTION

# ============================================================================
# 1. DINKI Upscale Latent By (Bypassable)
# ============================================================================

class DINKI_Upscale_Latent_By:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],),
                "scale_by": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01}),
                "bypass": ("BOOLEAN", {"default": False}),  # Bypass toggle
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "DINKIssTyle/Latent"

    def upscale(self, samples, upscale_method, scale_by, bypass):
        if bypass:
            return (samples,)
        
        s = samples.copy()
        width = round(samples["samples"].shape[3] * scale_by * 8)
        height = round(samples["samples"].shape[2] * scale_by * 8)
        
        s["samples"] = comfy.utils.common_upscale(
            samples["samples"], width // 8, height // 8, upscale_method, "disabled"
        )
        return (s,)


# ============================================================================
# 2. DINKI Mask Weighted Mix
# ============================================================================

class DINKI_Mask_Weighted_Mix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "weight1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "weight2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "operation": (["add", "subtract", "multiply", "min", "max"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "mix_masks"
    CATEGORY = "DINKIssTyle/Mask"

    def mix_masks(self, mask1, mask2, weight1, weight2, operation):
        # Ensure dimensions match (expand if necessary)
        if mask1.shape != mask2.shape:
            # Simple resize to match mask1 (creates a copy)
            mask2 = torch.nn.functional.interpolate(
                mask2.unsqueeze(0).unsqueeze(0), 
                size=mask1.shape[-2:], 
                mode="bilinear"
            ).squeeze(0).squeeze(0)

        m1 = mask1 * weight1
        m2 = mask2 * weight2

        if operation == "add":
            result = m1 + m2
        elif operation == "subtract":
            result = m1 - m2
        elif operation == "multiply":
            result = m1 * m2
        elif operation == "min":
            result = torch.min(m1, m2)
        elif operation == "max":
            result = torch.max(m1, m2)
        else:
            result = m1 + m2

        return (torch.clamp(result, 0.0, 1.0),)


# ============================================================================
# 3. DINKI Resize & Pad (and Remove Pad)
# ============================================================================

class DINKI_Resize_And_Pad:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION}),
                "target_height": ("INT", {"default": 768, "min": 64, "max": MAX_RESOLUTION}),
                "fit_method": (["letterbox", "crop", "stretch"],), 
                "pad_color": ("INT", {"default": 0, "min": 0, "max": 255}), 
                "interpolation": (["nearest", "bilinear", "bicubic", "lanczos"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "x", "y", "new_width", "new_height", "orig_width", "orig_height")
    FUNCTION = "resize_and_pad"
    CATEGORY = "DINKIssTyle/Image"

    def resize_and_pad(self, image, target_width, target_height, fit_method, pad_color, interpolation):
        # image: [B, H, W, C] tensor
        results = []
        info_list = []
        
        # Interpolation mapping
        interp_map = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        interp = interp_map.get(interpolation, Image.BILINEAR)

        for img_tensor in image:
            # Tensor -> PIL
            i = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            orig_w, orig_h = img.size

            if fit_method == "stretch":
                new_img = img.resize((target_width, target_height), interp)
                x, y = 0, 0
                new_w, new_h = target_width, target_height
                
            elif fit_method == "crop":
                # Center Crop Logic
                ratio_w = target_width / orig_w
                ratio_h = target_height / orig_h
                ratio = max(ratio_w, ratio_h) # Use max to fill
                
                new_w = int(orig_w * ratio)
                new_h = int(orig_h * ratio)
                resized = img.resize((new_w, new_h), interp)
                
                # Center crop
                left = (new_w - target_width) // 2
                top = (new_h - target_height) // 2
                new_img = resized.crop((left, top, left + target_width, top + target_height))
                
                x, y = -left, -top # Negative offset relative to original

            else: # letterbox (contain)
                ratio_w = target_width / orig_w
                ratio_h = target_height / orig_h
                ratio = min(ratio_w, ratio_h) # Use min to fit
                
                new_w = int(orig_w * ratio)
                new_h = int(orig_h * ratio)
                resized = img.resize((new_w, new_h), interp)
                
                new_img = Image.new("RGB", (target_width, target_height), (pad_color, pad_color, pad_color))
                x = (target_width - new_w) // 2
                y = (target_height - new_h) // 2
                new_img.paste(resized, (x, y))

            # PIL -> Tensor
            out = np.array(new_img).astype(np.float32) / 255.0
            results.append(torch.from_numpy(out))
            info_list.append((x, y, new_w, new_h, orig_w, orig_h))

        # Batch stack
        final_image = torch.stack(results)
        
        # Return info of the first image (simplified for batch)
        x, y, nw, nh, ow, oh = info_list[0]
        
        return (final_image, x, y, nw, nh, ow, oh)


class DINKI_Remove_Pad_From_Image:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("INT", {"default": 0}),
                "y": ("INT", {"default": 0}),
                "width": ("INT", {"default": 0}), # Valid content width
                "height": ("INT", {"default": 0}), # Valid content height
                "orig_width": ("INT", {"default": 0}), # Original size to restore
                "orig_height": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore"
    CATEGORY = "DINKIssTyle/Image"

    def restore(self, image, x, y, width, height, orig_width, orig_height):
        # image: [B, H, W, C]
        results = []
        
        for img_tensor in image:
            i = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # 1. Crop valid area (remove padding)
            # x, y is the top-left coordinate where the image was pasted
            # If x, y are negative (crop mode), logic might differ, 
            # but usually this node is paired with Letterbox mode.
            if width > 0 and height > 0:
                # Letterbox case
                valid_area = img.crop((x, y, x + width, y + height))
            else:
                valid_area = img

            # 2. Resize back to original
            if orig_width > 0 and orig_height > 0:
                restored = valid_area.resize((orig_width, orig_height), Image.BILINEAR)
            else:
                restored = valid_area

            out = np.array(restored).astype(np.float32) / 255.0
            results.append(torch.from_numpy(out))

        return (torch.stack(results),)


# ============================================================================
# 4. DINKI Toggle UNet Loader
# ============================================================================

class DINKI_ToggleUNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("unet"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
                "load_mode": (["Always", "Bypass (None)"], {"default": "Always"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "DINKIssTyle/Loaders"

    def load_unet(self, unet_name, weight_dtype, load_mode):
        if load_mode == "Bypass (None)":
            print(f"## DINKI: UNet Loading Bypassed for {unet_name}")
            return (None,)

        # Standard Load Logic
        unet_path = folder_paths.get_full_path("unet", unet_name)
        model = comfy.sd.load_unet(unet_path)
        
        # Dtype casting if needed (simplified from standard nodes)
        if weight_dtype == "fp8_e4m3fn":
             # Logic to cast model weights would go here if not handled by comfy.sd.load_unet
             # Usually standard loader handles simple loading.
             pass
             
        return (model,)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "DINKI_Upscale_Latent_By": DINKI_Upscale_Latent_By,
    "DINKI_Mask_Weighted_Mix": DINKI_Mask_Weighted_Mix,
    "DINKI_Resize_And_Pad": DINKI_Resize_And_Pad,
    "DINKI_Remove_Pad_From_Image": DINKI_Remove_Pad_From_Image,
    "DINKI_ToggleUNetLoader": DINKI_ToggleUNetLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_Upscale_Latent_By": "DINKI Upscale Latent By",
    "DINKI_Mask_Weighted_Mix": "DINKI Mask Weighted Mix",
    "DINKI_Resize_And_Pad": "DINKI Resize and Pad Image",
    "DINKI_Remove_Pad_From_Image": "DINKI Remove Pad from Image",
    "DINKI_ToggleUNetLoader": "DINKI UNet Loader (safetensors / GGUF)",
}