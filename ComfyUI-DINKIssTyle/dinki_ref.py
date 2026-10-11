"""ComfyUI-DINKIssTyle Image Ref Nodes (DKST PS)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch

import folder_paths
from .dinki_ref_core import (
    REF_FOLDER_NAME,
    blur_latent,
    calculate_token_count,
    get_ref_dir,
    list_ref_files,
    load_ref_safetensors,
    optimize_latent_2d,
    pool_latent_2d,
    save_ref_safetensors,
)

logger = logging.getLogger("DINKIssTyle.Ref")


def _load_image_from_gallery_item(item: Dict[str, Any]) -> Optional[torch.Tensor]:
    """Load image from disk and return [1, H, W, 3] float32 tensor in range [0, 1]."""
    filename = item.get("filename")
    if not filename:
        return None
    subfolder = item.get("subfolder", "")
    folder_type = item.get("type", "input")

    input_dir = folder_paths.get_input_directory()
    if folder_type == "temp":
        input_dir = folder_paths.get_temp_directory()
    elif folder_type == "output":
        input_dir = folder_paths.get_output_directory()

    if subfolder:
        img_path = os.path.join(input_dir, subfolder, filename)
    else:
        img_path = os.path.join(input_dir, filename)

    if not os.path.isfile(img_path):
        # Fallback search
        annotated = folder_paths.get_annotated_filepath(filename)
        if annotated and os.path.isfile(annotated):
            img_path = annotated
        else:
            logger.warning(f"Image not found: {filename}")
            return None

    try:
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception as e:
        logger.error(f"Error loading image {img_path}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 1. DINKI_Create_Image_Ref (Creator)
# ═══════════════════════════════════════════════════════════════════════════

class DINKI_Create_Image_Ref:
    """Create reference latents from multiple images with interactive grid UI and latent optimization."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "ref_name": ("STRING", {"default": "my_image_ref", "multiline": False}),
                "image_gallery": (
                    "DKST_IMAGE_REF_GRID",
                    {"default": "[]"},
                ),
                "mode": (
                    [
                        "Compressed (training)",
                        "Full Reference (encode)",
                    ],
                    {"default": "Compressed (training)"},
                ),
                "refinement_steps": (
                    "INT",
                    {"default": 150, "min": 0, "max": 1000, "step": 10},
                ),
                "pool_grid": (
                    "INT",
                    {"default": 4, "min": 2, "max": 32, "step": 2},
                ),
                "save_to_disk": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE_REF", "STRING")
    RETURN_NAMES = ("image_ref", "info")
    FUNCTION = "create_ref"
    CATEGORY = "DINKIssTyle/PS"
    TITLE = "DKST PS (Image Ref Creator)"
    DESCRIPTION = "Extract and compress reference latents from multiple images with an interactive grid gallery."

    def create_ref(
        self,
        vae,
        ref_name: str,
        image_gallery: str,
        mode: str,
        refinement_steps: int,
        pool_grid: int,
        save_to_disk: bool,
        images: Optional[torch.Tensor] = None,
    ):
        # 1. Collect all images
        collected: List[torch.Tensor] = []

        # From gallery widget JSON
        try:
            gallery_items = json.loads(image_gallery) if image_gallery else []
        except Exception:
            gallery_items = []

        for item in gallery_items:
            if isinstance(item, dict):
                tensor = _load_image_from_gallery_item(item)
                if tensor is not None:
                    collected.append(tensor)

        # From optional IMAGE input pin
        if images is not None and isinstance(images, torch.Tensor):
            for i in range(images.shape[0]):
                collected.append(images[i : i + 1])

        if not collected:
            raise ValueError("No images provided! Please upload images in the gallery or connect the 'images' input.")

        # Ensure uniform sizing for batch VAE encoding
        target_h, target_w = collected[0].shape[1], collected[0].shape[2]
        aligned_images = []
        for img in collected:
            if img.shape[1] != target_h or img.shape[2] != target_w:
                # Resize to match first image
                permuted = img.permute(0, 3, 1, 2)
                resized = torch.nn.functional.interpolate(
                    permuted, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
                aligned_images.append(resized.permute(0, 2, 3, 1))
            else:
                aligned_images.append(img)

        batched_pixels = torch.cat(aligned_images, dim=0)  # [B, H, W, 3]

        # 2. VAE Encode
        # ComfyUI vae.encode accepts [B, H, W, C]
        encoded = vae.encode(batched_pixels[:, :, :, :3])
        # Handle dict or tensor output
        if isinstance(encoded, dict) and "samples" in encoded:
            latent = encoded["samples"]
        elif isinstance(encoded, torch.Tensor):
            latent = encoded
        else:
            raise TypeError(f"Unexpected VAE output type: {type(encoded)}")

        # Ensure 4D [B, C, H, W]
        is_5d = False
        original_shape = list(latent.shape)
        if latent.dim() == 5:
            # [B, C, T, H, W] -> collapse or treat as 5D
            is_5d = True
            b, c, t, h, w = latent.shape
            latent_4d = latent.view(b * t, c, h, w)
        else:
            latent_4d = latent

        # 3. Compression / Optimization
        is_compressed = "Compressed" in mode
        if is_compressed:
            # Pool to grid
            pooled = pool_latent_2d(latent_4d, pool_grid, pool_grid)
            if refinement_steps > 0:
                refined = optimize_latent_2d(pooled, latent_4d, steps=refinement_steps)
            else:
                refined = pooled
            final_latent = refined
        else:
            final_latent = latent_4d

        if is_5d:
            final_latent = final_latent.view(b, c, t, final_latent.shape[-2], final_latent.shape[-1])

        token_count = calculate_token_count(final_latent)
        clean_name = ref_name.strip() or "image_ref"

        meta: Dict[str, Any] = {
            "name": clean_name,
            "mode": mode,
            "num_images": len(collected),
            "refinement_steps": refinement_steps if is_compressed else 0,
            "pool_grid": pool_grid if is_compressed else 0,
            "token_count": token_count,
            "original_shape": original_shape,
            "latent_shape": list(final_latent.shape),
        }

        # 4. Save to disk if requested
        saved_path = ""
        if save_to_disk:
            filename = f"{clean_name}.safetensors"
            dest = os.path.join(get_ref_dir(), filename)
            save_ref_safetensors(dest, final_latent, meta)
            saved_path = dest

        info_lines = [
            f"✅ Name: {clean_name}",
            f"📸 Images Processed: {len(collected)}",
            f"⚙️ Mode: {mode}",
            f"📐 Latent Shape: {tuple(final_latent.shape)}",
            f"🪙 Est. Tokens: {token_count}",
        ]
        if saved_path:
            info_lines.append(f"💾 Saved to: {saved_path}")

        image_ref = {
            "name": clean_name,
            "latent": final_latent,
            "meta": meta,
        }

        return (image_ref, "\n".join(info_lines))


# ═══════════════════════════════════════════════════════════════════════════
# 2. DINKI_Load_Image_Ref (Loader)
# ═══════════════════════════════════════════════════════════════════════════

class DINKI_Load_Image_Ref:
    """Load previously saved reference latents (.safetensors)."""

    @classmethod
    def INPUT_TYPES(cls):
        files = list_ref_files()
        if not files:
            files = ["None"]
        return {
            "required": {
                "ref_name": (files, {"default": files[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE_REF", "STRING")
    RETURN_NAMES = ("image_ref", "info")
    FUNCTION = "load_ref"
    CATEGORY = "DINKIssTyle/PS"
    TITLE = "DKST PS (Image Ref Loader)"
    DESCRIPTION = "Load saved reference files (.safetensors) from input/ref."

    def load_ref(self, ref_name: str):
        if ref_name in ("None", "", None):
            raise ValueError("No reference file selected or available in input/ref.")

        latent, meta = load_ref_safetensors(ref_name)
        token_count = meta.get("token_count", calculate_token_count(latent))
        name = meta.get("name", os.path.splitext(os.path.basename(ref_name))[0])

        info_lines = [
            f"📂 Loaded: {ref_name}",
            f"🏷️ Name: {name}",
            f"⚙️ Mode: {meta.get('mode', 'Unknown')}",
            f"📸 Images: {meta.get('num_images', '?')}",
            f"📐 Shape: {tuple(latent.shape)}",
            f"🪙 Est. Tokens: {token_count}",
        ]

        image_ref = {
            "name": name,
            "latent": latent,
            "meta": meta,
        }
        return (image_ref, "\n".join(info_lines))


# ═══════════════════════════════════════════════════════════════════════════
# 3. DINKI_Apply_Image_Ref (Applier)
# ═══════════════════════════════════════════════════════════════════════════

class DINKI_Apply_Image_Ref:
    """Inject reference latents into model attention/reference pathways."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "image_ref": ("IMAGE_REF",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_ref"
    CATEGORY = "DINKIssTyle/PS"
    TITLE = "DKST PS (Image Ref Apply)"
    DESCRIPTION = "Inject image reference tokens into the model with clean spatial-blur strength attenuation."

    def apply_ref(self, model, image_ref: Dict[str, Any], strength: float):
        if not isinstance(image_ref, dict) or "latent" not in image_ref:
            raise ValueError("Invalid IMAGE_REF provided.")

        latent: torch.Tensor = image_ref["latent"]

        # Apply strength attenuation
        if strength <= 0.0:
            # No-op if strength is 0
            return (model,)

        working_latent = latent
        if strength < 1.0:
            # Blend smoothly towards spatially blurred latent (preserves latent manifold)
            blurred = blur_latent(working_latent, factor=8)
            alpha = max(0.0, min(1.0, strength))
            working_latent = alpha * working_latent + (1.0 - alpha) * blurred
        elif strength > 1.0:
            # Scale slightly
            working_latent = working_latent * strength

        # Clone model to avoid mutating original
        patched_model = model.clone()

        # Attach to model's transformer/reference options
        opts = patched_model.model_options.setdefault("transformer_options", {})
        ref_list = opts.setdefault("ref_latents", [])
        ref_list.append(working_latent)

        # Support native MiniMax H3 / ref2video payload structure if present
        if hasattr(patched_model.model, "diffusion_model"):
            dm = patched_model.model.diffusion_model
            if hasattr(dm, "refs") or "refs" in opts:
                opts.setdefault("refs", []).append(working_latent)

        return (patched_model,)


# ═══════════════════════════════════════════════════════════════════════════
# 4. DINKI_Inspect_Image_Ref (Inspector & Preview)
# ═══════════════════════════════════════════════════════════════════════════

class DINKI_Inspect_Image_Ref:
    """Inspect reference metadata, token count, and optionally decode back to images for visual verification."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE_REF",),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("preview_images", "token_count", "info")
    FUNCTION = "inspect_ref"
    CATEGORY = "DINKIssTyle/PS"
    TITLE = "DKST PS (Image Ref Inspect)"
    DESCRIPTION = "Inspect reference tokens and decode latents back to images with VAE to inspect quality."

    def inspect_ref(self, image_ref: Dict[str, Any], vae=None):
        if not isinstance(image_ref, dict) or "latent" not in image_ref:
            raise ValueError("Invalid IMAGE_REF provided.")

        latent: torch.Tensor = image_ref["latent"]
        meta: Dict[str, Any] = image_ref.get("meta", {})
        token_count = meta.get("token_count", calculate_token_count(latent))

        info_lines = [
            f"🔍 Name: {image_ref.get('name', 'Unnamed')}",
            f"⚙️ Mode: {meta.get('mode', 'Unknown')}",
            f"📐 Stored Latent Shape: {tuple(latent.shape)}",
            f"🪙 Est. Tokens: {token_count}",
            f"📊 Parameters/Values: {latent.numel():,} floats",
        ]
        for k, v in meta.items():
            if k not in ("name", "mode", "token_count", "latent_shape"):
                info_lines.append(f"  • {k}: {v}")

        # VAE decoding for visual verification
        preview_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        if vae is not None:
            try:
                # If 5D [B, C, T, H, W], collapse to 4D for decoding
                dec_latent = latent
                if dec_latent.dim() == 5:
                    b, c, t, h, w = dec_latent.shape
                    dec_latent = dec_latent.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

                decoded = vae.decode(dec_latent)
                if isinstance(decoded, torch.Tensor):
                    preview_image = decoded
                elif isinstance(decoded, dict) and "samples" in decoded:
                    preview_image = decoded["samples"]
                info_lines.append(f"✨ Decoded Preview Size: {preview_image.shape[1]}x{preview_image.shape[2]} ({preview_image.shape[0]} frame(s))")
            except Exception as e:
                info_lines.append(f"⚠️ VAE Decode failed: {e}")

        return (preview_image, token_count, "\n".join(info_lines))
