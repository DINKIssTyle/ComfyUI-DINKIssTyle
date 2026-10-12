"""ComfyUI-DINKIssTyle Image Ref Creator (DKST PS)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps
import torch

import folder_paths
from .dinki_ref_core import (
    calculate_token_count,
    get_ref_dir,
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
                    {"default": "Full Reference (encode)", "tooltip": "Full Reference saves VAE-encoded images. Legacy Compressed mode pools latents without training model weights or a LoRA and can discard face detail."},
                ),
                "refinement_steps": (
                    "INT",
                    {"default": 150, "min": 0, "max": 1000, "step": 10, "tooltip": "Latent reconstruction refinement steps; ignored in Full Reference mode."},
                ),
                "pool_grid": (
                    "INT",
                    {"default": 4, "min": 2, "max": 32, "step": 2, "tooltip": "Compressed latent grid size. Small grids discard face detail; ignored in Full Reference mode."},
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
    DESCRIPTION = "Save VAE image references from a gallery. Retained for future use with the gallery layout; legacy compression refines latent reconstruction without training model weights or a LoRA."

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

        # Ensure uniform sizing for a single stored latent tensor.
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

        # 2. Encode each still image independently. Video VAEs such as H3
        # interpret an IMAGE batch as consecutive frames, not separate refs.
        encoded_images = []
        for pixels in aligned_images:
            encoded = vae.encode(pixels[:, :, :, :3])
            if isinstance(encoded, dict):
                encoded = encoded.get("samples")
            if not isinstance(encoded, torch.Tensor) or encoded.dim() not in (4, 5) or encoded.shape[0] != 1:
                raise ValueError("The VAE must encode each reference image to one 4D or 5D latent.")
            encoded_images.append(encoded)
        latent = torch.cat(encoded_images, dim=0)

        # Ensure 4D [B, C, H, W]
        is_5d = False
        original_shape = list(latent.shape)
        if latent.dim() == 5:
            # [B, C, T, H, W] -> collapse or treat as 5D
            is_5d = True
            b, c, t, h, w = latent.shape
            latent_4d = latent.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
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
            final_latent = final_latent.reshape(b, t, c, final_latent.shape[-2], final_latent.shape[-1]).permute(0, 2, 1, 3, 4)

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
            "reference_layout": "independent_images",
            "vae_class": type(getattr(vae, "first_stage_model", vae)).__name__,
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
