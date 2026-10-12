"""VAE encoding, latent refinement and storage helpers for Image Ref Creator."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

import folder_paths

META_KEY = "refmod_meta"
REF_FOLDER_NAME = "ref"


def get_ref_dir() -> str:
    """Return default directory path (input/ref) for saving ref files."""
    input_dir = folder_paths.get_input_directory()
    ref_dir = os.path.join(input_dir, REF_FOLDER_NAME)
    os.makedirs(ref_dir, exist_ok=True)
    return ref_dir


# ═══════════════════════════════════════════════════════════════════════════
# Latent Processing & Optimization
# ═══════════════════════════════════════════════════════════════════════════

def pool_latent_2d(z: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """Average-pool latent [B, C, H, W] to a compact grid [B, C, grid_h, grid_w]."""
    if z.dim() != 4:
        raise ValueError(f"Expected 4D latent [B, C, H, W], got {z.shape}")
    work = z.cpu() if z.device.type == "mps" else z
    pooled = F.adaptive_avg_pool2d(work.float(), (grid_h, grid_w))
    return pooled.to(device=z.device, dtype=z.dtype)


def optimize_latent_2d(
    z_small: torch.Tensor,
    z_full: torch.Tensor,
    steps: int = 150,
    lr: float = 0.02,
) -> torch.Tensor:
    """Model-free reconstruction refinement for compressed latent.

    Optimizes z_small directly using Adam so that bilinear upsampling minimizes MSE with z_full.
    """
    if steps <= 0:
        return z_small

    with torch.inference_mode(False), torch.set_grad_enabled(True):
        # ComfyUI executes nodes in inference_mode. Allocate ordinary tensors
        # here as well as enabling gradients, so backward can save its inputs.
        device = z_full.device
        target = z_full.detach().to(device=device, dtype=torch.float32).clone()
        param = nn.Parameter(z_small.detach().to(device=device, dtype=torch.float32).clone())
        opt = torch.optim.Adam([param], lr=lr)
        target_size = (target.shape[2], target.shape[3])
        for _ in range(steps):
            opt.zero_grad()
            up = F.interpolate(param, size=target_size, mode="bilinear", align_corners=False)
            loss = F.mse_loss(up, target)
            loss.backward()
            opt.step()

    return param.detach().to(device=z_small.device, dtype=z_small.dtype)


def calculate_token_count(latent: torch.Tensor, patch_size=None) -> int:
    """Estimate tokens for the supported reference families, or an explicit patch.

    Qwen 2.1 (64 channels) and FLUX.2 (128) use one spatial latent cell
    per token. Other VAEs are estimated using 2x2 patches. The channel-based
    estimate does not verify the compatibility of a saved file with a model.
    """
    if latent.dim() in (4, 5):
        if patch_size is None:
            patch_size = 1 if latent.shape[1] in (64, 128) else 2
        ph, pw = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size[-2:]
        if ph <= 0 or pw <= 0:
            raise ValueError("Reference patch size must be positive.")
        b = latent.shape[0]
        t = latent.shape[2] if latent.dim() == 5 else 1
        h, w = latent.shape[-2:]
        return b * t * ((h + ph - 1) // ph) * ((w + pw - 1) // pw)
    return int(latent.numel() / 64)


# ═══════════════════════════════════════════════════════════════════════════
# Serialization (.safetensors)
# ═══════════════════════════════════════════════════════════════════════════

def save_ref_safetensors(
    filepath: str,
    latent: torch.Tensor,
    meta: Dict[str, Any],
) -> str:
    """Save latent tensor and metadata to .safetensors file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tensors = {"latent": latent.contiguous()}
    encoded_meta = {
        META_KEY: json.dumps(meta, ensure_ascii=False),
        "kind": "image_ref",
    }
    save_file(tensors, filepath, metadata=encoded_meta)
    return filepath
