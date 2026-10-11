"""Core logic for DKST PS Image Ref nodes (No-training reference latent extraction & application)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import folder_paths

META_KEY = "refmod_meta"


def get_ref_dir() -> str:
    """Return default directory path (input/ref) for saving ref files."""
    input_dir = folder_paths.get_input_directory()
    ref_dir = os.path.join(input_dir, "ref")
    os.makedirs(ref_dir, exist_ok=True)
    return ref_dir


def list_ref_files() -> List[str]:
    """Return list of available ref .safetensors files in input/ref."""
    ref_dir = get_ref_dir()
    if not os.path.isdir(ref_dir):
        return []
    files = [
        f for f in os.listdir(ref_dir)
        if f.endswith(".safetensors") and os.path.isfile(os.path.join(ref_dir, f))
    ]
    return sorted(files)


# ═══════════════════════════════════════════════════════════════════════════
# Latent Processing & Optimization
# ═══════════════════════════════════════════════════════════════════════════

def blur_latent(z: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Heavy spatial low-pass (downsample then upsample) used for strength < 1.0.

    Preserves the latent manifold smoothly without introducing high-frequency noise textures.
    """
    if z.device.type == "mps":
        return blur_latent(z.cpu(), factor).to(z.device)

    # 4D: [B, C, H, W]
    if z.dim() == 4:
        b, c, h, w = z.shape
        sh, sw = max(1, h // factor), max(1, w // factor)
        down = F.adaptive_avg_pool2d(z.float(), (sh, sw))
        up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
        return up.to(z.dtype)

    # 5D: [B, C, T, H, W] (e.g. video / temporal latent)
    if z.dim() == 5:
        b, c, t, h, w = z.shape
        sh, sw = max(1, h // factor), max(1, w // factor)
        down = F.adaptive_avg_pool3d(z.float(), (t, sh, sw))
        up = F.interpolate(down, size=(t, h, w), mode="trilinear", align_corners=False)
        return up.to(z.dtype)

    return z


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

    device = z_full.device
    target = z_full.clone().float().to(device)
    param = nn.Parameter(z_small.clone().float().to(device))
    opt = torch.optim.Adam([param], lr=lr)
    target_size = (target.shape[2], target.shape[3])

    with torch.inference_mode(False), torch.set_grad_enabled(True):
        for _ in range(steps):
            opt.zero_grad()
            up = F.interpolate(param, size=target_size, mode="bilinear", align_corners=False)
            loss = F.mse_loss(up, target)
            loss.backward()
            opt.step()

    return param.detach().to(device=z_small.device, dtype=z_small.dtype)


def calculate_token_count(latent: torch.Tensor) -> int:
    """Calculate effective patch token count (e.g., 2x2 spatial patch per token)."""
    if latent.dim() == 4:
        # [B, C, H, W] -> (H/2) * (W/2) * B
        b, _, h, w = latent.shape
        return b * max(1, (h // 2)) * max(1, (w // 2))
    elif latent.dim() == 5:
        # [B, C, T, H, W]
        b, _, t, h, w = latent.shape
        return b * t * max(1, (h // 2)) * max(1, (w // 2))
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


def load_ref_safetensors(filepath: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load latent and metadata from .safetensors file."""
    if not os.path.isabs(filepath):
        cand = os.path.join(get_ref_dir(), filepath)
        if os.path.isfile(cand):
            filepath = cand
        elif not os.path.isfile(filepath):
            annotated = folder_paths.get_annotated_filepath(filepath)
            if annotated and os.path.isfile(annotated):
                filepath = annotated
            else:
                raise FileNotFoundError(f"Reference file not found: {filepath} (checked in {get_ref_dir()})")
    elif not os.path.isfile(filepath):
        raise FileNotFoundError(f"Reference file not found: {filepath}")

    metadata: Dict[str, Any] = {}
    with safe_open(filepath, framework="pt") as f:
        meta_dict = f.metadata()
        if meta_dict and META_KEY in meta_dict:
            try:
                metadata = json.loads(meta_dict[META_KEY])
            except Exception:
                pass
    loaded = load_file(filepath)
    if "latent" not in loaded:
        raise KeyError(f"File {filepath} does not contain 'latent' tensor.")
    return loaded["latent"], metadata
