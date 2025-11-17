# ComfyUI/custom_nodes/ComfyUI-DINKIssTyle/dinki_ai_oversaturation_fix.py
# DINKI AI Oversaturation Fix

import numpy as np
import torch

def _to_numpy(img_t: torch.Tensor) -> np.ndarray:
    if img_t.device.type != "cpu":
        img_t = img_t.cpu()
    return img_t.numpy().astype(np.float32, copy=False)

def _to_tensor(img_n: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.clip(img_n, 0.0, 1.0).astype(np.float32, copy=False))

def _rgb_to_hsv(img):
    eps = 1e-7
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    v = maxc
    s = np.where(maxc > eps, delta / (maxc + eps), 0.0)

    rc = (maxc - r) / (delta + eps)
    gc = (maxc - g) / (delta + eps)
    bc = (maxc - b) / (delta + eps)

    h = np.where(delta <= eps, 0.0, 
                 np.where(maxc == r, bc - gc,
                          np.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc)))
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)

def _hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i = i.astype(np.int32) % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)

def _luma(img):
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

def _is_skin_tone(h, s, v):
    h_cond = (h >= 0.014) & (h <= 0.07)
    s_cond = (s >= 0.1) & (s <= 0.6)
    v_cond = (v >= 0.3) & (v <= 0.95)
    return h_cond & s_cond & v_cond

def _desaturate_highlights(img, reduction, highlight_threshold, preserve_skin):
    hsv = _rgb_to_hsv(img)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    t_low = max(highlight_threshold - 0.15, 0.0)
    t_high = highlight_threshold
    mask = np.clip((v - t_low) / (t_high - t_low + 1e-6), 0.0, 1.0)

    if preserve_skin:
        skin_mask = _is_skin_tone(h, s, v)
        mask = np.where(skin_mask, mask * 0.3, mask)

    s_new = s * (1.0 - mask * reduction)
    hsv_new = np.stack([h, s_new, v], axis=-1)
    return _hsv_to_rgb(hsv_new)

def _global_desat(img, reduction):
    hsv = _rgb_to_hsv(img)
    hsv[..., 1] *= (1.0 - reduction)
    return _hsv_to_rgb(hsv)

def _chroma_limit(img, max_chroma):
    rgb = img.copy()
    sum_rgb = np.sum(rgb, axis=-1, keepdims=True)
    sum_rgb = np.maximum(sum_rgb, 1e-6)
    chroma = rgb / sum_rgb

    max_comp = np.max(chroma, axis=-1)
    over = max_comp > max_chroma

    if np.any(over):
        scale = np.where(over[..., None], max_chroma / (max_comp[..., None] + 1e-6), 1.0)
        chroma_limited = chroma * scale
        rgb_out = chroma_limited * sum_rgb
        return np.clip(rgb_out, 0.0, 1.0)
    return rgb

def _lerp(a, b, t):
    return a * (1.0 - t) + b * t

class DINKI_AIOversaturationFix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fix_enabled": ("BOOLEAN", {"default": True}),
                "mode": (["desaturate_highlights", "global_desat", "chroma_limit", "auto"],),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "saturation_reduction": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "highlight_threshold": ("FLOAT", {"default": 0.82, "min": 0.5, "max": 1.0, "step": 0.01}),
                "max_chroma": ("FLOAT", {"default": 0.58, "min": 0.2, "max": 0.9, "step": 0.02}),
                "preserve_skin_tones": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "DINKIssTyle/Color"

    def _process_one(self, img_np, mode, saturation_reduction, highlight_threshold, max_chroma, preserve_skin_tones):
        if mode == "desaturate_highlights":
            result = _desaturate_highlights(img_np, saturation_reduction, highlight_threshold, preserve_skin_tones)
        elif mode == "global_desat":
            result = _global_desat(img_np, saturation_reduction)
        elif mode == "chroma_limit":
            result = _chroma_limit(img_np, max_chroma)
        elif mode == "auto":
            step1 = _desaturate_highlights(img_np, saturation_reduction * 0.7, highlight_threshold, preserve_skin_tones)
            result = _chroma_limit(step1, max_chroma)
        else:
            result = img_np.copy()
        return result

    def apply(self, image, fix_enabled, mode, strength, saturation_reduction, highlight_threshold, max_chroma, preserve_skin_tones):
        if not fix_enabled:
            return (image,)

        assert image.dim() == 4 and image.shape[-1] in (3, 4), "Input must be [B,H,W,C] RGB(A)."

        has_alpha = (image.shape[-1] == 4)
        if has_alpha:
            rgb = image[..., :3]
            alpha = image[..., 3:4]
        else:
            rgb = image
            alpha = None

        batch_out = []
        for i in range(rgb.shape[0]):
            np_img = _to_numpy(rgb[i])
            if np_img.shape[-1] == 1:
                np_img = np.repeat(np_img, 3, axis=-1)

            fixed = self._process_one(
                np_img,
                mode=mode,
                saturation_reduction=saturation_reduction,
                highlight_threshold=highlight_threshold,
                max_chroma=max_chroma,
                preserve_skin_tones=preserve_skin_tones
            )
            blended = _lerp(np_img, fixed, strength)
            batch_out.append(_to_tensor(blended))

        out_rgb = torch.stack(batch_out, dim=0)
        if has_alpha:
            out = torch.cat([out_rgb, alpha.clone()], dim=-1)
        else:
            out = out_rgb

        return (out,)