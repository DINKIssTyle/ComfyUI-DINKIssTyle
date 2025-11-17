# ComfyUI/custom_nodes/ComfyUI-DINKIssTyle/dinki_auto_adjustment.py

import numpy as np
import torch

def _to_numpy(img_t: torch.Tensor) -> np.ndarray:
    if img_t.device.type != "cpu":
        img_t = img_t.cpu()
    return img_t.numpy().astype(np.float32, copy=False)

def _to_tensor(img_n: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.clip(img_n, 0.0, 1.0).astype(np.float32, copy=False))

def _safe_percentile(arr, q, axis=None, max_samples=100_000):
    if arr.size > max_samples:
        flat = arr.ravel()
        idx = np.random.default_rng(seed=42).choice(flat.size, max_samples, replace=False)
        arr = flat[idx]
    return np.percentile(arr, q, axis=axis)

def _luma(img):
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

def _auto_tone_protected(img, clip_percent):
    cp = np.clip(clip_percent, 0.0, 5.0)
    lows, highs = _safe_percentile(img.reshape(-1, 3), [cp, 100 - cp], axis=0)
    
    # 보호: 너무 좁은 범위는 확장하지 않음 (예: 0.8~0.9 → 전체 0-1로 확장 X)
    range_min = 0.05  # 최소 5% 범위 보장
    diffs = np.maximum(highs - lows, range_min)
    
    stretched = (img - lows) / diffs
    # 추가 보호: 원본 평균 밝기 보존 (gain 조정)
    orig_mean = np.mean(img)
    new_mean = np.mean(stretched)
    if new_mean > 1e-3:
        stretched = stretched * (orig_mean / new_mean)
    return np.clip(stretched, 0.0, 1.0)

def _auto_contrast_protected(img, clip_percent):
    cp = np.clip(clip_percent, 0.0, 5.0)
    luma = _luma(img)
    lo = _safe_percentile(luma, cp)
    hi = _safe_percentile(luma, 100 - cp)
    
    diff = max(hi - lo, 0.05)  # 최소 대비 보장
    # 공통 스케일로 스트레치
    stretched = (img - lo) / diff
    
    # 평균 밝기 보존
    orig_mean = np.mean(luma)
    new_luma = _luma(stretched)
    new_mean = np.mean(new_luma)
    if new_mean > 1e-3:
        stretched = stretched * (orig_mean / new_mean)
    return np.clip(stretched, 0.0, 1.0)

def _auto_color_chromaticity(img, clip_percent):
    luma = _luma(img)
    # 중간톤: luma 30% ~ 70% (조금 더 넓게)
    lo_l = _safe_percentile(luma, 30.0)
    hi_l = _safe_percentile(luma, 70.0)
    mask = (luma >= lo_l) & (luma <= hi_l)
    if not np.any(mask):
        return img.copy()
    
    mean_rgb = np.mean(img[mask], axis=0)
    total = np.sum(mean_rgb)
    if total < 1e-4:
        return img.copy()
    
    chroma = mean_rgb / total
    ideal = np.array([1/3, 1/3, 1/3], dtype=np.float32)
    gains = ideal / (chroma + 1e-6)
    # 더 보수적인 gain 제한
    gains = np.clip(gains, 0.6, 1.6)
    
    corrected = img * gains
    return np.clip(corrected, 0.0, 1.0)

def _lerp(a, b, t):
    return a * (1.0 - t) + b * t

class DINKI_Auto_Adjustment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enable_auto_tone": ("BOOLEAN", {"default": False}),
                "enable_auto_contrast": ("BOOLEAN", {"default": False}),
                "enable_auto_color": ("BOOLEAN", {"default": True}),
                "clip_percent": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "DINKIssTyle/Color"

    def _process_full(self, img, enable_auto_tone, enable_auto_contrast, enable_auto_color, clip_percent):
        out = img.copy()
        if enable_auto_tone:
            out = _auto_tone_protected(out, clip_percent)
        if enable_auto_contrast:
            out = _auto_contrast_protected(out, clip_percent)
        if enable_auto_color:
            out = _auto_color_chromaticity(out, clip_percent)
        return out

    def apply(self, image, enable_auto_tone, enable_auto_contrast, enable_auto_color, clip_percent, strength):
        assert image.dim() == 4 and image.shape[-1] in (3, 4)
        has_alpha = image.shape[-1] == 4
        rgb = image[..., :3]
        alpha = image[..., 3:4] if has_alpha else None

        results = []
        for i in range(rgb.shape[0]):
            np_img = _to_numpy(rgb[i])
            if np_img.shape[-1] == 1:
                np_img = np.repeat(np_img, 3, axis=-1)

            full = self._process_full(np_img, enable_auto_tone, enable_auto_contrast, enable_auto_color, clip_percent)
            blended = _lerp(np_img, full, strength)
            results.append(_to_tensor(blended))

        out_rgb = torch.stack(results, dim=0)
        out = torch.cat([out_rgb, alpha], dim=-1) if has_alpha else out_rgb
        return (out,)