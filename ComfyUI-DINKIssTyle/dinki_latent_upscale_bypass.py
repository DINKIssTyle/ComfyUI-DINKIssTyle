# ComfyUI/custom_nodes/dinki_latent_upscale_bypass/dinki_latent_upscale_bypass.py
import torch
from comfy.utils import common_upscale   # Comfy 기본 업스케일 함수 사용


def _resize_any(mask, width, height, method: str):
    """
    noise_mask 등 추가 텐서도 latent와 같은 방식으로 리사이즈.
    텐서 차원이 몇 차원이든 상관없이 마지막 두 축(H,W)만 바꿉니다.
    """
    if mask is None:
        return None
    return common_upscale(mask, width, height, method, "disabled")


class DINKI_Upscale_Latent_By:
    """
    Upscale Latent By + 실행 토글

    - 입력/출력: LATENT(dict)
        { 'samples': Tensor(..., H, W), (선택) 'noise_mask': Tensor(..., H, W), ... }

    - enabled=True  : 업스케일 실행
    - enabled=False : 바이패스 (입력을 그대로 반환)

    Comfy 기본 LatentUpscaleBy 와 같은 방식으로 마지막 두 축(H,W)만 업스케일합니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_method": ([
                    "nearest-exact",
                    "bilinear",
                    "area",
                    "bicubic",
                    "bislerp",
                ], {"default": "nearest-exact"}),
                "scale_by": ("FLOAT", {
                    "default": 1.50,
                    "min": 0.01,
                    "max": 8.0,
                    "step": 0.01,
                }),
                # ✅ 켜져 있을 때만 실행
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # 결과 해상도를 n의 배수로 맞추기 (기본 8, 0/1이면 스냅 끔)
                "snap_to_multiple": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                }),
            },
        }

    # ⭐ NEW: FLOAT 출력 추가 (latent_scale)
    RETURN_TYPES = ("LATENT", "FLOAT")
    RETURN_NAMES = ("samples", "latent_scale")
    FUNCTION = "apply"
    CATEGORY = "DINKIssTyle/Upscale"

    def apply(self, samples, upscale_method, scale_by, enabled, snap_to_multiple=8):
        # 🔁 토글 OFF → 통과 + latent_scale=1.0 (업스케일 없음)
        if not enabled:
            return (samples, 1.0)

        # 배율이 사실상 1이면 굳이 업스케일 안 함
        if abs(scale_by - 1.0) < 1e-6:
            return (samples, 1.0)

        s = samples.copy()
        x = s["samples"]

        if not isinstance(x, torch.Tensor):
            raise ValueError(f"DINKI_Upscale_Latent_By: LATENT['samples'] is not a tensor: {type(x)}")

        # ▶ 여기서는 차원 수를 신경 쓰지 않고, 마지막 두 축만 사용
        height = x.shape[-2]
        width  = x.shape[-1]

        # 새 크기 계산
        new_w = max(1, int(round(width  * float(scale_by))))
        new_h = max(1, int(round(height * float(scale_by))))

        # n의 배수로 스냅 (모델/vae 호환성 위해)
        if snap_to_multiple and snap_to_multiple > 1:
            m = int(snap_to_multiple)
            new_w = (new_w + m - 1) // m * m
            new_h = (new_h + m - 1) // m * m

        # Comfy 기본과 동일: 마지막 두 축만 확장
        s["samples"] = common_upscale(x, new_w, new_h, upscale_method, "disabled")

        # noise_mask 등도 같이 리사이즈 (있을 경우)
        if "noise_mask" in s and s["noise_mask"] is not None:
            s["noise_mask"] = _resize_any(s["noise_mask"], new_w, new_h, upscale_method)

        # ⭐ NEW: 실제 적용된 유효 스케일 (스냅 반영)
        # latent 해상도 기준이지만, VAE가 가로/세로에 같은 배율을 쓰기 때문에
        # 이미지에서도 동일한 비율로 적용된다고 볼 수 있음.
        effective_scale = new_w / float(width)

        return (s, float(effective_scale))
