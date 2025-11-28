import torch
import math

class DINKI_Auto_Adjustment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enable_auto_tone": ("BOOLEAN", {"default": False}),
                "enable_auto_contrast": ("BOOLEAN", {"default": False}),
                "enable_auto_color": ("BOOLEAN", {"default": True}),
                "enable_skin_tone": ("BOOLEAN", {"default": False}), # [신규] 스킨톤 보정 옵션
                "clip_percent": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 10.0, "step": 0.05
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

    def _get_luma(self, img):
        # Rec.709 Luma
        return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).unsqueeze(-1)

    def _match_brightness_gamma(self, src, target_mean):
        """
        [개선 1] Linear Multiplier 대신 Gamma Correction 사용
        밝기를 맞출 때 0.0(검정)과 1.0(흰색)을 고정하고 중간톤만 움직여서
        하이라이트 클리핑(날라감) 현상을 방지함.
        """
        curr_mean = torch.mean(src, dim=(1, 2, 3), keepdim=True)
        mask = (curr_mean > 1e-3) & (curr_mean < 1.0 - 1e-3) & (target_mean > 1e-3)
        gamma = torch.log(target_mean + 1e-6) / torch.log(curr_mean + 1e-6)
        gamma = torch.clamp(gamma, 0.5, 2.0)
        gamma = torch.where(mask, gamma, torch.ones_like(gamma))
        return torch.pow(src.clamp(min=1e-6), gamma)

    def _apply_auto_tone(self, img, clip_percent):
        """ RGB 채널별 독립 스트레칭 + 감마 밝기 복원 """
        B, H, W, C = img.shape
        flat = img.view(B, -1, C)

        cp = clip_percent / 100.0
        lows = torch.quantile(flat, cp, dim=1, keepdim=True).view(B, 1, 1, C)
        highs = torch.quantile(flat, 1.0 - cp, dim=1, keepdim=True).view(B, 1, 1, C)

        diffs = torch.maximum(highs - lows, torch.tensor(1e-5, device=img.device))

        stretched = (img - lows) / diffs
        stretched = torch.clamp(stretched, 0.0, 1.0)
        
        orig_mean = torch.mean(img, dim=(1, 2, 3), keepdim=True)
        stretched = self._match_brightness_gamma(stretched, orig_mean)

        return stretched

    def _apply_auto_contrast(self, img, clip_percent):
        """ Luma 기준 글로벌 스트레칭 """
        luma = self._get_luma(img)
        B, H, W, _ = luma.shape
        flat_luma = luma.view(B, -1)

        cp = clip_percent / 100.0
        lo = torch.quantile(flat_luma, cp, dim=1, keepdim=True).view(B, 1, 1, 1)
        hi = torch.quantile(flat_luma, 1.0 - cp, dim=1, keepdim=True).view(B, 1, 1, 1)

        diff = torch.maximum(hi - lo, torch.tensor(1e-5, device=img.device))
        
        stretched = (img - lo) / diff
        stretched = torch.clamp(stretched, 0.0, 1.0)

        orig_mean = torch.mean(luma, dim=(1, 2, 3), keepdim=True)
        stretched = self._match_brightness_gamma(stretched, orig_mean)

        return stretched

    def _apply_auto_color(self, img):
        """ [개선 2] Weighted Gray World """
        luma = self._get_luma(img) 
        weight = torch.exp(-torch.pow(luma - 0.5, 2) / (2 * 0.25**2))
        
        weighted_sum = torch.sum(img * weight, dim=(1, 2), keepdim=True)
        weight_sum = torch.sum(weight, dim=(1, 2), keepdim=True) + 1e-6
        
        mean_rgb = weighted_sum / weight_sum
        target_gray = torch.mean(mean_rgb, dim=-1, keepdim=True)
        
        gains = target_gray / (mean_rgb + 1e-6)
        gains = torch.clamp(gains, 0.8, 1.25)
        
        return torch.clamp(img * gains, 0.0, 1.0)

    def _apply_skin_tone(self, img):
        """
        [신규] Skin Tone Vector Alignment
        이미지를 YCbCr로 변환하여 피부색 영역(Mask)을 감지하고,
        해당 영역의 평균 색상이 '이상적인 스킨톤 라인'에 오도록
        색차(Cb, Cr) 평면을 미세하게 회전시킴.
        """
        device = img.device
        
        # 1. RGB to YCbCr (Rec.709 Standard)
        # Y: 0~1, Cb: -0.5~0.5, Cr: -0.5~0.5 approx
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        cb = (b - y) / 1.8556
        cr = (r - y) / 1.5748
        
        # 2. 피부색 감지 (Skin Mask)
        # 일반적인 피부색은 Cb가 음수(Blue 부족), Cr이 양수(Red 과잉)인 영역에 위치
        # 범위: Cb(-0.15 ~ -0.05), Cr(0.05 ~ 0.15) 근처 (Normalized RGB 기준)
        # Soft Mask를 사용하여 경계면 아티팩트 방지
        
        # 중심점 (Skin Center approx)
        center_cb, center_cr = -0.10, 0.10
        # 거리 계산 (Euclidean Distance in CbCr plane)
        dist = torch.sqrt((cb - center_cb)**2 + (cr - center_cr)**2)
        # 가중치 (거리가 가까울수록 1, 멀어질수록 0)
        skin_weight = torch.exp(-dist**2 / (2 * 0.05**2))
        
        # 3. 현재 피부색의 평균 각도 계산
        sum_weight = torch.sum(skin_weight, dim=(1, 2), keepdim=True) + 1e-6
        mean_cb = torch.sum(cb * skin_weight, dim=(1, 2), keepdim=True) / sum_weight
        mean_cr = torch.sum(cr * skin_weight, dim=(1, 2), keepdim=True) / sum_weight
        
        # 현재 각도 (atan2)
        curr_angle = torch.atan2(mean_cr, mean_cb)
        
        # 4. 목표 각도 (Ideal Skin Tone Line)
        # 벡터스코프에서 스킨톤 라인은 약 123도~135도 (2.14~2.35 rad) 부근 (Top-Left Quadrant)
        target_angle = torch.tensor(2.18, device=device) # 약 125도
        
        # 회전해야 할 각도 (Delta)
        # 너무 큰 회전은 오탐지일 수 있으므로 제한 (최대 +/- 15도)
        delta_theta = target_angle - curr_angle
        delta_theta = torch.clamp(delta_theta, -0.26, 0.26) 
        
        # 배치별로 델타가 다르므로 브로드캐스팅 준비
        sin_theta = torch.sin(delta_theta)
        cos_theta = torch.cos(delta_theta)
        
        # 5. CbCr 회전 (Global Rotation)
        # 피부색만 돌리는 게 아니라, 전체 이미지의 Tint를 조정하여 피부를 맞춤 (자연스러움)
        # Cb' = Cb * cos - Cr * sin
        # Cr' = Cb * sin + Cr * cos
        cb_new = cb * cos_theta - cr * sin_theta
        cr_new = cb * sin_theta + cr * cos_theta
        
        # 6. YCbCr to RGB
        # R = Y + 1.5748 * Cr
        # G = Y - 0.1873 * Cb - 0.4681 * Cr
        # B = Y + 1.8556 * Cb
        r_new = y + 1.5748 * cr_new
        g_new = y - 0.1873 * cb_new - 0.4681 * cr_new
        b_new = y + 1.8556 * cb_new
        
        out = torch.stack([r_new, g_new, b_new], dim=-1)
        return torch.clamp(out, 0.0, 1.0)

    def apply(self, image, enable_auto_tone, enable_auto_contrast, enable_auto_color, enable_skin_tone, clip_percent, strength):
        t_img = image
        if t_img.shape[-1] == 4:
            rgb = t_img[..., :3]
            alpha = t_img[..., 3:4]
        else:
            rgb = t_img
            alpha = None
            
        out = rgb.clone()
        
        # 순서: Color/Skin -> Tone -> Contrast
        # 색상(White Balance/Skin)을 먼저 잡아야 밝기 보정이 정확함
        
        if enable_auto_color:
            out = self._apply_auto_color(out)

        # [신규] 스킨톤 보정
        if enable_skin_tone:
            out = self._apply_skin_tone(out)
            
        if enable_auto_tone:
            out = self._apply_auto_tone(out, clip_percent)
            
        if enable_auto_contrast:
            out = self._apply_auto_contrast(out, clip_percent)
            
        # Strength Blending
        if strength < 1.0:
            out = torch.lerp(rgb, out, strength)
            
        if alpha is not None:
            out = torch.cat([out, alpha], dim=-1)
            
        return (out,)
