import torch

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
        # 현재 평균 밝기 (배치별)
        curr_mean = torch.mean(src, dim=(1, 2, 3), keepdim=True)
        
        # 0이나 1에 너무 가까우면 보정 제외 (안전장치)
        mask = (curr_mean > 1e-3) & (curr_mean < 1.0 - 1e-3) & (target_mean > 1e-3)
        
        # Gamma 공식: target = current ^ gamma  =>  gamma = log(target) / log(current)
        gamma = torch.log(target_mean + 1e-6) / torch.log(curr_mean + 1e-6)
        
        # 감마값이 너무 극단적이지 않게 제한 (0.5 ~ 2.0 정도)
        gamma = torch.clamp(gamma, 0.5, 2.0)
        
        # 마스크 적용 (조건 불만족 시 gamma=1.0)
        gamma = torch.where(mask, gamma, torch.ones_like(gamma))
        
        # 감마 적용: value ^ gamma
        # pow 연산은 음수에서 NaN이 나오므로 0.0 이상 보장 필요
        return torch.pow(src.clamp(min=1e-6), gamma)

    def _apply_auto_tone(self, img, clip_percent):
        """ RGB 채널별 독립 스트레칭 + 감마 밝기 복원 """
        B, H, W, C = img.shape
        flat = img.view(B, -1, C)

        cp = clip_percent / 100.0
        # 채널별 min/max 찾기
        lows = torch.quantile(flat, cp, dim=1, keepdim=True).view(B, 1, 1, C)
        highs = torch.quantile(flat, 1.0 - cp, dim=1, keepdim=True).view(B, 1, 1, C)

        # 0으로 나누기 방지 및 최소 대비 보장
        diffs = torch.maximum(highs - lows, torch.tensor(1e-5, device=img.device))

        # Linear Stretch
        stretched = (img - lows) / diffs
        stretched = torch.clamp(stretched, 0.0, 1.0)
        
        # [개선] 밝기 복원 (Gamma 방식)
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
        
        # RGB 전체에 동일한 scale/offset 적용
        stretched = (img - lo) / diff
        stretched = torch.clamp(stretched, 0.0, 1.0)

        # [개선] 밝기 복원 (Gamma 방식)
        orig_mean = torch.mean(luma, dim=(1, 2, 3), keepdim=True)
        stretched = self._match_brightness_gamma(stretched, orig_mean)

        return stretched

    def _apply_auto_color(self, img):
        """
        [개선 2] Weighted Gray World
        단순 마스크(30~70%) 대신, 50% 회색에 가까울수록 가중치를 높게 주는 방식을 사용.
        색상 왜곡을 줄이고 더 자연스러운 화이트 밸런스를 찾음.
        """
        luma = self._get_luma(img) # [B, H, W, 1]
        
        # 가중치 계산: 중간톤(0.5)일수록 1.0, 멀어질수록 0.0에 가까워짐 (Gaussian curve approximation)
        # exp(-((x-0.5)^2) / (2 * sigma^2))
        # sigma=0.25 정도로 설정하여 섀도우/하이라이트 배제
        weight = torch.exp(-torch.pow(luma - 0.5, 2) / (2 * 0.25**2))
        
        # 가중 평균 계산
        weighted_sum = torch.sum(img * weight, dim=(1, 2), keepdim=True) # [B, 1, 1, 3]
        weight_sum = torch.sum(weight, dim=(1, 2), keepdim=True) + 1e-6  # [B, 1, 1, 1]
        
        mean_rgb = weighted_sum / weight_sum # 가중 평균 색상
        
        # 회색(Gray) 타겟: 평균의 평균
        target_gray = torch.mean(mean_rgb, dim=-1, keepdim=True) # [B, 1, 1, 1]
        
        # Gain 계산
        gains = target_gray / (mean_rgb + 1e-6)
        
        # 안전장치: Gain이 너무 과하면(색이 완전히 틀어지면) 제한
        gains = torch.clamp(gains, 0.8, 1.25) # 0.6~1.6보다 더 보수적으로 잡음
        
        return torch.clamp(img * gains, 0.0, 1.0)

    def apply(self, image, enable_auto_tone, enable_auto_contrast, enable_auto_color, clip_percent, strength):
        t_img = image
        if t_img.shape[-1] == 4:
            rgb = t_img[..., :3]
            alpha = t_img[..., 3:4]
        else:
            rgb = t_img
            alpha = None
            
        out = rgb.clone()
        
        # 순서: Color -> Tone -> Contrast (일반적인 보정 순서)
        # 색감을 먼저 잡아야 톤 보정이 정확해짐
        
        if enable_auto_color:
            out = self._apply_auto_color(out)
            
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