import torch
import torch.nn.functional as F

class DINKI_Deband:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                
                # [추가됨] 켜기/끄기 토글 버튼 (Boolean)
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Bypass"}),
                
                "threshold": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "radius": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
                "grain": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_deband"
    CATEGORY = "DINKIssTyle/Image"

    def _box_filter(self, x, r):
        """ Box Filter (Mean Filter) using Avg Pool """
        ch = x.shape[1]
        k_size = 2 * r + 1
        padded = F.pad(x, (r, r, r, r), mode='reflect')
        return F.avg_pool2d(padded, kernel_size=k_size, stride=1, padding=0)

    def _guided_filter(self, x, r, eps):
        """ Guided Image Filtering (Batch Optimized) """
        mean_x = self._box_filter(x, r)
        mean_xx = self._box_filter(x * x, r)
        var_x = mean_xx - mean_x * mean_x
        a = var_x / (var_x + eps)
        b = mean_x - a * mean_x
        mean_a = self._box_filter(a, r)
        mean_b = self._box_filter(b, r)
        return mean_a * x + mean_b

    def apply_deband(self, image, enabled, threshold, radius, grain, iterations):
        """
        image: [B, H, W, 3]
        enabled: Boolean (True/False)
        """
        # [핵심] 토글이 꺼져있으면(False) 원본 이미지를 그대로 반환 (Bypass)
        if not enabled:
            return (image,)

        # Preprocessing: [B, H, W, C] -> [B, C, H, W]
        x = image.permute(0, 3, 1, 2)
        
        # Threshold mapping
        t_val = threshold / 255.0
        eps = t_val * t_val
        
        # Iterative Filtering
        out = x
        for _ in range(iterations):
            out = self._guided_filter(out, radius, eps)
        
        # Dithering (Grain)
        if grain > 0:
            noise_scale = grain / 255.0
            noise = (torch.rand_like(out) - 0.5) * 2.0 * noise_scale
            out = out + noise
            
        out = torch.clamp(out, 0.0, 1.0)
        
        # Postprocessing: [B, C, H, W] -> [B, H, W, C]
        result = out.permute(0, 2, 3, 1)
        
        return (result,)

# Node Registration
NODE_CLASS_MAPPINGS = {
    "DINKI_Deband": DINKI_Deband
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_Deband": "DINKI Deband"
}