import torch
import torch.nn.functional as F

class DINKI_Mask_Weighted_Mix:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # 5개의 마스크와 강도 슬라이더를 정의
        inputs = {
            "required": {},
            "optional": {}
        }
        for i in range(1, 6):
            inputs["optional"][f"mask_{i}"] = ("MASK",)
            # strength: 0.0(검은색/투명) ~ 1.0(흰색/불투명)
            inputs["optional"][f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
        
        return inputs

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mixed_mask",)
    FUNCTION = "mix_masks"
    CATEGORY = "DINKI/Mask"

    def mix_masks(self, mask_1=None, strength_1=1.0, 
                        mask_2=None, strength_2=1.0,
                        mask_3=None, strength_3=1.0, 
                        mask_4=None, strength_4=1.0,
                        mask_5=None, strength_5=1.0):

        # 루프 처리를 위해 리스트로 묶음
        masks_inputs = [
            (mask_1, strength_1), (mask_2, strength_2),
            (mask_3, strength_3), (mask_4, strength_4),
            (mask_5, strength_5)
        ]

        final_mask = None

        for mask, strength in masks_inputs:
            if mask is not None:
                # 1. 강도 적용 (mask * strength)
                # 마스크가 1.0(흰색)이어도 strength가 0.5면 0.5(회색)가 됨
                weighted_mask = mask * strength

                # 2. 첫 번째 마스크면 캔버스로 사용
                if final_mask is None:
                    final_mask = weighted_mask
                else:
                    # 3. 크기 불일치 시 자동 리사이즈 (첫 번째 마스크 기준)
                    if final_mask.shape[-2:] != weighted_mask.shape[-2:]:
                        # interpolate를 위해 차원 추가 (Batch, Channel, H, W)
                        # ComfyUI 마스크는 보통 (Batch, H, W) 형태임
                        wm_dim = weighted_mask.unsqueeze(1) 
                        target_h, target_w = final_mask.shape[-2], final_mask.shape[-1]
                        
                        wm_resized = F.interpolate(wm_dim, size=(target_h, target_w), mode="bilinear", align_corners=False)
                        weighted_mask = wm_resized.squeeze(1)

                    # 4. 병합 (Maximum 방식)
                    # Add(더하기) 대신 Max(최대값)를 사용하여 겹치는 부분이 하얗게 타버리는 현상 방지
                    # 예: 얼굴(0.3) 위에 눈(1.0)이 겹치면 눈 부분은 1.0이 됨 (더하면 1.3이 되어버림)
                    final_mask = torch.max(final_mask, weighted_mask)

        # 연결된 마스크가 하나도 없을 경우 (에러 방지용 빈 마스크)
        if final_mask is None:
            final_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")

        return (final_mask,)

# 노드 등록용 (단일 파일 사용 시)
NODE_CLASS_MAPPINGS = {
    "DINKI_Mask_Weighted_Mix": DINKI_Mask_Weighted_Mix
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_Mask_Weighted_Mix": "DINKI Mask Weighted Mix"
}