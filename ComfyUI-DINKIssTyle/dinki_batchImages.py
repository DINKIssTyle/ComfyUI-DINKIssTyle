import torch
import torch.nn.functional as F

class DINKI_BatchImages:
    @classmethod
    def INPUT_TYPES(cls):
        # 1~10번 이미지 입력을 동적으로 생성
        optional_inputs = {}
        for i in range(1, 11):
            optional_inputs[f"image{i}"] = ("IMAGE",)
            
        return {
            "required": {
                # label_on/off를 사용하여 UI상에서 스위치 상태를 직관적으로 표시
                "batch_image": ("BOOLEAN", {"default": True, "label_on": "multiple", "label_off": "single"}),
            },
            "optional": optional_inputs
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "DINKIssTyle/Image"

    def run(self, batch_image, **kwargs):
        # 1. 입력된 모든 이미지를 순서대로 수집 (None 제외)
        valid_images = []
        for i in range(1, 11):
            img = kwargs.get(f"image{i}")
            if img is not None:
                valid_images.append(img)

        # 2. 이미지가 하나도 없으면 안전하게 None 리턴
        if not valid_images:
            return (None,)

        # 3. Single 모드 (Multiple 꺼짐): 첫 번째 이미지만 통과
        if not batch_image:
            return (valid_images[0],)

        # 4. Multiple 모드 (Multiple 켜짐): 오토 리사이징 및 병합
        else:
            # 첫 번째 이미지를 기준 사이즈로 설정 (B, H, W, C)
            target_img = valid_images[0]
            target_h = target_img.shape[1]
            target_w = target_img.shape[2]
            
            processed_images = [target_img] # 첫 번째 이미지는 그대로 넣음

            for img in valid_images[1:]:
                # 크기가 다르면 리사이징 수행
                if img.shape[1] != target_h or img.shape[2] != target_w:
                    # PyTorch interpolate를 쓰기 위해 (B, H, W, C) -> (B, C, H, W)로 변경
                    img = img.movedim(-1, 1)
                    
                    # 리사이징 (bilinear 보간법 사용)
                    img = F.interpolate(img, size=(target_h, target_w), mode="bilinear", align_corners=False)
                    
                    # 다시 원래대로 (B, C, H, W) -> (B, H, W, C)로 변경
                    img = img.movedim(1, -1)
                
                processed_images.append(img)

            # 병합 (Batch Concatenation)
            batch = torch.cat(processed_images, dim=0)
            return (batch,)

# 요청하신 매핑 규칙 적용
NODE_CLASS_MAPPINGS = {
    "DINKI_BatchImages": DINKI_BatchImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_BatchImages": "DINKI Batch Images",
}