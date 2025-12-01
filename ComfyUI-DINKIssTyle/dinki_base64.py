import torch
import numpy as np
from PIL import Image
import io
import base64
import folder_paths
import os
import random

# --------------------------------------------------------------------------------
# Node 1: 이미지를 입력받아 Base64 String으로 변환 및 출력
# --------------------------------------------------------------------------------
class DINKI_Img2Base64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base64_string",)
    FUNCTION = "encode_image"
    CATEGORY = "DINKIssTyle/Image"
    
    def encode_image(self, image):
        # ComfyUI의 이미지는 Tensor (Batch, H, W, C) 형태이며 0-1 범위입니다.
        # 첫 번째 이미지만 처리 (배치 처리 시 너무 길어질 수 있음)
        img_tensor = image[0] 
        
        # Tensor -> PIL Image 변환
        i = 255. * img_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # Buffer에 저장 후 Base64 인코딩
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # 편의를 위해 prefix 없이 순수 데이터만 출력하거나, 필요시 prefix 추가 가능
        # 여기서는 순수 Base64 String만 반환합니다.
        return (img_str,)

# --------------------------------------------------------------------------------
# Node 2: Base64 String을 텍스트로 직접 입력 (Multiline)
# --------------------------------------------------------------------------------
class DINKI_Base64Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base64_string": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base64_string",)
    FUNCTION = "pass_string"
    CATEGORY = "DINKIssTyle/Image"

    def pass_string(self, base64_string):
        return (base64_string,)

# --------------------------------------------------------------------------------
# Node 3: Base64 String을 입력받아 이미지를 보여주는 뷰어 (Preview 기능 포함)
# --------------------------------------------------------------------------------
class DINKI_Base64Viewer:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base64_string": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode_and_view"
    CATEGORY = "DINKIssTyle/Image"
    OUTPUT_NODE = True

    def decode_and_view(self, base64_string):
        try:
            # 헤더 제거 (data:image/png;base64, 등)
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]

            # Base64 디코딩
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            
            # PIL -> Tensor 변환 (ComfyUI 포맷)
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,] # Add batch dimension

            # Preview를 위해 임시 폴더에 이미지 저장
            filename = f"dinki_b64_{random.randint(1, 1000000)}.png"
            image.save(os.path.join(self.output_dir, filename))

            # UI에 이미지 정보를 전달하여 미리보기 표시
            results = [
                {"filename": filename, "subfolder": "", "type": self.type}
            ]
            
            return {"ui": {"images": results}, "result": (image_tensor,)}

        except Exception as e:
            print(f"[DINKI Error] Base64 decoding failed: {e}")
            # 에러 발생 시 빈 검정 이미지 반환 (크래시 방지)
            empty_img = torch.zeros((1, 512, 512, 3))
            return {"ui": {"images": []}, "result": (empty_img,)}
