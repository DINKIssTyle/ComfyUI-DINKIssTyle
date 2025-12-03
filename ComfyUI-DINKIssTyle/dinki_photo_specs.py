import math

class DINKI_photo_specifications:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "megapixels": (["1MP", "2MP", "3MP", "4MP"], {"default": "1MP"}),
                "aspect_ratio": (
                    [
                        # --- Basic ---
                        "Basic 1:1", 
                        "Bacic 1:2",
                        "Bacic 1.5:2",  
                        "Basic 9:16", 
                        "Basic 10:16", 
                        # --- Photo ---
                        "Photo 3:4", 
                        "Photo 3.5:5", 
                        "Photo 4:6", 
                        "Photo 5:7", 
                        "Photo 6:8", 
                        "Photo 8:10", 
                        "Photo 10:13", 
                        "Photo 10:15", 
                        "Photo 11:14",
                        # --- Cinema / Film ---
                        "35mm Academy 1.37:1",
                        "35mm Flat 1.85:1",
                        "35mm Scope (Anamorphic) 2.39:1",
                        "70mm Todd-AO 2.20:1",
                        "IMAX 70mm 1.43:1",
                        "Super 35 1.85:1",
                        "Super 35 2.39:1",
                        "Super 16 1.66:1",
                        "Super 16 1.78:1",
                    ],
                    {"default": "Basic 1:1"}
                ),
                "orientation": (["Portrait", "Landscape"], {"default": "Portrait"}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "info_string")
    FUNCTION = "calculate_resolution"
    CATEGORY = "DINKIssTyle/Image"
    
    DESCRIPTION = "Selects the optimal resolution based on megapixels and aspect ratio. (Calculated in multiples of 8)"

    def calculate_resolution(self, megapixels, aspect_ratio, orientation):
        # 1. 목표 픽셀 수 설정 (Base: 1024x1024 = 1,048,576 pixel for 1MP)
        mp_multiplier = int(megapixels.replace("MP", ""))
        target_area = 1024 * 1024 * mp_multiplier

        # 2. 비율 파싱 로직 수정
        # 입력값이 "35mm Academy 1.37:1" 처럼 들어오므로, 공백으로 자른 후 마지막 부분만 가져옵니다.
        # 예: "Photo 3:4" -> ["Photo", "3:4"] -> "3:4"
        # 예: "35mm Academy 1.37:1" -> [..., "Academy", "1.37:1"] -> "1.37:1"
        ratio_string = aspect_ratio.split(" ")[-1]
        
        ratio_parts = ratio_string.split(":")
        w_ratio = float(ratio_parts[0])
        h_ratio = float(ratio_parts[1])
        
        # 실제 비율 값 (Width / Height)
        target_ratio = w_ratio / h_ratio

        # 3. 너비와 높이 계산
        height_val = math.sqrt(target_area / target_ratio)
        width_val = height_val * target_ratio

        # 4. 8의 배수로 보정 (반올림)
        width = round(width_val / 8) * 8
        height = round(height_val / 8) * 8

        # 5. 방향(Orientation) 적용
        is_portrait = "Portrait" in orientation
        
        if is_portrait:
            if width > height:
                width, height = height, width
        else: # Landscape
            if width < height:
                width, height = height, width

        # 6. 정보 텍스트 생성 (선택한 옵션 이름 전체를 포함)
        info_string = f"{width}x{height} ({aspect_ratio}, {megapixels})"

        return (width, height, info_string)

# ComfyUI 노드 등록
NODE_CLASS_MAPPINGS = {
    "DINKI_photo_specifications": DINKI_photo_specifications
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_photo_specifications": "DINKI Photo Specifications"
}