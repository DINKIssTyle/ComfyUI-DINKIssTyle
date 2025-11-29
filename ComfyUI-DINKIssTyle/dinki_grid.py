import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw

class DINKI_Grid:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # 1~10번 이미지 입력 (Optional로 설정하여 연결 안 된 경우 처리)
        img_inputs = {f"image_{i}": ("IMAGE",) for i in range(1, 11)}
        
        return {
            "required": {
                "cols": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "frame_thickness": ("INT", {"default": 10, "min": 0, "max": 500, "step": 1}),
                "bg_color_hex": ("STRING", {"default": "#000000", "multiline": False}),
                "resize_method": (["Keep Ratio (Fit)", "Keep Ratio (Crop)", "Stretch", "No Resize (Top-Left)"],),
                "limit_output": ("BOOLEAN", {"default": False}),
                "max_output_width": ("INT", {"default": 3840, "min": 512, "max": 8192}),
                "max_output_height": ("INT", {"default": 2160, "min": 512, "max": 8192}),
            },
            "optional": img_inputs
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_grid"
    CATEGORY = "DINKIssTyle/Image"

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        else:
            return (0, 0, 0) # Default to black on error

    def tensor_to_pil(self, img_tensor):
        # Batch, H, W, C -> PIL Image
        # 만약 배치(여러 장)가 들어오면 첫 번째 장만 사용
        return Image.fromarray(np.clip(255. * img_tensor[0].cpu().numpy(), 0, 255).astype(np.uint8))

    def pil_to_tensor(self, img_pil):
        img = np.array(img_pil).astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0) # (1, H, W, C)

    def generate_grid(self, cols, rows, frame_thickness, bg_color_hex, resize_method, 
                     limit_output, max_output_width, max_output_height, **kwargs):
        
        # 1. 입력 이미지 수집 (image_1 ~ image_10)
        images = []
        for i in range(1, 11):
            key = f"image_{i}"
            if key in kwargs and kwargs[key] is not None:
                images.append(self.tensor_to_pil(kwargs[key]))

        if not images:
            # 이미지가 하나도 없으면 512x512 검은 화면 반환
            return (torch.zeros((1, 512, 512, 3)),)

        # 2. 기준 셀 크기 결정 (1번 이미지 기준)
        base_w, base_h = images[0].size
        
        # 프레임 두께를 포함한 셀 크기 (이미지 영역 + 프레임)
        # 1번 이미지의 크기를 '프레임을 제외한 순수 이미지 영역'으로 볼지, 
        # '셀 전체 크기'로 볼지에 따라 다르지만, 여기서는 1번 이미지 크기 = 셀 크기로 설정합니다.
        # 즉, 프레임이 있으면 이미지가 그만큼 작아집니다.
        cell_w, cell_h = base_w, base_h
        
        # 실제 이미지가 들어갈 영역 크기
        content_w = max(1, cell_w - (frame_thickness * 2))
        content_h = max(1, cell_h - (frame_thickness * 2))

        bg_color = self.hex_to_rgb(bg_color_hex)

        # 3. 전체 캔버스 생성
        grid_w = cell_w * cols
        grid_h = cell_h * rows
        canvas = Image.new("RGB", (grid_w, grid_h), bg_color)

        # 4. 이미지 배치
        for idx in range(cols * rows):
            # 그리드 좌표 계산
            col_idx = idx % cols
            row_idx = idx // cols
            
            x_offset = col_idx * cell_w
            y_offset = row_idx * cell_h

            # 입력된 이미지가 존재하는 경우에만 처리 (나머지는 배경색 유지)
            if idx < len(images):
                img = images[idx]
                
                # 리사이즈 로직
                processed_img = None
                
                if resize_method == "Stretch":
                    processed_img = img.resize((content_w, content_h), Image.Resampling.LANCZOS)
                    
                elif resize_method == "Keep Ratio (Fit)":
                    # 비율 유지하며 박스 안에 맞춤 (남는 공간 배경색)
                    # ImageOps.contain은 원본보다 커지지 않으므로 수동 계산
                    ratio = min(content_w / img.width, content_h / img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    processed_img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                elif resize_method == "Keep Ratio (Crop)":
                    # 비율 유지하며 박스를 꽉 채움 (넘치는 부분 잘림)
                    processed_img = ImageOps.fit(img, (content_w, content_h), method=Image.Resampling.LANCZOS)
                
                elif resize_method == "No Resize (Top-Left)":
                    # 리사이즈 없이 원본 그대로 사용 (크면 잘림)
                    processed_img = img
                
                # 이미지 붙이기 (프레임 안쪽 중앙 정렬 또는 좌상단)
                if processed_img:
                    if resize_method == "No Resize (Top-Left)":
                        # 좌상단 정렬 (프레임 오프셋만 적용)
                        paste_x = x_offset + frame_thickness
                        paste_y = y_offset + frame_thickness
                        # 캔버스 밖으로 나가는 부분은 crop됨 (PIL 기본 동작)
                        canvas.paste(processed_img, (paste_x, paste_y))
                    else:
                        # 중앙 정렬 (Fit의 경우 중요)
                        paste_x = x_offset + frame_thickness + (content_w - processed_img.width) // 2
                        paste_y = y_offset + frame_thickness + (content_h - processed_img.height) // 2
                        canvas.paste(processed_img, (paste_x, paste_y))

        # 5. 전체 출력 크기 제한 (Downscale)
        if limit_output:
            if canvas.width > max_output_width or canvas.height > max_output_height:
                ratio = min(max_output_width / canvas.width, max_output_height / canvas.height)
                new_w = int(canvas.width * ratio)
                new_h = int(canvas.height * ratio)
                # 다운스케일링은 LANCZOS가 품질이 좋음
                canvas = canvas.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return (self.pil_to_tensor(canvas),)
