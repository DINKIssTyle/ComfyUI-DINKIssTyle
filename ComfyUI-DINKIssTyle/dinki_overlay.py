import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps

class DINKI_Overlay:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        positions = ["Top-Left", "Top-Center", "Top-Right", "Center", "Bottom-Left", "Bottom-Center", "Bottom-Right"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "text_content": ("STRING", {"multiline": True, "default": "Created with [AI Model] via ComfyUI."}),
                
                # --- 모드 선택 ---
                "enable_text": ("BOOLEAN", {"default": True, "label_on": "Text On", "label_off": "Text Off"}),
                "enable_overlay_image": ("BOOLEAN", {"default": True, "label_on": "Image On", "label_off": "Image Off"}),

                # --- 텍스트 설정 ---
                "text_position": (positions, {"default": "Bottom-Right"}),
                "text_margin_percent": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "text_size_percent": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "text_color_hex": ("STRING", {"default": "#FFFFFF"}),
                # [변경됨] 0~255 INT -> 0.0~100.0 FLOAT
                "text_opacity": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 0.1, "display": "slider"}), 

                # --- 오버레이 이미지 설정 ---
                "overlay_position": (positions, {"default": "Top-Left"}),
                "overlay_margin_percent": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "overlay_size_percent": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                # [변경됨] 0~255 INT -> 0.0~100.0 FLOAT
                "overlay_opacity": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 0.1, "display": "slider"}),
            },
            "optional": {
                "overlay_image": ("IMAGE",),
                "overlay_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_overlay"
    CATEGORY = "DINKIssTyle/Image"
    DESCRIPTION = "Adds text and/or image overlays. Opacity is controlled by percentage (0-100%)."

    def apply_overlay(self, image, text_content, enable_text, enable_overlay_image,
                      text_position, text_margin_percent, text_size_percent, text_color_hex, text_opacity,
                      overlay_position, overlay_margin_percent, overlay_size_percent, overlay_opacity,
                      overlay_image=None, overlay_mask=None):

        result_images = []
        
        # 폰트 로딩 함수 (OS별 폰트 자동 탐색)
        def get_font(size):
            current_dir = os.path.dirname(os.path.realpath(__file__))
            font_path = os.path.join(current_dir, "fonts", "NanumGothicBold.ttf") 
            
            try:
                return ImageFont.truetype(font_path, size)
            except IOError:
                try:
                    return ImageFont.truetype("malgun.ttf", size) # Windows
                except IOError:
                    try:
                        return ImageFont.truetype("AppleGothic.ttf", size) # Mac
                    except IOError:
                        try:
                            return ImageFont.truetype("arial.ttf", size) # Linux/Basic
                        except IOError:
                            return ImageFont.load_default()

        # 좌표 계산 함수
        def calculate_xy(base_w, base_h, target_w, target_h, pos_str, margin_pct):
            margin_x = int(base_w * (margin_pct / 100))
            margin_y = int(base_h * (margin_pct / 100))
            
            x, y = 0, 0
            
            if pos_str == "Top-Left":
                x, y = margin_x, margin_y
            elif pos_str == "Top-Center":
                x = (base_w - target_w) // 2
                y = margin_y
            elif pos_str == "Top-Right":
                x = base_w - target_w - margin_x
                y = margin_y
            elif pos_str == "Center":
                x = (base_w - target_w) // 2
                y = (base_h - target_h) // 2
            elif pos_str == "Bottom-Left":
                x = margin_x
                y = base_h - target_h - margin_y
            elif pos_str == "Bottom-Center":
                x = (base_w - target_w) // 2
                y = base_h - target_h - margin_y
            elif pos_str == "Bottom-Right":
                x = base_w - target_w - margin_x
                y = base_h - target_h - margin_y
                
            return x, y

        for i in range(len(image)):
            img_tensor = image[i]
            img_pil = Image.fromarray(np.clip(255. * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            base_w, base_h = img_pil.size
            
            txt_layer = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(txt_layer)

            # --- 텍스트 처리 ---
            if enable_text and text_content and isinstance(text_content, str) and text_content.strip():
                font_size = int(base_h * (text_size_percent / 100))
                font_size = max(1, font_size)
                font = get_font(font_size)
                
                bbox = draw.textbbox((0, 0), text_content, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                tx, ty = calculate_xy(base_w, base_h, text_w, text_h, text_position, text_margin_percent)
                tx = tx - bbox[0]
                ty = ty - bbox[1]

                try:
                    from PIL import ImageColor
                    color_rgb = ImageColor.getrgb(text_color_hex)
                except:
                    color_rgb = (255, 255, 255)
                
                # [변경됨] 퍼센트(0-100)를 0-255 정수로 변환
                text_alpha_val = int((text_opacity / 100.0) * 255)
                draw.text((tx, ty), text_content, font=font, fill=color_rgb + (text_alpha_val,))

            # --- 오버레이 이미지 처리 ---
            if enable_overlay_image and overlay_image is not None:
                ov_tensor = overlay_image[i] if i < len(overlay_image) else overlay_image[0]
                ov_pil = Image.fromarray(np.clip(255. * ov_tensor.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
                
                # 마스크 처리
                if overlay_mask is not None:
                    mask_tensor = overlay_mask[i] if i < len(overlay_mask) else overlay_mask[0]
                    mask_np = np.clip(255. * mask_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
                    mask_pil = Image.fromarray(mask_np, mode='L')
                    
                    if mask_pil.size != ov_pil.size:
                        mask_pil = mask_pil.resize(ov_pil.size, Image.LANCZOS)
                    
                    ov_pil.putalpha(mask_pil)

                target_ov_w = int(base_w * (overlay_size_percent / 100))
                if target_ov_w > 0:
                    aspect_ratio = ov_pil.height / ov_pil.width
                    target_ov_h = int(target_ov_w * aspect_ratio)
                    
                    ov_pil = ov_pil.resize((target_ov_w, target_ov_h), Image.LANCZOS)
                    
                    # [변경됨] 퍼센트(0-100) 기반 투명도 처리
                    if overlay_opacity < 100.0:
                        alpha = ov_pil.split()[3]
                        alpha = ImageOps.scale(alpha, 1.0)
                        # 알파 채널 값에 퍼센트를 곱함
                        alpha = alpha.point(lambda p: p * (overlay_opacity / 100.0))
                        ov_pil.putalpha(alpha)

                    ox, oy = calculate_xy(base_w, base_h, target_ov_w, target_ov_h, overlay_position, overlay_margin_percent)
                    txt_layer.paste(ov_pil, (ox, oy), ov_pil)

            out_pil = Image.alpha_composite(img_pil, txt_layer)
            out_tensor = torch.from_numpy(np.array(out_pil.convert('RGB')).astype(np.float32) / 255.0).unsqueeze(0)
            result_images.append(out_tensor)

        if len(result_images) > 1:
            return (torch.cat(result_images, dim=0),)
        return (result_images[0],)
