import torch
import numpy as np
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
                "text_opacity": ("INT", {"default": 255, "min": 0, "max": 255}),

                # --- 오버레이 이미지 설정 ---
                "overlay_position": (positions, {"default": "Top-Left"}),
                "overlay_margin_percent": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "overlay_size_percent": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "overlay_opacity": ("INT", {"default": 255, "min": 0, "max": 255}),
            },
            "optional": {
                "overlay_image": ("IMAGE",),
                "overlay_mask": ("MASK",), # [추가됨] 투명도 처리를 위한 마스크 입력
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_overlay"
    CATEGORY = "DINKIssTyle/Image"
    DESCRIPTION = "Adds text and/or image overlays. Connect 'MASK' from Load Image to 'overlay_mask' for transparent PNGs."

    def apply_overlay(self, image, text_content, enable_text, enable_overlay_image,
                      text_position, text_margin_percent, text_size_percent, text_color_hex, text_opacity,
                      overlay_position, overlay_margin_percent, overlay_size_percent, overlay_opacity,
                      overlay_image=None, overlay_mask=None): # overlay_mask 추가

        result_images = []
        
        def get_font(size):
            try:
                return ImageFont.truetype("arial.ttf", size)
            except IOError:
                try:
                    return ImageFont.truetype("DejaVuSans.ttf", size)
                except IOError:
                    return ImageFont.load_default()

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

            # 텍스트 처리
            if enable_text and text_content and isinstance(text_content, str) and text_content.strip():
                font_size = int(base_h * (text_size_percent / 100))
                font_size = max(1, font_size)
                font = get_font(font_size)
                
                bbox = draw.textbbox((0, 0), text_content, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                tx, ty = calculate_xy(base_w, base_h, text_w, text_h, text_position, text_margin_percent)
                
                try:
                    from PIL import ImageColor
                    color_rgb = ImageColor.getrgb(text_color_hex)
                except:
                    color_rgb = (255, 255, 255)
                
                draw.text((tx, ty), text_content, font=font, fill=color_rgb + (text_opacity,))

            # 오버레이 이미지 처리
            if enable_overlay_image and overlay_image is not None:
                # 1. 오버레이 이미지(RGB) 가져오기
                ov_tensor = overlay_image[i] if i < len(overlay_image) else overlay_image[0]
                ov_pil = Image.fromarray(np.clip(255. * ov_tensor.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
                
                # 2. 마스크(Alpha) 적용 로직 추가
                if overlay_mask is not None:
                    # 마스크 배치 처리
                    mask_tensor = overlay_mask[i] if i < len(overlay_mask) else overlay_mask[0]
                    # 마스크가 [H, W] 형태라면 [H, W, 1]로 변환 필요할 수 있음 (numpy 변환 과정에서 처리)
                    mask_np = np.clip(255. * mask_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
                    mask_pil = Image.fromarray(mask_np, mode='L') # Grayscale 이미지로 변환
                    
                    # 마스크 크기가 이미지와 다를 경우를 대비해 리사이즈 (보통은 같음)
                    if mask_pil.size != ov_pil.size:
                        mask_pil = mask_pil.resize(ov_pil.size, Image.LANCZOS)
                    
                    # RGB 이미지에 알파 채널로 마스크 주입
                    ov_pil.putalpha(mask_pil)

                # 3. 리사이즈 및 합성
                target_ov_w = int(base_w * (overlay_size_percent / 100))
                if target_ov_w > 0:
                    aspect_ratio = ov_pil.height / ov_pil.width
                    target_ov_h = int(target_ov_w * aspect_ratio)
                    
                    ov_pil = ov_pil.resize((target_ov_w, target_ov_h), Image.LANCZOS)
                    
                    # 전체 투명도(Opacity) 조절
                    if overlay_opacity < 255:
                        alpha = ov_pil.split()[3]
                        alpha = ImageOps.scale(alpha, 1.0)
                        alpha = alpha.point(lambda p: p * (overlay_opacity / 255))
                        ov_pil.putalpha(alpha)

                    ox, oy = calculate_xy(base_w, base_h, target_ov_w, target_ov_h, overlay_position, overlay_margin_percent)
                    txt_layer.paste(ov_pil, (ox, oy), ov_pil) # 투명도 유지하며 붙여넣기

            out_pil = Image.alpha_composite(img_pil, txt_layer)
            out_tensor = torch.from_numpy(np.array(out_pil.convert('RGB')).astype(np.float32) / 255.0).unsqueeze(0)
            result_images.append(out_tensor)

        if len(result_images) > 1:
            return (torch.cat(result_images, dim=0),)
        return (result_images[0],)