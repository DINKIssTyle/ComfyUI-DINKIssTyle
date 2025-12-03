import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

class DINKI_Overlay:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        positions = ["Top-Left", "Top-Center", "Top-Right", "Center", "Bottom-Left", "Bottom-Center", "Bottom-Right"]
        alignments = ["Left", "Center", "Right"]
        
        # 1. fonts 폴더 스캔하여 파일 목록 가져오기
        current_dir = os.path.dirname(os.path.realpath(__file__))
        fonts_dir = os.path.join(current_dir, "fonts")
        font_files = []
        
        if os.path.exists(fonts_dir):
            # .ttf, .otf 파일만 리스트업
            font_files = [f for f in os.listdir(fonts_dir) if f.lower().endswith(('.ttf', '.otf'))]
            font_files.sort() # 이름순 정렬
        
        # 폰트가 하나도 없으면 기본값 설정
        if not font_files:
            font_files = ["Default"]

        return {
            "required": {
                "image": ("IMAGE",),
                "text_content": ("STRING", {"multiline": True, "default": "Created with [AI Model] via ComfyUI."}),
                
                # --- 모드 선택 ---
                "enable_text": ("BOOLEAN", {"default": True, "label_on": "Text On", "label_off": "Text Off"}),
                "enable_overlay_image": ("BOOLEAN", {"default": True, "label_on": "Image On", "label_off": "Image Off"}),

                # --- 텍스트 기본 설정 ---
                "font_name": (font_files, {"default": font_files[0] if font_files else "Default"}),
                
                "text_position": (positions, {"default": "Center"}),
                "text_align": (alignments, {"default": "Center"}), 
                "text_wrap_percent": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "0 to disable wrapping"}),
                
                # [추가됨] 줄 간격 조절 (배수)
                "line_spacing_multiplier": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 5.0, "step": 0.1}),

                "text_margin_percent": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "text_size_percent": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "text_color_hex": ("STRING", {"default": "#FFFFFF"}),
                "text_opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1, "display": "slider"}),
                
                # --- 텍스트 꾸밈: Stroke (테두리) ---
                "enable_stroke": ("BOOLEAN", {"default": False, "label_on": "Stroke On", "label_off": "Stroke Off"}),
                "stroke_color_hex": ("STRING", {"default": "#000000"}),
                "stroke_width": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),

                # --- 텍스트 꾸밈: Shadow (그림자) ---
                "enable_shadow": ("BOOLEAN", {"default": False, "label_on": "Shadow On", "label_off": "Shadow Off"}),
                "shadow_color_hex": ("STRING", {"default": "#000000"}),
                "shadow_offset_x": ("INT", {"default": 5, "min": -100, "max": 100}),
                "shadow_offset_y": ("INT", {"default": 5, "min": -100, "max": 100}),
                "shadow_spread": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "shadow_opacity": ("INT", {"default": 70, "min": 0, "max": 100, "step": 1, "display": "slider"}),

                # --- 오버레이 이미지 설정 ---
                "overlay_position": (positions, {"default": "Top-Right"}),
                "overlay_margin_percent": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "overlay_size_percent": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "overlay_opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1, "display": "slider"}),
            },
            "optional": {
                "overlay_image": ("IMAGE",),
                "overlay_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_overlay"
    CATEGORY = "DINKIssTyle/Image"
    DESCRIPTION = "Adds text (align, wrap, spacing) and image overlays."

    def apply_overlay(self, image, text_content, enable_text, enable_overlay_image,
                      font_name,
                      text_position, text_align, text_wrap_percent, line_spacing_multiplier, 
                      text_margin_percent, text_size_percent, text_color_hex, text_opacity,
                      enable_stroke, stroke_color_hex, stroke_width,
                      enable_shadow, shadow_color_hex, shadow_offset_x, shadow_offset_y, shadow_spread, shadow_opacity,
                      overlay_position, overlay_margin_percent, overlay_size_percent, overlay_opacity,
                      overlay_image=None, overlay_mask=None):

        result_images = []
        
        # --- Helper Functions ---
        def get_font(filename, size):
            current_dir = os.path.dirname(os.path.realpath(__file__))
            font_path = os.path.join(current_dir, "fonts", filename) 
            try: return ImageFont.truetype(font_path, size)
            except IOError:
                try: return ImageFont.truetype("arial.ttf", size)
                except: return ImageFont.load_default()

        def get_rgb(hex_code, default=(255, 255, 255)):
            try:
                from PIL import ImageColor
                return ImageColor.getrgb(hex_code)
            except: return default

        def calculate_xy(base_w, base_h, target_w, target_h, pos_str, margin_pct):
            margin_x = int(base_w * (margin_pct / 100))
            margin_y = int(base_h * (margin_pct / 100))
            x, y = 0, 0
            if pos_str == "Top-Left": x, y = margin_x, margin_y
            elif pos_str == "Top-Center": x, y = (base_w - target_w) // 2, margin_y
            elif pos_str == "Top-Right": x, y = base_w - target_w - margin_x, margin_y
            elif pos_str == "Center": x, y = (base_w - target_w) // 2, (base_h - target_h) // 2
            elif pos_str == "Bottom-Left": x, y = margin_x, base_h - target_h - margin_y
            elif pos_str == "Bottom-Center": x, y = (base_w - target_w) // 2, base_h - target_h - margin_y
            elif pos_str == "Bottom-Right": x, y = base_w - target_w - margin_x, base_h - target_h - margin_y
            return x, y

        # --- Text Wrapping Logic ---
        def wrap_text_pixel(text, font, max_pixel_width, draw_obj, stroke_w):
            if max_pixel_width <= 0: return text.split('\n')
            
            wrapped_lines = []
            paragraphs = text.split('\n')
            
            for para in paragraphs:
                words = para.split(' ')
                if not words:
                    wrapped_lines.append("")
                    continue
                
                current_line = words[0]
                for word in words[1:]:
                    test_line = current_line + " " + word
                    bbox = draw_obj.textbbox((0, 0), test_line, font=font, stroke_width=stroke_w)
                    w = bbox[2] - bbox[0]
                    
                    if w <= max_pixel_width:
                        current_line = test_line
                    else:
                        wrapped_lines.append(current_line)
                        current_line = word
                wrapped_lines.append(current_line)
                
            return wrapped_lines

        # --- Main Processing Loop ---
        for i in range(len(image)):
            img_tensor = image[i]
            img_pil = Image.fromarray(np.clip(255. * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            base_w, base_h = img_pil.size
            
            txt_layer = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(txt_layer)

            # 1. Text Processing
            if enable_text and text_content and isinstance(text_content, str) and text_content.strip():
                font_size = int(base_h * (text_size_percent / 100))
                font_size = max(1, font_size)
                font = get_font(font_name, font_size)
                
                # Determine Wrapping Width
                wrap_px = 0
                if text_wrap_percent > 0:
                    wrap_px = int(base_w * (text_wrap_percent / 100))

                s_width_calc = stroke_width if enable_stroke else 0
                
                # Apply Wrapping
                lines = wrap_text_pixel(text_content, font, wrap_px, draw, s_width_calc)
                
                # Calculate dimensions for the WHOLE text block
                line_widths = []
                max_text_w = 0
                
                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font, stroke_width=s_width_calc)
                    w = bbox[2] - bbox[0]
                    line_widths.append(w)
                    max_text_w = max(max_text_w, w)
                
                # [수정됨] Line Height Calculation with Multiplier
                sample_bbox = draw.textbbox((0, 0), "Aj", font=font)
                base_line_height = sample_bbox[3] - sample_bbox[1]
                
                # 사용자가 입력한 multiplier 적용
                line_spacing = int(base_line_height * line_spacing_multiplier)
                
                total_text_h = line_spacing * (len(lines) - 1) + base_line_height # 마지막 줄은 높이만 계산 (옵션) or 전체 동일 간격
                # 간단하게 전체 간격으로 통일
                total_text_h = line_spacing * len(lines)
                
                # Calculate Origin
                tx_origin, ty_origin = calculate_xy(base_w, base_h, max_text_w, total_text_h, text_position, text_margin_percent)

                text_rgb = get_rgb(text_color_hex, (255, 255, 255))
                text_alpha_val = int((text_opacity / 100.0) * 255)
                stroke_rgb = get_rgb(stroke_color_hex, (0, 0, 0))
                shadow_rgb = get_rgb(shadow_color_hex, (0, 0, 0))
                shadow_alpha_val = int((shadow_opacity / 100.0) * 255)

                shadow_layer = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
                shadow_draw = ImageDraw.Draw(shadow_layer)
                
                current_y = ty_origin

                # Draw Line by Line
                for idx, line in enumerate(lines):
                    line_w = line_widths[idx]
                    
                    # Alignment Logic
                    align_offset_x = 0
                    if text_align == "Center":
                        align_offset_x = (max_text_w - line_w) // 2
                    elif text_align == "Right":
                        align_offset_x = max_text_w - line_w
                    
                    final_x = tx_origin + align_offset_x

                    # 1. Stroke
                    if enable_stroke:
                        draw.text((final_x, current_y), line, font=font, fill=stroke_rgb + (text_alpha_val,), 
                                  stroke_width=stroke_width, stroke_fill=stroke_rgb + (text_alpha_val,))

                    # 2. Text
                    draw.text((final_x, current_y), line, font=font, fill=text_rgb + (text_alpha_val,))
                    
                    # 3. Shadow
                    if enable_shadow:
                        sx = final_x + shadow_offset_x
                        sy = current_y + shadow_offset_y
                        shadow_draw.text((sx, sy), line, font=font, fill=shadow_rgb + (shadow_alpha_val,), 
                                         stroke_width=s_width_calc if enable_stroke else 0, 
                                         stroke_fill=shadow_rgb + (shadow_alpha_val,))

                    # [수정됨] 간격만큼 이동
                    current_y += line_spacing

                # Blur & Merge Shadow
                if enable_shadow and shadow_spread > 0:
                    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_spread))
                
                if enable_shadow:
                    txt_layer = Image.alpha_composite(shadow_layer, txt_layer) 

            # 2. Overlay Image Processing
            if enable_overlay_image and overlay_image is not None:
                ov_tensor = overlay_image[i] if i < len(overlay_image) else overlay_image[0]
                ov_pil = Image.fromarray(np.clip(255. * ov_tensor.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
                
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
                    
                    if overlay_opacity < 100:
                        alpha = ov_pil.split()[3]
                        alpha = ImageOps.scale(alpha, 1.0)
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