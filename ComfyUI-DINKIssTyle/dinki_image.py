# ComfyUI/custom_nodes/ComfyUI-DINKIssTyle/dinki_image_nodes.py

import os
import torch
import numpy as np
import comfy.utils
from PIL import Image, ImageDraw, ImageFont


class DINKI_ImageSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {"optional": {f"image_{i}": ("IMAGE",) for i in range(1, 9)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select_latest_image"
    CATEGORY = "DINKIssTyle/Util"

    def select_latest_image(self, **kwargs):
        for i in range(8, 0, -1):
            image = kwargs.get(f"image_{i}")
            if image is not None:
                return (image,)
        # 비어있을 때는 1x1 dummy
        return (torch.zeros(1, 1, 1, 3),)


class DINKI_CrossOutputSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image_out_1", "image_out_2")
    FUNCTION = "process"
    CATEGORY = "DINKIssTyle/Util"

    def process(self, image_1, image_2, invert=False):
        # invert가 True면 교차 출력 (2,1), 아니면 그대로 (1,2)
        if invert:
            return (image_2, image_1)
        else:
            return (image_1, image_2)


class DINKI_ImagePreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "images": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 384, "min": 64, "max": 4096}),
                "bg_gray": ("INT", {"default": 42, "min": 0, "max": 255}),
                "fg_gray": ("INT", {"default": 220, "min": 0, "max": 255}),
                "font_size": ("INT", {"default": 32, "min": 6, "max": 256}),
                "font_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "TTF/TTC 경로(비우면 자동 탐색)"
                }),
                "auto_fit": ("BOOLEAN", {"default": True}),
                "line_spacing": ("INT", {"default": 4, "min": 0, "max": 100}),   # 줄 간격(px)
                "placeholder_text": ("STRING", {"default": "No Image", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "preview"
    CATEGORY = "DINKIssTyle/Util"

    # ===== 내부 유틸 =====
    def _find_font(self, font_path_hint: str):
        if font_path_hint and os.path.exists(font_path_hint):
            return font_path_hint
        candidates = [
            "DejaVuSans.ttf",
            os.path.join(os.getcwd(), "DejaVuSans.ttf"),
            os.path.join(os.getcwd(), "ComfyUI", "DejaVuSans.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _load_font(self, font_path, size):
        ttf = self._find_font(font_path)
        try:
            if ttf:
                return ImageFont.truetype(ttf, size), ttf
            return ImageFont.truetype("DejaVuSans.ttf", size), "DejaVuSans.ttf"
        except Exception:
            return ImageFont.load_default(), None  # 고정 크기

    def _measure_multiline(self, draw, lines, font, line_spacing):
        """여러 줄의 총 폭/높이와 각 라인의 bbox를 계산"""
        bboxes, widths, heights = [], [], []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            bboxes.append(bbox)
            widths.append(w)
            heights.append(h)
        total_h = sum(heights) + line_spacing * max(0, len(lines) - 1)
        max_w = max(widths) if widths else 0
        return max_w, total_h, widths, heights

    def _placeholder(self, w, h, bg, fg, text, font_size, font_path, auto_fit, line_spacing):
        img = Image.new("RGB", (w, h), (bg, bg, bg))
        draw = ImageDraw.Draw(img)

        # 멀티라인 분해
        lines = (text or "").split("\n")
        # 폰트 로딩
        font, ttf = self._load_font(font_path, font_size)

        # 자동 맞춤(멀티라인 기준)
        if auto_fit and ttf is not None:
            max_w_allowed = int(w * 0.9)
            max_h_allowed = int(h * 0.9)
            fs = font.size
            # 너무 큰 경우 줄여서 전체가 영역 안에 들어오게
            while fs > 6:
                max_w_line, total_h, _, _ = self._measure_multiline(draw, lines, font, line_spacing)
                if max_w_line <= max_w_allowed and total_h <= max_h_allowed:
                    break
                fs -= 2
                try:
                    font = ImageFont.truetype(ttf, fs)
                except Exception:
                    font = ImageFont.load_default()
                    break

        # 다시 측정
        max_w_line, total_h, widths, heights = self._measure_multiline(draw, lines, font, line_spacing)

        # 시작 Y: 중앙 정렬
        y = (h // 2) - (total_h // 2)

        # 각 줄 그리기 (수평 중앙)
        for i, line in enumerate(lines):
            tw = widths[i]
            th = heights[i]
            x = w // 2  # anchor="mm" 사용
            draw.text((x, y + th // 2), line, fill=(fg, fg, fg), font=font, anchor="mm")
            y += th + line_spacing

        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = arr[None, ...]
        return torch.from_numpy(arr)

    def preview(
        self,
        images=None,
        width=512,
        height=384,
        bg_gray=42,
        fg_gray=220,
        font_size=32,
        font_path="",
        auto_fit=True,
        line_spacing=4,
        placeholder_text="No Image",
    ):
        if images is None:
            out = self._placeholder(
                width, height, bg_gray, fg_gray,
                placeholder_text, font_size, font_path,
                auto_fit, line_spacing
            )
        else:
            out = (
                torch.from_numpy(images.astype(np.float32, copy=False))
                if isinstance(images, np.ndarray)
                else images
            )
        return (out,)




class DINKI_Image_Resize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 0, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 0, "max": 8192, "step": 8}),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "keep_proportion": ("BOOLEAN", {"default": False, "label_on": "True (Fit)", "label_off": "False (Stretch)"}),
                "condition": ([
                    "always", 
                    "downscale_if_bigger", 
                    "upscale_if_smaller", 
                    "if_bigger_area", 
                    "if_smaller_area"
                ],),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height")
    FUNCTION = "resize_image"
    CATEGORY = "DINKIssTyle/Image"

    def resize_image(self, width, height, interpolation, keep_proportion, condition, image=None):
        # 1. 이미지가 입력되지 않았을 때 (Bypass 포함) 처리
        if image is None:
            print("[🅳INKIssTyle - Warning]: No image input for DINKI Image Resize. Sending 1x1 Dummy.")
            # 1x1 검은색 텐서 생성 (B, H, W, C)
            empty_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32, device="cpu")
            return (empty_image, 1, 1)

        # 현재 이미지 크기 가져오기
        # ComfyUI Image Shape: [Batch, Height, Width, Channel]
        _, cur_h, cur_w, _ = image.shape

        # 2. 목표 크기 계산 (비율 유지 여부에 따라)
        target_w = width
        target_h = height

        if keep_proportion:
            # 비율을 유지하면서 목표 박스 안에 맞춤 (Fit)
            ratio = min(width / cur_w, height / cur_h)
            target_w = round(cur_w * ratio)
            target_h = round(cur_h * ratio)

        # 3. 리사이즈 조건(Condition) 체크
        should_resize = False

        if condition == "always":
            should_resize = True
            
        elif condition == "downscale_if_bigger":
            # 목표 크기가 현재보다 작을 때만 (축소)
            if target_w < cur_w or target_h < cur_h:
                should_resize = True
                
        elif condition == "upscale_if_smaller":
            # 목표 크기가 현재보다 클 때만 (확대)
            if target_w > cur_w or target_h > cur_h:
                should_resize = True
                
        elif condition == "if_bigger_area":
            # 현재 픽셀 수가 목표 픽셀 수보다 많으면 (총 면적 기준 축소)
            if (cur_w * cur_h) > (target_w * target_h):
                should_resize = True
                
        elif condition == "if_smaller_area":
            # 현재 픽셀 수가 목표 픽셀 수보다 적으면 (총 면적 기준 확대)
            if (cur_w * cur_h) < (target_w * target_h):
                should_resize = True

        # 4. 리사이즈 수행 또는 원본 반환
        if should_resize:
            # 이미지 채널 변경 [B,H,W,C] -> [B,C,H,W] (comfy.utils.common_upscale용)
            samples = image.movedim(-1, 1)
            
            # 리사이즈 실행
            resized_samples = comfy.utils.common_upscale(
                samples, target_w, target_h, interpolation, "disabled"
            )
            
            # 다시 채널 복구 [B,C,H,W] -> [B,H,W,C]
            final_image = resized_samples.movedim(1, -1)
            return (final_image, target_w, target_h)
        else:
            # 조건에 맞지 않으면 원본 그대로 출력
            return (image, cur_w, cur_h)