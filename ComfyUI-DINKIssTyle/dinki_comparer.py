import os
import torch
import numpy as np
from PIL import Image
import imageio
import folder_paths
import random # Temp 파일 충돌 방지용

class DINKI_Image_Comparer_MOV:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory() # Temp 폴더 경로 추가
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                # width, height -> max_width, max_height 로 변경
                "max_width": ("INT", {"default": 1280, "min": 0, "max": 8192, "step": 1}),
                "max_height": ("INT", {"default": 1280, "min": 0, "max": 8192, "step": 1}),
                # resize_method 삭제됨 (항상 원본 비율 유지)
                "resampling": (["lanczos", "bilinear", "bicubic", "nearest"],),
                "sweep_duration": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "pause_duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120}),
                "format": (["mp4", "gif", "webp"],),
                #"format": (["mp4", "gif", "webp", "webm"],),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100}),
                "loops": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                
                # [New] 미리보기 모드 추가
                "preview_mode": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "DINKI_Compare"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "compare_images"
    OUTPUT_NODE = True
    CATEGORY = "DINKIssTyle/Video"

    def get_resized_image(self, tensor_img, target_w, target_h, resampling):
        # Tensor (Batch, H, W, C) -> PIL
        i = 255. * tensor_img.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)[0])

        # 이미 크기가 같다면 바로 리턴
        if img.width == target_w and img.height == target_h:
            return img

        resample_filter = {
            "lanczos": Image.Resampling.LANCZOS,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "nearest": Image.Resampling.NEAREST
        }[resampling]

        # 강제 리사이즈 (비율 계산은 이미 호출부에서 완료됨)
        return img.resize((target_w, target_h), resample=resample_filter)

    def compare_images(self, image_a, image_b, max_width, max_height, resampling, 
                       sweep_duration, pause_duration, fps, format, quality, loops, preview_mode, filename_prefix):
        
        # 1. 크기 계산 로직 (비율 유지)
        # image_a의 원본 크기 가져오기
        orig_h, orig_w = image_a.shape[1], image_a.shape[2]
        
        target_w = orig_w
        target_h = orig_h

        # 제한 설정이 0이 아니고, 이미지가 제한보다 클 경우에만 리사이즈
        if (max_width > 0 and orig_w > max_width) or (max_height > 0 and orig_h > max_height):
            # 가로, 세로 중 더 많이 줄여야 하는 비율을 찾음
            width_ratio = max_width / orig_w if max_width > 0 else 999
            height_ratio = max_height / orig_h if max_height > 0 else 999
            
            scale = min(width_ratio, height_ratio)
            
            target_w = int(orig_w * scale)
            target_h = int(orig_h * scale)

        # 비디오 인코딩(H.264 등)을 위해 가로/세로는 반드시 짝수여야 함
        if target_w % 2 != 0: target_w -= 1
        if target_h % 2 != 0: target_h -= 1

        # 안전장치: 최소 64px
        target_w = max(64, target_w)
        target_h = max(64, target_h)

        # 2. 이미지 준비 (계산된 크기로 리사이즈)
        img1 = self.get_resized_image(image_a, target_w, target_h, resampling)
        img2 = self.get_resized_image(image_b, target_w, target_h, resampling) # B도 A의 비율에 강제 맞춤
        
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        frames = []
        sweep_frames = int(sweep_duration * fps)
        pause_frames = int(pause_duration * fps)
        
        # --- 애니메이션 생성 로직 (이전과 동일) ---
        
        # Phase 1: Hold A
        for _ in range(pause_frames):
            frames.append(arr1)
            
        # Phase 2: Sweep A -> B
        for i in range(sweep_frames):
            progress = (i + 1) / sweep_frames
            split_x = int(target_w * progress)
            split_x = max(0, min(target_w, split_x))
            
            if split_x == 0: frame = arr1
            elif split_x == target_w: frame = arr2
            else:
                frame = np.concatenate((arr2[:, :split_x, :], arr1[:, split_x:, :]), axis=1)
            
            if 0 < split_x < target_w:
                line_thickness = max(2, int(target_w * 0.003))
                start = max(0, split_x - line_thickness // 2)
                end = min(target_w, split_x + line_thickness // 2)
                frame = frame.copy()
                frame[:, start:end, :] = 255
            frames.append(frame)

        # Phase 3: Hold B
        for _ in range(pause_frames):
            frames.append(arr2)
            
        # Phase 4: Sweep B -> A
        for i in range(sweep_frames):
            progress = (i + 1) / sweep_frames
            split_x = target_w - int(target_w * progress)
            split_x = max(0, min(target_w, split_x))
            
            if split_x == 0: frame = arr1
            elif split_x == target_w: frame = arr2
            else:
                frame = np.concatenate((arr2[:, :split_x, :], arr1[:, split_x:, :]), axis=1)
            
            if 0 < split_x < target_w:
                line_thickness = max(2, int(target_w * 0.003))
                start = max(0, split_x - line_thickness // 2)
                end = min(target_w, split_x + line_thickness // 2)
                frame = frame.copy()
                frame[:, start:end, :] = 255
            frames.append(frame)

        # 3. Save Output (수정됨: Preview Mode 지원)
        
        if preview_mode:
            # 미리보기: temp 폴더 사용
            output_dir = self.temp_dir
            type_name = "temp"
            current_prefix = f"temp_{filename_prefix}_{random.randint(1, 100000)}"
        else:
            # 저장: output 폴더 사용
            output_dir = self.output_dir
            type_name = "output"
            current_prefix = filename_prefix

        # 경로 생성
        full_output_folder, filename, counter, subfolder, current_prefix = \
            folder_paths.get_save_image_path(current_prefix, output_dir, image_width=target_w, image_height=target_h)
            
        file_ext = format
        if format == 'animated webp': file_ext = 'webp'
        
        file_name_with_ext = f"{filename}_{counter:05}_.{file_ext}"
        full_path = os.path.join(full_output_folder, file_name_with_ext)

        writer_kwargs = {}
        if format == 'mp4':
            writer_kwargs = {
                'fps': fps, 
                'macro_block_size': None,
                'ffmpeg_params': ['-crf', str(max(0, 51 - (quality // 2)))]
            }
        elif format in ['gif', 'webp']:
            writer_kwargs = {
                'fps': fps, 
                'loop': loops,
                'quality': quality if format == 'webp' else None,
            }
            if format == 'gif': writer_kwargs['quantizer'] = 'nq'

        imageio.mimsave(full_path, frames, format=format, **writer_kwargs)
        
        if not preview_mode:
            print(f"🅳INKIssTyle - Comparer saved to: {full_path}")

        # UI 업데이트 및 파일 경로 리턴 (UI에 보여주기 위해 딕셔너리 구조 사용)
        return {"ui": {"images": [{"filename": file_name_with_ext, "subfolder": subfolder, "type": type_name}]}, 
                "result": (full_path,)}

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("NaN")