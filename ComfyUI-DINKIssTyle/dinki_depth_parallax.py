import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import folder_paths
import os
import math
import imageio
import random

# --------------------------------------------------------------------------------
# Node 4: DINKI Depth Parallax MOV (All-in-One: Wiggle & SBS & Preview)
# --------------------------------------------------------------------------------
class DINKI_DepthParallax_MOV:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),      # 원본 이미지
                "depth_map": ("IMAGE",),  # 뎁스 맵
                
                # --- 3D 효과 파라미터 ---
                "amount": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.005}), 
                "frames": ("INT", {"default": 24, "min": 2, "max": 120}), 
                "mode": (["horizontal", "vertical", "circle", "figure8", "sbs_parallel", "sbs_cross"],), 
                "phase": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0}),
                "focus_depth": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "normalize_depth": ("BOOLEAN", {"default": True}),

                # --- 저장 및 포맷 파라미터 ---
                "fps": ("INT", {"default": 15, "min": 1, "max": 60}),
                "format": (["webp", "gif", "mp4", "png", "jpg"],),
                "quality": ("INT", {"default": 90, "min": 1, "max": 100}),
                
                # [New] 미리보기 모드: 켜면 temp 폴더에 저장 (디스크 절약)
                "preview_mode": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "DINKI_3D"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "generate_and_save"
    OUTPUT_NODE = True
    CATEGORY = "DINKIssTyle/Video"

    def generate_and_save(self, image, depth_map, amount, frames, mode, phase, focus_depth, normalize_depth,
                          fps, format, quality, preview_mode, filename_prefix):
        
        # 1. 이미지 및 뎁스맵 전처리
        img_tensor = image[0].permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
        B, C, H, W = img_tensor.shape
        device = img_tensor.device

        depth_tensor = depth_map[0].permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
        
        if depth_tensor.shape[2:] != img_tensor.shape[2:]:
            depth_tensor = F.interpolate(depth_tensor, size=(H, W), mode="bilinear", align_corners=False)
        
        depth_tensor = torch.mean(depth_tensor, dim=1, keepdim=True) # (1, 1, H, W)

        if normalize_depth:
            d_min = torch.min(depth_tensor)
            d_max = torch.max(depth_tensor)
            if d_max - d_min > 1e-5:
                depth_tensor = (depth_tensor - d_min) / (d_max - d_min)

        # 2. Grid 생성
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0) 

        generated_frames = []

        # --------------------------------------------------------------------------------
        # 로직 분기: SBS (2-Frame Wiggle) vs 일반 애니메이션
        # --------------------------------------------------------------------------------
        if "sbs" in mode:
            d_val = depth_tensor.permute(0, 2, 3, 1)
            disparity = (d_val - focus_depth)

            # 왼쪽 눈
            new_grid_l = base_grid.clone()
            new_grid_l[..., 0] += disparity[..., 0] * amount 
            warped_l = F.grid_sample(img_tensor, new_grid_l, mode='bicubic', padding_mode='reflection', align_corners=False)
            img_l = np.clip(255. * warped_l.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 255).astype(np.uint8)

            # 오른쪽 눈
            new_grid_r = base_grid.clone()
            new_grid_r[..., 0] -= disparity[..., 0] * amount
            warped_r = F.grid_sample(img_tensor, new_grid_r, mode='bicubic', padding_mode='reflection', align_corners=False)
            img_r = np.clip(255. * warped_r.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 255).astype(np.uint8)

            # SBS 모드는 2프레임 생성
            if mode == "sbs_parallel":
                generated_frames.append(img_l)
                generated_frames.append(img_r)
            else: # sbs_cross
                generated_frames.append(img_r)
                generated_frames.append(img_l)

        else:
            # 일반 Wiggle
            for i in range(frames):
                t = (i / frames) * 2 * math.pi * phase
                offset_x = 0
                offset_y = 0
                
                if mode == "horizontal":
                    offset_x = -math.sin(t) * amount 
                elif mode == "vertical":
                    offset_y = -math.cos(t) * amount
                elif mode == "circle":
                    offset_x = -math.sin(t) * amount
                    offset_y = -math.cos(t) * amount
                elif mode == "figure8":
                    offset_x = -math.sin(t) * amount
                    offset_y = -math.sin(t * 2) * (amount * 0.5)

                d_val = depth_tensor.permute(0, 2, 3, 1)
                disparity = (d_val - focus_depth)

                new_grid = base_grid.clone()
                new_grid[..., 0] += disparity[..., 0] * offset_x
                new_grid[..., 1] += disparity[..., 0] * offset_y

                warped = F.grid_sample(img_tensor, new_grid, mode='bicubic', padding_mode='reflection', align_corners=False)
                i_np = 255. * warped.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img_pil = np.clip(i_np, 0, 255).astype(np.uint8)
                generated_frames.append(img_pil)

        # --------------------------------------------------------------------------------
        # 4. 저장 위치 결정 (Preview Mode 로직 추가)
        # --------------------------------------------------------------------------------
        save_h, save_w, _ = generated_frames[0].shape
        
        if preview_mode:
            # 미리보기: temp 폴더 사용
            output_dir = self.temp_dir
            type_name = "temp"
            # 파일명 충돌 방지를 위해 랜덤 숫자 추가 또는 별도 prefix
            current_prefix = f"temp_{filename_prefix}_{random.randint(1, 100000)}"
        else:
            # 저장: output 폴더 사용
            output_dir = self.output_dir
            type_name = "output"
            current_prefix = filename_prefix

        # 경로 생성
        full_output_folder, filename, counter, subfolder, current_prefix = \
            folder_paths.get_save_image_path(current_prefix, output_dir, save_w, save_h)
        
        file_ext = format
        file_name_with_ext = f"{filename}_{counter:05}_.{file_ext}"
        full_path = os.path.join(full_output_folder, file_name_with_ext)

        # 파일 저장 실행
        try:
            if file_ext in ['png', 'jpg', 'jpeg']:
                imageio.imwrite(full_path, generated_frames[0], quality=quality if file_ext != 'png' else None)
            
            elif file_ext == 'mp4':
                writer_kwargs = {
                    'fps': fps, 
                    'macro_block_size': None, 
                    'ffmpeg_params': ['-crf', str(max(0, 51 - (quality // 2)))]
                }
                imageio.mimsave(full_path, generated_frames, format=format, **writer_kwargs)
            
            elif file_ext in ['gif', 'webp']:
                writer_kwargs = {
                    'fps': fps, 
                    'loop': 0, 
                    'quality': quality if file_ext == 'webp' else None,
                }
                if file_ext == 'gif': writer_kwargs['quantizer'] = 'nq'
                
                imageio.mimsave(full_path, generated_frames, format=format, **writer_kwargs)

            if not preview_mode:
                print(f"DINKI Result saved to: {full_path}")
            
        except Exception as e:
            print(f"Error saving result: {e}")
            return ("",)

        # 5. UI 업데이트 (type에 따라 temp/output 구분됨)
        return {"ui": {"images": [{"filename": file_name_with_ext, "subfolder": subfolder, "type": type_name}]}, 
                "result": (full_path,)}
