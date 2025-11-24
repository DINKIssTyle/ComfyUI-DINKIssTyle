import os
import numpy as np
import torch
import torch.nn.functional as F
import folder_paths

# --- [설정] input/luts 폴더 사용 ---
input_dir = folder_paths.get_input_directory()
luts_dir = os.path.join(input_dir, "luts")

if not os.path.exists(luts_dir):
    try:
        os.makedirs(luts_dir, exist_ok=True)
        print(f"Created directory: {luts_dir}")
    except Exception as e:
        print(f"Failed to create luts directory in input: {e}")

folder_paths.add_model_folder_path("luts", luts_dir)
# -------------------------------

class DINKI_Color_Lut:
    """
    ComfyUI Custom Node for applying 3D LUTs (.cube)
    - Path: ComfyUI/input/luts
    - GPU accelerated using torch.nn.functional.grid_sample
    """
    
    _loaded_luts = {}

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # 1. 파일 목록 가져오기
        file_list = folder_paths.get_filename_list("luts")
        if not file_list:
            file_list = []

        # 2. "-- None --" 옵션을 맨 앞에 추가
        # (사용자가 LUT 적용을 원치 않을 때 선택)
        file_list = ["-- None --"] + file_list

        return {
            "required": {
                "image": ("IMAGE",),
                "lut_name": (file_list,),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_lut"
    CATEGORY = "DINKIssTyle/Color"

    def read_lut_file(self, lut_path):
        """ .cube 파일을 읽어서 PyTorch Tensor로 변환 """
        with open(lut_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        size = -1
        data_lines = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('LUT_3D_SIZE'):
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        size = int(parts[1])
                except ValueError:
                    pass
                continue
            
            if not (line[0].isdigit() or line[0] == '-'):
                continue

            data_lines.append(line)

        if size == -1:
            raise ValueError(f"LUT_3D_SIZE not found in {lut_path}")

        try:
            full_text = " ".join(data_lines)
            lut_data = np.fromstring(full_text, sep=' ', dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to parse LUT data: {e}")

        expected_count = size * size * size * 3
        
        if len(lut_data) != expected_count:
            if len(lut_data) > expected_count:
                lut_data = lut_data[:expected_count]
            else:
                raise ValueError(f"Insufficient data in LUT file. Expected {expected_count}, got {len(lut_data)}")

        # Reshape & Transpose for grid_sample
        lut_np = lut_data.reshape(size, size, size, 3)
        lut_tensor = torch.from_numpy(lut_np).permute(3, 0, 1, 2).unsqueeze(0)
        
        return lut_tensor

    def get_lut_tensor(self, lut_name):
        lut_path = folder_paths.get_full_path("luts", lut_name)
        
        if lut_path is None:
            print(f"[Error] LUT file not found: {lut_name}")
            return None

        if lut_path in self._loaded_luts:
            return self._loaded_luts[lut_path]

        try:
            lut_tensor = self.read_lut_file(lut_path)
            self._loaded_luts[lut_path] = lut_tensor
            return lut_tensor
        except Exception as e:
            print(f"[Error] Failed to load LUT {lut_name}: {e}")
            return None

    def apply_lut(self, image, lut_name, strength):
        # 3. None 체크: 선택된 값이 None이면 원본 그대로 반환
        if not lut_name or lut_name == "-- None --":
            return (image,)

        lut_tensor = self.get_lut_tensor(lut_name)
        if lut_tensor is None:
            return (image,)

        device = image.device
        if lut_tensor.device != device:
            lut_tensor = lut_tensor.to(device)

        # grid shape: [B, 1, H, W, 3]
        grid = image.unsqueeze(1) 
        grid = grid * 2.0 - 1.0
        
        # grid_sample (Trilinear Interpolation)
        processed = F.grid_sample(lut_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        # 차원 복원: [B, 3, 1, H, W] -> [B, H, W, 3]
        processed = processed.permute(0, 2, 3, 4, 1).squeeze(1)

        # Strength 적용
        strength = float(strength)
        if strength < 1.0:
            result = torch.lerp(image, processed, strength)
        else:
            result = processed

        return (result,)

NODE_CLASS_MAPPINGS = {
    "DINKI_Color_Lut": DINKI_Color_Lut
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_Color_Lut": "DINKI Color LUT"
}