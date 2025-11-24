import os
import io
import numpy as np
import torch
import torch.nn.functional as F
import folder_paths
from PIL import Image
from server import PromptServer
from aiohttp import web

# --- [설정] input/luts 폴더 사용 ---
input_dir = folder_paths.get_input_directory()
luts_dir = os.path.join(input_dir, "luts")

if not os.path.exists(luts_dir):
    try:
        os.makedirs(luts_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create luts directory in input: {e}")

folder_paths.add_model_folder_path("luts", luts_dir)
# -------------------------------

class DINKI_Color_Lut_Preview:
    """
    ComfyUI Custom Node: DINKI Color LUT Preview
    - Supports Interactive Preview via API
    - Caches the last input image to allow LUT switching without re-running workflow
    """
    
    _loaded_luts = {}
    # API 프리뷰를 위해 마지막 입력 이미지를 저장하는 정적 변수
    last_input_tensor = None 

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        file_list = folder_paths.get_filename_list("luts")
        if not file_list:
            file_list = []
        
        # LUT 선택 안 함 옵션 추가
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
        with open(lut_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        size = -1
        data_lines = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'): continue
            
            if line.startswith('LUT_3D_SIZE'):
                try:
                    parts = line.split()
                    if len(parts) >= 2: size = int(parts[1])
                except: pass
                continue
            
            if not (line[0].isdigit() or line[0] == '-'): continue
            data_lines.append(line)

        if size == -1: raise ValueError(f"LUT_3D_SIZE error")

        full_text = " ".join(data_lines)
        lut_data = np.fromstring(full_text, sep=' ', dtype=np.float32)

        expected = size * size * size * 3
        if len(lut_data) > expected:
            lut_data = lut_data[:expected]
        
        lut_np = lut_data.reshape(size, size, size, 3)
        return torch.from_numpy(lut_np).permute(3, 0, 1, 2).unsqueeze(0)

    def get_lut_tensor(self, lut_name):
        lut_path = folder_paths.get_full_path("luts", lut_name)
        if not lut_path: return None
        if lut_path in self._loaded_luts: return self._loaded_luts[lut_path]

        try:
            t = self.read_lut_file(lut_path)
            self._loaded_luts[lut_path] = t
            return t
        except Exception as e:
            print(f"[Error] Failed to load LUT {lut_name}: {e}")
            return None

    def apply_lut(self, image, lut_name, strength):
        # [핵심] 들어온 이미지 캐싱 (배치 중 첫장만, CPU로 이동하여 VRAM 절약)
        DINKI_Color_Lut_Preview.last_input_tensor = image[0:1].clone().cpu()

        if not lut_name or lut_name == "-- None --":
            return (image,)

        lut_tensor = self.get_lut_tensor(lut_name)
        if lut_tensor is None:
            return (image,)

        device = image.device
        if lut_tensor.device != device:
            lut_tensor = lut_tensor.to(device)

        # Grid Sample
        grid = image.unsqueeze(1) * 2.0 - 1.0
        processed = F.grid_sample(lut_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
        processed = processed.permute(0, 2, 3, 4, 1).squeeze(1)

        # Strength Lerp
        strength = float(strength)
        if strength < 1.0:
            result = torch.lerp(image, processed, strength)
        else:
            result = processed

        return (result,)

    # --- API용 프리뷰 처리 로직 ---
    @staticmethod
    def process_preview(lut_name, strength):
        if DINKI_Color_Lut_Preview.last_input_tensor is None:
            return None

        img = DINKI_Color_Lut_Preview.last_input_tensor
        node = DINKI_Color_Lut_Preview()
        
        # apply_lut 재사용
        result_tuple = node.apply_lut(img, lut_name, strength)
        result_tensor = result_tuple[0]

        # Tensor -> Numpy -> PIL
        # 0~1 범위를 0~255로 변환
        result_np = np.clip(255. * result_tensor.squeeze(0).numpy(), 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(result_np)
        
        # 리사이즈 (가로 1024px 유지)
        max_size = 2048
        if pil_img.width > max_size:
            ratio = max_size / pil_img.width
            new_height = int(pil_img.height * ratio)
            pil_img = pil_img.resize((max_size, new_height), Image.BILINEAR)

        buff = io.BytesIO()
        
        # --- [수정됨] JPEG -> PNG ---
        # PNG는 quality 파라미터가 필요 없습니다.
        # compress_level=4 정도로 설정하면 속도와 압축률 균형이 좋습니다 (기본값은 6)
        pil_img.save(buff, format="PNG", compress_level=4) 
        
        return buff.getvalue()


# --- API Route 등록 ---
@PromptServer.instance.routes.post("/dinki/preview_lut")
async def preview_lut_route(request):
    data = await request.json()
    lut_name = data.get("lut_name")
    strength = data.get("strength", 1.0)
    
    if DINKI_Color_Lut_Preview.last_input_tensor is None:
        return web.Response(status=400, text="No cached image found. Please run the workflow once.")

    img_bytes = DINKI_Color_Lut_Preview.process_preview(lut_name, strength)
    
    if img_bytes:
        # --- [수정됨] image/jpeg -> image/png ---
        return web.Response(body=img_bytes, content_type='image/png')
    else:
        return web.Response(status=500, text="Processing failed")


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "DINKI_Color_Lut_Preview": DINKI_Color_Lut_Preview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_Color_Lut_Preview": "DINKI Color LUT Preview"
}