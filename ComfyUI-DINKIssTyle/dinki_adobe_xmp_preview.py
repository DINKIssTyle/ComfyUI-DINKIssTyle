import os
import io
import torch
import numpy as np
import folder_paths
import xml.etree.ElementTree as ET
from PIL import Image
from server import PromptServer
from aiohttp import web

# --- [설정] input/adobe_xmp 폴더 사용 ---
input_dir = folder_paths.get_input_directory()
xmp_dir = os.path.join(input_dir, "adobe_xmp")

if not os.path.exists(xmp_dir):
    try:
        os.makedirs(xmp_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create adobe_xmp directory: {e}")

folder_paths.add_model_folder_path("adobe_xmp", xmp_dir)
# -------------------------------

# --- Helper Functions (v2.1과 동일) ---

def _rgb_to_hsv_torch(img):
    """
    [Improved] RGB to HSV with better numerical stability
    """
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    max_val, _ = torch.max(img, dim=-1)
    min_val, _ = torch.min(img, dim=-1)
    
    # Epsilon을 1e-6에서 1e-5로 늘려 안정성 확보
    diff = max_val - min_val + 1e-5

    # Hue calculation
    h = torch.zeros_like(max_val)
    
    # 등호 비교(==)의 불안정성을 해소하기 위해 우선순위 마스킹 적용
    mask_r = (max_val == r)
    mask_g = (max_val == g) & (~mask_r)
    mask_b = (max_val == b) & (~mask_r) & (~mask_g)

    # 마스크된 영역만 계산
    h[mask_r] = (g[mask_r] - b[mask_r]) / diff[mask_r] % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4
    h = h / 6.0

    # Saturation
    s = torch.zeros_like(max_val)
    mask_nz = (max_val > 0)
    s[mask_nz] = diff[mask_nz] / max_val[mask_nz]

    v = max_val
    return h, s, v

def _hsv_to_rgb_torch(h, s, v):
    h = h * 6.0; c = v * s; x = c * (1 - torch.abs((h % 2) - 1)); m = v - c
    z = torch.zeros_like(h)
    out = torch.stack([
        torch.where((h < 1), c, torch.where((h < 2), x, torch.where((h < 3), z, torch.where((h < 4), z, torch.where((h < 5), x, c))))),
        torch.where((h < 1), x, torch.where((h < 2), c, torch.where((h < 3), c, torch.where((h < 4), x, torch.where((h < 5), z, z))))),
        torch.where((h < 1), z, torch.where((h < 2), z, torch.where((h < 3), x, torch.where((h < 4), c, torch.where((h < 5), c, x)))))
    ], dim=-1)
    return out + m.unsqueeze(-1)

def _calculate_pchip_lut(points, num_entries=256):
    points = sorted(points, key=lambda p: p[0])
    x = np.array([p[0] for p in points], dtype=np.float32)
    y = np.array([p[1] for p in points], dtype=np.float32)
    if len(points) < 2: return np.linspace(0, 1, num_entries, dtype=np.float32)
    dx = x[1:] - x[:-1]; dy = y[1:] - y[:-1]; dx[dx == 0] = 1e-6; m = dy / dx
    t = np.zeros_like(x); t[0] = m[0]; t[-1] = m[-1]
    if len(x) > 2:
        mask = np.sign(m[:-1]) * np.sign(m[1:]) > 0
        w1 = 2 * dx[1:] + dx[:-1]; w2 = dx[1:] + 2 * dx[:-1]
        t_middle = np.zeros_like(m[:-1])
        t_middle[mask] = (w1[mask] + w2[mask]) / ((w1[mask] / m[:-1][mask]) + (w2[mask] / m[1:][mask]))
        t[1:-1] = t_middle
    xi = np.linspace(0, 255, num_entries, dtype=np.float32)
    indices = np.searchsorted(x, xi) - 1; indices = np.clip(indices, 0, len(x) - 2)
    h = xi - x[indices]; d = x[indices+1] - x[indices]; norm_t = h / d
    t2 = norm_t * norm_t; t3 = t2 * norm_t
    h00 = 2 * t3 - 3 * t2 + 1; h10 = t3 - 2 * t2 + norm_t
    h01 = -2 * t3 + 3 * t2; h11 = t3 - t2
    yi = (h00 * y[indices] + h10 * d * t[indices] + h01 * y[indices+1] + h11 * d * t[indices+1])
    return np.clip(yi / 255.0, 0.0, 1.0).astype(np.float32)

def _apply_lut_torch(img_channel, lut_tensor):
    scaled = img_channel * 255.0
    idx_floor = torch.floor(scaled).long().clamp(0, 255)
    idx_ceil = torch.ceil(scaled).long().clamp(0, 255)
    weight = scaled - idx_floor.float()
    val_floor = lut_tensor[idx_floor]; val_ceil = lut_tensor[idx_ceil]
    return torch.lerp(val_floor, val_ceil, weight)
# ------------------------------------


class DINKI_Adobe_XMP_Preview:
    """
    ComfyUI Custom Node: DINKI Adobe XMP Preview
    - Interactive preview via API, caches last input image.
    - Includes: Exposure, Contrast, HSL, Tone Curve, Vignette, Grain
    """
    
    # [핵심] 마지막 입력 이미지 캐싱용
    last_input_tensor = None

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        file_list = folder_paths.get_filename_list("adobe_xmp")
        if not file_list: file_list = []
        file_list = ["-- None --"] + file_list

        return {
            "required": {
                "image": ("IMAGE",),
                "xmp_file": (file_list,),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_preset"
    CATEGORY = "DINKIssTyle/Color"

    # --- Parsing & Logic (v2.1과 동일) ---
    def parse_xmp(self, file_path):
        params = {
            "Exposure": 0.0, "Contrast": 0, "Vibrance": 0, "Saturation": 0,
            "VignetteAmount": 0, "VignetteMidpoint": 50, "VignetteFeather": 50,
            "GrainAmount": 0, "GrainSize": 25,
            "ToneCurve": None, "ToneCurveRed": None, "ToneCurveGreen": None, "ToneCurveBlue": None,
            "HSL_Hue": {}, "HSL_Sat": {}, "HSL_Lum": {}
        }
        color_names = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
        def parse_seq(text_seq):
            try:
                vals = text_seq.replace('\n', '').split(',')
                vals = [float(x.strip()) for x in vals if x.strip()]
                points = []
                for i in range(0, len(vals), 2): points.append((vals[i], vals[i+1]))
                return points
            except: return None
        try:
            tree = ET.parse(file_path); root = tree.getroot()
            descriptions = root.findall(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
            for desc in descriptions:
                for key, value in desc.attrib.items():
                    tag = key.split('}')[-1] if '}' in key else key
                    if tag == "Exposure2012": params["Exposure"] = float(value)
                    elif tag == "Contrast2012": params["Contrast"] = float(value)
                    elif tag == "Vibrance": params["Vibrance"] = float(value)
                    elif tag == "Saturation": params["Saturation"] = float(value)
                    elif tag == "PostCropVignetteAmount": params["VignetteAmount"] = float(value)
                    elif tag == "PostCropVignetteMidpoint": params["VignetteMidpoint"] = float(value)
                    elif tag == "PostCropVignetteFeather": params["VignetteFeather"] = float(value)
                    elif tag == "GrainAmount": params["GrainAmount"] = float(value)
                    elif tag == "GrainSize": params["GrainSize"] = float(value)
                    for c_name in color_names:
                        if tag == f"HueAdjustment{c_name}": params["HSL_Hue"][c_name] = float(value)
                        elif tag == f"SaturationAdjustment{c_name}": params["HSL_Sat"][c_name] = float(value)
                        elif tag == f"LuminanceAdjustment{c_name}": params["HSL_Lum"][c_name] = float(value)
                for child in desc:
                    tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if tag in ["ToneCurve", "ToneCurvePV2012"]:
                        seq = child.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Seq")
                        if seq: params["ToneCurve"] = parse_seq(", ".join([li.text for li in seq.findall(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li")]))
                    for color in ["Red", "Green", "Blue"]:
                        if tag in [f"ToneCurve{color}", f"ToneCurvePV2012{color}"]:
                            seq = child.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Seq")
                            if seq: params[f"ToneCurve{color}"] = parse_seq(", ".join([li.text for li in seq.findall(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li")]))
        except Exception as e: print(f"[Warning] Failed to parse XMP {file_path}: {e}")
        return params

    def apply_hsl(self, img, p, device):
        """ 
        Applies HSL adjustments with Low-Saturation Protection (Fix for highlight artifacts)
        """
        # HSL 설정이 없으면 패스
        has_hsl = (len(p["HSL_Hue"]) + len(p["HSL_Sat"]) + len(p["HSL_Lum"])) > 0
        if not has_hsl:
            return img

        # 1. RGB -> HSV
        h, s, v = _rgb_to_hsv_torch(img) # [B,H,W] 0..1

        # 2. Define Color Centers
        color_centers = {
            "Red": 0.0, "Orange": 35/360.0, "Yellow": 60/360.0, "Green": 120/360.0,
            "Aqua": 180/360.0, "Blue": 240/360.0, "Purple": 275/360.0, "Magenta": 315/360.0
        }
        color_names = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
        width = 45/360.0 

        total_hue_shift = torch.zeros_like(h)
        total_sat_scale = torch.zeros_like(s)
        total_val_scale = torch.zeros_like(v)

        for c_name in color_names:
            center = color_centers[c_name]
            
            h_adj = p["HSL_Hue"].get(c_name, 0.0) / 100.0 * (30/360.0)
            s_adj = p["HSL_Sat"].get(c_name, 0.0) / 100.0
            l_adj = p["HSL_Lum"].get(c_name, 0.0) / 100.0

            if h_adj == 0 and s_adj == 0 and l_adj == 0:
                continue

            # Calculate Mask (Circular Distance)
            diff = torch.abs(h - center)
            dist = torch.min(diff, 1.0 - diff)
            
            # Weight: 해당 색상 영역인지 확인
            weight = torch.clamp(1.0 - (dist / width), 0.0, 1.0)
            
            total_hue_shift += weight * h_adj
            total_sat_scale += weight * s_adj
            total_val_scale += weight * l_adj

        # [핵심 수정] Saturation Protection Mask
        # 채도(s)가 낮으면(흰색/회색) HSL 변형을 적용하지 않음 (0.05 이하는 무시)
        # 이렇게 하면 하늘 같은 하이라이트 영역에서 Hue가 튀는 현상이 사라짐
        protection_mask = torch.clamp((s - 0.02) * 20.0, 0.0, 1.0) # Smooth transition

        # 3. Apply Adjustments (weighted by protection_mask)
        # Hue Shift가 가장 큰 문제이므로 강력하게 마스킹
        h_new = (h + total_hue_shift * protection_mask) % 1.0
        
        # Saturation/Val은 덜 민감하지만 안전하게 같이 마스킹
        s_new = torch.clamp(s * (1.0 + total_sat_scale * protection_mask), 0.0, 1.0)
        v_new = torch.clamp(v * (1.0 + total_val_scale * protection_mask), 0.0, 1.0)

        # 4. Convert back to RGB
        return _hsv_to_rgb_torch(h_new, s_new, v_new)

    # --- Main Execution ---
    def apply_preset(self, image, xmp_file, strength):
        # [핵심] 이미지 캐싱 (CPU로 이동하여 저장)
        DINKI_Adobe_XMP_Preview.last_input_tensor = image[0:1].clone().cpu()

        if not xmp_file or xmp_file == "-- None --": return (image,)
        xmp_path = folder_paths.get_full_path("adobe_xmp", xmp_file)
        if not xmp_path: return (image,)

        p = self.parse_xmp(xmp_path)
        device = image.device
        out = image.clone()

        # 1. Exposure & Contrast
        if p["Exposure"] != 0: out = out * torch.pow(2.0, torch.tensor(p["Exposure"], device=device))
        if p["Contrast"] != 0:
            c_val = p["Contrast"] / 100.0
            scale = 1.0 + c_val if c_val > 0 else 1.0 / (1.0 - c_val)
            out = (out - 0.5) * scale + 0.5
            out = torch.clamp(out, 0.0, 1.0)

        # 2. Tone Curve
        if p["ToneCurve"]:
            lut_t = torch.from_numpy(_calculate_pchip_lut(p["ToneCurve"])).to(device)
            for c in range(3): out[..., c] = _apply_lut_torch(out[..., c], lut_t)
        for i, key in enumerate(["ToneCurveRed", "ToneCurveGreen", "ToneCurveBlue"]):
            if p[key]:
                lut_t = torch.from_numpy(_calculate_pchip_lut(p[key])).to(device)
                out[..., i] = _apply_lut_torch(out[..., i], lut_t)
        out = torch.clamp(out, 0.0, 1.0)

        # 3. HSL
        out = self.apply_hsl(out, p, device)

        # 4. Vibrance & Saturation
        if p["Saturation"] != 0 or p["Vibrance"] != 0:
            luma = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
            luma = luma.unsqueeze(-1)
            max_ch, _ = torch.max(out, dim=-1, keepdim=True); min_ch, _ = torch.min(out, dim=-1, keepdim=True)
            curr_sat = max_ch - min_ch
            sat_mul = 1.0 + (p["Saturation"] / 100.0)
            vib_val = p["Vibrance"] / 100.0
            vib_mul = 1.0 + (vib_val * (1.0 - curr_sat)) if vib_val >= 0 else 1.0 + vib_val
            out = luma + (out - luma) * (sat_mul * vib_mul)
            out = torch.clamp(out, 0.0, 1.0)

        # 5. Vignette
        if p["VignetteAmount"] != 0:
            B, H, W, C = out.shape
            y = torch.linspace(-1, 1, H, device=device); x = torch.linspace(-1, 1, W, device=device)
            mesh_y, mesh_x = torch.meshgrid(y, x, indexing='ij')
            dist = torch.sqrt(mesh_x**2 + mesh_y**2)
            midpoint = p["VignetteMidpoint"] / 100.0
            dist_norm = torch.clamp((dist - midpoint) / (1.5 - midpoint + 1e-6), 0.0, 1.0)
            feather = p["VignetteFeather"] / 100.0
            if feather > 0: dist_norm = torch.pow(dist_norm, 1.0 / (feather + 0.1))
            amount = p["VignetteAmount"] / 100.0
            vignette_factor = 1.0 + (amount * dist_norm)
            out = out * vignette_factor.unsqueeze(0).unsqueeze(-1)
            out = torch.clamp(out, 0.0, 1.0)

        # 6. Grain
        if p["GrainAmount"] > 0:
            amount = p["GrainAmount"] / 100.0; size = max(p["GrainSize"] / 100.0, 0.01)
            noise = torch.randn_like(out)
            if size > 0.3:
                down_factor = 1.0 / (1.0 + size * 2.0)
                dH, dW = int(out.shape[1] * down_factor), int(out.shape[2] * down_factor)
                small_noise = torch.randn((out.shape[0], 3, dH, dW), device=device)
                noise = torch.nn.functional.interpolate(small_noise, size=(out.shape[1], out.shape[2]), mode='bilinear').permute(0, 2, 3, 1)
            out = out + (noise * (amount * 0.15))
            out = torch.clamp(out, 0.0, 1.0)

        if strength < 1.0: out = torch.lerp(image, out, strength)
        return (out,)

    # --- [핵심] API 프리뷰 처리 로직 ---
    @staticmethod
    def process_preview(xmp_file, strength):
        if DINKI_Adobe_XMP_Preview.last_input_tensor is None: return None
        img = DINKI_Adobe_XMP_Preview.last_input_tensor
        node = DINKI_Adobe_XMP_Preview()
        
        # 캐시된 이미지로 apply_preset 실행
        result_tuple = node.apply_preset(img, xmp_file, strength)
        result_tensor = result_tuple[0]

        # Tensor -> PNG 변환
        result_np = np.clip(255. * result_tensor.squeeze(0).numpy(), 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(result_np)
        
        # 리사이즈 (속도 최적화)
        max_size = 1024
        if pil_img.width > max_size:
            ratio = max_size / pil_img.width
            new_height = int(pil_img.height * ratio)
            pil_img = pil_img.resize((max_size, new_height), Image.BILINEAR)

        buff = io.BytesIO()
        pil_img.save(buff, format="PNG", compress_level=4)
        return buff.getvalue()


# --- API Route 등록 ---
@PromptServer.instance.routes.post("/dinki/preview_xmp")
async def preview_xmp_route(request):
    data = await request.json()
    xmp_file = data.get("xmp_file")
    strength = data.get("strength", 1.0)
    
    if DINKI_Adobe_XMP_Preview.last_input_tensor is None:
        return web.Response(status=400, text="No cached image found. Please run the workflow once.")

    img_bytes = DINKI_Adobe_XMP_Preview.process_preview(xmp_file, strength)
    
    if img_bytes:
        return web.Response(body=img_bytes, content_type='image/png')
    else:
        return web.Response(status=500, text="Processing failed")


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "DINKI_Adobe_XMP_Preview": DINKI_Adobe_XMP_Preview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_Adobe_XMP_Preview": "DINKI Adobe XMP Preview"
}