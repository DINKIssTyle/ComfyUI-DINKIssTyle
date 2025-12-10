import os
import io
import math
import numpy as np
import torch
import torch.nn.functional as F
import xml.etree.ElementTree as ET
import folder_paths
from PIL import Image
from server import PromptServer
from aiohttp import web

# ============================================================================
# [Folder Setup] Input Directories
# ============================================================================
input_dir = folder_paths.get_input_directory()

# 1. Adobe XMP
xmp_dir = os.path.join(input_dir, "adobe_xmp")
if not os.path.exists(xmp_dir):
    try:
        os.makedirs(xmp_dir, exist_ok=True)
    except Exception as e:
        print(f"[🅳INKIssTyle - Failed] to create adobe_xmp directory: {e}")
folder_paths.add_model_folder_path("adobe_xmp", xmp_dir)

# 2. LUTs
luts_dir = os.path.join(input_dir, "luts")
if not os.path.exists(luts_dir):
    try:
        os.makedirs(luts_dir, exist_ok=True)
    except Exception as e:
        print(f"[🅳INKIssTyle - Failed] to create luts directory: {e}")
folder_paths.add_model_folder_path("luts", luts_dir)


# ============================================================================
# [Shared Helpers] Torch Color Conversions & Curves (for XMP)
# ============================================================================

def _rgb_to_hsv_torch(img):
    """ RGB to HSV with numerical stability (Torch) """
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    max_val, _ = torch.max(img, dim=-1)
    min_val, _ = torch.min(img, dim=-1)
    
    diff = max_val - min_val + 1e-5

    # Hue calculation
    h = torch.zeros_like(max_val)
    mask_r = (max_val == r)
    mask_g = (max_val == g) & (~mask_r)
    mask_b = (max_val == b) & (~mask_r) & (~mask_g)

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
    """ HSV to RGB (Torch) """
    h = h * 6.0; c = v * s; x = c * (1 - torch.abs((h % 2) - 1)); m = v - c
    z = torch.zeros_like(h)
    out = torch.stack([
        torch.where((h < 1), c, torch.where((h < 2), x, torch.where((h < 3), z, torch.where((h < 4), z, torch.where((h < 5), x, c))))),
        torch.where((h < 1), x, torch.where((h < 2), c, torch.where((h < 3), c, torch.where((h < 4), x, torch.where((h < 5), z, z))))),
        torch.where((h < 1), z, torch.where((h < 2), z, torch.where((h < 3), x, torch.where((h < 4), c, torch.where((h < 5), c, x)))))
    ], dim=-1)
    return out + m.unsqueeze(-1)

def _calculate_pchip_lut(points, num_entries=256):
    """ NumPy PCHIP implementation for Tone Curve """
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


# ============================================================================
# [Class 1] DINKI Adobe XMP (Base)
# ============================================================================

class DINKI_adobe_xmp:
    def __init__(self): pass

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

    def parse_xmp(self, file_path):
        params = {
            "Exposure": 0.0, "Contrast": 0, "Vibrance": 0, "Saturation": 0,
            "VignetteAmount": 0, "VignetteMidpoint": 50, "VignetteFeather": 50, "VignetteRoundness": 0,
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
        except Exception as e: print(f"[🅳INKIssTyle - Warning] Failed to parse XMP {file_path}: {e}")
        return params

    def apply_hsl(self, img, p, device):
        has_hsl = (len(p["HSL_Hue"]) + len(p["HSL_Sat"]) + len(p["HSL_Lum"])) > 0
        if not has_hsl: return img

        h, s, v = _rgb_to_hsv_torch(img)
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
            h_adj = p["HSL_Hue"].get(c_name, 0.0) / 100.0 * (30/360.0)
            s_adj = p["HSL_Sat"].get(c_name, 0.0) / 100.0
            l_adj = p["HSL_Lum"].get(c_name, 0.0) / 100.0
            if h_adj == 0 and s_adj == 0 and l_adj == 0: continue
            center = color_centers[c_name]
            diff = torch.abs(h - center)
            dist = torch.min(diff, 1.0 - diff)
            weight = torch.clamp(1.0 - (dist / width), 0.0, 1.0)
            total_hue_shift += weight * h_adj
            total_sat_scale += weight * s_adj
            total_val_scale += weight * l_adj

        protection_mask = torch.clamp((s - 0.02) * 20.0, 0.0, 1.0)
        h_new = (h + total_hue_shift * protection_mask) % 1.0
        s_new = torch.clamp(s * (1.0 + total_sat_scale * protection_mask), 0.0, 1.0)
        v_new = torch.clamp(v * (1.0 + total_val_scale * protection_mask), 0.0, 1.0)
        return _hsv_to_rgb_torch(h_new, s_new, v_new)

    def apply_preset(self, image, xmp_file, strength):
        if not xmp_file or xmp_file == "-- None --": return (image,)
        xmp_path = folder_paths.get_full_path("adobe_xmp", xmp_file)
        if not xmp_path: return (image,)

        p = self.parse_xmp(xmp_path)
        device = image.device
        out = image.clone()

        if p["Exposure"] != 0: out = out * torch.pow(2.0, torch.tensor(p["Exposure"], device=device))
        if p["Contrast"] != 0:
            c_val = p["Contrast"] / 100.0
            scale = 1.0 + c_val if c_val > 0 else 1.0 / (1.0 - c_val)
            out = (out - 0.5) * scale + 0.5
            out = torch.clamp(out, 0.0, 1.0)

        if p["ToneCurve"]:
            lut = _calculate_pchip_lut(p["ToneCurve"])
            lut_t = torch.from_numpy(lut).to(device)
            for c in range(3): out[..., c] = _apply_lut_torch(out[..., c], lut_t)
        for i, key in enumerate(["ToneCurveRed", "ToneCurveGreen", "ToneCurveBlue"]):
            if p[key]:
                lut = _calculate_pchip_lut(p[key])
                lut_t = torch.from_numpy(lut).to(device)
                out[..., i] = _apply_lut_torch(out[..., i], lut_t)
        out = torch.clamp(out, 0.0, 1.0)

        out = self.apply_hsl(out, p, device)

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


# ============================================================================
# [Class 2] DINKI Adobe XMP Preview (Interactive + API)
# ============================================================================

class DINKI_Adobe_XMP_Preview(DINKI_adobe_xmp):
    last_input_tensor = None

    def __init__(self): super().__init__()

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
    FUNCTION = "apply_preset_preview"
    CATEGORY = "DINKIssTyle/Color"

    def apply_preset_preview(self, image, xmp_file, strength):
        # Cache image
        DINKI_Adobe_XMP_Preview.last_input_tensor = image[0:1].clone().cpu()
        return self.apply_preset(image, xmp_file, strength)

    @staticmethod
    def process_preview(xmp_file, strength):
        if DINKI_Adobe_XMP_Preview.last_input_tensor is None: return None
        img = DINKI_Adobe_XMP_Preview.last_input_tensor
        node = DINKI_Adobe_XMP_Preview()
        
        result_tuple = node.apply_preset(img, xmp_file, strength)
        result_tensor = result_tuple[0]

        result_np = np.clip(255. * result_tensor.squeeze(0).numpy(), 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(result_np)
        
        max_size = 1024
        if pil_img.width > max_size:
            ratio = max_size / pil_img.width
            new_height = int(pil_img.height * ratio)
            pil_img = pil_img.resize((max_size, new_height), Image.BILINEAR)

        buff = io.BytesIO()
        pil_img.save(buff, format="PNG", compress_level=4)
        return buff.getvalue()

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


# ============================================================================
# [Class 3] DINKI AI Oversaturation Fix
# ============================================================================

class DINKI_AIOversaturationFix:
    def _to_numpy(self, img_t: torch.Tensor) -> np.ndarray:
        if img_t.device.type != "cpu": img_t = img_t.cpu()
        return img_t.numpy().astype(np.float32, copy=False)

    def _to_tensor(self, img_n: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.clip(img_n, 0.0, 1.0).astype(np.float32, copy=False))

    def _rgb_to_hsv_np(self, img):
        eps = 1e-7
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        maxc = np.maximum(np.maximum(r, g), b); minc = np.minimum(np.minimum(r, g), b)
        delta = maxc - minc
        v = maxc
        s = np.where(maxc > eps, delta / (maxc + eps), 0.0)
        rc = (maxc - r) / (delta + eps); gc = (maxc - g) / (delta + eps); bc = (maxc - b) / (delta + eps)
        h = np.where(delta <= eps, 0.0, np.where(maxc == r, bc - gc, np.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc)))
        h = (h / 6.0) % 1.0
        return np.stack([h, s, v], axis=-1)

    def _hsv_to_rgb_np(self, hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6.0); f = (h * 6.0) - i
        p = v * (1.0 - s); q = v * (1.0 - s * f); t = v * (1.0 - s * (1.0 - f))
        i = i.astype(np.int32) % 6
        r = np.choose(i, [v, q, p, p, t, v]); g = np.choose(i, [t, v, v, q, p, p]); b = np.choose(i, [p, p, t, v, v, q])
        return np.stack([r, g, b], axis=-1)

    def _is_skin_tone(self, h, s, v):
        return (h >= 0.014) & (h <= 0.07) & (s >= 0.1) & (s <= 0.6) & (v >= 0.3) & (v <= 0.95)

    def _desaturate_highlights(self, img, reduction, highlight_threshold, preserve_skin):
        hsv = self._rgb_to_hsv_np(img)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        t_low = max(highlight_threshold - 0.15, 0.0); t_high = highlight_threshold
        mask = np.clip((v - t_low) / (t_high - t_low + 1e-6), 0.0, 1.0)
        if preserve_skin:
            skin_mask = self._is_skin_tone(h, s, v)
            mask = np.where(skin_mask, mask * 0.3, mask)
        s_new = s * (1.0 - mask * reduction)
        return self._hsv_to_rgb_np(np.stack([h, s_new, v], axis=-1))

    def _chroma_limit(self, img, max_chroma):
        rgb = img.copy()
        sum_rgb = np.sum(rgb, axis=-1, keepdims=True); sum_rgb = np.maximum(sum_rgb, 1e-6)
        chroma = rgb / sum_rgb
        max_comp = np.max(chroma, axis=-1)
        over = max_comp > max_chroma
        if np.any(over):
            scale = np.where(over[..., None], max_chroma / (max_comp[..., None] + 1e-6), 1.0)
            return np.clip(chroma * scale * sum_rgb, 0.0, 1.0)
        return rgb

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fix_enabled": ("BOOLEAN", {"default": True}),
                "mode": (["desaturate_highlights", "global_desat", "chroma_limit", "auto"],),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "saturation_reduction": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "highlight_threshold": ("FLOAT", {"default": 0.82, "min": 0.5, "max": 1.0, "step": 0.01}),
                "max_chroma": ("FLOAT", {"default": 0.58, "min": 0.2, "max": 0.9, "step": 0.02}),
                "preserve_skin_tones": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "DINKIssTyle/Color"

    def apply(self, image, fix_enabled, mode, strength, saturation_reduction, highlight_threshold, max_chroma, preserve_skin_tones):
        if not fix_enabled: return (image,)
        rgb = image[..., :3]; alpha = image[..., 3:4] if image.shape[-1] == 4 else None
        batch_out = []
        for i in range(rgb.shape[0]):
            np_img = self._to_numpy(rgb[i])
            if np_img.shape[-1] == 1: np_img = np.repeat(np_img, 3, axis=-1)
            
            if mode == "desaturate_highlights":
                result = self._desaturate_highlights(np_img, saturation_reduction, highlight_threshold, preserve_skin_tones)
            elif mode == "global_desat":
                hsv = self._rgb_to_hsv_np(np_img); hsv[..., 1] *= (1.0 - saturation_reduction)
                result = self._hsv_to_rgb_np(hsv)
            elif mode == "chroma_limit":
                result = self._chroma_limit(np_img, max_chroma)
            elif mode == "auto":
                step1 = self._desaturate_highlights(np_img, saturation_reduction * 0.7, highlight_threshold, preserve_skin_tones)
                result = self._chroma_limit(step1, max_chroma)
            else: result = np_img.copy()
            
            blended = np_img * (1.0 - strength) + result * strength
            batch_out.append(self._to_tensor(blended))
        
        out_rgb = torch.stack(batch_out, dim=0)
        return (torch.cat([out_rgb, alpha.clone()], dim=-1) if alpha is not None else out_rgb,)


# ============================================================================
# [Class 4] DINKI Auto Adjustment
# ============================================================================

class DINKI_Auto_Adjustment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enable_auto_tone": ("BOOLEAN", {"default": False}),
                "enable_auto_contrast": ("BOOLEAN", {"default": False}),
                "enable_auto_color": ("BOOLEAN", {"default": True}),
                "enable_skin_tone": ("BOOLEAN", {"default": False}),
                "clip_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.05}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "DINKIssTyle/Color"

    def _get_luma(self, img):
        return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).unsqueeze(-1)

    def _match_brightness_gamma(self, src, target_mean):
        curr_mean = torch.mean(src, dim=(1, 2, 3), keepdim=True)
        mask = (curr_mean > 1e-3) & (curr_mean < 1.0 - 1e-3) & (target_mean > 1e-3)
        gamma = torch.log(target_mean + 1e-6) / torch.log(curr_mean + 1e-6)
        gamma = torch.clamp(gamma, 0.5, 2.0)
        gamma = torch.where(mask, gamma, torch.ones_like(gamma))
        return torch.pow(src.clamp(min=1e-6), gamma)

    def _apply_skin_tone(self, img):
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        cb = (b - y) / 1.8556; cr = (r - y) / 1.5748
        
        dist = torch.sqrt((cb + 0.10)**2 + (cr - 0.10)**2)
        skin_weight = torch.exp(-dist**2 / (2 * 0.05**2))
        
        sum_weight = torch.sum(skin_weight, dim=(1, 2), keepdim=True) + 1e-6
        mean_cb = torch.sum(cb * skin_weight, dim=(1, 2), keepdim=True) / sum_weight
        mean_cr = torch.sum(cr * skin_weight, dim=(1, 2), keepdim=True) / sum_weight
        
        curr_angle = torch.atan2(mean_cr, mean_cb)
        target_angle = torch.tensor(2.18, device=img.device)
        delta_theta = torch.clamp(target_angle - curr_angle, -0.26, 0.26)
        
        sin_theta = torch.sin(delta_theta); cos_theta = torch.cos(delta_theta)
        cb_new = cb * cos_theta - cr * sin_theta
        cr_new = cb * sin_theta + cr * cos_theta
        
        r_new = y + 1.5748 * cr_new
        g_new = y - 0.1873 * cb_new - 0.4681 * cr_new
        b_new = y + 1.8556 * cb_new
        return torch.clamp(torch.stack([r_new, g_new, b_new], dim=-1), 0.0, 1.0)

    def apply(self, image, enable_auto_tone, enable_auto_contrast, enable_auto_color, enable_skin_tone, clip_percent, strength):
        rgb = image[..., :3]; alpha = image[..., 3:4] if image.shape[-1] == 4 else None
        out = rgb.clone()

        if enable_auto_color:
            luma = self._get_luma(out)
            weight = torch.exp(-torch.pow(luma - 0.5, 2) / (2 * 0.25**2))
            mean_rgb = torch.sum(out * weight, dim=(1, 2), keepdim=True) / (torch.sum(weight, dim=(1, 2), keepdim=True) + 1e-6)
            gains = torch.clamp(torch.mean(mean_rgb, dim=-1, keepdim=True) / (mean_rgb + 1e-6), 0.8, 1.25)
            out = torch.clamp(out * gains, 0.0, 1.0)

        if enable_skin_tone: out = self._apply_skin_tone(out)
        
        if enable_auto_tone or enable_auto_contrast:
            cp = clip_percent / 100.0
            orig_mean = torch.mean(out, dim=(1, 2, 3), keepdim=True)
            if enable_auto_tone:
                flat = out.view(out.shape[0], -1, 3)
                lows = torch.quantile(flat, cp, dim=1, keepdim=True).view(out.shape[0], 1, 1, 3)
                highs = torch.quantile(flat, 1.0 - cp, dim=1, keepdim=True).view(out.shape[0], 1, 1, 3)
                out = torch.clamp((out - lows) / torch.maximum(highs - lows, torch.tensor(1e-5, device=out.device)), 0, 1)
                out = self._match_brightness_gamma(out, orig_mean)
            if enable_auto_contrast:
                luma_flat = self._get_luma(out).view(out.shape[0], -1)
                lo = torch.quantile(luma_flat, cp, dim=1, keepdim=True).view(out.shape[0], 1, 1, 1)
                hi = torch.quantile(luma_flat, 1.0 - cp, dim=1, keepdim=True).view(out.shape[0], 1, 1, 1)
                out = torch.clamp((out - lo) / torch.maximum(hi - lo, torch.tensor(1e-5, device=out.device)), 0, 1)
                out = self._match_brightness_gamma(out, orig_mean)

        if strength < 1.0: out = torch.lerp(rgb, out, strength)
        return (torch.cat([out, alpha], dim=-1) if alpha is not None else out,)


# ============================================================================
# [Class 5] DINKI Color LUT (Base)
# ============================================================================

class DINKI_Color_Lut:
    _loaded_luts = {}
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):
        file_list = folder_paths.get_filename_list("luts")
        if not file_list: file_list = []
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

    def get_lut_tensor(self, lut_name):
        lut_path = folder_paths.get_full_path("luts", lut_name)
        if not lut_path: return None
        if lut_path in self._loaded_luts: return self._loaded_luts[lut_path]

        try:
            with open(lut_path, 'r', encoding='utf-8') as f: lines = f.readlines()
            size = -1; data_lines = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if line.startswith('LUT_3D_SIZE'):
                    try: size = int(line.split()[1])
                    except: pass
                    continue
                if not (line[0].isdigit() or line[0] == '-'): continue
                data_lines.append(line)
            
            if size == -1: raise ValueError("LUT_3D_SIZE not found")
            lut_data = np.fromstring(" ".join(data_lines), sep=' ', dtype=np.float32)
            expected = size * size * size * 3
            if len(lut_data) > expected: lut_data = lut_data[:expected]
            
            lut_tensor = torch.from_numpy(lut_data.reshape(size, size, size, 3)).permute(3, 0, 1, 2).unsqueeze(0)
            self._loaded_luts[lut_path] = lut_tensor
            return lut_tensor
        except Exception as e:
            print(f"[🅳INKIssTyle - Error] Failed to load LUT {lut_name}: {e}")
            return None

    def apply_lut(self, image, lut_name, strength):
        if not lut_name or lut_name == "-- None --": return (image,)
        lut_tensor = self.get_lut_tensor(lut_name)
        if lut_tensor is None: return (image,)

        device = image.device
        if lut_tensor.device != device: lut_tensor = lut_tensor.to(device)

        grid = image.unsqueeze(1) * 2.0 - 1.0
        processed = F.grid_sample(lut_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
        processed = processed.permute(0, 2, 3, 4, 1).squeeze(1)

        result = torch.lerp(image, processed, float(strength)) if strength < 1.0 else processed
        return (result,)


# ============================================================================
# [Class 6] DINKI Color LUT Preview (Interactive + API)
# ============================================================================

class DINKI_Color_Lut_Preview(DINKI_Color_Lut):
    last_input_tensor = None
    
    def __init__(self): super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        file_list = folder_paths.get_filename_list("luts")
        if not file_list: file_list = []
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
    FUNCTION = "apply_lut_preview"
    CATEGORY = "DINKIssTyle/Color"

    def apply_lut_preview(self, image, lut_name, strength):
        # Cache image
        DINKI_Color_Lut_Preview.last_input_tensor = image[0:1].clone().cpu()
        return self.apply_lut(image, lut_name, strength)

    @staticmethod
    def process_preview(lut_name, strength):
        if DINKI_Color_Lut_Preview.last_input_tensor is None: return None
        img = DINKI_Color_Lut_Preview.last_input_tensor
        node = DINKI_Color_Lut_Preview()
        
        result_tuple = node.apply_lut(img, lut_name, strength)
        result_tensor = result_tuple[0]

        result_np = np.clip(255. * result_tensor.squeeze(0).numpy(), 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(result_np)
        
        max_size = 2048
        if pil_img.width > max_size:
            ratio = max_size / pil_img.width
            new_height = int(pil_img.height * ratio)
            pil_img = pil_img.resize((max_size, new_height), Image.BILINEAR)

        buff = io.BytesIO()
        pil_img.save(buff, format="PNG", compress_level=4)
        return buff.getvalue()

@PromptServer.instance.routes.post("/dinki/preview_lut")
async def preview_lut_route(request):
    data = await request.json()
    lut_name = data.get("lut_name")
    strength = data.get("strength", 1.0)
    
    if DINKI_Color_Lut_Preview.last_input_tensor is None:
        return web.Response(status=400, text="No cached image found. Please run the workflow once.")

    img_bytes = DINKI_Color_Lut_Preview.process_preview(lut_name, strength)
    if img_bytes:
        return web.Response(body=img_bytes, content_type='image/png')
    else:
        return web.Response(status=500, text="Processing failed")


# ============================================================================
# [Class 7] DINKI Deband
# ============================================================================

class DINKI_Deband:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Bypass"}),
                "threshold": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "radius": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
                "grain": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_deband"
    CATEGORY = "DINKIssTyle/Color"

    def _box_filter(self, x, r):
        k_size = 2 * r + 1
        padded = F.pad(x, (r, r, r, r), mode='reflect')
        return F.avg_pool2d(padded, kernel_size=k_size, stride=1, padding=0)

    def _guided_filter(self, x, r, eps):
        mean_x = self._box_filter(x, r)
        mean_xx = self._box_filter(x * x, r)
        var_x = mean_xx - mean_x * mean_x
        a = var_x / (var_x + eps)
        b = mean_x - a * mean_x
        mean_a = self._box_filter(a, r); mean_b = self._box_filter(b, r)
        return mean_a * x + mean_b

    def apply_deband(self, image, enabled, threshold, radius, grain, iterations):
        if not enabled: return (image,)
        x = image.permute(0, 3, 1, 2)
        eps = (threshold / 255.0) ** 2
        out = x
        for _ in range(iterations): out = self._guided_filter(out, radius, eps)
        if grain > 0:
            noise = (torch.rand_like(out) - 0.5) * 2.0 * (grain / 255.0)
            out = out + noise
        return (torch.clamp(out, 0.0, 1.0).permute(0, 2, 3, 1),)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "DINKI_adobe_xmp": DINKI_adobe_xmp,
    "DINKI_Adobe_XMP_Preview": DINKI_Adobe_XMP_Preview,
    "DINKI_AIOversaturationFix": DINKI_AIOversaturationFix,
    "DINKI_Auto_Adjustment": DINKI_Auto_Adjustment,
    "DINKI_Color_Lut": DINKI_Color_Lut,
    "DINKI_Color_Lut_Preview": DINKI_Color_Lut_Preview,
    "DINKI_Deband": DINKI_Deband
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_adobe_xmp": "DINKI Adobe XMP",
    "DINKI_Adobe_XMP_Preview": "DINKI Adobe XMP Preview",
    "DINKI_AIOversaturationFix": "DINKI AI Oversaturation Fix",
    "DINKI_Auto_Adjustment": "DINKI Auto Adjustment",
    "DINKI_Color_Lut": "DINKI Color LUT",
    "DINKI_Color_Lut_Preview": "DINKI Color LUT Preview",
    "DINKI_Deband": "DINKI Deband"
}