import os
import torch
import numpy as np
import folder_paths
import xml.etree.ElementTree as ET

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

# --- Helper Functions: Color Space & Curve ---

def _rgb_to_hsv_torch(img):
    """
    PyTorch implementation of RGB to HSV
    img: [B, H, W, 3] range 0..1
    Returns: h, s, v (each [B, H, W])
    """
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    max_val, _ = torch.max(img, dim=-1)
    min_val, _ = torch.min(img, dim=-1)
    diff = max_val - min_val + 1e-6

    # Hue calculation
    h = torch.zeros_like(max_val)
    mask_r = (max_val == r)
    mask_g = (max_val == g)
    mask_b = (max_val == b)

    h[mask_r] = (g[mask_r] - b[mask_r]) / diff[mask_r] % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4
    h = h / 6.0 # Normalize to 0..1

    # Saturation
    s = torch.zeros_like(max_val)
    mask_nz = (max_val > 0)
    s[mask_nz] = diff[mask_nz] / max_val[mask_nz]

    # Value
    v = max_val
    return h, s, v

def _hsv_to_rgb_torch(h, s, v):
    """
    PyTorch implementation of HSV to RGB
    h, s, v: [B, H, W] range 0..1
    Returns: [B, H, W, 3]
    """
    h = h * 6.0
    c = v * s
    x = c * (1 - torch.abs((h % 2) - 1))
    m = v - c

    z = torch.zeros_like(h)
    
    # 조건별 RGB 조합
    # 0 <= h < 1: c, x, 0
    # 1 <= h < 2: x, c, 0
    # ...
    
    # 효율적인 구현을 위해 stack 사용
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
    
    if len(points) < 2:
        return np.linspace(0, 1, num_entries, dtype=np.float32)

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dx[dx == 0] = 1e-6
    m = dy / dx

    t = np.zeros_like(x)
    t[0] = m[0]; t[-1] = m[-1]
    
    if len(x) > 2:
        mask = np.sign(m[:-1]) * np.sign(m[1:]) > 0
        w1 = 2 * dx[1:] + dx[:-1]
        w2 = dx[1:] + 2 * dx[:-1]
        t_middle = np.zeros_like(m[:-1])
        t_middle[mask] = (w1[mask] + w2[mask]) / ((w1[mask] / m[:-1][mask]) + (w2[mask] / m[1:][mask]))
        t[1:-1] = t_middle

    xi = np.linspace(0, 255, num_entries, dtype=np.float32)
    indices = np.searchsorted(x, xi) - 1
    indices = np.clip(indices, 0, len(x) - 2)
    
    h = xi - x[indices]
    d = x[indices+1] - x[indices]
    norm_t = h / d
    t2 = norm_t * norm_t
    t3 = t2 * norm_t
    
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + norm_t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    
    yi = (h00 * y[indices] + h10 * d * t[indices] + h01 * y[indices+1] + h11 * d * t[indices+1])
    return np.clip(yi / 255.0, 0.0, 1.0).astype(np.float32)

def _apply_lut_torch(img_channel, lut_tensor):
    scaled = img_channel * 255.0
    idx_floor = torch.floor(scaled).long().clamp(0, 255)
    idx_ceil = torch.ceil(scaled).long().clamp(0, 255)
    weight = scaled - idx_floor.float()
    val_floor = lut_tensor[idx_floor]
    val_ceil = lut_tensor[idx_ceil]
    return torch.lerp(val_floor, val_ceil, weight)


class DINKI_adobe_xmp:
    """
    ComfyUI Custom Node: DINKI Adobe XMP Loader
    Includes: Exposure, Contrast, HSL, Tone Curve, Vignette, Grain
    """

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

    def parse_xmp(self, file_path):
        params = {
            "Exposure": 0.0, "Contrast": 0, "Vibrance": 0, "Saturation": 0,
            "VignetteAmount": 0, "VignetteMidpoint": 50, "VignetteFeather": 50, "VignetteRoundness": 0,
            "GrainAmount": 0, "GrainSize": 25,
            "ToneCurve": None, "ToneCurveRed": None, "ToneCurveGreen": None, "ToneCurveBlue": None,
            
            # HSL Dictionaries: Key=ColorName, Value=Adjustment(-100~100)
            "HSL_Hue": {},
            "HSL_Sat": {},
            "HSL_Lum": {}
        }
        
        # Adobe Color Names mapping
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
            tree = ET.parse(file_path)
            root = tree.getroot()
            descriptions = root.findall(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
            
            for desc in descriptions:
                for key, value in desc.attrib.items():
                    tag = key.split('}')[-1] if '}' in key else key
                    
                    # Basic
                    if tag == "Exposure2012": params["Exposure"] = float(value)
                    elif tag == "Contrast2012": params["Contrast"] = float(value)
                    elif tag == "Vibrance": params["Vibrance"] = float(value)
                    elif tag == "Saturation": params["Saturation"] = float(value)
                    
                    # Vignette & Grain
                    elif tag == "PostCropVignetteAmount": params["VignetteAmount"] = float(value)
                    elif tag == "PostCropVignetteMidpoint": params["VignetteMidpoint"] = float(value)
                    elif tag == "PostCropVignetteFeather": params["VignetteFeather"] = float(value)
                    elif tag == "GrainAmount": params["GrainAmount"] = float(value)
                    elif tag == "GrainSize": params["GrainSize"] = float(value)

                    # HSL Parsing
                    # Pattern: HueAdjustmentRed, SaturationAdjustmentBlue, LuminanceAdjustmentOrange etc.
                    for c_name in color_names:
                        if tag == f"HueAdjustment{c_name}": params["HSL_Hue"][c_name] = float(value)
                        elif tag == f"SaturationAdjustment{c_name}": params["HSL_Sat"][c_name] = float(value)
                        elif tag == f"LuminanceAdjustment{c_name}": params["HSL_Lum"][c_name] = float(value)

                # Tone Curve Parsing
                for child in desc:
                    tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if tag in ["ToneCurve", "ToneCurvePV2012"]:
                        seq = child.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Seq")
                        if seq: params["ToneCurve"] = parse_seq(", ".join([li.text for li in seq.findall(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li")]))
                    
                    for color in ["Red", "Green", "Blue"]:
                        if tag in [f"ToneCurve{color}", f"ToneCurvePV2012{color}"]:
                            seq = child.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Seq")
                            if seq: params[f"ToneCurve{color}"] = parse_seq(", ".join([li.text for li in seq.findall(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li")]))

        except Exception as e:
            print(f"[Warning] Failed to parse XMP {file_path}: {e}")
        return params

    def apply_hsl(self, img, p, device):
        """ 
        Applies HSL adjustments using Soft Masking
        """
        # Check if any HSL adjustments exist
        has_hsl = (len(p["HSL_Hue"]) + len(p["HSL_Sat"]) + len(p["HSL_Lum"])) > 0
        if not has_hsl:
            return img

        # 1. Convert RGB to HSV
        h, s, v = _rgb_to_hsv_torch(img) # [B,H,W] 0..1

        # 2. Define Color Centers (Adobe Approximations in 0..1 space)
        # Red(0), Orange(35), Yellow(60), Green(120), Aqua(180), Blue(240), Purple(275), Magenta(315)
        color_centers = {
            "Red": 0.0, "Orange": 35/360.0, "Yellow": 60/360.0, "Green": 120/360.0,
            "Aqua": 180/360.0, "Blue": 240/360.0, "Purple": 275/360.0, "Magenta": 315/360.0
        }
        color_names = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
        
        # Influence Widths (approximate)
        width = 45/360.0 

        total_hue_shift = torch.zeros_like(h)
        total_sat_scale = torch.zeros_like(s)
        total_val_scale = torch.zeros_like(v)
        total_weight = torch.zeros_like(h) + 1e-6

        for c_name in color_names:
            center = color_centers[c_name]
            
            # Hue Adjust (-100..100) -> (-0.1..0.1 approx in normalized hue)
            # Adobe Hue shifts are roughly +/- 30 degrees max
            h_adj = p["HSL_Hue"].get(c_name, 0.0) / 100.0 * (30/360.0)
            
            # Sat Adjust (-100..100) -> scale factor (-1.0 .. 1.0)
            s_adj = p["HSL_Sat"].get(c_name, 0.0) / 100.0
            
            # Lum Adjust (-100..100) -> scale factor
            l_adj = p["HSL_Lum"].get(c_name, 0.0) / 100.0

            if h_adj == 0 and s_adj == 0 and l_adj == 0:
                continue

            # Calculate Mask (Circular Distance)
            # dist = min(|a-b|, 1-|a-b|)
            diff = torch.abs(h - center)
            dist = torch.min(diff, 1.0 - diff)
            
            # Gaussian-like linear falloff
            # Weight = 1 at center, 0 at >width
            weight = torch.clamp(1.0 - (dist / width), 0.0, 1.0)
            
            # Accumulate changes
            total_hue_shift += weight * h_adj
            total_sat_scale += weight * s_adj
            total_val_scale += weight * l_adj
            
            # Normalize mask sum (optional, simplifed here as additive)
            # total_weight += weight

        # Apply Adjustments
        h_new = (h + total_hue_shift) % 1.0
        s_new = torch.clamp(s * (1.0 + total_sat_scale), 0.0, 1.0)
        v_new = torch.clamp(v * (1.0 + total_val_scale), 0.0, 1.0)

        # 3. Convert back to RGB
        return _hsv_to_rgb_torch(h_new, s_new, v_new)


    def apply_preset(self, image, xmp_file, strength):
        if not xmp_file or xmp_file == "-- None --": return (image,)
        xmp_path = folder_paths.get_full_path("adobe_xmp", xmp_file)
        if not xmp_path: return (image,)

        p = self.parse_xmp(xmp_path)
        device = image.device
        out = image.clone()

        # 1. Exposure
        if p["Exposure"] != 0:
            out = out * torch.pow(2.0, torch.tensor(p["Exposure"], device=device))

        # 2. Contrast
        if p["Contrast"] != 0:
            c_val = p["Contrast"] / 100.0
            scale = 1.0 + c_val if c_val > 0 else 1.0 / (1.0 - c_val)
            out = (out - 0.5) * scale + 0.5
            out = torch.clamp(out, 0.0, 1.0)

        # 3. Tone Curve (Master & RGB)
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

        # 4. [NEW] HSL Adjustment
        out = self.apply_hsl(out, p, device)

        # 5. Vibrance & Saturation (Global)
        if p["Saturation"] != 0 or p["Vibrance"] != 0:
            luma = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
            luma = luma.unsqueeze(-1)
            max_ch, _ = torch.max(out, dim=-1, keepdim=True)
            min_ch, _ = torch.min(out, dim=-1, keepdim=True)
            curr_sat = max_ch - min_ch
            
            sat_mul = 1.0 + (p["Saturation"] / 100.0)
            vib_val = p["Vibrance"] / 100.0
            vib_mul = 1.0 + (vib_val * (1.0 - curr_sat)) if vib_val >= 0 else 1.0 + vib_val
            
            out = luma + (out - luma) * (sat_mul * vib_mul)
            out = torch.clamp(out, 0.0, 1.0)

        # 6. Vignette
        if p["VignetteAmount"] != 0:
            B, H, W, C = out.shape
            y = torch.linspace(-1, 1, H, device=device)
            x = torch.linspace(-1, 1, W, device=device)
            mesh_y, mesh_x = torch.meshgrid(y, x, indexing='ij')
            dist = torch.sqrt(mesh_x**2 + mesh_y**2)
            
            midpoint = p["VignetteMidpoint"] / 100.0
            dist_norm = (dist - midpoint) / (1.5 - midpoint + 1e-6)
            dist_norm = torch.clamp(dist_norm, 0.0, 1.0)
            
            feather = p["VignetteFeather"] / 100.0
            if feather > 0: dist_norm = torch.pow(dist_norm, 1.0 / (feather + 0.1))

            amount = p["VignetteAmount"] / 100.0
            vignette_factor = 1.0 + (amount * dist_norm)
            out = out * vignette_factor.unsqueeze(0).unsqueeze(-1)
            out = torch.clamp(out, 0.0, 1.0)

        # 7. Grain
        if p["GrainAmount"] > 0:
            amount = p["GrainAmount"] / 100.0
            size = max(p["GrainSize"] / 100.0, 0.01)
            noise = torch.randn_like(out)
            if size > 0.3:
                down_factor = 1.0 / (1.0 + size * 2.0)
                dH, dW = int(out.shape[1] * down_factor), int(out.shape[2] * down_factor)
                small_noise = torch.randn((out.shape[0], 3, dH, dW), device=device)
                noise = torch.nn.functional.interpolate(small_noise, size=(out.shape[1], out.shape[2]), mode='bilinear').permute(0, 2, 3, 1)
            out = out + (noise * (amount * 0.15))
            out = torch.clamp(out, 0.0, 1.0)

        if strength < 1.0:
            out = torch.lerp(image, out, strength)

        return (out,)