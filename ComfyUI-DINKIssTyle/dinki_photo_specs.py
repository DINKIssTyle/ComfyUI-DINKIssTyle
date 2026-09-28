import math

class DINKI_photo_specifications:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resolution": (["Image", "Custom"], {"default": "Custom"}),
                "resolution_multiple": ("INT", {"default": 8, "min": 4, "max": 128, "step": 4}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 64.0, "step": 0.01, "round": 0.01}),
                "aspect_ratio": (
                    [
                        # --- Basic ---
                        "Basic 1:1", 
                        "Bacic 1:2",
                        "Bacic 1.5:2",  
                        "Basic 9:16", 
                        "Basic 10:16", 
                        # --- Photo ---
                        "Photo 3:4", 
                        "Photo 3.5:5", 
                        "Photo 4:6", 
                        "Photo 5:7", 
                        "Photo 6:8", 
                        "Photo 8:10", 
                        "Photo 10:13", 
                        "Photo 10:15", 
                        "Photo 11:14",
                        # --- Cinema / Film ---
                        "35mm Academy 1.37:1",
                        "35mm Flat 1.85:1",
                        "35mm Scope (Anamorphic) 2.39:1",
                        "70mm Todd-AO 2.20:1",
                        "IMAX 70mm 1.43:1",
                        "Super 35 1.85:1",
                        "Super 35 2.39:1",
                        "Super 16 1.66:1",
                        "Super 16 1.78:1",
                    ],
                    {"default": "Basic 1:1"}
                ),
                "orientation": ("BOOLEAN", {
                    "default": False,
                    "label_off": "Portrait",
                    "label_on": "Landscape",
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "info_string")
    FUNCTION = "calculate_resolution"
    CATEGORY = "DINKIssTyle/Image"
    
    DESCRIPTION = "Calculates a target resolution from an image or a custom aspect ratio and rounds each dimension to the selected multiple."

    def calculate_resolution(self, megapixels, aspect_ratio, orientation, resolution="Custom", resolution_multiple=8, image=None):
        # 1. 목표 픽셀 수 설정 (Base: 1024x1024 = 1,048,576 pixel for 1MP)
        mp_multiplier = float(str(megapixels).removesuffix("MP"))
        megapixels_label = f"{mp_multiplier:g}MP"
        target_area = 1024 * 1024 * mp_multiplier

        # Image uses the source image's aspect ratio and direction. Custom keeps
        # the existing aspect-ratio and orientation controls.
        if resolution == "Image":
            if image is None:
                raise ValueError("Photo Specs: connect an image when resolution is Image.")
            if len(image.shape) != 4 or image.shape[1] < 1 or image.shape[2] < 1:
                raise ValueError("Photo Specs: image must have shape [batch, height, width, channels].")
            source_height, source_width = int(image.shape[1]), int(image.shape[2])
            target_ratio = source_width / source_height
        elif resolution == "Custom":
            ratio_string = aspect_ratio.split(" ")[-1]
            w_ratio, h_ratio = map(float, ratio_string.split(":"))
            target_ratio = w_ratio / h_ratio
        else:
            raise ValueError(f"Photo Specs: unknown resolution mode {resolution!r}.")

        # 3. 너비와 높이 계산
        height_val = math.sqrt(target_area / target_ratio)
        width_val = height_val * target_ratio

        # 4. 선택한 배수로 보정 (반올림)
        multiple = int(resolution_multiple)
        if multiple < 4 or multiple > 128 or multiple % 4:
            raise ValueError("Photo Specs: resolution_multiple must be a multiple of 4 from 4 to 128.")
        width = max(multiple, round(width_val / multiple) * multiple)
        height = max(multiple, round(height_val / multiple) * multiple)

        if resolution == "Custom":
            # Accept saved/API values from the former dropdown as well.
            is_portrait = orientation == "Portrait" if isinstance(orientation, str) else not orientation
            if is_portrait and width > height:
                width, height = height, width
            elif not is_portrait and width < height:
                width, height = height, width

        if resolution == "Image":
            divisor = math.gcd(source_width, source_height)
            ratio_label = f"{source_width // divisor}:{source_height // divisor}"
            info_string = (f"{width}x{height} (Image {source_width}x{source_height}, "
                           f"{ratio_label}, {megapixels_label}, multiple {multiple})")
        elif multiple == 8:
            # Preserve the text supplied by existing workflows at the default.
            info_string = f"{width}x{height} ({aspect_ratio}, {megapixels_label})"
        else:
            info_string = f"{width}x{height} ({aspect_ratio}, {megapixels_label}, multiple {multiple})"

        return (width, height, info_string)

# ComfyUI 노드 등록
NODE_CLASS_MAPPINGS = {
    "DINKI_photo_specifications": DINKI_photo_specifications
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_photo_specifications": "DKST Image (Photo Specs)"
}
