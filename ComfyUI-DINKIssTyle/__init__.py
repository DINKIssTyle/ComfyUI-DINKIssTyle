# ComfyUI/custom_nodes/ComfyUI-DINKIssTyle/__init__.py

import os
import importlib.util
import subprocess
import sys

# --- Package Installation Checks ---
packages = ["imageio"]

def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False
    return spec is not None

def install_package(package):
    print(f"## 🅳INKIssTyle: Installing missing package: {package}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for package in packages:
    if not is_installed(package):
        install_package(package)


# --- Node Imports ---

# 1. Prompt 관련 노드
from .dinki_prompt import (
    PromptLoader,
    DINKI_PromptSelector,
    DINKI_PromptSelectorLive,
    DINKI_random_prompt,
    prompt_loader,
    DINKI_Sampler_Preset,
)

# 2. Color & Correction 관련 노드 (dinki_color.py)
from .dinki_color import (
    DINKI_adobe_xmp,
    DINKI_Adobe_XMP_Preview,
    DINKI_AIOversaturationFix,
    DINKI_Auto_Adjustment,
    DINKI_Color_Lut,
    DINKI_Color_Lut_Preview,
    DINKI_Deband,
)

# 3. Processing 관련 노드 (통합됨: dinki_ps.py)
from .dinki_ps import (
    DINKI_Upscale_Latent_By,
    DINKI_Mask_Weighted_Mix,
    DINKI_Resize_And_Pad,
    DINKI_Remove_Pad_From_Image,
    DINKI_ToggleUNetLoader,
    DINKI_Empty_Or_Image_Latent,
)

# 4. 이미지 관련 노드 (통합됨: dinki_image.py)
from .dinki_image import (
    DINKI_ImageSelector,
    DINKI_CrossOutputSwitch,
    DINKI_ImagePreview,
    DINKI_Image_Resize,
)

# 5. 뷰어 관련 노드
from .dinki_viewer import (
    DINKI_Video_Player,
)

# 6. 스위치 관련
from .dinki_switch import (
    DINKI_Node_Switch,
    DINKI_String_Switch_RT,
)

# 7. 도구 관련
from .dinki_tool import (
    DINKI_Note,
    DINKI_Node_Check,
    DINKI_Anchor,
    DINKI_Auto_Focus,
)


from .dinki_lmstudio import DINKI_LMStudio
from .dinki_batchImages import DINKI_BatchImages
from .dinki_photo_specs import DINKI_photo_specifications
from .dinki_overlay import DINKI_Overlay
from .dinki_comparer import DINKI_Image_Comparer_MOV
from .dinki_grid import DINKI_Grid
from .dinki_base64 import DINKI_Img2Base64, DINKI_Base64Input, DINKI_Base64Viewer
from .dinki_depth_parallax import DINKI_DepthParallax_MOV





"""
@author: DINKIssTyle
@title: DINKIssTyle Nodes
@nickname: DINKIssTyle
@description: A collection of useful utility nodes for ComfyUI.
"""

# --- Node Registration ---

NODE_CLASS_MAPPINGS = {
    # Prompt
    "DINKI_PromptSelector": DINKI_PromptSelector,
    "DINKI_PromptSelectorLive": DINKI_PromptSelectorLive,
    "DINKI_random_prompt": DINKI_random_prompt,
    "DINKI_Sampler_Preset": DINKI_Sampler_Preset,

    # Color & Correction
    "DINKI_adobe_xmp": DINKI_adobe_xmp,
    "DINKI_Adobe_XMP_Preview": DINKI_Adobe_XMP_Preview,
    "DINKI_AIOversaturationFix": DINKI_AIOversaturationFix,
    "DINKI_Auto_Adjustment": DINKI_Auto_Adjustment,
    "DINKI_Color_Lut": DINKI_Color_Lut,
    "DINKI_Color_Lut_Preview": DINKI_Color_Lut_Preview,
    "DINKI_Deband": DINKI_Deband,

    # Photoshop/Processing (dinki_ps)
    "DINKI_Upscale_Latent_By": DINKI_Upscale_Latent_By,
    "DINKI_Mask_Weighted_Mix": DINKI_Mask_Weighted_Mix,
    "DINKI_Resize_And_Pad": DINKI_Resize_And_Pad,
    "DINKI_Remove_Pad_From_Image": DINKI_Remove_Pad_From_Image,
    "DINKI_ToggleUNetLoader": DINKI_ToggleUNetLoader,
    "DINKI_Empty_Or_Image_Latent": DINKI_Empty_Or_Image_Latent,

    # Viewer
    "DINKI_Video_Player": DINKI_Video_Player,

    # Switch
    "DINKI_Node_Switch": DINKI_Node_Switch,
    "DINKI_String_Switch_RT": DINKI_String_Switch_RT,

    # Tool
    "DINKI_Note": DINKI_Note,
    "DINKI_Node_Check": DINKI_Node_Check,
    "DINKI_Anchor": DINKI_Anchor,
    "DINKI_Auto_Focus": DINKI_Auto_Focus,


    # Others
    "DINKI_ImageSelector": DINKI_ImageSelector,
    "DINKI_CrossOutputSwitch": DINKI_CrossOutputSwitch,
    "DINKI_ImagePreview": DINKI_ImagePreview,
    "DINKI_LMStudio": DINKI_LMStudio,
    "DINKI_BatchImages": DINKI_BatchImages,
    "DINKI_photo_specifications": DINKI_photo_specifications,
    "DINKI_Overlay": DINKI_Overlay,
    "DINKI_Image_Comparer_MOV": DINKI_Image_Comparer_MOV,
    "DINKI_Grid": DINKI_Grid,
    "DINKI_Img2Base64": DINKI_Img2Base64,
    "DINKI_Base64Input": DINKI_Base64Input,
    "DINKI_Base64Viewer": DINKI_Base64Viewer,
    "DINKI_DepthParallax_MOV": DINKI_DepthParallax_MOV,
    "DINKI_Image_Resize": DINKI_Image_Resize,


}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Prompt
    "DINKI_PromptSelector": "DINKI CSV Prompt Selector",
    "DINKI_PromptSelectorLive": "DINKI CSV Prompt Selector (Live)",
    "DINKI_random_prompt": "DINKI Random Prompt",
    "DINKI_Sampler_Preset": "DINKI Sampler Preset",

    # Color & Correction
    "DINKI_adobe_xmp": "DINKI Adobe XMP",
    "DINKI_Adobe_XMP_Preview": "DINKI Adobe XMP Preview",
    "DINKI_AIOversaturationFix": "DINKI AI Oversaturation Fix",
    "DINKI_Auto_Adjustment": "DINKI Auto Adjustment",
    "DINKI_Color_Lut": "DINKI Color LUT",
    "DINKI_Color_Lut_Preview": "DINKI Color LUT Preview",
    "DINKI_Deband": "DINKI Deband",

    # Photoshop/Processing
    "DINKI_Upscale_Latent_By": "DINKI Upscale Latent By",
    "DINKI_Mask_Weighted_Mix": "DINKI Mask Weighted Mix",
    "DINKI_Resize_And_Pad": "DINKI Resize and Pad Image",
    "DINKI_Remove_Pad_From_Image": "DINKI Remove Pad from Image",
    "DINKI_ToggleUNetLoader": "DINKI UNet Loader (safetensors / GGUF)",
    "DINKI_Empty_Or_Image_Latent": "DINKI Empty or Image Latent",

    # Viewer
    "DINKI_Video_Player": "DINKI MP4/WEBM/GIF Player",

    # Switch
    "DINKI_Node_Switch": "DINKI Node Switch",
    "DINKI_String_Switch_RT": "DINKI String Switch RT",

    # Tool
    "DINKI_Note": "DINKI Note",
    "DINKI_Node_Check": "DINKI Node Check",
    "DINKI_Anchor": "DINKI Anchor",
    "DINKI_Auto_Focus": "DINKI Auto Focus",


    # Others
    "DINKI_ImageSelector": "DINKI Image Selector",
    "DINKI_CrossOutputSwitch": "DINKI Cross Output Switch",
    "DINKI_ImagePreview": "DINKI Image Signal",
    "DINKI_LMStudio": "DINKI LM Studio Assistant",
    "DINKI_BatchImages": "DINKI Batch Images",
    "DINKI_photo_specifications": "DINKI Photo Specifications",
    "DINKI_Overlay": "DINKI Overlay",
    "DINKI_Image_Comparer_MOV": "DINKI Image Comparer (to Video)",
    "DINKI_Grid": "DINKI Grid",
    "DINKI_Img2Base64": "DINKI Image To Base64",
    "DINKI_Base64Input": "DINKI Base64 String Input",
    "DINKI_Base64Viewer": "DINKI Base64 Image Viewer",
    "DINKI_DepthParallax_MOV": "DINKI Depth Parallax",
    "DINKI_Image_Resize": "DINKI Image Resize",


}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']