# ComfyUI/custom_nodes/ComfyUI-DINKIssTyle/__init__.py

import os
import time
from datetime import datetime

from server import PromptServer
from aiohttp import web
import folder_paths

from .dinki_auto_adjustment import DINKI_Auto_Adjustment
from .dinki_ai_oversaturation_fix import DINKI_AIOversaturationFix
from .dinki_latent_upscale_bypass import DINKI_Upscale_Latent_By
from .dinki_toggle_unet_loader import DINKI_ToggleUNetLoader
from .dinki_resize_pad import DINKI_Resize_And_Pad, DINKI_Remove_Pad_From_Image
from .dinki_prompt_nodes import (
    PromptLoader,
    DINKI_PromptSelector,
    DINKI_PromptSelectorLive,
    prompt_loader,
)
from .dinki_image_nodes import (
    DINKI_ImageSelector,
    DINKI_CrossOutputSwitch,
    DINKI_ImagePreview,
)
from .dinki_lmstudio import DINKI_LMStudio
from .dinki_batchImages import DINKI_BatchImages
from .dinki_node_switch import DINKI_Node_Switch


"""
@author: DINKIssTyle
@title: DINKIssTyle Nodes
@nickname: DINKIssTyle
@description: A collection of useful utility nodes for ComfyUI.
"""

@PromptServer.instance.routes.get("/get-csv-prompts")
async def get_csv_prompts(request):
    prompt_titles = []
    base_path = folder_paths.get_input_directory()
    file_path = os.path.join(base_path, "prompt_list.csv")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(',', 1)
                        if len(parts) == 2:
                            prompt_titles.append(parts[0].strip())
        except Exception as e:
            print(f"❌ DINKI_PromptSelector API Error: {e}")
    return web.json_response(sorted(prompt_titles))


@PromptServer.instance.routes.get("/dinki/prompts")
async def dinki_get_prompts(request):
    prompt_loader.load_prompts()
    return web.json_response(prompt_loader.prompt_data)


# --- Node Registration ---

NODE_CLASS_MAPPINGS = {
    "DINKI_Resize_And_Pad": DINKI_Resize_And_Pad,
    "DINKI_Remove_Pad_From_Image": DINKI_Remove_Pad_From_Image,
    "DINKI_PromptSelector": DINKI_PromptSelector,
    "DINKI_PromptSelectorLive": DINKI_PromptSelectorLive,
    "DINKI_ImageSelector": DINKI_ImageSelector,
    "DINKI_CrossOutputSwitch": DINKI_CrossOutputSwitch,
    "DINKI_ImagePreview": DINKI_ImagePreview,
    "DINKI_Auto_Adjustment": DINKI_Auto_Adjustment,
    "DINKI_AIOversaturationFix": DINKI_AIOversaturationFix,
    "DINKI_Upscale_Latent_By": DINKI_Upscale_Latent_By,
    "DINKI_ToggleUNetLoader": DINKI_ToggleUNetLoader,
    "DINKI_LMStudio": DINKI_LMStudio,
    "DINKI_BatchImages": DINKI_BatchImages,
    "DINKI_Node_Switch": DINKI_Node_Switch,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_Resize_And_Pad": "DINKI Resize and Pad Image",
    "DINKI_Remove_Pad_From_Image": "DINKI Remove Pad from Image",
    "DINKI_PromptSelector": "DINKI CSV Prompt Selector",
    "DINKI_PromptSelectorLive": "DINKI CSV Prompt Selector (Live)",
    "DINKI_ImageSelector": "DINKI Image Selector",
    "DINKI_CrossOutputSwitch": "DINKI Cross Output Switch",
    "DINKI_ImagePreview": "DINKI Image Preview",
    "DINKI_Auto_Adjustment": "DINKI Auto Adjustment",
    "DINKI_AIOversaturationFix": "DINKI AI Oversaturation Fix",
    "DINKI_Upscale_Latent_By": "DINKI Upscale Latent By",
    "DINKI_ToggleUNetLoader": "DINKI UNet Loader (safetensors / GGUF)",
    "DINKI_LMStudio": "DINKI LM Studio Assistant",
    "DINKI_BatchImages": "DINKI Batch Images",
    "DINKI_Node_Switch": "DINKI Node Switch",

}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
