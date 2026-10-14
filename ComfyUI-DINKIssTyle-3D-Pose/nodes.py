import numpy as np
import torch
from .schema import RATIOS, validate_scene
from .storage import load_render


class PoseImage:
    CATEGORY = "DINKIssTyle/3D Pose"
    RETURN_TYPES = ("IMAGE", "MASK", "DKST_POSE")
    RETURN_NAMES = ("image", "mask", "pose")
    FUNCTION = "render"
    DESCRIPTION = "Edit a 3D mannequin and render its camera. Open the editor before first use."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "heads": (list(RATIOS), {"default": 7}),
            "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 1}),
            "scene_json": ("STRING", {"default": "", "multiline": True}),
            "render_id": ("STRING", {"default": ""}),
        }}

    def render(self, heads, width, height, scene_json, render_id):
        scene = validate_scene(scene_json)
        if scene["ratio"] != heads or scene["output"] != {"width": width, "height": height}:
            raise ValueError("Pose settings changed. Queue from the browser to update the render.")
        pixels = np.asarray(load_render(scene_json, render_id), dtype=np.float32) / 255.0
        return (torch.from_numpy(pixels[..., :3].copy()).unsqueeze(0),
                torch.from_numpy((1.0 - pixels[..., 3]).copy()).unsqueeze(0),
                {"scene_json":scene_json,"render_id":render_id})

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Check artifacts even when ComfyUI would otherwise return a cached image.
        image = load_render(kwargs["scene_json"], kwargs["render_id"])
        image.close()
        return kwargs["render_id"]
