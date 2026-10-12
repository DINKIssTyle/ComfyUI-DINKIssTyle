"""Face dataset exports use CPU detection independently of the generation queue."""
import asyncio
import json
from pathlib import Path
import threading

from aiohttp import web
import folder_paths
from server import PromptServer

from .face_crop.core import ACTIVE, FaceCropService

_service = None
_service_lock = threading.Lock()


def service():
    global _service
    with _service_lock:
        if _service is None:
            roots = {"input": folder_paths.get_input_directory(), "output": folder_paths.get_output_directory(),
                     "temp": folder_paths.get_temp_directory()}
            _service = FaceCropService(roots["output"], roots, Path(roots["input"]) / "dkst_face_crop/jobs")
        return _service


def public_state(state):
    return {**state, "output_dir": str(service().output_dir(state))}


def summary(state):
    return (f'{state["status"]} · {state["processed"]}/{state["total"]} images/cells\n'
            f'{len(state["results"])} crops · {len(state["skipped"])} skipped\n'
            f'{service().output_dir(state)}' + (f'\n{state["error"]}' if state.get("error") else ""))


class DINKI_Face_Crop_Export:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image_gallery": ("DKST_IMAGE_REF_GRID", {"default": "[]"}),
            "output_folder": ("STRING", {"default": "DKST_FaceCrops/my_person", "tooltip": "Relative to ComfyUI output. Each export creates a new dated folder."}),
            "filename_prefix": ("STRING", {"default": "face"}),
            "face_selection": (["All faces", "Largest face"],),
            "crop_shape": (["BBox ratio", "Square"],),
            "padding_percent": ("FLOAT", {"default": 30.0, "min": 0, "max": 200, "step": 1, "tooltip": "30 adds 30% of face width on each side and 30% of height above/below. Crops stay within the image or grid cell."}),
            "limit_size": ("BOOLEAN", {"default": True, "tooltip": "Downscale oversized crops only. Never enlarge small crops."}),
            "max_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64, "tooltip": "Longest edge in pixels; used only when limit_size is enabled."}),
            "grid_split": (["Off", "4x4"], {"tooltip": "4x4 detects each cell independently and confines padding to that cell."}),
            "confidence": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 0.99, "step": 0.01}),
        }, "optional": {"result_job_id": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "folder", "info")
    OUTPUT_IS_LIST = (True, False, False)
    OUTPUT_NODE = True
    FUNCTION = "report"
    CATEGORY = "DINKIssTyle/PS"
    DESCRIPTION = "Export padded face crops using the Execute button. Queue/Run sends the latest export to downstream nodes as a list of images, preserving different sizes."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def report(self, result_job_id="", **kwargs):
        import numpy as np
        from PIL import Image
        import torch
        state = service().status(result_job_id)
        if state["status"] in ACTIVE:
            raise ValueError("Face export is still running. Wait for it to finish before Queue/Run.")
        if not state["results"]:
            raise ValueError(summary(state) + "\nNo crops to output. Review skipped entries or lower confidence and execute again.")
        images = []
        for result in state["results"]:
            with Image.open(service().result_path(state, result["filename"])) as image:
                images.append(torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0).unsqueeze(0))
        return (images, str(service().output_dir(state)), summary(state))


@PromptServer.instance.routes.post("/dinki/face_crop/jobs")
async def mutate_job(request):
    try:
        raw = bytearray()
        async for chunk in request.content.iter_chunked(64 * 1024):
            raw.extend(chunk)
            if len(raw) > 2 * 1024 * 1024:
                raise ValueError("Face-crop request is too large.")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        action = data.get("action")
        if action == "start":
            state = await asyncio.to_thread(service().start, data)
        elif action == "cancel":
            state = service().cancel(data.get("id"))
        else:
            raise ValueError("Unknown face-crop action.")
        return web.json_response(public_state(state))
    except (ValueError, OSError) as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.get("/dinki/face_crop/jobs")
async def get_job(request):
    try:
        state = await asyncio.to_thread(service().status, request.query.get("id"))
        return web.json_response(public_state(state))
    except (ValueError, OSError) as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.get("/dinki/face_crop/crop")
async def get_crop(request):
    try:
        state = service().status(request.query.get("id"))
        path = service().result_path(state, request.query.get("file"))
        return web.FileResponse(path)
    except (ValueError, OSError) as error:
        return web.json_response({"error": str(error)}, status=404)


@PromptServer.instance.routes.get("/dinki/face_crop/manifest")
async def get_manifest(request):
    try:
        state = service().status(request.query.get("id"))
        return web.json_response(state, headers={"Content-Disposition": 'attachment; filename="manifest.json"'})
    except (ValueError, OSError) as error:
        return web.json_response({"error": str(error)}, status=404)
