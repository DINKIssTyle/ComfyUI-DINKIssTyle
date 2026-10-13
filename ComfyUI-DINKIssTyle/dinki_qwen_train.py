"""Gallery-driven Qwen 2.1 LoRA training. Importing this module loads no trainer."""
import asyncio
import json
from pathlib import Path
import subprocess
import threading

from aiohttp import web
import folder_paths
from server import PromptServer

from .qwen21_trainer.config import ACTIVE
from .qwen21_trainer.jobs import Jobs
from .dinki_qwen_caption import NONE as CAPTION_NONE, caption_aliases, generate_caption
from .training_guard import GPU_LOCK, active_training, register_trainer, require_other_trainers_idle
from . import dinki_training_shared  # Combined generation/training status endpoint.

MANAGER = None
MANAGER_LOCK = threading.Lock()


def manager():
    global MANAGER
    with MANAGER_LOCK:
        if MANAGER is None:
            lora_roots = folder_paths.get_folder_paths("loras")
            MANAGER = Jobs(Path(folder_paths.get_input_directory()) / "dkst_qwen21",
                           lora_roots[0], {"input": folder_paths.get_input_directory(),
                                           "output": folder_paths.get_output_directory(),
                                           "temp": folder_paths.get_temp_directory()})
    return MANAGER


register_trainer("qwen21", lambda: manager())
dinki_training_shared.register_runtime_setup("/dinki/qwen21", "qwen21",
                                           Path(__file__).parent / "qwen21_trainer/runner.py")


class DINKI_Qwen21_LoRA_Trainer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project_name": ("STRING", {"default": "my_person"}),
            "image_gallery": ("DKST_IMAGE_REF_GRID", {"default": "[]", "captions": True}),
            "trigger_word": ("STRING", {"default": "dkst_person"}),
            "preset": (["Fast", "Standard"], {"default": "Fast"}),
            "epochs": ("INT", {"default": 30, "min": 1, "max": 1000}),
            "resolution": ("INT", {"default": 704, "min": 256, "max": 2048, "step": 16,
                                    "tooltip": "Bucket area target, preserving photo aspect ratios. 704 is about 0.5 MP."}),
            "precision": (["auto", "bf16", "int8", "nf4"],),
            "compile_blocks": (["off", "auto", "on"],),
            "sample_every": ("INT", {"default": 5, "min": 0, "max": 1000,
                                      "tooltip": "Preview every N epochs. 0 disables previews."}),
            "sample_prompt": ("STRING", {"default": "Portrait of {trigger}, waving hello, natural light", "multiline": True}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**32-1, "control_after_generate": False}),
        }, "optional": {
            "caption_connection_alias": (caption_aliases(), {"default": CAPTION_NONE,
                "tooltip": "Optional. Uses Settings > DKST > LLM only when Auto Caption is clicked."}),
            "caption_model": ("STRING", {"default": "",
                "tooltip": "Optional. Enter the exact name of an image-capable LLM for automatic captions."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_path", "info")
    FUNCTION = "report"
    CATEGORY = "DINKIssTyle/PS"
    OUTPUT_NODE = True
    DESCRIPTION = "Train a Qwen Image 2.1 person LoRA using the Start Training button. Queueing only reports status."

    @classmethod
    def VALIDATE_INPUTS(cls, caption_connection_alias=CAPTION_NONE, **kwargs):
        # A removed optional alias must not block ordinary, manual-caption training.
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def report(self, project_name, **kwargs):
        history = manager().history(project_name)
        state = history[0] if history else {"status": "No training job. Use Configure Models and Start Training."}
        path = state.get("lora_path", "")
        info = f"Qwen Image 2.1 LoRA: {state['status']}\n{path or state.get('error', '')}"
        return {"ui": {"text": [info]}, "result": (path, info)}


async def payload(request):
    limit = 2 * 1024 * 1024
    body = bytearray()
    # StreamReader.read(n) may return a partial network packet, before EOF.
    while chunk := await request.content.read(min(65536, limit + 1 - len(body))):
        body.extend(chunk)
        if len(body) > limit:
            raise ValueError("Training request is too large.")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    return data


def busy_generation():
    running, queued = PromptServer.instance.prompt_queue.get_current_queue()
    return bool(running or queued)


def start_job(data, resume_id=None):
    with GPU_LOCK:
        dinki_training_shared.require_runtime_idle()
        if busy_generation():
            raise ValueError("Finish or clear the ComfyUI generation queue before starting training.")
        require_other_trainers_idle("qwen21")
        return _start_job(data, resume_id)


def _start_job(data, resume_id=None):
    if busy_generation():
        raise ValueError("Finish or clear the ComfyUI generation queue before starting training.")
    service = manager()
    def release_models():
        if busy_generation():
            raise ValueError("The ComfyUI queue changed while preparing photos. Clear it and retry.")
        # Release ComfyUI's cached models before the worker loads its own models.
        import comfy.model_management
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
    return service.resume(resume_id, release_models) if resume_id else service.start(data, release_models)


@PromptServer.instance.routes.get("/dinki/qwen21/training")
async def get_training(request):
    try:
        service = manager()
        job_id = request.query.get("id")
        if job_id:
            return web.json_response(service.status(job_id))
        return web.json_response({"jobs": service.history(request.query.get("project")), "active": service.active()})
    except (ValueError, OSError) as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.post("/dinki/qwen21/training")
async def change_training(request):
    try:
        data = await payload(request)
        action = data.get("action")
        if action == "start":
            result = await asyncio.to_thread(start_job, data)
        elif action == "resume":
            result = await asyncio.to_thread(start_job, {}, data.get("id"))
        elif action in {"pause", "cancel"}:
            result = manager().control(data.get("id"), action)
        else:
            raise ValueError("Unknown training action.")
        return web.json_response(result)
    except (ValueError, TypeError, OSError) as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.post("/dinki/qwen21/caption")
async def caption(request):
    try:
        data = await payload(request)
        service = manager()
        if service.active() or active_training(exclude="qwen21"):
            raise ValueError("Finish or pause the active training job before generating photo captions.")
        return web.json_response(await asyncio.to_thread(generate_caption, service, data))
    except (ValueError, TypeError, OSError) as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.get("/dinki/qwen21/settings")
async def get_settings(request):
    return web.json_response(manager().settings())


def model_choices():
    """Use ComfyUI's registry, including extra_model_paths and legacy aliases."""
    choices = {key: [] for key in ("python", "dit", "vae", "text_encoder", "training_adapter", "turbo_lora")}
    warnings = []
    groups = {"dit": ("diffusion_models", "unet"), "vae": ("vae",),
              "text_encoder": ("text_encoders", "clip"),
              "training_adapter": ("loras",), "turbo_lora": ("loras",)}
    for key, categories in groups.items():
        seen = set()
        for category in categories:
            try:
                for name in folder_paths.get_filename_list(category):
                    path = folder_paths.get_full_path(category, name)
                    if not path or not Path(path).is_file():
                        continue
                    path = str(Path(path).resolve())
                    if path in seen:
                        continue
                    seen.add(path)
                    choices[key].append({"label": f"{name} [{category}]", "path": path})
            except KeyError:
                continue  # Older ComfyUI versions may have only the legacy category.
            except OSError as error:
                warnings.append(f"{category}: {error}")
        choices[key].sort(key=lambda entry: entry["label"].casefold())
    runtime = Path(__file__).parent / "qwen21_trainer/.venv"
    for path in (runtime / "Scripts/python.exe", runtime / "bin/python"):
        if path.is_file():
            choices["python"].append({"label": "DKST training environment (.venv)", "path": str(path.absolute())})
    # Do not resolve a venv Python symlink: its invoked path selects the environment.
    return {"choices": choices, "warnings": warnings}


@PromptServer.instance.routes.get("/dinki/qwen21/models")
async def get_models(request):
    return web.json_response(await asyncio.to_thread(model_choices))


@PromptServer.instance.routes.post("/dinki/qwen21/settings")
async def set_settings(request):
    try:
        data = await payload(request)
        if manager().active():
            raise ValueError("Finish the active training job before changing runtime settings.")
        config = manager().save_settings(data)
        return web.json_response({"ok": True, "settings": config})
    except (ValueError, TypeError, OSError) as error:
        return web.json_response({"error": str(error)}, status=400)


def check_runtime():
    config = manager().settings()
    if not config.get("python"):
        raise ValueError("Configure the training Python environment first.")
    runner = Path(__file__).parent / "qwen21_trainer/runner.py"
    result = subprocess.run([config["python"], "-u", str(runner), "--check"], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=60)
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "message": (result.stderr or result.stdout)[-3000:] or "Runtime check failed."}


@PromptServer.instance.routes.post("/dinki/qwen21/check")
async def check(request):
    try:
        dinki_training_shared.require_runtime_idle()
        if manager().active():
            raise ValueError("Finish the active job before checking the runtime.")
        return web.json_response(await asyncio.to_thread(check_runtime))
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.get("/dinki/qwen21/sample")
async def sample(request):
    try:
        path = manager().job_file(request.query.get("id"))
        job = json.loads(path.read_text(encoding="utf-8"))
        root = Path(job["output_dir"]).resolve()
        photo = (root / request.query.get("file", "")).resolve()
        if not photo.is_relative_to(root) or photo.suffix.lower() != ".png" or not photo.is_file():
            raise ValueError("Sample image not found.")
        return web.FileResponse(photo)
    except (ValueError, OSError) as error:
        return web.json_response({"error": str(error)}, status=404)


@PromptServer.instance.routes.get("/dinki/qwen21/log")
async def full_log(request):
    try:
        job = manager().job_file(request.query.get("id"))
        kind = request.query.get("kind", "train")
        if kind not in {"train", "worker"}:
            raise ValueError("Unknown log type.")
        path = job.parent / f"{kind}.log"
        if not path.is_file():
            raise ValueError(f"No {kind} log has been written yet.")
        headers = {"Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store"}
        if request.query.get("download") == "1":
            headers["Content-Disposition"] = f'attachment; filename="qwen21-{job.parent.name}-{kind}.log"'
        return web.FileResponse(path, headers=headers)
    except (ValueError, TypeError, OSError) as error:
        return web.json_response({"error": str(error)}, status=404)
