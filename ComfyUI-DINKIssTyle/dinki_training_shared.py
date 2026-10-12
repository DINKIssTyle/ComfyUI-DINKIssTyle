"""Common training routes; no worker/GPU dependencies are imported here."""
import asyncio
import json
from pathlib import Path
import subprocess

from aiohttp import web
import folder_paths
from server import PromptServer

from .training_guard import GPU_LOCK, active_training, require_other_trainers_idle


@PromptServer.instance.routes.get("/dinki/training")
async def all_training(request):
    state = active_training()
    return web.json_response({"active": state})


def start_exclusive(name, get_service, data, resume_id=None):
    with GPU_LOCK:
        running, queued = PromptServer.instance.prompt_queue.get_current_queue()
        if running or queued:
            raise ValueError("Finish or clear the ComfyUI generation queue before starting training.")
        require_other_trainers_idle(name)
        service = get_service()

        def release_models():
            running, queued = PromptServer.instance.prompt_queue.get_current_queue()
            if running or queued:
                raise ValueError("The ComfyUI queue changed while preparing photos. Clear it and retry.")
            import comfy.model_management
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()

        return service.resume(resume_id, release_models) if resume_id else service.start(data, release_models)


def runtime_choices(runtime_dir, extra_groups=None, fallback_runtime=None):
    groups = {"dit": ("diffusion_models", "unet"), "vae": ("vae",),
              "text_encoder": ("text_encoders", "clip"),
              "training_adapter": ("loras",), "turbo_lora": ("loras",)}
    groups.update(extra_groups or {})
    choices, warnings = {"python": []}, []
    for key, categories in groups.items():
        choices[key], seen = [], set()
        for category in categories:
            try:
                for name in folder_paths.get_filename_list(category):
                    path = folder_paths.get_full_path(category, name)
                    if not path or not Path(path).is_file():
                        continue
                    path = str(Path(path).resolve())
                    if path not in seen:
                        seen.add(path)
                        choices[key].append({"label": f"{name} [{category}]", "path": path})
            except KeyError:
                continue
            except OSError as error:
                warnings.append(f"{category}: {error}")
        choices[key].sort(key=lambda entry: entry["label"].casefold())
    for directory, label in [(runtime_dir, "DKST training environment (.venv)"),
                             (fallback_runtime, "DKST Qwen training environment (compatible with H3)")]:
        if directory is None:
            continue
        for path in (Path(directory) / "Scripts/python.exe", Path(directory) / "bin/python"):
            if path.is_file():
                choices["python"].append({"label": label, "path": str(path.absolute())})
    return {"choices": choices, "warnings": warnings}


def register_training_routes(endpoint, name, get_service, runner, read_payload, model_choices, generate_caption):
    routes = PromptServer.instance.routes

    def response_error(error, status=400):
        return web.json_response({"error": str(error)}, status=status)

    @routes.get(endpoint + "/training")
    async def get_training(request):
        try:
            service = get_service()
            if request.query.get("id"):
                return web.json_response(service.status(request.query["id"]))
            return web.json_response({"jobs": service.history(request.query.get("project")), "active": service.active()})
        except (ValueError, OSError) as error:
            return response_error(error)

    @routes.post(endpoint + "/training")
    async def change_training(request):
        try:
            data = await read_payload(request)
            action = data.get("action")
            if action in {"start", "resume"}:
                if action == "resume" and not data.get("id"):
                    raise ValueError("Select a job to resume.")
                result = await asyncio.to_thread(start_exclusive, name, get_service, data,
                                                data.get("id") if action == "resume" else None)
            elif action in {"pause", "cancel"}:
                result = get_service().control(data.get("id"), action)
            else:
                raise ValueError("Unknown training action.")
            return web.json_response(result)
        except (ValueError, TypeError, OSError) as error:
            return response_error(error)

    @routes.post(endpoint + "/caption")
    async def caption(request):
        try:
            data = await read_payload(request)
            if active_training():
                raise ValueError("Finish or pause the active training job before generating photo captions.")
            return web.json_response(await asyncio.to_thread(generate_caption, get_service(), data))
        except (ValueError, TypeError, OSError) as error:
            return response_error(error)

    @routes.get(endpoint + "/settings")
    async def get_settings(request):
        return web.json_response(get_service().settings())

    @routes.post(endpoint + "/settings")
    async def set_settings(request):
        try:
            data = await read_payload(request)
            return web.json_response({"ok": True, "settings": get_service().save_settings(data)})
        except (ValueError, TypeError, OSError) as error:
            return response_error(error)

    @routes.get(endpoint + "/models")
    async def get_models(request):
        return web.json_response(await asyncio.to_thread(model_choices))

    @routes.post(endpoint + "/check")
    async def check(request):
        try:
            if active_training():
                raise ValueError("Finish or pause the active job before checking the runtime.")
            python = get_service().settings().get("python")
            if not python:
                raise ValueError("Configure the training Python environment first.")
            result = await asyncio.to_thread(subprocess.run, [python, "-u", str(runner), "--check"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
            try:
                data = json.loads(result.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                data = {"ok": False, "message": (result.stderr or result.stdout)[-3000:] or "Runtime check failed."}
            return web.json_response(data)
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            return response_error(error)

    @routes.get(endpoint + "/sample")
    async def sample(request):
        try:
            job = json.loads(get_service().job_file(request.query.get("id")).read_text(encoding="utf-8"))
            root = Path(job["output_dir"]).resolve()
            photo = (root / request.query.get("file", "")).resolve()
            if not photo.is_relative_to(root) or photo.suffix.lower() != ".png" or not photo.is_file():
                raise ValueError("Sample image not found.")
            return web.FileResponse(photo)
        except (ValueError, OSError) as error:
            return response_error(error, 404)

    @routes.get(endpoint + "/log")
    async def full_log(request):
        try:
            job = get_service().job_file(request.query.get("id"))
            kind = request.query.get("kind", "train")
            if kind not in {"train", "worker"}:
                raise ValueError("Unknown log type.")
            path = job.parent / f"{kind}.log"
            if not path.is_file():
                raise ValueError(f"No {kind} log has been written yet.")
            headers = {"Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store"}
            if request.query.get("download") == "1":
                headers["Content-Disposition"] = f'attachment; filename="{name}-{job.parent.name}-{kind}.log"'
            return web.FileResponse(path, headers=headers)
        except (ValueError, TypeError, OSError) as error:
            return response_error(error, 404)
