"""Comfy-side project preparation and persistent worker jobs (no Torch imports)."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid

from .config import ACTIVE, model_signature
from . import config as qwen_config
from .runner import write_json


class Jobs:
    def __init__(self, root, output_root, image_roots, *, config=qwen_config, runner=None,
                 output_prefix="DKST_Qwen21", label="Qwen"):
        self.root = Path(root).resolve()
        self.output_root = Path(output_root).resolve()
        self.image_roots = {k: Path(v).resolve() for k, v in image_roots.items()}
        self.lock = threading.RLock()
        self.processes = {}
        self.preparing = False
        self.config = config
        self.runner = Path(runner) if runner else Path(__file__).with_name("runner.py")
        self.output_prefix = output_prefix
        self.label = label

    def settings(self):
        path = self.root / "settings.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def save_settings(self, data):
        config = self.config.validate_runtime(data)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock:
            if self.active():
                raise ValueError("Finish the active training job before changing runtime settings.")
            write_json(self.root / "settings.json", config)
        return config

    def resolve_image(self, item):
        if not isinstance(item, dict):
            raise ValueError("Invalid gallery image.")
        root = self.image_roots.get(item.get("type", "input"))
        filename, subfolder = item.get("filename"), item.get("subfolder", "")
        if root is None or not isinstance(filename, str) or not filename or not isinstance(subfolder, str):
            raise ValueError("Invalid gallery image path.")
        path = (root / subfolder / filename).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Gallery images must exist inside ComfyUI's input, temp or output directory.")
        return path

    def job_file(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("Invalid job ID.")
        for path in self.root.glob(f"projects/*/jobs/{job_id}/job.json"):
            return path
        raise ValueError("Training job not found.")

    def active(self):
        if self.preparing:
            return "preparing"
        for path in self.root.glob("projects/*/jobs/*/status.json"):
            state = json.loads(path.read_text(encoding="utf-8"))
            process = self.processes.get(state["id"])
            if state.get("status") in ACTIVE:
                # A detached coordinator keeps its own heartbeat across server restarts.
                if (process is not None and process.poll() is None
                        or process is None and time.time() - state.get("heartbeat", 0) < 120):
                    return state["id"]
        return None

    def prepare(self, data):
        from PIL import Image, ImageOps
        options = self.config.training_options(data)
        runtime = self.config.validate_runtime(self.settings())
        if hasattr(self.config, "validate_job"):
            self.config.validate_job(options, runtime)
        items = data.get("image_gallery", [])
        if isinstance(items, str):
            try:
                items = json.loads(items)
            except ValueError:
                raise ValueError("Invalid gallery JSON.") from None
        if not isinstance(items, list) or not 1 <= len(items) <= 256:
            raise ValueError("Add between 1 and 256 training photos.")
        photos = []
        for i, item in enumerate(items):
            path = self.resolve_image(item)
            with Image.open(path) as source:
                photo = ImageOps.exif_transpose(source)
                photo = photo.convert("RGBA" if "A" in photo.getbands() else "RGB")
                buffer = io.BytesIO()
                photo.save(buffer, format="PNG")
            pixels = buffer.getvalue()
            description = item.get("caption", "")
            if not isinstance(description, str) or len(description) > 4000:
                raise ValueError("Each photo caption must contain at most 4000 characters.")
            description = " ".join(description.split())
            trigger = options["trigger_word"]
            caption = description if description.startswith(trigger) else f"{trigger}, {description}".rstrip(", ")
            photos.append((f"{i:04d}_{hashlib.sha256(pixels).hexdigest()[:16]}", pixels, caption))
        job_id = uuid.uuid4().hex
        directory = self.root / "projects" / options["project_name"] / "jobs" / job_id
        images = directory / "images"
        images.mkdir(parents=True)
        signatures = {key: model_signature(runtime[key]) for key in self.config.MODEL_KEYS if runtime[key]}
        fingerprint = hashlib.sha256(json.dumps({
            "engine": self.config.ENGINE_VERSION, "images": [name for name, _, _ in photos],
            "resolution": options["resolution"], "vae": signatures["vae"],
        }, sort_keys=True).encode()).hexdigest()[:24]
        cache = directory.parents[1] / "cache" / fingerprint
        cache.mkdir(parents=True, exist_ok=True)
        output = self.output_root / self.output_prefix / options["project_name"] / job_id
        output.mkdir(parents=True)
        for name, pixels, caption in photos:
            (images / (name + ".png")).write_bytes(pixels)
            (images / (name + ".txt")).write_text(caption, encoding="utf-8")
        config = directory / "dataset.toml"
        config.write_text(self.config.dataset_toml(images, cache, options["resolution"]), encoding="utf-8")
        prompts = directory / "sample_prompts.txt"
        prompts.write_text(options["sample_prompt"].replace("{trigger}", options["trigger_word"]) + "\n", encoding="utf-8")
        job = dict(id=job_id, options=options, runtime=runtime, model_signatures=signatures,
                   image_dir=str(images), cache_dir=str(cache), dataset_config=str(config),
                   output_dir=str(output), sample_prompts=str(prompts), num_images=len(photos),
                   created=time.time(), engine_version=self.config.ENGINE_VERSION)
        write_json(directory / "job.json", job)
        return directory / "job.json"

    def launch(self, path):
        job = json.loads(path.read_text(encoding="utf-8"))
        write_json(path.parent / "status.json", dict(id=job["id"], status="starting", heartbeat=time.time(), error=""))
        runner = self.runner
        try:
            with (path.parent / "worker.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen([job["runtime"]["python"], "-u", str(runner), "--job", str(path)],
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=os.name != "nt",
                                           creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
        except OSError as error:
            write_json(path.parent / "status.json", dict(id=job["id"], status="failed", error=str(error), heartbeat=time.time()))
            raise ValueError("Could not launch the configured training Python. Check the runtime settings.") from error
        self.processes[job["id"]] = process
        return self.status(job["id"])

    def start(self, data, before_launch=None):
        with self.lock:
            if self.active():
                raise ValueError(f"Another {self.label} training job is already active.")
            self.preparing = True
            try:
                path = self.prepare(data)
                if before_launch:
                    before_launch()
                return self.launch(path)
            finally:
                self.preparing = False

    def resume(self, job_id, before_launch=None):
        with self.lock:
            if self.active():
                raise ValueError(f"Another {self.label} training job is already active.")
            path = self.job_file(job_id)
            job = json.loads(path.read_text(encoding="utf-8"))
            if self.status(job_id)["status"] == "completed":
                raise ValueError("This job is complete. Start a new job for new settings or more epochs.")
            for key, signature in job["model_signatures"].items():
                if model_signature(job["runtime"][key]) != signature:
                    raise ValueError("Model files changed; start a new job instead of resuming.")
            output = Path(job["output_dir"])
            states = sorted(p for p in output.glob("*-state") if p.is_dir()
                            and all((p / name).is_file() for name in
                                    ("training_state.json", "lora.safetensors", "optimizer.pt", "rng.pt")))
            if states:
                job["resume_state"] = str(states[-1])
            elif self.status(job_id)["status"] in {"paused", "failed", "cancelled", "interrupted"}:
                job.pop("resume_state", None)  # a stop during caching can safely repeat caching
            if before_launch:
                before_launch()
            for control in (path.parent / ".cancel_requested", output / ".pause_requested"):
                control.unlink(missing_ok=True)
            write_json(path, job)
            return self.launch(path)

    def control(self, job_id, action):
        with self.lock:
            path = self.job_file(job_id)
            state = self.status(job_id)
            if state["status"] not in ACTIVE:
                raise ValueError("This job is not running.")
            job = json.loads(path.read_text(encoding="utf-8"))
            if action == "pause":
                (Path(job["output_dir"]) / ".pause_requested").touch()
            elif action == "cancel":
                (path.parent / ".cancel_requested").touch()
            else:
                raise ValueError("Unknown training action.")
            return self.status(job_id)

    def status(self, job_id):
        path = self.job_file(job_id)
        job = json.loads(path.read_text(encoding="utf-8"))
        status_path = path.parent / "status.json"
        state = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {"id": job_id, "status": "prepared"}
        process = self.processes.get(job_id)
        if state["status"] in ACTIVE and ((process is not None and process.poll() is not None)
                or time.time() - state.get("heartbeat", 0) >= 120 and process is None):
            state.update(status="interrupted", error="Worker stopped or its heartbeat expired. Inspect the log before resuming.")
        log = path.parent / "train.log"
        if not log.is_file():
            log = path.parent / "worker.log"
        state["log"] = self.tail(log)
        progress = re.findall(r"epoch (\d+)/(\d+)\s+avr_loss=([\d.e+-]+)\s+step=(\d+)", state["log"])
        if progress:
            epoch, total, loss, step = progress[-1]
            state.update(epoch=int(epoch), epochs=int(total), loss=loss, step=int(step))
        if (path.parent / ".cancel_requested").exists() and state["status"] in ACTIVE:
            state["status"] = "cancel_requested"
        elif (Path(job["output_dir"]) / ".pause_requested").exists() and state["status"] in ACTIVE:
            state["status"] = "pause_requested"
        output = Path(job["output_dir"])
        state.update(project_name=job["options"]["project_name"], num_images=job["num_images"],
                     output_dir=str(output), created=job["created"],
                     checkpoints=[p.name for p in sorted(output.glob("*.safetensors")) if not p.name.startswith(".")],
                     samples=[str(p.relative_to(output)) for p in sorted(output.rglob("*.png"))][-12:])
        final = output / (job["options"]["project_name"] + ".safetensors")
        if final.is_file() and state["status"] == "completed":
            state["lora_path"] = str(final)
        return state

    def history(self, project=None):
        files = sorted(self.root.glob("projects/*/jobs/*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        results = []
        for path in files:
            if project and path.parents[2].name != project:
                continue
            state = self.status(path.parent.name)
            results.append({k: v for k, v in state.items() if k != "log"})
            if len(results) == 30:
                break
        return results

    @staticmethod
    def tail(path):
        if not path.is_file():
            return ""
        with path.open("rb") as log:
            log.seek(max(0, path.stat().st_size - 16000))
            return log.read().decode("utf-8", errors="replace").replace("\r", "\n")[-16000:]
