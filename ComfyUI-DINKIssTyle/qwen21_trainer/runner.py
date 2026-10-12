"""Isolated coordinator: each cache/training stage uses a fresh Python process.

python runner.py --check
python runner.py --job /absolute/path/job.json
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback

if __package__:
    from .config import commands, model_signature, validate_runtime
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import commands, model_signature, validate_runtime

VENDOR = Path(__file__).resolve().parent / "vendor"


def write_json(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        temp = Path(file.name)
    try:
        for attempt in range(6):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                # A Windows reader may briefly hold the old status file open.
                if os.name != "nt" or attempt == 5:
                    raise
                time.sleep(0.02)
    finally:
        temp.unlink(missing_ok=True)


def check_environment():
    import torch
    import accelerate
    import transformers
    import bitsandbytes
    import safetensors
    import einops
    import toml
    import voluptuous
    import cv2
    sys.path.insert(0, str(VENDOR / "src"))
    from fizgig.families.registry import FAMILIES
    from fizgig.dataset.config import ConfigSanitizer
    from fizgig.families.train import train_family
    driver = FAMILIES["qwen_image21"].load_driver()
    assert set(FAMILIES) == {"qwen_image21"}
    return {"ok": bool(torch.cuda.is_available()), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA GPU",
            "message": "Ready" if torch.cuda.is_available() else "Training requires an NVIDIA CUDA GPU."}


def invalidate_text_cache(job, architecture="qwenimage21"):
    """Plain T2I upstream skip-existing only checks the filename. Also check
    captions and encoder identity, while retaining the independent VAE caches.
    """
    from safetensors import safe_open
    cache = Path(job["cache_dir"])
    marker = cache / "text_encoder.json"
    signature = model_signature(job["runtime"]["text_encoder"])
    previous = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else None
    images = Path(job["image_dir"])
    for path in cache.glob(f"*_{architecture}_te.safetensors"):
        stem = path.name.removesuffix(f"_{architecture}_te.safetensors")
        caption_file = images / (stem + ".txt")
        valid = False
        if previous == signature and caption_file.is_file():
            try:
                with safe_open(str(path), framework="pt") as stored:
                    valid = (stored.metadata() or {}).get("caption1") == caption_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        if not valid:
            path.unlink()
    return signature


def terminate_stage(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def validate_lora(path):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as stored:
        keys = set(stored.keys())
        down = [k for k in keys if k.startswith("transformer.") and k.endswith(".lora_A.weight")]
        if not down or any(k.replace(".lora_A.weight", ".lora_B.weight") not in keys for k in down):
            raise RuntimeError("The final file has no complete Qwen LoRA weight pairs.")


def run_job(path, engine=None):
    # Model-specific entry points reuse process control without sharing GPU model imports.
    runtime_validator = engine.validate_runtime if engine else validate_runtime
    environment_check = engine.check_environment if engine else check_environment
    command_builder = engine.commands if engine else commands
    text_invalidator = engine.invalidate_text_cache if engine else invalidate_text_cache
    artifact_validator = engine.validate_lora if engine else validate_lora
    vendor = engine.VENDOR if engine else VENDOR
    job = json.loads(path.read_text(encoding="utf-8"))
    status_path = path.parent / "status.json"
    state = {"id": job["id"], "status": "starting", "pid": os.getpid(), "error": ""}
    cancel = path.parent / ".cancel_requested"
    pause = Path(job["output_dir"]) / ".pause_requested"
    final = Path(job["output_dir"]) / (job["options"]["project_name"] + ".safetensors")
    process = None

    def update(status):
        state.update(status=status, heartbeat=time.time())
        write_json(status_path, state)

    with (path.parent / "train.log").open("a", encoding="utf-8", buffering=1) as log:
        try:
            update("starting")
            runtime_validator(job["runtime"])
            for key, signature in job["model_signatures"].items():
                if model_signature(job["runtime"][key]) != signature:
                    raise ValueError(f"The {key} model file changed since this job was prepared. Start a new job.")
            env_result = environment_check()
            if not env_result["ok"]:
                raise RuntimeError(env_result["message"])
            if engine and hasattr(engine, "validate_models"):
                engine.validate_models(job)
            stage_commands = command_builder(job, vendor)
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            # Don't leak another ComfyUI/Fizgig installation into the worker's imports.
            env["PYTHONPATH"] = str(vendor / "src")
            # Each stage exits and releases its encoder/GPU allocations.
            for stage in ("latents", "text", "training"):
                if cancel.exists():
                    update("cancelled")
                    return
                if pause.exists() and stage != "training":
                    update("paused")
                    return
                signature = text_invalidator(job) if stage == "text" else None
                log.write(f"\n[DKST] Stage: {stage}\n")
                update(stage)
                process = subprocess.Popen(stage_commands[stage], stdout=log, stderr=subprocess.STDOUT,
                                           env=env, cwd=str(vendor), start_new_session=os.name != "nt",
                                           creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
                while process.poll() is None:
                    if cancel.exists():
                        terminate_stage(process)
                        update("cancelled")
                        return
                    update("pause_requested" if pause.exists() else stage)
                    time.sleep(1)
                if process.returncode:
                    raise RuntimeError(f"{stage} process exited with code {process.returncode}. See the training log.")
                if signature is not None:
                    write_json(Path(job["cache_dir"]) / "text_encoder.json", signature)
                if pause.exists() and not (stage == "training" and final.is_file()):
                    update("paused")
                    return
            if not final.is_file():
                raise RuntimeError("The trainer exited without writing the final LoRA.")
            artifact_validator(final)
            state["lora_path"] = str(final)
            update("completed")
        except BaseException as error:
            if process is not None and process.poll() is None:
                terminate_stage(process)
            traceback.print_exc(file=log)
            state["error"] = str(error)
            update("failed")
            raise


def main():
    parser = argparse.ArgumentParser(description="DKST Qwen Image 2.1 LoRA worker")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--check", action="store_true")
    choice.add_argument("--job", type=Path)
    args = parser.parse_args()
    if args.check:
        try:
            result = check_environment()
        except Exception as error:
            result = {"ok": False, "message": str(error)}
        print(json.dumps(result))
        return 0 if result["ok"] else 1
    run_job(args.job.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
