"""H3 worker entry point; independent vendor imports and the shared coordinator."""
import argparse
import json
from pathlib import Path
import sys

if __package__ and __package__ != "minimax_trainer":
    from .config import commands, validate_runtime
    from ..qwen21_trainer import runner as coordinator
else:
    # Invoked directly by the configured venv, without importing custom-node __init__.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from minimax_trainer.config import commands, validate_runtime
    from qwen21_trainer import runner as coordinator

VENDOR = Path(__file__).resolve().parent / "vendor"


def check_environment():
    coordinator.check_requirements(Path(__file__).parent / "requirements.txt")
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
    from fizgig.families.train import train_family
    from fizgig.minimax.embedder import build_qwen3_te
    from fizgig.minimax.vae import MiniMaxH3VideoVAEEncoder, MiniMaxH3VideoVAEDecoder
    assert set(FAMILIES) == {"minimax"}, "The worker loaded a different Fizgig installation."
    FAMILIES["minimax"].load_driver()
    return {"ok": bool(torch.cuda.is_available()), "family": "minimax", "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA GPU",
            "message": "Ready" if torch.cuda.is_available() else "Training requires an NVIDIA CUDA GPU."}


def invalidate_text_cache(job):
    return coordinator.invalidate_text_cache(job, architecture="minimaxh3")


def validate_models(job):
    """Inspect headers before GPU allocation; upstream VAE loads ignore decoder/encoder extras."""
    import torch
    from safetensors import safe_open
    from fizgig.minimax.vae import MiniMaxH3VideoVAEEncoder, MiniMaxH3VideoVAEDecoder
    runtime, options = job["runtime"], job["options"]
    dit = runtime["ref_dit"] if options["training_base"] == "ref2va" else runtime["dit"]
    with safe_open(dit, framework="pt", device="cpu") as stored:
        for name, expected in {"video_patch_proj.weight": [5376, 96], "condition_proj.weight": [5376, 5120]}.items():
            if name not in stored.keys() or stored.get_slice(name).get_shape() != expected:
                raise ValueError("The selected DiT is not a supported MiniMax H3 checkpoint.")
    with safe_open(runtime["text_encoder"], framework="pt", device="cpu") as stored:
        name = "model.embed_tokens.weight"
        if name not in stored.keys() or stored.get_slice(name).get_shape()[1:] != [5120]:
            raise ValueError("H3 requires its Qwen3-VL-32B text encoder, not the Qwen Image 8B encoder.")
    with torch.device("meta"):
        models = [MiniMaxH3VideoVAEEncoder()]
        if options["sample_every"]:
            models.append(MiniMaxH3VideoVAEDecoder())
    with safe_open(runtime["vae"], framework="pt", device="cpu") as stored:
        keys = set(stored.keys())
        for model in models:
            bad = [name for name, parameter in model.named_parameters()
                   if name not in keys or stored.get_slice(name).get_shape() != list(parameter.shape)]
            if bad:
                raise ValueError(f"Select the MiniMax H3 Video VAE. Missing/incompatible weights: {', '.join(bad[:5])}.")


def validate_lora(path):
    import re
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as stored:
        keys = set(stored.keys())
        down = [k for k in keys if re.fullmatch(r"lora_unet_blocks_\d+_.+\.lora_down\.weight", k)]
        if not down or any(k.replace(".lora_down.weight", ".lora_up.weight") not in keys for k in down):
            raise RuntimeError("The final file has no complete H3 LoRA weight pairs.")


def main():
    parser = argparse.ArgumentParser(description="DKST MiniMax H3 image LoRA worker")
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
    coordinator.run_job(args.job.resolve(), engine=sys.modules[__name__])
    return 0


if __name__ == "__main__":
    sys.exit(main())
