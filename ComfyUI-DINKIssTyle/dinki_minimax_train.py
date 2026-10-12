"""MiniMax H3 photo LoRA node. GPU code runs only in its configured worker."""
from pathlib import Path
import threading

import folder_paths

from .qwen21_trainer.jobs import Jobs
from .minimax_trainer import config
from .dinki_qwen_train import DINKI_Qwen21_LoRA_Trainer, payload
from .dinki_qwen_caption import generate_caption
from .dinki_training_shared import register_training_routes, runtime_choices
from .training_guard import register_trainer

MANAGER = None
MANAGER_LOCK = threading.Lock()
RUNTIME = Path(__file__).parent / "minimax_trainer"


def manager():
    global MANAGER
    with MANAGER_LOCK:
        if MANAGER is None:
            MANAGER = Jobs(Path(folder_paths.get_input_directory()) / "dkst_minimax_h3",
                folder_paths.get_folder_paths("loras")[0],
                {"input": folder_paths.get_input_directory(), "output": folder_paths.get_output_directory(),
                 "temp": folder_paths.get_temp_directory()}, config=config, runner=RUNTIME / "runner.py",
                output_prefix="DKST_MinimaxH3", label="MiniMax H3")
    return MANAGER


register_trainer("minimax_h3", lambda: manager())


class DINKI_Minimax_H3_LoRA_Trainer(DINKI_Qwen21_LoRA_Trainer):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        required = inputs["required"]
        required["epochs"] = ("INT", {"default": 50, "min": 1, "max": 1000})
        required["resolution"] = ("INT", {"default": 512, "min": 256, "max": 2048, "step": 32,
            "tooltip": "Bucket area target, preserving photo aspect ratios. 512 is about 0.25 MP."})
        required["precision"] = (["auto", "int8", "nf4"],)
        required.update({
            "training_base": (["fl2va", "ref2va"], {"default": "fl2va",
                "tooltip": "Choose the same H3 base used by your generation workflow. ref2va needs its own DiT."}),
            "training_mode": (["Default", "More Blocks"], {"default": "Default",
                "tooltip": "Default trains blocks 20–49. More Blocks trains 6–49. Text token refiner stays frozen."}),
            "training_structure": (["Likeness and Style", "Model default"], {"default": "Likeness and Style",
                "tooltip": "Likeness and Style allocates 60% of steps to low noise; Model default uses 8%."}),
        })
        return inputs

    DESCRIPTION = "Train a MiniMax H3 person LoRA from photos using Start Training. Queueing only reports status."

    def report(self, project_name, **kwargs):
        history = manager().history(project_name)
        state = history[0] if history else {"status": "No training job. Use Configure Models and Start Training."}
        path = state.get("lora_path", "")
        info = f"MiniMax H3 LoRA: {state['status']}\n{path or state.get('error', '')}"
        return {"ui": {"text": [info]}, "result": (path, info)}


def model_choices():
    return runtime_choices(RUNTIME / ".venv", {"ref_dit": ("diffusion_models", "unet")},
                           fallback_runtime=Path(__file__).parent / "qwen21_trainer/.venv")


register_training_routes("/dinki/minimax_h3", "minimax_h3", lambda: manager(), RUNTIME / "runner.py",
                         payload, model_choices, generate_caption)
