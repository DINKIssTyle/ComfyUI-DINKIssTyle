"""Model-only LoRA stack for the DKST PS category."""

import json
import math

import folder_paths
import nodes


class DINKI_Multi_LoRA_Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_stack": (
                    "DKST_LORA_STACK",
                    {
                        "default": "[]",
                        "lora_names": ["None", *folder_paths.get_filename_list("loras")],
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_loras"
    CATEGORY = "DINKIssTyle/PS"
    TITLE = "DKST PS (Multi LoRA Loader)"
    DESCRIPTION = "Apply enabled LoRAs to a model in the order shown. CLIP is unchanged."

    def __init__(self):
        self._loaders = {}

    def load_loras(self, model, lora_stack):
        try:
            rows = json.loads(lora_stack)
        except (TypeError, ValueError) as exc:
            raise ValueError("LoRA stack must be valid JSON.") from exc
        if not isinstance(rows, list):
            raise ValueError("LoRA stack must be a list.")

        active = []
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise ValueError(f"LoRA row {index} must be an object.")
            if row.get("enabled", True) is not True:
                continue
            name = row.get("name", "None")
            if name in ("", "None", None):
                continue
            if not isinstance(name, str):
                raise ValueError(f"LoRA row {index} has an invalid name.")
            try:
                strength = float(row.get("strength_model", 1.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"LoRA row {index} has an invalid strength_model.") from exc
            if not math.isfinite(strength) or not -100.0 <= strength <= 100.0:
                raise ValueError(f"LoRA row {index} strength_model must be between -100 and 100.")
            if strength != 0:
                active.append((name, strength))

        for name, strength in active:
            loader = self._loaders.get(name)
            if loader is None:
                loader = nodes.LoraLoaderModelOnly()
                self._loaders[name] = loader
            model = loader.load_lora_model_only(model, name, strength)[0]
        return (model,)
