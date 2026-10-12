"""Validated H3 still-image recipes and shell-free stage commands."""
from pathlib import Path

if __package__ == "minimax_trainer":
    from qwen21_trainer.config import ACTIVE, dataset_toml, integer, model_signature, safe_name
else:
    from ..qwen21_trainer.config import ACTIVE, dataset_toml, integer, model_signature, safe_name

ENGINE_VERSION = "minimax-33454e1b-v1"
MODEL_KEYS = ("dit", "ref_dit", "vae", "text_encoder", "training_adapter", "turbo_lora")
PRESETS = {"Fast": 8, "Standard": 16}


def training_options(data):
    preset = data.get("preset", "Fast")
    base = data.get("training_base", "fl2va")
    mode = data.get("training_mode", "Default")
    structure = data.get("training_structure", "Likeness and Style")
    precision = data.get("precision", "auto")
    compile_blocks = data.get("compile_blocks", "off")
    if (preset not in PRESETS or base not in {"fl2va", "ref2va"}
            or mode not in {"Default", "More Blocks"}
            or structure not in {"Likeness and Style", "Model default"}
            or precision not in {"auto", "int8", "nf4"}
            or compile_blocks not in {"off", "auto", "on"}):
        raise ValueError("Unknown H3 training option.")
    trigger = data.get("trigger_word", "dkst_person")
    if not isinstance(trigger, str) or not trigger.strip() or len(trigger) > 128 or "\n" in trigger:
        raise ValueError("Enter a trigger word (up to 128 characters, one line).")
    prompt = data.get("sample_prompt", "Portrait of {trigger}, waving hello, natural light")
    if not isinstance(prompt, str) or len(prompt) > 4000:
        raise ValueError("Sample prompt must be text of at most 4000 characters.")
    resolution = integer(data, "resolution", 512, 256, 2048)
    if resolution % 32:
        raise ValueError("H3 resolution must be a multiple of 32.")
    return dict(project_name=safe_name(data.get("project_name", "my_person")), trigger_word=trigger.strip(),
                preset=preset, training_base=base, training_mode=mode, training_structure=structure,
                epochs=integer(data, "epochs", 50, 1, 1000), resolution=resolution,
                precision=precision, compile_blocks=compile_blocks,
                sample_every=integer(data, "sample_every", 5, 0, 1000), sample_prompt=prompt,
                seed=integer(data, "seed", 42, 0, 2**32-1))


def validate_runtime(data):
    result = {}
    for key in ("python", *MODEL_KEYS):
        value = data.get(key, "")
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a file path.")
        value = value.strip()
        if not value and key in {"ref_dit", "training_adapter", "turbo_lora"}:
            result[key] = ""
            continue
        path = Path(value).expanduser()
        if not value or not path.is_absolute() or not path.is_file():
            raise ValueError(f"Select an existing absolute file path for {key}.")
        result[key] = str(path.absolute() if key == "python" else path.resolve())
    return result


def validate_job(options, runtime):
    if options["training_base"] == "ref2va" and not runtime["ref_dit"]:
        raise ValueError("Select the H3 ref2va DiT in Configure Models before training on ref2va.")


def commands(job, vendor):
    options, runtime = job["options"], job["runtime"]
    validate_job(options, runtime)
    common = [runtime["python"], "-u"]
    cache = common + [str(vendor / "src/fizgig/families/cache.py"), "--family", "minimax",
        "--dataset_config", job["dataset_config"], "--skip_existing", "--num_workers", "2", "--device", "cuda"]
    rank = PRESETS[options["preset"]]
    blocks = "20-49" if options["training_mode"] == "Default" else "6-49"
    dit = runtime["ref_dit"] if options["training_base"] == "ref2va" else runtime["dit"]
    train = common + [str(vendor / "src/fizgig/families/train.py"), "--family", "minimax",
        "--dataset_config", job["dataset_config"], "--dit", dit,
        "--vae", runtime["vae"], "--text_encoder", runtime["text_encoder"],
        "--output_dir", job["output_dir"], "--output_name", options["project_name"],
        "--network_type", "lora", "--network_dim", str(rank), "--network_alpha", str(rank),
        "--learning_rate", "1e-6", "--optimizer_type", "automagic3", "--ema_decay", "0.98",
        "--max_train_epochs", str(options["epochs"]), "--save_every_n_epochs", "1",
        "--save_state", "--save_state_on_train_end", "--keep_last_n_states", "2", "--log_per_image_loss",
        "--precision", options["precision"], "--blocks_to_swap", "-1", "--compile_blocks", options["compile_blocks"],
        "--seed", str(options["seed"]), "--trigger_word", options["trigger_word"], "--train_blocks", blocks]
    for option in (f"photo_blocks={blocks}", "train_token_refiner=0", "caption_dropout=0.05",
                   "lownoise_pct=60" if options["training_structure"] == "Likeness and Style" else "lownoise_pct=8",
                   "preview_frames=1", "preview_audio=0", "distill=0"):
        train += ["--family_option", option]
    if runtime["training_adapter"]:
        train += ["--training_adapter", runtime["training_adapter"]]
    if options["sample_every"]:
        # Upstream's Turbo is for fl2va previews. Keep ref2va previews on the ordinary schedule.
        turbo = runtime["turbo_lora"] if options["training_base"] == "fl2va" else ""
        train += ["--sample_prompts", job["sample_prompts"], "--sample_every_n_epochs", str(options["sample_every"]),
                  "--sample_width", "768", "--sample_height", "768", "--sample_seed", str(options["seed"]),
                  "--sample_steps", "6" if turbo else "20", "--sample_cfg_scale", "1"]
        if turbo:
            train += ["--speed_lora", turbo, "--speed_lora_strength", "0.75"]
    if job.get("resume_state"):
        train += ["--resume", job["resume_state"]]
    return {"latents": cache + ["--stage", "latents", "--model", runtime["vae"], "--batch_size", "1"],
            "text": cache + ["--stage", "text", "--model", runtime["text_encoder"], "--batch_size", "16"],
            "training": train}
