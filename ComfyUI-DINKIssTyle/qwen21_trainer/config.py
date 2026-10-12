"""Small, dependency-free validation and command builder for the Qwen worker."""
import json
from pathlib import Path
import re

ENGINE_VERSION = "qwen21-33454e1b-v1"
MODEL_KEYS = ("dit", "vae", "text_encoder", "training_adapter", "turbo_lora")
PRESETS = {"Fast": (8, 2e-4, 4e-4), "Standard": (16, 1e-4, 2e-4)}
ACTIVE = {"starting", "latents", "text", "training", "pause_requested", "cancel_requested"}


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[\w-]{1,64}", value):
        raise ValueError("Project name must contain 1–64 letters, numbers, underscores or hyphens.")
    if value.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                         *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("This project name is reserved by Windows.")
    return value


def integer(data, name, default, low, high):
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}.")
    return value


def training_options(data):
    preset = data.get("preset", "Fast")
    if preset not in PRESETS:
        raise ValueError("Choose Fast or Standard.")
    trigger = data.get("trigger_word", "dkst_person")
    if not isinstance(trigger, str) or not trigger.strip() or len(trigger) > 128 or "\n" in trigger:
        raise ValueError("Enter a trigger word (up to 128 characters, one line).")
    precision = data.get("precision", "auto")
    compile_blocks = data.get("compile_blocks", "off")
    if precision not in {"auto", "bf16", "int8", "nf4"} or compile_blocks not in {"off", "auto", "on"}:
        raise ValueError("Unknown precision or compilation option.")
    prompt = data.get("sample_prompt", "Portrait of {trigger}, waving hello, natural light")
    if not isinstance(prompt, str) or len(prompt) > 4000:
        raise ValueError("Sample prompt must be text of at most 4000 characters.")
    return dict(project_name=safe_name(data.get("project_name", "my_person")), preset=preset,
                trigger_word=trigger.strip(), epochs=integer(data, "epochs", 30, 1, 1000),
                resolution=integer(data, "resolution", 704, 256, 2048),
                seed=integer(data, "seed", 42, 0, 2**32-1), precision=precision,
                compile_blocks=compile_blocks, sample_every=integer(data, "sample_every", 5, 0, 1000),
                sample_prompt=prompt)


def model_signature(path):
    p = Path(path).resolve()
    stat = p.stat()
    return dict(path=str(p), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def validate_runtime(data):
    result = {}
    for key in ("python", *MODEL_KEYS):
        value = data.get(key, "")
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a file path.")
        value = value.strip()
        if not value and key == "turbo_lora":
            result[key] = ""
            continue
        p = Path(value).expanduser()
        if not value or not p.is_absolute() or not p.is_file():
            raise ValueError(f"Select an existing absolute file path for {key}.")
        result[key] = str(p.absolute() if key == "python" else p.resolve())
    return result


def dataset_toml(image_dir, cache_dir, resolution):
    # JSON string escaping is also valid for the basic TOML strings used here.
    q = lambda p: json.dumps(Path(p).as_posix(), ensure_ascii=False)
    return (f'[general]\nresolution = [{resolution}, {resolution}]\n'
            'caption_extension = ".txt"\nbatch_size = 1\nnum_repeats = 1\n'
            'enable_bucket = true\nbucket_no_upscale = true\n\n[[datasets]]\n'
            f'image_directory = {q(image_dir)}\ncache_directory = {q(cache_dir)}\n')


def commands(job, vendor):
    """Return argv lists, never shell command strings. Model selection is fixed."""
    options, runtime = job["options"], job["runtime"]
    common = [runtime["python"], "-u"]
    cache = common + [str(vendor / "src/fizgig/families/cache.py"), "--family", "qwen_image21",
                      "--dataset_config", job["dataset_config"], "--skip_existing", "--num_workers", "2",
                      "--batch_size", "1", "--device", "cuda"]
    rank, lr_min, lr_max = PRESETS[options["preset"]]
    train = common + [str(vendor / "src/fizgig/families/train.py"), "--family", "qwen_image21",
        "--dataset_config", job["dataset_config"], "--dit", runtime["dit"],
        "--vae", runtime["vae"], "--text_encoder", runtime["text_encoder"],
        "--training_adapter", runtime["training_adapter"],
        "--output_dir", job["output_dir"], "--output_name", options["project_name"],
        "--network_type", "lora", "--network_dim", str(rank), "--network_alpha", str(rank),
        "--adaptive_lr", "--adaptive_lr_min", str(lr_min), "--adaptive_lr_max", str(lr_max),
        "--max_train_epochs", str(options["epochs"]), "--save_every_n_epochs", "1",
        "--save_state", "--save_state_on_train_end", "--keep_last_n_states", "2",
        "--ema_decay", "0.98", "--optimizer_type", "adamw8bit", "--log_per_image_loss",
        "--precision", options["precision"], "--blocks_to_swap", "-1",
        "--compile_blocks", options["compile_blocks"], "--seed", str(options["seed"]),
        "--trigger_word", options["trigger_word"]]
    if options["sample_every"]:
        train += ["--sample_prompts", job["sample_prompts"], "--sample_every_n_epochs", str(options["sample_every"]),
                  "--sample_width", "704", "--sample_height", "704", "--sample_seed", str(options["seed"]),
                  "--sample_steps", "8" if runtime["turbo_lora"] else "25", "--sample_cfg_scale",
                  "1" if runtime["turbo_lora"] else "3"]
        if runtime["turbo_lora"]:
            train += ["--speed_lora", runtime["turbo_lora"]]
    if job.get("resume_state"):
        train += ["--resume", job["resume_state"]]
    return {"latents": cache + ["--stage", "latents", "--model", runtime["vae"]],
            "text": cache + ["--stage", "text", "--model", runtime["text_encoder"]], "training": train}
