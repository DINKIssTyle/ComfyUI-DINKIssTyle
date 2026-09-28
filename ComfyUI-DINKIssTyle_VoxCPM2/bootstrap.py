"""Install missing Python dependencies in ComfyUI's own environment."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path


REQUIRED_MODULES = (
    "voxcpm",
    "whisper",
    "huggingface_hub",
    "soundfile",
    "scipy",
    "torchaudio",
    "av",
)
REQUIREMENTS = Path(__file__).resolve().parent / "requirements.txt"


def _missing_modules() -> list[str]:
    missing = []
    for module in REQUIRED_MODULES:
        try:
            if importlib.util.find_spec(module) is None:
                missing.append(module)
        except ModuleNotFoundError:
            missing.append(module)
    return missing


def ensure_dependencies() -> None:
    missing = _missing_modules()
    if not missing:
        return

    print(f"[DKST VoxCPM2] Installing missing dependencies: {', '.join(missing)}", flush=True)
    command = [
        sys.executable, "-m", "pip", "install",
        "--disable-pip-version-check", "--no-input", "-r", str(REQUIREMENTS),
    ]
    try:
        subprocess.check_call(command)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"DKST VoxCPM2 dependency installation failed in {sys.executable}. "
            f"Install {REQUIREMENTS} with that Python interpreter, then restart ComfyUI."
        ) from exc

    importlib.invalidate_caches()
    still_missing = _missing_modules()
    if still_missing:
        raise RuntimeError(
            f"DKST VoxCPM2 dependencies are still unavailable: {', '.join(still_missing)}. "
            "Restart ComfyUI after installation."
        )
    print("[DKST VoxCPM2] Python dependencies installed.", flush=True)
