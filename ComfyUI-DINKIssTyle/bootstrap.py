"""Prepare host dependencies without letting a failed install abort all nodes."""
import importlib
from importlib import metadata
import logging
import os
import subprocess
import sys

LOGGER = logging.getLogger(__name__)
DEPENDENCIES = (
    ("numpy", "numpy"), ("PIL", "Pillow"), ("safetensors", "safetensors"),
    ("requests", "requests"), ("imageio", "imageio"),
    ("imageio_ffmpeg", "imageio-ffmpeg"), ("av", "av"), ("psutil", "psutil"),
    ("cv2", "opencv-python-headless>=4.8.0"),
)


def opencv_requirement():
    # All four distributions expose cv2; don't add a second one to a working env.
    for distribution in ("opencv-contrib-python-headless", "opencv-python-headless",
                         "opencv-contrib-python", "opencv-python"):
        try:
            metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
        return distribution + ">=4.8.0"
    return "opencv-python-headless>=4.8.0"


def dependency_error(module):
    try:
        imported = importlib.import_module(module)
        if module == "cv2" and not hasattr(imported, "FaceDetectorYN_create"):
            return "The installed OpenCV build has no YuNet face detector."
        if module == "cv2" and tuple(int(part) for part in imported.__version__.split(".")[:2]) < (4, 8):
            return "The bundled YuNet model requires OpenCV 4.8 or newer."
    except Exception as error:
        return str(error)
    return None


def ensure_dependencies(install=None):
    if install is None:
        install = os.environ.get("DKST_AUTO_INSTALL", "1").lower() not in {"0", "false", "no"}
    failures = {}
    for module, requirement in DEPENDENCIES:
        error = dependency_error(module)
        if error is None:
            continue
        if module == "cv2":
            requirement = opencv_requirement()
        if install:
            LOGGER.warning("DKST: preparing %s with %s", requirement, sys.executable)
            try:
                command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
                if module == "cv2":
                    command.append("--upgrade")
                subprocess.run(command + [requirement], check=True, timeout=180)
                importlib.invalidate_caches()
                # A previously imported cv2 without YuNet cannot be safely reloaded.
                if module == "cv2" and module in sys.modules:
                    error = "OpenCV was updated. Restart ComfyUI to load the new build."
                else:
                    error = dependency_error(module)
            except (OSError, subprocess.SubprocessError) as failure:
                error = str(failure)
        if error is not None:
            failures[module] = error
            LOGGER.error("DKST: %s unavailable: %s. Run %s -m pip install \"%s\" and restart ComfyUI. "
                         "Other node modules will still be loaded.", module, error, sys.executable, requirement)
    return failures


def main():
    failures = ensure_dependencies(install=True)
    if failures:
        raise SystemExit(1)
    print("DKST: host dependencies ready. Qwen/MiniMax: use Prepare Runtime in Configure Models "
          "to install their separate CUDA environment.")


if __name__ == "__main__":
    main()
