"""ComfyUI VoxCPM2 speech synthesis, cloning, and model management nodes."""

from .bootstrap import ensure_dependencies

ensure_dependencies()

from .nodes import Downloader, ReferenceAudio, TTSCloning
from . import api  # Register voice upload and transcript endpoints.

NODE_CLASS_MAPPINGS = {
    "DKST_VoxCPM2_Downloader": Downloader,
    "DKST_VoxCPM2_ReferenceAudio": ReferenceAudio,
    "DKST_VoxCPM2_TTSCloning": TTSCloning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DKST_VoxCPM2_Downloader": "DKST VoxCPM2 (Downloader)",
    "DKST_VoxCPM2_ReferenceAudio": "DKST VoxCPM2 (Reference Audio)",
    "DKST_VoxCPM2_TTSCloning": "DKST VoxCPM2 (TTS & Cloning)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
