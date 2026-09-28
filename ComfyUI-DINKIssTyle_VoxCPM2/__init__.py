"""ComfyUI VoxCPM2 speech synthesis, cloning, and model management nodes."""

from .bootstrap import ensure_dependencies

ensure_dependencies()

from .nodes import Downloader, TTS, Cloning

NODE_CLASS_MAPPINGS = {
    "DKST_VoxCPM2_Downloader": Downloader,
    "DKST_VoxCPM2_TTS": TTS,
    "DKST_VoxCPM2_Cloning": Cloning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DKST_VoxCPM2_Downloader": "DKST VoxCPM2 (Downloader)",
    "DKST_VoxCPM2_TTS": "DKST VoxCPM2 (TTS)",
    "DKST_VoxCPM2_Cloning": "DKST VoxCPM2 (Cloning)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
