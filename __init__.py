"""ComfyUI entry point when this repository is cloned into custom_nodes."""
import importlib

_nodes = importlib.import_module(".ComfyUI-DINKIssTyle", __name__)
NODE_CLASS_MAPPINGS = _nodes.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _nodes.NODE_DISPLAY_NAME_MAPPINGS
WEB_DIRECTORY = "./ComfyUI-DINKIssTyle/js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
