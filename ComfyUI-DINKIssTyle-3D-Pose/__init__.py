"""Standalone 3D pose editor for ComfyUI."""
from .nodes import PoseImage
from .openpose import OpenPoseImage
from . import api  # Register only this pack's HTTP routes.

NODE_CLASS_MAPPINGS = {"DKST_3D_Pose":PoseImage,"DKST_3D_OpenPose":OpenPoseImage}
NODE_DISPLAY_NAME_MAPPINGS = {"DKST_3D_Pose":"DKST 3D Pose","DKST_3D_OpenPose":"DKST 3D Pose → OpenPose"}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
