"""Standalone 3D pose editor for ComfyUI."""
from .nodes import PoseImage
from .openpose import OpenPoseImage
from .depth import DepthImage
from .camera_info import CameraInfo
from . import api  # Register only this pack's HTTP routes.

NODE_CLASS_MAPPINGS = {"DKST_3D_Pose":PoseImage,"DKST_3D_OpenPose":OpenPoseImage,"DKST_3D_Depth":DepthImage,"DKST_3D_CameraInfo":CameraInfo}
NODE_DISPLAY_NAME_MAPPINGS = {"DKST_3D_Pose":"DKST 3D Pose","DKST_3D_OpenPose":"DKST 3D Pose → OpenPose","DKST_3D_Depth":"DKST 3D Pose → Depth","DKST_3D_CameraInfo":"DKST 3D Pose → Camera Info"}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
