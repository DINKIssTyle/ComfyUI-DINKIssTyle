"""Immutable, restart-safe render artifacts; no user supplied paths."""
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from PIL import Image
import folder_paths
from .schema import ID_PATTERN, MAX_BYTES, scene_hash, validate_scene
from .pose_data import validate_pose


def directory():
    path = Path(folder_paths.get_input_directory()) / "dkst_3d_pose"
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path(render_id):
    if not isinstance(render_id, str) or not ID_PATTERN.fullmatch(render_id):
        raise ValueError("Invalid pose render ID")
    return directory() / f"{render_id}.png"


def _atomic(path, data):
    fd, name = tempfile.mkstemp(prefix=".pose-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _render_digest(raw, data, pose=None):
    content=raw.encode()+b"\0"+data
    if pose is not None:
        content+=b"\0"+json.dumps(pose,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def save_render(raw, data, pose=None):
    scene = validate_scene(raw)
    if pose is not None:
        validate_pose(pose,scene["output"]["width"],scene["output"]["height"])
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Render must be a PNG under 32 MiB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.size != (scene["output"]["width"], scene["output"]["height"]):
                raise ValueError("PNG dimensions do not match the scene")
            image.verify()
    except Image.DecompressionBombError as exc:
        raise ValueError("PNG dimensions exceed the allowed output size") from exc
    render_id = _render_digest(raw,data,pose)
    path = artifact_path(render_id)
    # Metadata is the commit marker. Every ID refers to one scene and one PNG.
    _atomic(path, data)
    metadata={"scene_hash":scene_hash(raw),"scene":raw}
    if pose is not None:metadata["pose"]=pose
    _atomic(path.with_suffix(".json"),json.dumps(metadata,allow_nan=False).encode())
    return {"render_id": render_id, "scene_hash": scene_hash(raw)}


def _read_artifact(raw, render_id):
    scene = validate_scene(raw)
    path = artifact_path(render_id)
    if not path.exists() or not path.with_suffix(".json").exists():
        raise ValueError("Pose render is missing. Open the browser workflow and queue again to render it.")
    metadata = json.loads(path.with_suffix(".json").read_text())
    data = path.read_bytes()
    pose=metadata.get("pose")
    if metadata.get("scene_hash") != scene_hash(raw) or _render_digest(raw,data,pose) != render_id:
        raise ValueError("Pose changed or render is damaged. Open the editor and render again.")
    if pose is not None:validate_pose(pose,scene["output"]["width"],scene["output"]["height"])
    return scene,metadata,data


def load_render(raw, render_id):
    scene,_,data=_read_artifact(raw,render_id)
    with Image.open(io.BytesIO(data)) as image:
        if image.size != (scene["output"]["width"], scene["output"]["height"]):
            raise ValueError("Pose render size mismatch")
        return image.convert("RGBA")


def load_pose(raw,render_id):
    _,metadata,_=_read_artifact(raw,render_id)
    if metadata.get("pose") is None:
        raise ValueError("OpenPose data is missing. Queue this workflow from the browser to rebuild the pose render.")
    return metadata["pose"]
