"""Versioned scene validation shared by upload and execution."""
import hashlib
import json
import math
import re
from pathlib import Path

VERSION = 2
RENDERER_VERSION = "dkst-pose-5-makehuman"
RATIOS = (2, 3, 6, 7, 8, 9)
BASE_JOINTS = ("root", "pelvis", "waist", "chest", "neck", "head")
LIMBS = ("clavicle", "shoulder", "elbow", "wrist", "hip", "knee", "ankle", "toe")
JOINTS = BASE_JOINTS + tuple(f"{side}_{joint}" for side in ("L", "R") for joint in LIMBS) + tuple(
    f"{side}_{finger}_{i}" for side in ("L", "R") for finger in ("thumb", "index", "middle", "ring", "pinky") for i in range(3))
MAX_BYTES = 32 * 1024 * 1024
ID_PATTERN = re.compile(r"[a-f0-9]{64}\Z")


def defaults():
    return json.loads((Path(__file__).parent / "web/default_scene.json").read_text())


def _keys(obj, keys, label):
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise ValueError(f"Invalid {label} fields")


def _number(n, lo, hi, label):
    if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or not lo <= n <= hi:
        raise ValueError(f"Invalid {label}")


def _vector(value, length, lo, hi, label):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"Invalid {label}")
    for n in value:
        _number(n, lo, hi, label)


def validate_scene(raw):
    if not isinstance(raw, str) or len(raw.encode()) > 128 * 1024:
        raise ValueError("Scene JSON must be a string under 128 KiB")
    try:
        scene = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid scene JSON") from exc
    _keys(scene, ("version", "renderer", "ratio", "root", "joints", "camera", "appearance", "output", "objects"), "scene")
    if type(scene["version"]) is not int or scene["version"] != VERSION or scene["renderer"] != RENDERER_VERSION:
        raise ValueError("Unsupported scene version. Create a new scene with this development build.")
    if type(scene["ratio"]) is not int or scene["ratio"] not in RATIOS:
        raise ValueError("Choose 2, 3, 6, 7, 8 or 9 heads")
    _vector(scene["root"], 3, -100, 100, "root position")
    _keys(scene["joints"], JOINTS, "joints")
    for name, q in scene["joints"].items():
        _vector(q, 4, -1.001, 1.001, name)
        if abs(sum(x*x for x in q) - 1) > .002:
            raise ValueError(f"Non-unit joint rotation: {name}")
    cam = scene["camera"]
    _keys(cam, ("projection", "position", "quaternion", "target", "focal_mm", "ortho"), "camera")
    if cam["projection"] not in ("perspective", "orthographic"):
        raise ValueError("Invalid projection")
    _vector(cam["position"], 3, -100, 100, "camera position")
    _vector(cam["target"], 3, -100, 100, "camera target")
    _vector(cam["quaternion"], 4, -1.001, 1.001, "camera rotation")
    if abs(sum(x*x for x in cam["quaternion"]) - 1) > .002:
        raise ValueError("Non-unit camera rotation")
    _number(cam["focal_mm"], 8, 300, "focal length")
    _number(cam["ortho"], .1, 50, "orthographic size")
    appearance = scene["appearance"]
    _keys(appearance, ("background", "body", "transparent", "face", "floor", "light", "simple_hands", "simple_feet", "centerlines", "contours", "joint_markers", "outline", "guide_color", "line_width"), "appearance")
    for key in ("background", "body", "guide_color"):
        if not isinstance(appearance[key], str) or not re.fullmatch(r"#[a-fA-F0-9]{6}", appearance[key]):
            raise ValueError(f"Invalid {key} color")
    for key in ("transparent", "face", "floor", "simple_hands", "simple_feet", "centerlines", "contours", "joint_markers", "outline"):
        if type(appearance[key]) is not bool:
            raise ValueError(f"Invalid {key}")
    _number(appearance["light"], .1, 5, "light intensity")
    _number(appearance["line_width"], .5, 8, "guide line width")
    if not isinstance(scene['objects'],list) or len(scene['objects'])>32:raise ValueError('Use at most 32 objects')
    ids=set()
    for obj in scene['objects']:
        _keys(obj,('id','type','name','color','position','quaternion','size','visible'),'object')
        if not isinstance(obj['id'],str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',obj['id']) or obj['id'] in ids:raise ValueError('Invalid or duplicate object ID')
        ids.add(obj['id'])
        if obj['type']!='box' or not isinstance(obj['name'],str) or not 1<=len(obj['name'])<=80:raise ValueError('Invalid box name/type')
        if not isinstance(obj['color'],str) or not re.fullmatch(r'#[a-fA-F0-9]{6}',obj['color']):raise ValueError('Invalid box color')
        _vector(obj['position'],3,-100,100,'box position');_vector(obj['size'],3,.01,50,'box size')
        _vector(obj['quaternion'],4,-1.001,1.001,'box rotation')
        if abs(sum(x*x for x in obj['quaternion'])-1)>.002:raise ValueError('Invalid box rotation')
        if type(obj['visible']) is not bool:raise ValueError('Invalid box visibility')
    _keys(scene["output"], ("width", "height"), "output")
    for key in ("width", "height"):
        n = scene["output"][key]
        if type(n) is not int or not 64 <= n <= 4096:
            raise ValueError("Output size must be 64–4096 pixels")
    return scene


def scene_hash(raw):
    # Hash the exact validated UTF-8 snapshot. The browser uses this same wire format.
    validate_scene(raw)
    return hashlib.sha256(raw.encode()).hexdigest()
