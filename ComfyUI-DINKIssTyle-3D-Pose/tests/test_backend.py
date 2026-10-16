import io
import json
import numpy as np
import pytest
import torch
from PIL import Image
from pose_pack.schema import defaults, validate_scene, scene_hash
from pose_pack.storage import save_render, load_render, artifact_path
from pose_pack.nodes import PoseImage
from pose_pack.openpose import OpenPoseImage


def fixture(transparent=True):
    scene = defaults()
    scene["output"] = {"width": 64, "height": 80}
    scene["appearance"]["transparent"] = transparent
    pixels = np.zeros((80, 64, 4), dtype=np.uint8)
    pixels[...,:3] = (45,120,200)
    pixels[...,3] = 255
    if transparent:
        pixels[:10,:,3] = 0
        pixels[10:20,:,3] = 128
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return json.dumps(scene), stream.getvalue()


def test_standalone_registration():
    import pose_pack
    assert pose_pack.NODE_CLASS_MAPPINGS['DKST_3D_Pose'] is PoseImage
    assert pose_pack.NODE_CLASS_MAPPINGS['DKST_3D_OpenPose'] is OpenPoseImage
    assert set(pose_pack.NODE_CLASS_MAPPINGS)=={'DKST_3D_Pose','DKST_3D_OpenPose','DKST_3D_Depth','DKST_3D_CameraInfo'}
    assert pose_pack.WEB_DIRECTORY == "./web"


@pytest.mark.parametrize("ratio", [2,3,6,7,8,9])
def test_ratios(ratio):
    scene=defaults(); scene["ratio"]=ratio
    assert validate_scene(json.dumps(scene))["ratio"] == ratio


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(ratio=5), lambda s:s.update(ratio=True), lambda s:s.update(version=1),
    lambda s:s["joints"].update(head=[0,0,0,0]), lambda s:s["joints"].update(head=[float('nan'),0,0,1]),
    lambda s:s["output"].update(width=4097), lambda s:s["output"].update(width=64.5),
    lambda s:s["appearance"].update(transparent="false"), lambda s:s["appearance"].update(body="red"),
    lambda s:s["camera"].update(focal_mm=0), lambda s:s["camera"].update(quaternion=[0,0,0,0]),
    lambda s:s.update(extra="unknown"), lambda s:s["joints"].pop("head"),
])
def test_bad_scenes(mutation):
    scene=defaults();mutation(scene)
    with pytest.raises(ValueError):validate_scene(json.dumps(scene))


def test_rgb_mask_tensors_and_restart_safe_read():
    raw,data=fixture(); saved=save_render(raw,data)
    rgb,mask,pose=PoseImage().render(7,64,80,raw,saved["render_id"])
    assert pose=={"scene_json":raw,"render_id":saved["render_id"]}
    assert rgb.shape==(1,80,64,3) and mask.shape==(1,80,64)
    assert rgb.dtype==mask.dtype==torch.float32
    assert torch.allclose(rgb[0,0,0],torch.tensor([45,120,200])/255)
    assert mask[0,0,0]==1 and mask[0,30,0]==0
    assert mask[0,10,0].item()==pytest.approx(1-128/255)
    assert load_render(raw,saved["render_id"]).size==(64,80)
    assert saved["scene_hash"]==scene_hash(raw)
    assert PoseImage.IS_CHANGED(scene_json=raw,render_id=saved["render_id"])==saved["render_id"]


def test_opaque_mask():
    raw,data=fixture(False); saved=save_render(raw,data)
    assert PoseImage().render(7,64,80,raw,saved["render_id"])[1].sum()==0


def test_stale_and_corrupt_renders_rejected():
    raw,data=fixture(); saved=save_render(raw,data); scene=json.loads(raw);scene["ratio"]=8
    with pytest.raises(ValueError,match="changed"):load_render(json.dumps(scene),saved["render_id"])
    path=artifact_path(saved["render_id"]);path.write_bytes(data+b"damaged")
    with pytest.raises(ValueError,match="damaged"):load_render(raw,saved["render_id"])
    path.unlink()
    with pytest.raises(ValueError,match="missing"):load_render(raw,saved["render_id"])


def test_dimensions_must_agree():
    raw,data=fixture();scene=json.loads(raw);scene["output"]["width"]=65
    with pytest.raises(ValueError,match="dimensions"):save_render(json.dumps(scene),data)
    saved=save_render(raw,data)
    with pytest.raises(ValueError,match="settings changed"):PoseImage().render(8,64,80,raw,saved["render_id"])


@pytest.mark.parametrize("identifier", ["../escape", "abc", "A"*64, "a"*63+"/", None])
def test_path_traversal(identifier):
    with pytest.raises(ValueError):artifact_path(identifier)


def test_non_png_and_invalid_json():
    raw,_=fixture()
    with pytest.raises(OSError):save_render(raw,b"not a png")
    with pytest.raises(ValueError):validate_scene("{broken")
