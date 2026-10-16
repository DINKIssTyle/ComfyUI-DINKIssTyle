import copy
import io
import json
import numpy as np
import pytest
from PIL import Image
from pose_pack.schema import defaults
from pose_pack.pose_data import validate_pose, select_pose, pixel_document
from pose_pack.storage import save_render, load_pose, artifact_path
from pose_pack.openpose import OpenPoseImage, draw_openpose, BODY_EDGES, HAND_EDGES


def document():
    return {"version":1.3,"canvas_width":64,"canvas_height":80,"people":[{
        "pose_keypoints_2d":[.2,.2,1]+[0]*51,"hand_left_keypoints_2d":[0]*63,
        "hand_right_keypoints_2d":[0]*63,"face_keypoints_2d":[],
    }]}


def artifact(pose=None):
    scene=defaults();scene["output"]={"width":64,"height":80};raw=json.dumps(scene)
    stream=io.BytesIO();Image.new("RGB",(64,80)).save(stream,format="PNG")
    saved=save_render(raw,stream.getvalue(),pose)
    return {"scene_json":raw,"render_id":saved["render_id"]}


def test_openpose_image_normalized_keypoints_and_pixel_json():
    reference=artifact(document());image,keypoints,raw=OpenPoseImage().render(reference)
    assert image.shape==(1,80,64,3)
    assert image[0,16,13,0]>.9 and image[0,16,13,1]<.02
    assert image[0,79,63].sum()==0
    assert keypoints==[document()]
    pixels=json.loads(raw);assert pixels["people"][0]["pose_keypoints_2d"][:3]==[12.8,16.0,1]
    assert len(BODY_EDGES)==17 and len(HAND_EDGES)==20


def test_filters_do_not_mutate_saved_landmarks():
    doc=document();doc["people"][0]["hand_left_keypoints_2d"][:3]=[.3,.4,1]
    original=copy.deepcopy(doc);ref=artifact(doc)
    image,kp,_=OpenPoseImage().render(ref,include_body=False,include_hands=False)
    assert image.sum()==0
    assert kp[0]["people"][0]["pose_keypoints_2d"]==[0]*54
    assert load_pose(**{ "raw":ref["scene_json"],"render_id":ref["render_id"]})==original
    assert select_pose(doc,include_head=False)["people"][0]["pose_keypoints_2d"][:3]==[0,0,0]


@pytest.mark.parametrize("mutation",[
    lambda d:d.update(canvas_width=65),lambda d:d.update(version=2),lambda d:d.update(people=[]),
    lambda d:d["people"][0].update(pose_keypoints_2d=[0]*51),
    lambda d:d["people"][0]["pose_keypoints_2d"].__setitem__(0,float("nan")),
    lambda d:d["people"][0]["pose_keypoints_2d"].__setitem__(0,1.5),
    lambda d:d["people"][0]["pose_keypoints_2d"].__setitem__(2,0),
])
def test_invalid_projection_rejected(mutation):
    doc=document();mutation(doc)
    with pytest.raises(ValueError):validate_pose(doc,64,80)


def test_missing_projection_is_rejected_and_saved_projection_is_immutable():
    ref=artifact()
    with pytest.raises(ValueError,match="missing"):OpenPoseImage().render(ref)
    old_path=artifact_path(ref["render_id"]).with_suffix(".json")
    old_metadata=json.loads(old_path.read_text());old_metadata["pose"]=None;old_path.write_text(json.dumps(old_metadata))
    with pytest.raises(ValueError,match="missing"):OpenPoseImage().render(ref)
    ref=artifact(document());path=artifact_path(ref["render_id"]).with_suffix(".json")
    metadata=json.loads(path.read_text());metadata["pose"]["people"][0]["pose_keypoints_2d"][0]=.8;path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match="damaged"):OpenPoseImage().render(ref)


def test_pose_payload_changes_artifact_identity():
    one=document();two=document();two["people"][0]["pose_keypoints_2d"][0]=.25
    assert artifact(one)["render_id"]!=artifact(two)["render_id"]


@pytest.mark.parametrize("line,radius",[(0,4),(33,4),(4,0),(4,33),(True,4)])
def test_invalid_drawing_options(line,radius):
    with pytest.raises(ValueError):draw_openpose(document(),line,radius)


def test_no_connections_drawn_to_missing_points_or_image_origin():
    doc=document();person=doc["people"][0];person["pose_keypoints_2d"]=[0]*54
    person["pose_keypoints_2d"][6:9]=[.8,.3,1]  # isolated right shoulder
    image=draw_openpose(doc)
    assert np.asarray(image)[:10,:10].sum()==0;image.close()
    assert pixel_document(doc)["people"][0]["pose_keypoints_2d"][6:9]==[51.2,24.0,1]


def test_floor_grid_can_be_drawn_without_polluting_standard_keypoint_json():
    doc=document();doc['floor_grid']=[[.1,.8,.9,.8,1],[.5,.6,.5,1,0]];ref=artifact(doc)
    grid,keypoints,raw=OpenPoseImage().render(ref,include_body=False,include_hands=False)
    plain,_,_=OpenPoseImage().render(ref,include_body=False,include_hands=False,include_floor=False)
    assert grid.sum()>0 and plain.sum()==0
    assert 'floor_grid' not in keypoints[0] and 'floor_grid' not in json.loads(raw)


@pytest.mark.parametrize('line',[[0,0,1,1,2],[0,0,1,1],[0,0,float('nan'),1,0],[-1,0,1,1,1]])
def test_invalid_floor_segments_rejected(line):
    doc=document();doc['floor_grid']=[line]
    with pytest.raises(ValueError):validate_pose(doc)
