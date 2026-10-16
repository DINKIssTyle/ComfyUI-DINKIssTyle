import io
import json
import math
import numpy as np
import pytest
from PIL import Image
from pose_pack.schema import defaults,validate_scene
from pose_pack.storage import save_render,load_depth,artifact_path
from pose_pack.depth import DepthImage
from pose_pack.camera_info import camera_document,camera_text,CameraInfo


def artifact(depth):
    scene=defaults();scene['output']={'width':64,'height':64};raw=json.dumps(scene)
    stream=io.BytesIO();Image.new('RGB',(64,64)).save(stream,format='PNG')
    saved=save_render(raw,stream.getvalue(),depth=depth.astype('<f4').tobytes())
    return {'scene_json':raw,'render_id':saved['render_id']}


def test_depth_distance_order_inversion_background_and_manual_range():
    depth=np.zeros((64,64),dtype=np.float32);depth[:,1:20]=2;depth[:,20:40]=4;depth[:,40:]=6
    ref=artifact(depth);image,mask=DepthImage().render(ref)
    assert image.shape==(1,64,64,3) and mask.shape==(1,64,64)
    assert image[0,0,1,0]==1 and image[0,0,25,0]==.5 and image[0,0,45,0]==0
    assert mask[0,0,0]==1 and mask[0,0,25]==0
    flipped,_=DepthImage().render(ref,invert=True);assert flipped[0,0,45,0]==1 and flipped[0,0,0,0]==0
    manual,_=DepthImage().render(ref,range_mode='Manual range',near_distance=0.5,far_distance=10)
    assert manual[0,0,25,0].item()==pytest.approx(1-3.5/9.5)
    with pytest.raises(ValueError,match='Near'):DepthImage().render(ref,range_mode='Manual range',near_distance=10,far_distance=2)


@pytest.mark.parametrize('distance',[0,3])
def test_empty_and_single_plane_depth_stay_finite(distance):
    image,mask=DepthImage().render(artifact(np.full((64,64),distance,dtype=np.float32)))
    assert np.isfinite(image.numpy()).all()
    assert image.max()==(1 if distance else 0)


def test_depth_is_immutable_and_dimensions_are_validated():
    depth=np.ones((64,64),dtype=np.float32);ref=artifact(depth)
    path=artifact_path(ref['render_id']).with_suffix('.depth.bin');path.write_bytes(depth.astype('<f4').tobytes()+b'bad')
    with pytest.raises(ValueError,match='damaged'):load_depth(ref['scene_json'],ref['render_id'])
    with pytest.raises(ValueError,match='dimensions'):artifact(np.zeros((63,64),dtype=np.float32))
    for value in [float('nan'),float('inf'),-1,.001,201]:
        depth[0,0]=value
        with pytest.raises(ValueError,match='depth'):artifact(depth)


def test_camera_text_uses_actual_quaternion_with_down_tilt_and_right_roll():
    # q = rotationX(-45 degrees) * rotationZ(-30 degrees).
    a,b=math.radians(-45)/2,math.radians(-30)/2
    s=defaults();s['camera']['quaternion']=[math.sin(a)*math.cos(b),-math.sin(a)*math.sin(b),math.cos(a)*math.sin(b),math.cos(a)*math.cos(b)]
    doc=camera_document(s)
    assert doc['tilt_deg']==pytest.approx(-45)
    assert doc['roll_deg']==pytest.approx(30)
    assert 'The camera tilts down 45 degrees and rolls 30 degrees to the right.' in camera_text(doc)
    ref=artifact(np.ones((64,64),dtype=np.float32));text,raw=CameraInfo().describe(ref)['result']
    assert json.loads(raw)['projection']=='perspective' and '50 mm equivalent' in text


def test_camera_ui_and_string_outputs_are_identical_and_never_empty():
    value=CameraInfo().describe(artifact(np.ones((64,64),dtype=np.float32)))
    text,raw=value['result']
    assert type(text) is type(raw) is str and text.strip() and raw.strip()
    assert value['ui']=={'text':[text],'camera_text':[text],'camera_json':[raw]}
    assert CameraInfo.OUTPUT_NODE is True


def test_camera_poles_and_orthographic_do_not_invent_heading_or_lens():
    s=defaults();s['camera']['quaternion']=[math.sqrt(.5),0,0,math.sqrt(.5)]
    doc=camera_document(s);assert doc['heading_deg'] is None and doc['roll_deg'] is None
    assert 'tilts up 90 degrees' in camera_text(doc)
    s['camera']['projection']='orthographic';doc=camera_document(s)
    assert doc['focal_length_35mm_equiv_mm'] is None and doc['vertical_fov_deg'] is None
    assert 'Lens:' not in camera_text(doc)


def test_box_schema_rejects_bad_geometry_duplicate_ids_and_hidden_state_types():
    s=defaults();box=dict(id='box1',type='box',name='red box',color='#ef3038',position=[1,.5,0],quaternion=[0,0,0,1],size=[1,1,1],visible=True)
    s['objects']=[box];assert validate_scene(json.dumps(s))['objects']==[box]
    s['objects'].append(dict(box))
    with pytest.raises(ValueError,match='duplicate'):validate_scene(json.dumps(s))
    s['objects']=[box]
    for key,value in [('size',[1,-1,1]),('quaternion',[0,0,0,0]),('visible','false')]:
        old=box[key];box[key]=value
        with pytest.raises(ValueError):validate_scene(json.dumps(s))
        box[key]=old
