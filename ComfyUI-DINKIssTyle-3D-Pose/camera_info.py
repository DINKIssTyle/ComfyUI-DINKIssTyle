"""Deterministic camera description. Angles are derived from the view basis."""
import json
import math
from .schema import validate_scene


def camera_document(scene):
    c=scene['camera'];x,y,z,w=c['quaternion']
    right=[1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w)]
    up=[2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w)]
    forward=[-2*(x*z+y*w),-2*(y*z-x*w),-(1-2*(x*x+y*y))]
    horizontal=math.hypot(forward[0],forward[2])
    tilt=math.degrees(math.asin(max(-1,min(1,forward[1]))))
    heading=math.degrees(math.atan2(forward[0],-forward[2])) if horizontal>1e-7 else None
    if heading is not None:
        ref_right=[-forward[2]/horizontal,0,forward[0]/horizontal]
        ref_up=[ref_right[1]*forward[2]-ref_right[2]*forward[1],ref_right[2]*forward[0]-ref_right[0]*forward[2],ref_right[0]*forward[1]-ref_right[1]*forward[0]]
        roll=-math.degrees(math.atan2(sum(a*b for a,b in zip(right,ref_up)),sum(a*b for a,b in zip(right,ref_right))))
    else:roll=None
    aspect=scene['output']['width']/scene['output']['height']
    h=math.hypot(36,24)/math.sqrt(1+aspect*aspect)
    perspective=c['projection']=='perspective'
    return dict(version=1,projection=c['projection'],position=c['position'],quaternion=c['quaternion'],forward=forward,up=up,
        tilt_deg=tilt,heading_deg=heading,roll_deg=roll,
        focal_length_35mm_equiv_mm=c['focal_mm'] if perspective else None,
        vertical_fov_deg=math.degrees(2*math.atan(h/(2*c['focal_mm']))) if perspective else None,
        horizontal_fov_deg=math.degrees(2*math.atan(h*aspect/(2*c['focal_mm']))) if perspective else None,
        orthographic_size=c['ortho'] if not perspective else None,aspect=aspect,output=scene['output'],
        convention='Y-up; forward local -Z; heading zero world -Z, positive toward +X; tilt positive up; roll positive clockwise looking through the camera; 35mm equivalence uses the 36x24mm diagonal.')


def camera_text(doc):
    number=lambda v:format(round(abs(v),1),'g')
    tilt,heading,roll=doc['tilt_deg'],doc['heading_deg'],doc['roll_deg']
    clauses=[]
    if abs(tilt)>=.05:clauses.append(f"tilts {'up' if tilt>0 else 'down'} {number(tilt)} degrees")
    else:clauses.append('looks horizontally')
    if heading is not None and abs(heading)>=.05:clauses.append(f"turns {number(heading)} degrees to the {'right' if heading>0 else 'left'} from world -Z")
    if roll is not None and abs(roll)>=.05:clauses.append(f"rolls {number(roll)} degrees to the {'right' if roll>0 else 'left'}")
    orientation='The camera '+(', '.join(clauses[:-1])+' and '+clauses[-1] if len(clauses)>1 else clauses[0])+'.'
    lens=f"Lens: {number(doc['focal_length_35mm_equiv_mm'])} mm equivalent (35 mm format, diagonal angle-of-view convention)." if doc['projection']=='perspective' else f"Orthographic size: {number(doc['orthographic_size'])} scene units."
    position=', '.join(f'{v:.2f}' for v in doc['position'])
    direction=', '.join(f'{v:.4f}' for v in doc['forward'])
    o=doc['output']
    return f"Projection: {doc['projection']}.\n{lens}\nOrientation: {orientation}\nPosition: ({position}) scene units.\nView direction: ({direction}).\nOutput: {o['width']} x {o['height']} pixels."


class CameraInfo:
    CATEGORY='DINKIssTyle/3D Pose'
    RETURN_TYPES=('STRING','STRING')
    RETURN_NAMES=('camera_text','camera_json')
    FUNCTION='describe'
    OUTPUT_NODE=True
    DESCRIPTION='Describe the actual render camera in English and structured JSON.'
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'pose':('DKST_POSE',)}}
    @classmethod
    def IS_CHANGED(cls,**kwargs):return 'camera-info-ui-2'
    def describe(self,pose):
        from .storage import load_render
        if not isinstance(pose,dict) or set(pose)!={'scene_json','render_id'}:raise ValueError('Connect the pose output from DKST 3D Pose')
        image=load_render(pose['scene_json'],pose['render_id']);image.close()
        doc=camera_document(validate_scene(pose['scene_json']))
        text=camera_text(doc)
        raw=json.dumps(doc,allow_nan=False)
        return {'ui':{'text':[text],'camera_text':[text],'camera_json':[raw]},'result':(text,raw)}
