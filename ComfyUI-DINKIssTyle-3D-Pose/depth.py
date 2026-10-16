"""Convert immutable linear camera distances into conditioning images."""
import numpy as np
import torch
from .storage import load_depth


class DepthImage:
    CATEGORY='DINKIssTyle/3D Pose'
    RETURN_TYPES=('IMAGE','MASK')
    RETURN_NAMES=('image','mask')
    FUNCTION='render'
    DESCRIPTION='Render linear camera depth. Near white is the default. Mask is 1 for background.'
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'pose':('DKST_POSE',),
            'range_mode':(['Scene range','Manual range'],),
            'near_distance':('FLOAT',{'default':.5,'min':.01,'max':200,'step':.01}),
            'far_distance':('FLOAT',{'default':10,'min':.01,'max':200,'step':.01}),
            'invert':('BOOLEAN',{'default':False,'tooltip':'Switch to near black / far white. Background stays black.'})}}
    def render(self,pose,range_mode='Scene range',near_distance=.5,far_distance=10,invert=False):
        if not isinstance(pose,dict) or set(pose)!={'scene_json','render_id'}:raise ValueError('Connect the pose output from DKST 3D Pose')
        depth=load_depth(pose['scene_json'],pose['render_id']);valid=depth>0
        if range_mode not in ('Scene range','Manual range'):raise ValueError('Invalid depth range mode')
        if type(invert) is not bool:raise ValueError('Invalid depth inversion')
        if range_mode=='Manual range':
            if not np.isfinite([near_distance,far_distance]).all() or not .01<=near_distance<far_distance<=200:raise ValueError('Near distance must be smaller than far distance (0.01–200).')
            lo,hi=near_distance,far_distance
        elif valid.any():lo,hi=depth[valid].min(),depth[valid].max()
        else:lo,hi=0,1
        value=np.clip((depth-lo)/(hi-lo),0,1) if hi>lo else np.zeros_like(depth)
        if not invert:value=1-value
        value[~valid]=0
        image=np.repeat(value[...,None],3,axis=-1).astype(np.float32)
        return torch.from_numpy(image).unsqueeze(0),torch.from_numpy((~valid).astype(np.float32)).unsqueeze(0)
