"""Optional source crops. Margin is a percentage of face size on each side."""
import math
import numpy as np


def validate(settings):
    if not isinstance(settings,dict) or set(settings)-{'enabled','margin'}:
        raise ValueError('Invalid face crop settings.')
    if not isinstance(settings.get('enabled',False),bool):
        raise ValueError('Face crop enabled must be boolean.')
    margin=settings.get('margin',30)
    if isinstance(margin,bool) or not isinstance(margin,(int,float)) or not math.isfinite(margin) or not 0<=margin<=200:
        raise ValueError('Face crop margin must be between 0 and 200 percent.')


def bounds(asset,size):
    settings=asset.get('crop',{});validate(settings)
    width,height=size
    if not settings.get('enabled',False):
        return [0,0,width,height]
    p=np.asarray(asset['points'])[:468]*[width-1,height-1]
    lo=np.maximum(p.min(0),0);hi=np.minimum(p.max(0)+1,[width,height])
    span=hi-lo
    if (span<=0).any():
        raise ValueError('Face crop lies outside the photo. Reset correspondence points.')
    extent=np.minimum([width,height],span*(1+settings.get('margin',30)/50))
    start=np.minimum(np.maximum(0,(lo+hi-extent)/2),np.array([width,height])-extent)
    end=np.minimum([width,height],np.ceil(start+extent)).astype(int)
    start=np.maximum(0,np.floor(start)).astype(int)
    if (end-start<2).any():
        raise ValueError('Face crop must be at least 2 pixels in each dimension.')
    return [int(start[0]),int(start[1]),int(end[0]),int(end[1])]


def points(original,bbox,size):
    left,top,right,bottom=bbox
    result=(np.asarray(original)*[size[0]-1,size[1]-1]-[left,top])/[right-left-1,bottom-top-1]
    # The frame protects the source canvas, rather than the uncropped photograph.
    result[-8:]=[[0,0],[.5,0],[1,0],[1,.5],[1,1],[.5,1],[0,1],[0,.5]]
    return result.tolist()


def sources(assets,sizes):
    boxes=[bounds(a,s) for a,s in zip(assets,sizes)]
    transformed=[{**a,'width':b[2]-b[0],'height':b[3]-b[1],'points':points(a['points'],b,s)} for a,b,s in zip(assets,boxes,sizes)]
    cropped_sizes=[[b[2]-b[0],b[3]-b[1]] for b in boxes]
    return transformed,boxes,cropped_sizes


def mask(service,identifier,bbox,size):
    from . import hair
    import cv2
    if bbox==[0,0,*size]:
        return identifier
    key=hair.token(['source-crop-mask',3,identifier,bbox,size])
    if not hair.mask_path(service,key).exists():
        raw=hair.read(service,identifier);mh,mw=raw.shape[:2]
        left,top,right,bottom=bbox;w,h=right-left,bottom-top
        factor=min(1,768/max(w,h));cw=max(2,round(w*factor));ch=max(2,round(h*factor))
        x,y=np.meshgrid(np.linspace(left,right-1,cw),np.linspace(top,bottom-1,ch))
        sampled=cv2.remap(raw,(x*(mw-1)/(size[0]-1)).astype(np.float32),(y*(mh-1)/(size[1]-1)).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        sampled[:,:,2]=hair.distance_field(sampled[:,:,1])
        hair.store(service,key,sampled)
    return key
