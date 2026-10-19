"""Cached face/hair segmentation and reproducible manual mask corrections."""
import hashlib
import json
import re
import numpy as np
from PIL import Image
from .topology import OVAL

MODEL_NAME = 'selfie_multiclass.tflite'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite'
MODEL_SHA256 = 'c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0'
REVISION = 1


def validate(settings, edits=None):
    if not isinstance(settings, dict) or set(settings)-{'enabled','protect_base'}:
        raise ValueError('Invalid hair protection settings.')
    if any(not isinstance(v,bool) for v in settings.values()):
        raise ValueError('Hair protection options must be boolean.')
    if edits is None:
        return
    if not isinstance(edits,dict) or set(edits)-{'strokes','forehead','temple','feather'}:
        raise ValueError('Invalid mask corrections.')
    for key,maximum,default in [('forehead',100,0),('temple',100,0),('feather',8,1)]:
        v=edits.get(key,default)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=maximum:
            raise ValueError('Invalid mask boundary control.')
    strokes=edits.get('strokes',[])
    if not isinstance(strokes,list) or len(strokes)>200:
        raise ValueError('A photo supports up to 200 mask strokes.')
    count=0
    for stroke in strokes:
        if not isinstance(stroke,dict) or set(stroke)!={'mode','radius','points'} or stroke['mode'] not in ('include','exclude','restore'):
            raise ValueError('Invalid mask stroke.')
        r=stroke['radius']
        if isinstance(r,bool) or not isinstance(r,(int,float)) or not np.isfinite(r) or not .002<=r<=.15:
            raise ValueError('Invalid brush radius.')
        p=np.asarray(stroke['points'],dtype=float)
        count+=len(p) if p.ndim else 0
        if p.ndim!=2 or p.shape[1]!=2 or not 1<=len(p)<=2000 or count>12000 or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
            raise ValueError('Invalid mask stroke points.')


def token(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def mask_path(service, identifier):
    if not isinstance(identifier,str) or not re.fullmatch('[0-9a-f]{64}',identifier):
        raise ValueError('Invalid mask identifier.')
    p=service.assets/'masks'/(identifier+'.png')
    if p.is_symlink():
        raise ValueError('Linked masks are not supported.')
    return p


def store(service, identifier, array):
    path=mask_path(service,identifier)
    if not path.exists():
        import uuid
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.'+uuid.uuid4().hex+'.part')
        try:
            Image.fromarray(np.uint8(np.clip(array,0,1)*255+.5)).save(temp,format='PNG')
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    return identifier


def read(service, identifier):
    with Image.open(mask_path(service,identifier)) as im:
        return np.asarray(im.convert('RGB'),dtype=np.float32)/255


def automatic(service, asset):
    """Segment a selected-head crop, retaining full-photo coordinates in the cache."""
    import cv2
    pts=np.asarray(asset['points'])[:468]
    # Crop is tied to the selected face and current correspondence points.
    lo=pts.min(0);hi=pts.max(0);span=hi-lo
    lo=np.maximum(0,lo-span*[.45,.65]);hi=np.minimum(1,hi+span*[.45,.25])
    identifier=token(['hair',REVISION,MODEL_SHA256,asset['id'],lo.tolist(),hi.tolist()])
    if mask_path(service,identifier).exists():
        return identifier
    raw=service.image(asset['id']);h,w=raw.shape[:2]
    x0,y0=np.floor(lo*[w-1,h-1]).astype(int);x1,y1=np.ceil(hi*[w-1,h-1]).astype(int)+1
    if x1-x0<8 or y1-y0<8:
        raise ValueError('Face crop is too small for hair analysis. Reset correspondence points.')
    crop=raw[y0:y1,x0:x1]
    scale=min(1,768/max(crop.shape[:2]));crop=cv2.resize(crop,(max(2,round(crop.shape[1]*scale)),max(2,round(crop.shape[0]*scale))))
    mp=service._mediapipe()
    result=service._hair_segmenter().segment(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(crop)))
    face=np.array(result.confidence_masks[3].numpy_view(),copy=True)
    hair=np.array(result.confidence_masks[1].numpy_view(),copy=True)
    # Make confident hair fully opaque, preserving a soft transition at ambiguous edges.
    hair=np.clip((hair-.15)/.65,0,1)
    face=np.clip((face-.08)/.65,0,1)*(1-hair)
    edge=768;mw=max(2,round(w*min(1,edge/max(w,h))));mh=max(2,round(h*min(1,edge/max(w,h))))
    out=np.zeros((mh,mw,3),np.float32)
    a,b=np.floor(lo*[mw-1,mh-1]).astype(int);c,d=np.ceil(hi*[mw-1,mh-1]).astype(int)+1
    out[b:d,a:c,0]=cv2.resize(face,(c-a,d-b));out[b:d,a:c,1]=cv2.resize(hair,(c-a,d-b))
    return store(service,identifier,out)


def corrected(raw, points, edits):
    import cv2
    validate({},edits)
    result=raw.copy();h,w=raw.shape[:2]
    oval=np.asarray(points)[OVAL]*[w-1,h-1];lo=oval.min(0);hi=oval.max(0);center=(lo+hi)/2;span=hi-lo
    forehead=edits.get('forehead',0)/100;temple=edits.get('temple',0)/100
    if forehead or temple:
        oval[:,1]+=span[1]*.2*forehead*np.clip((center[1]-oval[:,1])/max(span[1]*.5,1),0,1)**2
        oval[:,0]=center[0]+(oval[:,0]-center[0])*(1-.2*temple)
        boundary=np.zeros((h,w),np.uint8);cv2.fillPoly(boundary,[np.round(oval).astype(np.int32)],255)
        result[:,:,0]*=boundary/255
    for stroke in edits.get('strokes',[]):
        p=np.asarray(stroke['points'])*[w-1,h-1];r=stroke['radius']*max(w,h)
        # Analytic capsules give consistent round ends regardless of pointer event spacing.
        pairs=list(zip(p[:-1],p[1:])) or [(p[0],p[0])]
        for a,b in pairs:
            lo=np.maximum(0,np.floor(np.minimum(a,b)-r).astype(int));hi=np.minimum([w-1,h-1],np.ceil(np.maximum(a,b)+r).astype(int))
            x,y=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1));q=np.stack([x,y],-1)-a;v=b-a
            t=np.clip((q*v).sum(-1)/max(float(v@v),1e-8),0,1)
            hit=np.linalg.norm(q-t[:,:,None]*v,axis=-1)<=r
            region=(slice(lo[1],hi[1]+1),slice(lo[0],hi[0]+1))
            if stroke['mode']=='restore':result[region][hit]=raw[region][hit]
            else:result[region][hit]=[1,0,0] if stroke['mode']=='include' else [0,1,0]
    sigma=edits.get('feather',1)*max(w,h)/768
    if sigma>0:
        result=cv2.GaussianBlur(result,(0,0),max(sigma,.01))
    return np.clip(result,0,1)


def prepare(service, assets):
    import cv2
    ids=[];raw_ids=[];visibility=[]
    with service.lock:
        for asset in assets:
            edits=asset.get('mask_edit',{});validate({},edits)
            raw_id=automatic(service,asset)
            identifier=token(['corrected',REVISION,raw_id,asset['points'],edits])
            if not mask_path(service,identifier).exists():
                store(service,identifier,corrected(read(service,raw_id),asset['points'],edits))
            mask=read(service,identifier);h,w=mask.shape[:2]
            smooth=cv2.GaussianBlur(mask[:,:,1],(0,0),max(1,max(w,h)*.015))
            pts=np.asarray(asset['points'],np.float32)
            covered=cv2.remap(smooth,(pts[:,0]*(w-1)).reshape(1,-1),(pts[:,1]*(h-1)).reshape(1,-1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).ravel()
            score=np.maximum(.1,1-covered);score[468:]=1
            ids.append(identifier);raw_ids.append(raw_id);visibility.append(score.tolist())
    return ids,raw_ids,visibility
