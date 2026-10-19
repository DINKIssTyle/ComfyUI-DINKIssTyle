"""Local inverse warps shared by the face and its surrounding base context."""
import math
import numpy as np

EYE_KEYS = ('size','height','width','tilt','distance')
SCALARS = ('nose_height','nose_width','smile','upper_lip','lower_lip','mouth_width','mouth_height',
           'forehead','chin_height','jawline','face_width')


def validate(settings):
    if not isinstance(settings,dict) or set(settings)-({'enabled','eyes','eye_l','eye_r'}|set(SCALARS)):
        raise ValueError('Invalid Face Liquify settings.')
    if 'enabled' in settings and type(settings['enabled']) is not bool:
        raise ValueError('Face Liquify enabled flag must be a boolean.')
    def number(value):
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not -100<=value<=100:
            raise ValueError('Face Liquify values must be finite numbers between -100 and 100.')
    for key,value in settings.items():
        if key=='enabled':continue
        if key in ('eyes','eye_l','eye_r'):
            allowed=EYE_KEYS if key=='eyes' else EYE_KEYS[:4]
            if not isinstance(value,dict) or set(value)-set(allowed):raise ValueError('Invalid eye adjustment.')
            for item in value.values():number(item)
        else:number(value)


def build_warp(target, size, settings=None):
    settings={} if settings is None else settings
    validate(settings)
    if settings.get('enabled',True) is False:return None
    get=lambda key:settings.get(key,0)/100
    eyes=settings.get('eyes',{})
    points=np.asarray(target,dtype=float)*[size[0]-1,size[1]-1]
    left=points[[33,133]].mean(0);right=points[[362,263]].mean(0)
    scale=float(np.linalg.norm(right-left))
    if scale<1e-6:raise ValueError('Eye spacing is too small for Face Liquify.')
    xaxis=(right-left)/scale;yaxis=np.array([-xaxis[1],xaxis[0]])
    origin=(left+right)/2
    def point(index):
        d=points[index]-origin
        return [float((d[0]*xaxis[0]+d[1]*xaxis[1])/scale),float((d[0]*yaxis[0]+d[1]*yaxis[1])/scale)]
    def mean(ids):return np.mean([point(i) for i in ids],axis=0).tolist()
    ops=[]
    def add(center,radius,sx=1,sy=1,angle=0,dx=0,dy=0):
        c,s=math.cos(angle),math.sin(angle)
        matrix=[c/sx,s/sx,-s/sy,c/sy]
        if max(abs(sx-1),abs(sy-1),abs(angle),abs(dx),abs(dy))<1e-12:return
        ops.append({'center':center,'radius':radius,'matrix':matrix,'shift':[-dx,-dy]})
    for side,ids,sign in [('eye_l',[33,133],-1),('eye_r',[362,263],1)]:
        v=lambda key:settings.get(side,{}).get(key,eyes.get(key,0))/100
        add(mean(ids),[.48,.32],1+.30*v('size')+.22*v('width'),1+.30*v('size')+.25*v('height'),
            -sign*v('tilt')*math.pi/15,sign*eyes.get('distance',0)/100*.13)
    add(mean([1,2,98,327]),[.52,.65],1+.30*get('nose_width'),1+.25*get('nose_height'))
    mouth=mean([61,291])
    add(mouth,[.72,.44],1+.30*get('mouth_width'),1+.30*get('mouth_height'))
    for index,sign in [(61,-1),(291,1)]:add(point(index),[.30,.30],dx=sign*.025*get('smile'),dy=-.13*get('smile'))
    add(mean([0,13]),[.58,.18],sy=1+.25*get('upper_lip'),dy=-.045*get('upper_lip'))
    add(mean([14,17]),[.58,.18],sy=1+.25*get('lower_lip'),dy=.045*get('lower_lip'))
    add(point(10),[1.10,.65],dy=-.22*get('forehead'))
    add(point(152),[.80,.65],dy=.22*get('chin_height'))
    for index,sign in [(172,-1),(397,1)]:add(point(index),[.70,.70],dx=sign*.18*get('jawline'))
    add(mean([234,454]),[1.65,2.0],sx=1+.20*get('face_width'))
    if not ops:return None
    return {'origin':origin.tolist(),'axis':xaxis.tolist(),'scale':scale,'size':list(size),'ops':ops}


def source_coordinates(uv,warp):
    """Map normalized destination pixel centers to normalized source coordinates."""
    uv=np.asarray(uv,dtype=float)
    axis=np.array(warp['axis']);down=np.array([-axis[1],axis[0]])
    extent=np.array(warp['size'])-1
    d=uv*extent-np.array(warp['origin'])
    q=np.stack((d[...,0]*axis[0]+d[...,1]*axis[1],d[...,0]*down[0]+d[...,1]*down[1]),axis=-1)/warp['scale']
    displacement=np.zeros_like(q)
    for op in warp['ops']:
        delta=q-np.array(op['center'])
        r=delta/np.array(op['radius'])
        falloff=np.maximum(0,1-np.sum(r*r,axis=-1))**2
        m=op['matrix']
        transformed=np.stack((m[0]*delta[...,0]+m[1]*delta[...,1],m[2]*delta[...,0]+m[3]*delta[...,1]),axis=-1)
        change=transformed-delta+op['shift']
        displacement+=change*falloff[...,None]
    pin=np.clip(np.minimum(np.minimum(uv[...,0],1-uv[...,0]),np.minimum(uv[...,1],1-uv[...,1]))*25,0,1)
    displacement*=pin[...,None]
    result=uv+(displacement[...,0,None]*axis+displacement[...,1,None]*down)*warp['scale']/extent
    return np.clip(result,0,1)


def pin_hair(destination, source, protection):
    """Use a normalized hair-distance field to smoothly pin the displacement path."""
    if protection is None:
        return source
    import cv2
    h,w=protection.shape[:2]
    distance=np.ones(destination.shape[:-1],np.float32)*max(h,w)*.25
    for t in (0,.25,.5,.75,1):
        uv=destination+(source-destination)*t
        sampled=cv2.remap(protection,(uv[...,0]*(w-1)).astype(np.float32),
                          (uv[...,1]*(h-1)).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        distance=np.minimum(distance,sampled*max(h,w)*.25)
    travel=np.linalg.norm((source-destination)*[w-1,h-1],axis=-1)
    # A halo proportional to travel prevents the pinned field from folding at a hair edge.
    x=np.clip(distance/np.maximum(3,3*travel),0,1)
    mobility=x*x*(3-2*x)
    return destination+(source-destination)*mobility[...,None]


def apply_liquify(image,mask,target,reference_size,settings=None,check_cancel=None,protection=None,premultiplied=False):
    import cv2
    warp=build_warp(target,reference_size,settings)
    if warp is None:return image,mask
    h,w=image.shape[:2]
    output=np.empty_like(image);out_mask=np.empty_like(mask) if mask is not None else None
    for top in range(0,h,192):
        if check_cancel:check_cancel()
        bottom=min(h,top+192)
        x,y=np.meshgrid(np.arange(w)/max(w-1,1),np.arange(top,bottom)/max(h-1,1))
        destination=np.stack((x,y),-1)
        uv=pin_hair(destination,source_coordinates(destination,warp),protection).astype(np.float32)
        mx,my=uv[...,0]*(w-1),uv[...,1]*(h-1)
        output[top:bottom]=cv2.remap(image,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        if mask is not None:
            out_mask[top:bottom]=cv2.remap(mask,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    output[0]=image[0];output[-1]=image[-1];output[:,0]=image[:,0];output[:,-1]=image[:,-1]
    if mask is not None:
        out_mask[0]=mask[0];out_mask[-1]=mask[-1];out_mask[:,0]=mask[:,0];out_mask[:,-1]=mask[:,-1]
    if premultiplied and out_mask is not None:
        # Cubic overshoot must scale color and alpha together to avoid bright/dark edge rims.
        factor=np.clip(out_mask,0,1)/np.maximum(out_mask,1e-8)
        output*=factor[:,:,None]
    return np.clip(output,0,1),np.clip(out_mask,0,1) if out_mask is not None else None
