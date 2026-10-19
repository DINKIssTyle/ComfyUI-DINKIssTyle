"""Bounded-memory rasterizer using the browser's barycentric interpolation rules."""
import numpy as np


def render(images, assets, base, mesh, size, color_match=False, check_cancel=None, masks=None, liquify=None):
    import cv2
    w,h = size
    target = np.asarray(mesh['target']) * [w-1,h-1]
    tris = np.asarray(mesh['triangles'])
    output = np.empty((h,w,3),np.float32)
    mask = np.empty((h,w),np.float32)
    context = np.empty((h,w,3),np.float32)
    from .liquify import pin_hair
    from .hair import distance_field
    context_points=np.concatenate([np.asarray(assets[base]['points'])[:468],np.asarray(mesh['target'])[:468]])
    context_lo,context_hi=context_points.min(0),context_points.max(0)
    context_span=np.maximum(context_hi-context_lo,1e-5)
    protection=np.floor(distance_field(masks[base][:,:,1])*255+.5)/255 if masks is not None and mesh.get('protect_base',True) and np.any(masks[base][:,:,1]>.02) else None
    # Match colors once per source, never independently per tile (avoids tile seams).
    corrections = [(np.ones(3),np.zeros(3)) for _ in images]
    if color_match:
        for i,image in enumerate(images):
            if i == base:
                continue
            def skin_pixels(index):
                im = images[index]
                face = np.asarray(assets[index]['points'])[:468]*[im.shape[1]-1,im.shape[0]-1]
                lo = np.maximum(face.min(0).astype(int),0)
                hi = np.minimum(face.max(0).astype(int),[im.shape[1]-1,im.shape[0]-1])
                crop = im[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
                if not crop.size:
                    raise ValueError('Face points lie outside the photo.')
                pixels=cv2.resize(crop,(128,128)).reshape(-1,3).astype(np.float32)/255
                if masks is not None:
                    mh,mw=masks[index].shape[:2]
                    xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],128),np.linspace(lo[1],hi[1],128))
                    skin=cv2.remap(masks[index][:,:,0],(xx*(mw-1)/max(im.shape[1]-1,1)).astype(np.float32),
                                   (yy*(mh-1)/max(im.shape[0]-1,1)).astype(np.float32),cv2.INTER_LINEAR)
                    pixels=pixels[skin.ravel()>.8]
                return pixels if len(pixels)>=32 else None
            ref,src = skin_pixels(base),skin_pixels(i)
            if ref is None or src is None:
                continue
            gain = np.clip(ref.std(0)/np.maximum(src.std(0),.02),.5,2)
            corrections[i] = gain,ref.mean(0)-src.mean(0)*gain
    for top in range(0,h,192):
        if check_cancel:
            check_cancel()
        bottom = min(h,top+192)
        th = bottom-top
        ids = np.zeros((th,w,3),np.int32)
        bary = np.zeros((th,w,3),np.float32)
        valid = np.zeros((th,w),bool)
        for tri in tris:
            t = target[tri]
            lo = np.maximum(np.floor(t.min(0)).astype(int),[0,top])
            hi = np.minimum(np.ceil(t.max(0)).astype(int),[w-1,bottom-1])
            if (hi < lo).any():
                continue
            x,y = np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
            m = np.column_stack((t[0]-t[2],t[1]-t[2]))
            if abs(np.linalg.det(m)) < 1e-8:
                continue
            uv = (np.stack((x,y),-1)-t[2]) @ np.linalg.inv(m).T
            b = np.concatenate((uv,1-uv.sum(-1,keepdims=True)),-1)
            inside = (b >= -1e-5).all(-1)
            region = (slice(lo[1]-top,hi[1]-top+1),slice(lo[0],hi[0]+1))
            ids[region][inside] = tri
            bary[region][inside] = b[inside]
            valid[region][inside] = True
        alpha = (np.asarray(mesh['alpha'])[ids]*bary).sum(-1).astype(np.float32)*valid
        mixed = np.zeros((th,w,3),np.float32)
        total = np.zeros((th,w),np.float32)
        uncovered = np.ones((th,w),np.float32)
        xx,yy=np.meshgrid(np.arange(w)/max(w-1,1),np.arange(top,bottom)/max(h-1,1))
        destination=np.stack((xx,yy),-1).astype(np.float32)
        for i,image in enumerate(images):
            if check_cancel:
                check_cancel()
            sh,sw = image.shape[:2]
            pts = np.asarray(assets[i]['points'],dtype=np.float32)*[sw-1,sh-1]
            uv = (pts[ids]*bary[:,:,:,None]).sum(2).astype(np.float32)
            weight = (np.asarray(mesh['feature_weights'][i])[ids]*bary).sum(-1).astype(np.float32)*valid
            # Clamp at photo boundaries, same as browser CLAMP_TO_EDGE.
            uv[:,:,0] = np.clip(uv[:,:,0],0,sw-1)
            uv[:,:,1] = np.clip(uv[:,:,1],0,sh-1)
            warped = cv2.remap(image,uv[:,:,0],uv[:,:,1],cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)/255
            visibility = np.ones((th,w),np.float32)
            if masks is not None:
                mh,mw=masks[i].shape[:2]
                visibility=cv2.remap(masks[i][:,:,0],uv[:,:,0]*(mw-1)/max(sw-1,1),
                                    uv[:,:,1]*(mh-1)/max(sh-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            uncovered *= 1-visibility*(weight>1e-8)
            weight *= visibility
            if i==base:
                mapped=uv/np.array([sw-1,sh-1],np.float32)
                mapped=np.where(valid[:,:,None],mapped,destination)
                outside=np.maximum(np.maximum(context_lo-destination,destination-context_hi),0)/context_span
                fade=np.clip(np.max(outside,axis=-1)/.25,0,1)
                influence=1-fade*fade*(3-2*fade)
                mapped=destination+(mapped-destination)*influence[:,:,None]
                mapped=pin_hair(destination,mapped,protection).astype(np.float32)
                context[top:bottom]=cv2.remap(image,mapped[:,:,0]*(sw-1),mapped[:,:,1]*(sh-1),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)/255
            gain,offset = corrections[i]
            mixed += (warped*gain+offset)*weight[:,:,None]
            total += weight
        mixed /= np.maximum(total[:,:,None],1e-8)
        alpha *= (1-uncovered)*(total>1e-8)
        output[top:bottom] = np.clip(mixed,0,1)*alpha[:,:,None]
        mask[top:bottom] = np.clip(alpha,0,1)
    # The backing follows the same geometry: shrinking alpha must not reveal the old jaw.
    from .liquify import apply_liquify
    active=bool(np.any(mask>1e-8))
    reference=mesh.get('sizes',[size]*len(images))[base]
    output,mask=apply_liquify(output,mask,mesh['target'],reference,liquify or {},check_cancel,protection,premultiplied=True)
    if active:
        context,_=apply_liquify(context,None,mesh['target'],reference,liquify or {},check_cancel,protection)
    output=np.minimum(np.maximum(output,0),mask[:,:,None])
    bh,bw=images[base].shape[:2]
    for top in range(0,h,192):
        if check_cancel:check_cancel()
        bottom=min(h,top+192)
        xx,yy=np.meshgrid(np.arange(w),np.arange(top,bottom))
        u=(xx*(bw-1)/max(w-1,1)).astype(np.float32);v=(yy*(bh-1)/max(h-1,1)).astype(np.float32)
        base_image=cv2.remap(images[base],u,v,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)/255
        if masks is not None and mesh.get('protect_base',True):
            mh,mw=masks[base].shape[:2]
            hair=cv2.remap(masks[base][:,:,1],u*(mw-1)/max(bw-1,1),v*(mh-1)/max(bh-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        else:
            hair=np.zeros((bottom-top,w),np.float32)
        backing=context[top:bottom] if active else base_image
        composed=output[top:bottom]+backing*(1-mask[top:bottom,:,None])
        output[top:bottom]=np.clip(composed*(1-hair[:,:,None])+base_image*hair[:,:,None],0,1)
        mask[top:bottom]*=1-hair
    return output,mask
