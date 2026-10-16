import * as T from './vendor/three.module.mjs';
export const DEFORM_VERTEX=`
#include <common>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
varying float vDistance;
void main(){
 #include <skinbase_vertex>
 #include <begin_vertex>
 #include <morphtarget_vertex>
 #include <skinning_vertex>
 #include <project_vertex>
 vDistance=-mvPosition.z;
}`;
export function depthMaterial(){return new T.ShaderMaterial({vertexShader:DEFORM_VERTEX,fragmentShader:`
varying float vDistance;
void main(){
 float d=floor(clamp(vDistance/200.0,0.0,1.0)*16777215.0+.5);
 gl_FragColor=vec4(floor(d/65536.0),mod(floor(d/256.0),256.0),mod(d,256.0),255.0)/255.0;
}`,toneMapped:false});}

// RGB24 packing is portable across WebGL2 devices and is decoded to linear
// Float32 camera distances. Zero means background; all surfaces are >= near.
export function renderDepth(renderer,world,width,height,camera=world.camera){
    const target=new T.WebGLRenderTarget(width,height,{depthBuffer:true,stencilBuffer:false});
    target.texture.colorSpace=T.NoColorSpace;
    const material=depthMaterial(),scene=world.scene,oldOverride=scene.overrideMaterial,oldBG=scene.background;
    const previousTarget=renderer.getRenderTarget(),color=renderer.getClearColor(new T.Color()),alpha=renderer.getClearAlpha();
    const hidden=[];scene.traverse(o=>{if(o.visible&&(o.userData.helper||o.userData.outline||o===world.rig.face||world.rig.marks.includes(o))){hidden.push(o);o.visible=false;}});
    try{
        scene.overrideMaterial=material;scene.background=null;renderer.setClearColor(0,0);renderer.setRenderTarget(target);renderer.render(scene,camera);
        const packed=new Uint8Array(width*height*4);renderer.readRenderTargetPixels(target,0,0,width,height,packed);
        const data=new Float32Array(width*height);
        for(let y=0;y<height;y++)for(let x=0;x<width;x++){
            const a=((height-y-1)*width+x)*4;
            if(packed[a+3])data[y*width+x]=(packed[a]*65536+packed[a+1]*256+packed[a+2])*200/16777215;
        }
        return data;
    }finally{scene.overrideMaterial=oldOverride;scene.background=oldBG;hidden.forEach(o=>o.visible=true);renderer.setRenderTarget(previousTarget);renderer.setClearColor(color,alpha);target.dispose();material.dispose();}
}
export function drawDepthPreview(renderer,world){
    const canvas=renderer.domElement,w=canvas.width,h=canvas.height,data=renderDepth(renderer,world,w,h);
    let lo=Infinity,hi=0;for(const d of data)if(d>0){lo=Math.min(lo,d);hi=Math.max(hi,d);}
    const image=new Uint8Array(w*h*4);
    for(let y=0;y<h;y++)for(let x=0;x<w;x++){const d=data[y*w+x],v=d>0?Math.round(255*(hi>lo?1-(d-lo)/(hi-lo):1)):0,j=((h-y-1)*w+x)*4;image[j]=image[j+1]=image[j+2]=v;image[j+3]=255;}
    const texture=new T.DataTexture(image,w,h,T.RGBAFormat);texture.colorSpace=T.SRGBColorSpace;texture.needsUpdate=true;
    const s=new T.Scene(),c=new T.OrthographicCamera(-1,1,1,-1,0,1),g=new T.PlaneGeometry(2,2),m=new T.MeshBasicMaterial({map:texture,toneMapped:false});s.add(new T.Mesh(g,m));
    renderer.render(s,c);texture.dispose();g.dispose();m.dispose();
}
