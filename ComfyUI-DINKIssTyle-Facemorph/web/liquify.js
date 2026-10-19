// Kept numerically equivalent to core/liquify.py; applied to the finished composite.
export const EYE_CONTROLS=[['size','Eye size'],['height','Eye height'],['width','Eye width'],['tilt','Eye tilt']];
export const SECTIONS=[
 ['Nose',[['nose_height','Nose height'],['nose_width','Nose width']]],
 ['Mouth',[['smile','Smile'],['upper_lip','Upper lip'],['lower_lip','Lower lip'],['mouth_width','Mouth width'],['mouth_height','Mouth height']]],
 ['Face shape',[['forehead','Forehead'],['chin_height','Chin height'],['jawline','Jawline'],['face_width','Face width']]],
];
export function adjustment(settings,key,side){return Number(side?settings[side]?.[key]??settings.eyes?.[key]??0:settings[key]??0);}
export function setAdjustment(settings,key,value,side){
 if(!side){settings[key]=value;return;}
 settings[side]??={};settings[side][key]=value;
 if(side==='eyes')for(const child of ['eye_l','eye_r']){
  if(settings[child]){delete settings[child][key];if(!Object.keys(settings[child]).length)delete settings[child];}
 }
}
export function buildWarp(target,size,settings={}){
 if(settings.enabled===false)return null;
 const points=target.map(p=>[p[0]*(size[0]-1),p[1]*(size[1]-1)]);
 const avg=ids=>ids.reduce((a,i)=>[a[0]+points[i][0]/ids.length,a[1]+points[i][1]/ids.length],[0,0]);
 const left=avg([33,133]),right=avg([362,263]);
 const scale=Math.hypot(right[0]-left[0],right[1]-left[1]);
 if(scale<1e-6)throw new Error('Eye spacing is too small for Face Liquify.');
 const axis=[(right[0]-left[0])/scale,(right[1]-left[1])/scale],down=[-axis[1],axis[0]],origin=left.map((v,d)=>(v+right[d])/2);
 const point=i=>{const d=points[i].map((v,j)=>v-origin[j]);return [(d[0]*axis[0]+d[1]*axis[1])/scale,(d[0]*down[0]+d[1]*down[1])/scale];};
 const mean=ids=>ids.reduce((a,i)=>{const p=point(i);return [a[0]+p[0]/ids.length,a[1]+p[1]/ids.length];},[0,0]);
 const ops=[],get=key=>(settings[key]??0)/100,eyes=settings.eyes??{};
 function add(center,radius,sx=1,sy=1,angle=0,dx=0,dy=0){
  if(Math.max(Math.abs(sx-1),Math.abs(sy-1),Math.abs(angle),Math.abs(dx),Math.abs(dy))<1e-12)return;
  const c=Math.cos(angle),s=Math.sin(angle);ops.push({center,radius,matrix:[c/sx,s/sx,-s/sy,c/sy],shift:[-dx,-dy]});
 }
 for(const [side,ids,sign] of [['eye_l',[33,133],-1],['eye_r',[362,263],1]]){
  const v=key=>(settings[side]?.[key]??eyes[key]??0)/100;
  add(mean(ids),[.48,.32],1+.30*v('size')+.22*v('width'),1+.30*v('size')+.25*v('height'),-sign*v('tilt')*Math.PI/15,sign*(eyes.distance??0)/100*.13);
 }
 add(mean([1,2,98,327]),[.52,.65],1+.30*get('nose_width'),1+.25*get('nose_height'));
 add(mean([61,291]),[.72,.44],1+.30*get('mouth_width'),1+.30*get('mouth_height'));
 for(const [i,sign] of [[61,-1],[291,1]])add(point(i),[.30,.30],1,1,0,sign*.025*get('smile'),-.13*get('smile'));
 add(mean([0,13]),[.58,.18],1,1+.25*get('upper_lip'),0,0,-.045*get('upper_lip'));
 add(mean([14,17]),[.58,.18],1,1+.25*get('lower_lip'),0,0,.045*get('lower_lip'));
 add(point(10),[1.10,.65],1,1,0,0,-.22*get('forehead'));
 add(point(152),[.80,.65],1,1,0,0,.22*get('chin_height'));
 for(const [i,sign] of [[172,-1],[397,1]])add(point(i),[.70,.70],1,1,0,sign*.18*get('jawline'));
 add(mean([234,454]),[1.65,2.0],1+.20*get('face_width'));
 return ops.length?{origin,axis,scale,size,ops}:null;
}
export const HAIR_PIN=`
uniform sampler2D protectionMask;uniform vec2 protectionSize;uniform float protectHair;
float distanceAt(vec2 p){return texture2D(protectionMask,(clamp(p,0.,1.)*(protectionSize-1.)+.5)/protectionSize).b*max(protectionSize.x,protectionSize.y)*.25;}
vec2 pinHair(vec2 dst,vec2 src){
 if(protectHair<.5)return src;
 float distance=min(min(distanceAt(dst),distanceAt(src)),min(distanceAt(mix(dst,src,.25)),min(distanceAt(mix(dst,src,.5)),distanceAt(mix(dst,src,.75)))));
 float travel=length((src-dst)*(protectionSize-1.));
 return mix(dst,src,smoothstep(0.,max(3.,3.*travel),distance));
}`;
export const WARP_FRAGMENT=`precision highp float;
uniform sampler2D photo;uniform vec2 outputSize;uniform vec2 referenceSize;
uniform vec2 origin;uniform vec2 axis;uniform float scale;uniform int opCount;
uniform vec4 regions[16];uniform vec4 matrices[16];uniform vec2 shifts[16];varying vec2 vUV;
${HAIR_PIN}
void main(){
 vec2 uv=(vUV*outputSize-0.5)/(outputSize-1.0),down=vec2(-axis.y,axis.x);
 vec2 d=uv*(referenceSize-1.0)-origin;
 vec2 q=vec2(dot(d,axis),dot(d,down))/scale,displacement=vec2(0.);
 for(int i=0;i<16;i++){
  if(i>=opCount)break;
  vec2 delta=q-regions[i].xy,r=delta/regions[i].zw;
  float f=max(0.,1.-dot(r,r));f*=f;
  vec4 m=matrices[i];vec2 transformed=vec2(m.x*delta.x+m.y*delta.y,m.z*delta.x+m.w*delta.y);
  displacement+=(transformed-delta+shifts[i])*f;
 }
 float pin=clamp(min(min(uv.x,1.-uv.x),min(uv.y,1.-uv.y))*25.,0.,1.);
 vec2 src=clamp(uv+(axis*displacement.x+down*displacement.y)*scale*pin/(referenceSize-1.),0.,1.);
 src=pinHair(uv,src);
 vec2 sampleUV=(src*(outputSize-1.)+0.5)/outputSize;
 gl_FragColor=texture2D(photo,vec2(sampleUV.x,1.-sampleUV.y));
}`;
