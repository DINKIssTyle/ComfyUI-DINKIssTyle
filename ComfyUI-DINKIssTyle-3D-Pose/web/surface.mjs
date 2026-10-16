import * as T from './vendor/three.module.mjs';
import {authoredGeometry,prepareAuthoredRig} from './human_model.mjs';
const cache=new Map();

export function installSurface(rig,state){
    rig.root.position.set(0,0,0);for(const bone of Object.values(rig.joints))bone.quaternion.identity();
    rig.joints.L_shoulder.rotation.z=Math.PI/2;rig.joints.R_shoulder.rotation.z=-Math.PI/2;
    rig.root.updateMatrixWorld(true);
    const restoreNeutral=prepareAuthoredRig(rig);
    const bones=Object.values(rig.joints);
    const key=`${state.ratio}:${state.appearance.simple_hands}:${state.appearance.simple_feet}`;
    let data=cache.get(key);
    if(!data){data=authoredGeometry(rig,state,bones);cache.set(key,data);if(cache.size>8){const first=cache.keys().next().value;cache.delete(first);}}
    const geometry=data.geometry.clone();rig.resources.push(geometry);
    rig.surfaceMaterial=rig.material.clone();rig.resources.push(rig.surfaceMaterial);
    const uniforms={centerlines:{value:0},contours:{value:0},markers:{value:0},face:{value:0},lineWidth:{value:1.4},guideColor:{value:new T.Color()}};
    rig.surfaceUniforms=uniforms;
    rig.surfaceMaterial.onBeforeCompile=shader=>{
        Object.assign(shader.uniforms,uniforms);
        shader.vertexShader=shader.vertexShader.replace('#include <common>','#include <common>\nattribute vec4 guide; attribute float guideRegion; varying vec4 vGuide; varying float vRegion;').replace('#include <begin_vertex>','#include <begin_vertex>\nvGuide=guide;vRegion=guideRegion;');
        shader.fragmentShader=shader.fragmentShader.replace('#include <common>','#include <common>\nvarying vec4 vGuide; varying float vRegion; uniform float centerlines,contours,markers,face,lineWidth; uniform vec3 guideColor;')
            .replace('#include <color_fragment>',`#include <color_fragment>
            float front=step(0.0,vGuide.z);
            float dx=min(fwidth(vGuide.x),.003);
            float dy=min(fwidth(vGuide.y),.003);
            float dw=min(fwidth(vGuide.w),.003);
            float central=1.0-smoothstep(dx*lineWidth*.5,dx*(lineWidth*.5+1.0),abs(vGuide.x));
            float contour=1.0-smoothstep(dy*lineWidth*.5,dy*(lineWidth*.5+1.0),abs(vGuide.y));
            float joint=1.0-smoothstep(dw*lineWidth*.7,dw*(lineWidth*.7+1.0),vGuide.w);
            float head=step(.5,vRegion);
            float centerEnabled=mix(centerlines,face,head);
            float crossEnabled=mix(contours,face,head);
            diffuseColor.rgb=mix(diffuseColor.rgb,vec3(.29,.34,.40),contour*crossEnabled*.65);
            diffuseColor.rgb=mix(diffuseColor.rgb,guideColor,max(central*front*centerEnabled,joint*markers*(1.0-head)*.65));`);
    };
    const skeleton=new T.Skeleton(bones);rig.resources.push(skeleton);
    const mesh=new T.SkinnedMesh(geometry,rig.surfaceMaterial);mesh.bind(skeleton);mesh.userData.surface=true;rig.root.add(mesh);rig.surface=mesh;
    const outlineMaterial=new T.ShaderMaterial({side:T.BackSide,depthWrite:false,uniforms:{resolution:{value:new T.Vector2(768,1024)},lineWidth:{value:1.4}},vertexShader:`
    #include <common>
    #include <morphtarget_pars_vertex>
    #include <skinning_pars_vertex>
    uniform vec2 resolution; uniform float lineWidth;
    void main(){
     #include <beginnormal_vertex>
     #include <morphnormal_vertex>
     #include <skinbase_vertex>
     #include <skinnormal_vertex>
     #include <defaultnormal_vertex>
     #include <begin_vertex>
     #include <morphtarget_vertex>
     #include <skinning_vertex>
     #include <project_vertex>
     vec2 direction=transformedNormal.xy;
     direction/=max(length(direction),.0001);
     gl_Position.xy+=direction*lineWidth*2.0/resolution*gl_Position.w;
     gl_Position.z+=.00002*gl_Position.w;
    }`,fragmentShader:'void main(){gl_FragColor=vec4(.08,.12,.18,1.0);}',toneMapped:false});
    rig.resources.push(outlineMaterial);
    const outline=new T.SkinnedMesh(geometry,outlineMaterial);outline.bind(skeleton);outline.userData.outline=true;outline.renderOrder=1;
    outline.onBeforeRender=renderer=>renderer.getDrawingBufferSize(outlineMaterial.uniforms.resolution.value);
    rig.root.add(outline);rig.outline=outline;
    outline.frustumCulled=false;
    rig.correctiveNames=data.correctiveNames;
    // Keep the existing rig/handles, remove the rigid construction meshes.
    const simplified=[];
    for(const side of ['L','R']){if(state.appearance.simple_hands)rig.hands[side].simple.traverse(o=>{if(o.isMesh)simplified.push(o);});if(state.appearance.simple_feet)rig.feet[side].simple.traverse(o=>{if(o.isMesh)simplified.push(o);});}
    for(const m of rig.pickables)if(!rig.marks.includes(m)&&!simplified.includes(m))m.removeFromParent();
    rig.simplePickables=simplified;rig.face.visible=false;rig.pickables=[mesh,...simplified,...rig.marks];
    restoreNeutral();rig.apply(state);
}

export function updateSurface(rig,state){
    if(!rig.surface)return;
    rig.face.visible=false;
    const a=state.appearance,u=rig.surfaceUniforms;
    rig.surfaceMaterial.color.set(a.body);u.centerlines.value=a.centerlines?1:0;u.contours.value=a.contours?1:0;u.markers.value=a.joint_markers?1:0;
    u.face.value=a.face?1:0;u.lineWidth.value=a.line_width;u.guideColor.value.set(a.guide_color);
    rig.outline.visible=a.outline;rig.outline.material.uniforms.lineWidth.value=a.line_width;
    for(let i=0;i<rig.correctiveNames.length;i++){
        const q=rig.joints[rig.correctiveNames[i]].quaternion,angle=2*Math.acos(Math.min(1,Math.abs(q.w)));
        rig.surface.morphTargetInfluences[i]=Math.sin(Math.min(Math.PI,angle)/2);
        rig.outline.morphTargetInfluences[i]=rig.surface.morphTargetInfluences[i];
    }
    rig.surface.skeleton.update();rig.surface.computeBoundingBox();rig.surface.computeBoundingSphere();
}
