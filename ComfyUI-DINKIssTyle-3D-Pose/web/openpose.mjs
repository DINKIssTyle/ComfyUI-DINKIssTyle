import * as T from './vendor/three.module.mjs';
import {GRID_SPACING,GRID_MAJOR_EVERY,FLOOR_Y} from './floor.mjs';

// COCO_18 anatomical order (right/left refer to the person, never screen sides).
export const BODY_JOINTS=[null,null,'R_shoulder','R_elbow','R_wrist','L_shoulder','L_elbow','L_wrist',
    'R_hip','R_knee','R_ankle','L_hip','L_knee','L_ankle',null,null,null,null];
const FINGERS=['thumb','index','middle','ring','pinky'];
const round=n=>Math.round(n*1e8)/1e8;

export function projectPoint(point,camera) {
    const view=point.clone().applyMatrix4(camera.matrixWorldInverse);
    if(view.z>=-camera.near||view.z<-camera.far)return [0,0,0];
    const ndc=point.clone().project(camera);
    if(![ndc.x,ndc.y,ndc.z].every(Number.isFinite)||Math.abs(ndc.x)>1||Math.abs(ndc.y)>1||Math.abs(ndc.z)>1)return [0,0,0];
    return [round((ndc.x+1)/2),round((1-ndc.y)/2),1];
}
export function projectSegment(a,b,camera) {
    const matrix=new T.Matrix4().multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse);
    const start=new T.Vector4(a.x,a.y,a.z,1).applyMatrix4(matrix),end=new T.Vector4(b.x,b.y,b.z,1).applyMatrix4(matrix);
    const planes=v=>[v.w+v.x,v.w-v.x,v.w+v.y,v.w-v.y,v.w+v.z,v.w-v.z];
    const one=planes(start),two=planes(end);let low=0,high=1;
    for(let i=0;i<6;i++){
        if(one[i]<0&&two[i]<0)return null;
        if(one[i]<0)low=Math.max(low,one[i]/(one[i]-two[i]));
        if(two[i]<0)high=Math.min(high,one[i]/(one[i]-two[i]));
    }
    if(low>high)return null;
    const xy=t=>{const p=start.clone().lerp(end,t);if(Math.abs(p.w)<1e-8)return null;return [round(T.MathUtils.clamp((p.x/p.w+1)/2,0,1)),round(T.MathUtils.clamp((1-p.y/p.w)/2,0,1))];};
    const first=xy(low),last=xy(high);return first&&last?[...first,...last]:null;
}
export function projectFloor(camera) {
    const extent=8,count=Math.round(extent/GRID_SPACING),lines=[];
    for(let i=-count;i<=count;i++)for(const direction of [0,1]){
        const coord=i*GRID_SPACING;
        const a=direction?new T.Vector3(-extent,FLOOR_Y,coord):new T.Vector3(coord,FLOOR_Y,-extent);
        const b=direction?new T.Vector3(extent,FLOOR_Y,coord):new T.Vector3(coord,FLOOR_Y,extent);
        const projected=projectSegment(a,b,camera);if(projected)lines.push([...projected,i%GRID_MAJOR_EVERY===0?1:0]);
    }
    return lines;
}
export function projectOpenPose(world) {
    const rig=world.rig,camera=world.camera;
    rig.root.updateMatrixWorld(true);camera.updateMatrixWorld(true);
    const position=name=>rig.joints[name].getWorldPosition(new T.Vector3());
    const project=p=>projectPoint(p,camera);
    const body=BODY_JOINTS.map(name=>name?project(position(name)):[0,0,0]);
    body[1]=project(position('L_shoulder').add(position('R_shoulder')).multiplyScalar(.5));
    const head=rig.joints.head,h=rig.p.head,center=position('head');
    const headRotation=head.getWorldQuaternion(new T.Quaternion());
    const towardCamera=camera.position.clone().sub(center).normalize();
    const facing=new T.Vector3(0,0,1).applyQuaternion(headRotation).dot(towardCamera);
    const facial=(local,normal=null)=>{
        // These are synthetic surface landmarks. Hide the back of the head and
        // the far eye/ear in profile; body and hands deliberately remain X-ray.
        if(facing<-.25)return [0,0,0];
        if(normal&&new T.Vector3(...normal).normalize().applyQuaternion(headRotation).dot(towardCamera)<=0)return [0,0,0];
        return project(head.localToWorld(new T.Vector3(...local).multiplyScalar(h)));
    };
    body[0]=facial([0,-.10,.385]);
    body[14]=facial([-.16,.09,.31],[-.55,0,.85]);
    body[15]=facial([.16,.09,.31],[.55,0,.85]);
    body[16]=facial([-.34,.025,0],[-1,0,.35]);
    body[17]=facial([.34,.025,0],[1,0,.35]);
    const hand=side=>{
        if(world.state.appearance.simple_hands)return Array(63).fill(0);
        const points=[project(position(side+'_wrist'))];
        for(const finger of FINGERS){
            for(let i=0;i<3;i++)points.push(project(position(`${side}_${finger}_${i}`)));
            const tip=rig.fingerTips[`${side}_${finger}`];points.push(project(tip.joint.localToWorld(tip.local.clone())));
        }
        return points.flat();
    };
    const document={version:1.3,canvas_width:world.state.output.width,canvas_height:world.state.output.height,people:[{
        pose_keypoints_2d:body.flat(),face_keypoints_2d:[],hand_left_keypoints_2d:hand('L'),hand_right_keypoints_2d:hand('R'),
    }]};
    if(world.state.appearance.floor)document.floor_grid=projectFloor(camera);
    return document;
}
