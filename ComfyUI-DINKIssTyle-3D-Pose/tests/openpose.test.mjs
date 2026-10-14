import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
globalThis.fetch=async url=>({json:async()=>JSON.parse(await fs.readFile(url,'utf8'))});
const {DEFAULT,clone,quaternion,applyPreset,RATIOS}=await import('../web/state.mjs');
const {World}=await import('../web/renderer.mjs');
const {projectOpenPose,projectPoint,projectSegment}=await import('../web/openpose.mjs');
const T=await import('../web/vendor/three.module.mjs');
function world(ratio=8){const state=clone(DEFAULT);state.ratio=ratio;state.camera.projection='orthographic';state.camera.position=[0,1,5];state.camera.quaternion=[0,0,0,1];state.camera.ortho=2.4;return new World(state);}
const point=(values,i)=>values.slice(i*3,i*3+3);

test('COCO-18 and 21-point hands use anatomical right/left and image coordinates',()=>{
    for(const ratio of RATIOS){
        const w=world(ratio),doc=projectOpenPose(w),person=doc.people[0],body=person.pose_keypoints_2d;
        assert.equal(body.length,54);assert.equal(person.hand_left_keypoints_2d.length,63);assert.equal(person.hand_right_keypoints_2d.length,63);
        assert.ok(point(body,2)[0]<.5&&point(body,5)[0]>.5,'Person right appears on image left in front view');
        assert.ok(point(body,0)[1]<point(body,1)[1],'Nose must be above neck');
        assert.deepEqual(point(body,4),point(person.hand_right_keypoints_2d,0));
        assert.deepEqual(point(body,7),point(person.hand_left_keypoints_2d,0));
        const tip=w.rig.fingerTips.L_index;
        assert.deepEqual(point(person.hand_left_keypoints_2d,8),projectPoint(tip.joint.localToWorld(tip.local.clone()),w.camera));
        assert.ok(body.every(Number.isFinite));w.dispose();
    }
});
test('camera projection clips offscreen, behind-camera and near-plane points',()=>{
    const w=world();assert.deepEqual(projectPoint(new T.Vector3(0,1,0),w.camera),[.5,.5,1]);
    assert.deepEqual(projectPoint(new T.Vector3(100,1,0),w.camera),[0,0,0]);
    assert.deepEqual(projectPoint(new T.Vector3(0,1,6),w.camera),[0,0,0]);
    assert.deepEqual(projectPoint(new T.Vector3(0,1,4.999),w.camera),[0,0,0]);
    w.state.camera.projection='perspective';w.setState(w.state);
    assert.deepEqual(projectPoint(new T.Vector3(0,1,0),w.camera),[.5,.5,1]);w.dispose();
});
test('finger rotations and root position change projected points',()=>{
    const w=world(),before=projectOpenPose(w).people[0];w.state.joints.L_index_1=quaternion(-65,0,0);w.state.root=[.10,0,0];w.setState(w.state);
    const after=projectOpenPose(w).people[0];
    assert.notDeepEqual(point(before.hand_left_keypoints_2d,8),point(after.hand_left_keypoints_2d,8));
    assert.ok(point(after.pose_keypoints_2d,5)[0]>point(before.pose_keypoints_2d,5)[0]);w.dispose();
});
test('back-facing head has no false facial landmarks and profile hides far eye',()=>{
    const w=world();w.state.joints.head=quaternion(0,180,0);w.setState(w.state);
    const body=projectOpenPose(w).people[0].pose_keypoints_2d;
    for(const i of [0,14,15,16,17])assert.deepEqual(point(body,i),[0,0,0]);
    assert.equal(point(body,2)[2],1);
    w.state.joints.head=[0,0,0,1];w.state.camera.position=[5,1,0];w.state.camera.quaternion=quaternion(0,90,0);w.setState(w.state);
    const side=projectOpenPose(w).people[0].pose_keypoints_2d;
    assert.equal(point(side,14)[2],0);assert.equal(point(side,15)[2],1);w.dispose();
});
test('pose maps use the frozen scene and ignore editor-only marker visibility',()=>{
    const w=world();applyPreset(w.state,'Run');w.setState(w.state);const before=projectOpenPose(w);
    for(const mark of w.rig.marks)mark.visible=false;w.rig.face.visible=false;
    assert.deepEqual(projectOpenPose(w),before);w.dispose();
});
test('simplified hands keep body wrists and suppress all 21-point hand details',()=>{
    const w=world(),before=projectOpenPose(w);w.state.appearance.simple_hands=true;w.setState(w.state);
    const after=projectOpenPose(w);assert.deepEqual(after.people[0].pose_keypoints_2d,before.people[0].pose_keypoints_2d);
    assert.deepEqual(after.people[0].hand_left_keypoints_2d,Array(63).fill(0));assert.deepEqual(after.people[0].hand_right_keypoints_2d,Array(63).fill(0));w.dispose();
});
test('floor lines are frustum-clipped even when both original endpoints are offscreen',()=>{
    const w=world();assert.deepEqual(projectSegment(new T.Vector3(-20,1,0),new T.Vector3(20,1,0),w.camera),[0,.5,1,.5]);
    assert.equal(projectSegment(new T.Vector3(-20,100,0),new T.Vector3(20,100,0),w.camera),null);
    w.state.appearance.floor=true;w.setState(w.state);const doc=projectOpenPose(w);
    assert.ok(doc.floor_grid.length>0&&doc.floor_grid.length<=256);assert.ok(doc.floor_grid.every(line=>line.length===5&&line.every(n=>Number.isFinite(n)&&n>=0&&n<=1)));w.dispose();
});
