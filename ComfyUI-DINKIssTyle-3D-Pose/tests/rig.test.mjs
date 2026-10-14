import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
globalThis.fetch=async url=>({json:async()=>JSON.parse(await fs.readFile(url,'utf8'))});
const {DEFAULT,clone,quaternion,applyPreset,mirror,copySide,History,sceneString,validate,RATIOS}=await import('../web/state.mjs');
const {Rig,solveIK,proportions,sculptGeometry}=await import('../web/rig.mjs');
const T=await import('../web/vendor/three.module.mjs');

test('neutral standing height and head ratio are exact for all six proportions',()=>{
    for(const ratio of RATIOS){
        const state=clone(DEFAULT);state.ratio=ratio;const rig=new Rig(state),p=proportions(ratio);
        const head=rig.joints.head.getWorldPosition(new T.Vector3());
        assert.ok(Math.abs(head.y+p.head/2-2)<1e-9);
        assert.equal(2/p.head,ratio);
        assert.ok(rig.joints.L_ankle.getWorldPosition(new T.Vector3()).y>0);
        assert.equal(Object.keys(rig.joints).length,52);rig.dispose();
    }
});
test('proportion replacement keeps the pose and world position',()=>{
    const state=clone(DEFAULT);state.root=[1,2,3];applyPreset(state,'Run');
    const before=sceneString(state.joints);state.ratio=2;const rig=new Rig(state);rig.read(state);
    assert.equal(sceneString(state.joints),before);assert.deepEqual(state.root,[1,2,3]);rig.dispose();
});
test('mirror twice is identity including asymmetric fingers',()=>{
    const state=clone(DEFAULT);applyPreset(state,'Run');state.joints.L_index_1=quaternion(-43,8,10);
    const before=sceneString(state);mirror(state);mirror(state);assert.equal(sceneString(state),before);
    copySide(state,'L');assert.deepEqual(state.joints.R_index_1,[state.joints.L_index_1[0],-state.joints.L_index_1[1],-state.joints.L_index_1[2],state.joints.L_index_1[3]]);
});
test('presets preserve camera and body proportion',()=>{
    const state=clone(DEFAULT);state.ratio=3;const camera=clone(state.camera);applyPreset(state,'Sit');
    assert.deepEqual(state.camera,camera);assert.equal(state.ratio,3);validate(state);
});
test('history branches after undo',()=>{
    const state=clone(DEFAULT),h=new History(state);state.ratio=2;h.push(state);state.ratio=3;h.push(state);
    assert.equal(h.move(-1).ratio,2);state.ratio=9;h.push(state);assert.equal(h.move(1).ratio,9);
});
test('two-bone IK reaches goals and clamps unreachable goals for all ratios',()=>{
    for(const ratio of RATIOS) for(const limb of ['arm','leg']) for(const side of ['L','R']){
        const state=clone(DEFAULT);state.ratio=ratio;state.root=[.2,0,-.1];state.joints.root=quaternion(0,25,0);
        const rig=new Rig(state),names=limb==='arm'?['shoulder','elbow','wrist']:['hip','knee','ankle'];
        const upper=rig.joints[side+'_'+names[0]],lower=rig.joints[side+'_'+names[1]],end=rig.joints[side+'_'+names[2]];
        const start=upper.getWorldPosition(new T.Vector3()),reach=lower.position.length()+end.position.length();
        const goal=start.clone().add(new T.Vector3(.1,-.6,.25).normalize().multiplyScalar(reach*.75));
        const pole=start.clone().add(new T.Vector3(0,0,limb==='arm'?1:-1));solveIK(rig,side,limb,goal,pole);
        assert.ok(end.getWorldPosition(new T.Vector3()).distanceTo(goal)<1e-5,`${ratio} ${side} ${limb}`);
        solveIK(rig,side,limb,start.clone().add(new T.Vector3(0,-20,0)),pole);
        assert.ok(Math.abs(end.getWorldPosition(new T.Vector3()).distanceTo(start)-reach)<1e-4);
        rig.read(state);validate(state);rig.dispose();
    }
});
test('constrained IK keeps elbows and knees within their signed flexion limits',()=>{
    for(const limb of ['arm','leg']) {
        const rig=new Rig(clone(DEFAULT)),upper=rig.joints[limb==='arm'?'L_shoulder':'L_hip'];
        const start=upper.getWorldPosition(new T.Vector3());
        solveIK(rig,'L',limb,start.clone(),start.clone().add(new T.Vector3(0,0,1)),true);
        const joint=rig.joints[limb==='arm'?'L_elbow':'L_knee'];
        const rotation=new T.Euler().setFromQuaternion(joint.quaternion);
        const angle=T.MathUtils.radToDeg(rotation.x);
        assert.ok(limb==='arm'?angle>=-150.0001&&angle<=0:angle>=0&&angle<=150.0001);
        assert.ok(Math.abs(rotation.y)<1e-9&&Math.abs(rotation.z)<1e-9);rig.dispose();
    }
});
test('adult arm span stays close to standing height and wrists reach the pelvis',()=>{
    for(const ratio of [6,7,8,9]){
        const p=proportions(ratio),span=2*(p.shoulder+p.upperArm+p.lowerArm+p.hand);
        assert.ok(span>=1.90&&span<=2.16,`${ratio} heads: arm span ${span}`);
        const rig=new Rig({...clone(DEFAULT),ratio});
        const hip=rig.joints.L_hip.getWorldPosition(new T.Vector3()).y;
        const wrist=rig.joints.L_wrist.getWorldPosition(new T.Vector3()).y;
        assert.ok(Math.abs(wrist-hip)<p.head*.45,'Wrist should sit near hip level');
        assert.ok(p.pelvis/p.shoulder>=.68&&p.pelvis/p.shoulder<=.82,'Pelvis must balance shoulder width');
        rig.dispose();
    }
});
test('sculpted surfaces have finite outward normals and grounded soles',()=>{
    const geometry=sculptGeometry([[0,1,1],[1,1,1]],24,4);
    const vertex=2*25,normal=new T.Vector3().fromBufferAttribute(geometry.attributes.normal,vertex);
    assert.ok(normal.x>.99,'Side faces must point outward');geometry.dispose();
    for(const ratio of RATIOS){
        const rig=new Rig({...clone(DEFAULT),ratio});
        rig.root.traverse(o=>{if(o.geometry){
            for(const name of ['position','normal']) assert.ok(Array.from(o.geometry.attributes[name].array).every(Number.isFinite));
        }});
        const bounds=new T.Box3().setFromObject(rig.root);assert.ok(Math.abs(bounds.min.y)<1e-6);assert.ok(Math.abs(bounds.max.y-2)<1e-6);rig.dispose();
    }
});
test('old scenes migrate to the new model while preserving joint rotations and camera',()=>{
    const old=clone(DEFAULT);old.renderer='dkst-pose-1-three-180';applyPreset(old,'Run');
    const migrated=validate(old);assert.equal(migrated.renderer,DEFAULT.renderer);
    assert.deepEqual(migrated.joints,old.joints);assert.deepEqual(migrated.camera,old.camera);
    assert.equal(old.renderer,'dkst-pose-1-three-180','Migration must not mutate the source');
});
test('simplified mesh limbs hide detail without changing stored finger or toe rotations',()=>{
    const state=clone(DEFAULT);state.joints.L_index_1=quaternion(-45,0,0);state.joints.L_toe=quaternion(-20,0,0);
    const rig=new Rig(state),before=sceneString(state.joints);state.appearance.simple_hands=true;state.appearance.simple_feet=true;rig.apply(state);
    assert.equal(rig.hands.L.detailed.visible,false);assert.equal(rig.hands.L.simple.visible,true);
    assert.equal(rig.feet.L.detailed.visible,false);assert.equal(rig.feet.L.simple.visible,true);
    assert.ok(Math.abs(rig.bounds().min.y)<1e-6,'Hidden posed toes must not move the simplified sole above the floor');
    assert.ok(rig.visiblePickables().every(o=>!/^L_index_|^R_index_|_toe$/.test(o.userData.joint)));
    rig.read(state);assert.equal(sceneString(state.joints),before);
    state.appearance.simple_hands=false;state.appearance.simple_feet=false;rig.apply(state);
    assert.equal(rig.hands.L.detailed.visible,true);assert.equal(rig.feet.L.detailed.visible,true);rig.dispose();
});
test('previous model scenes receive explicit detail defaults when migrated',()=>{
    const old=clone(DEFAULT);old.renderer='dkst-pose-2-three-180';delete old.appearance.simple_hands;delete old.appearance.simple_feet;
    const migrated=validate(old);assert.equal(migrated.renderer,DEFAULT.renderer);assert.equal(migrated.appearance.simple_hands,false);assert.equal(migrated.appearance.simple_feet,false);
});
