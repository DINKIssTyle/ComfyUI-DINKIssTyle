import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
globalThis.fetch=async url=>({json:async()=>JSON.parse(await fs.readFile(url,'utf8'))});
const {DEFAULT,clone,quaternion}=await import('../web/state.mjs');
const {Rig}=await import('../web/rig.mjs');
const {lensFOV,focalFromFOV}=await import('../web/camera.mjs');
const {World}=await import('../web/renderer.mjs');
const {createBox}=await import('../web/objects.mjs');
const T=await import('../web/vendor/three.module.mjs');

test('lens conversion preserves diagonal equivalence for landscape and portrait',()=>{
    for(const aspect of [.5,.75,1,1.5,2])for(const mm of [18,50,135])assert.ok(Math.abs(focalFromFOV(lensFOV(mm,aspect),aspect)-mm)<1e-10);
    assert.ok(lensFOV(18,.75)>lensFOV(50,.75));
});
test('surface has one connected topology, normalized bone weights and bending corrections',()=>{
    const s=clone(DEFAULT),r=new Rig(s),g=r.surface.geometry,n=g.attributes.position.count;
    assert.ok(r.surface.isSkinnedMesh);assert.equal(r.surface.skeleton.bones.length,52);
    assert.equal(g.userData.modelSource,'MakeHuman CC0 authored base mesh');assert.equal(n,53514);
    const weights=g.attributes.skinWeight.array;for(let i=0;i<weights.length;i+=4)assert.ok(Math.abs(weights[i]+weights[i+1]+weights[i+2]+weights[i+3]-1)<1e-6);
    const parent=Int32Array.from({length:n},(_,i)=>i),find=i=>{while(parent[i]!==i){parent[i]=parent[parent[i]];i=parent[i];}return i;};
    for(let i=0;i<g.index.count;i+=3){const a=find(g.index.array[i]);parent[find(g.index.array[i+1])]=a;parent[find(g.index.array[i+2])]=a;}
    assert.equal(new Set(Array.from(parent,(_,i)=>find(i))).size,1);
    const before=r.surface.boundingBox.clone();s.joints.L_elbow=quaternion(-120,0,0);r.apply(s);
    assert.notDeepEqual(r.surface.boundingBox,before);assert.ok(r.surface.morphTargetInfluences.some(n=>n>.5));
    assert.ok(r.surface.geometry.attributes.normal.array.every(Number.isFinite));r.dispose();
});
test('box transforms and visibility affect scene bounds, deletion frees the selection mesh',()=>{
    const s=clone(DEFAULT),b=createBox(s,'red');b.position=[3,.5,0];b.size=[1,1,2];b.quaternion=quaternion(0,45,0);
    const w=new World(s);assert.ok(w.bounds().max.x>3.9);assert.equal(w.objects.picks().length,1);
    b.visible=false;w.setState(s);assert.ok(w.bounds().max.x<1);
    s.objects=[];w.setState(s);assert.equal(w.objects.meshes.size,0);w.dispose();
});
