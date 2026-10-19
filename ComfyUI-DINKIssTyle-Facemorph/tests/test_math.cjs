const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web/math.js'),'utf8');
const math=import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
function state(){return {base:0,assets:[{shape:50,feature:50,regions:{}},{shape:50,feature:50,regions:{}}]};}
test('weights normalize each vertex and zero weights fall back to base',async()=>{
 const {normalizedWeights}=await math,s=state();
 assert.deepEqual(normalizedWeights(s,[0,1,2],'feature'),[[.5,.5,.5],[.5,.5,.5]]);
 s.assets.forEach(a=>a.feature=0);
 assert.deepEqual(normalizedWeights(s,[0,1,2],'feature'),[[1,1,1],[0,0,0]]);
});
test('paired sliders control both sides; child overrides affect one side only',async()=>{
 const {normalizedWeights,setRegionWeight}=await math,s=state(),a=s.assets[1];
 for(const [pair,left,right] of [['eyes',1,2],['brows',6,7]]){
  setRegionWeight(a,pair,'feature',100);
  assert.deepEqual(normalizedWeights(s,[left,right],'feature'),[[1/3,1/3],[2/3,2/3]]);
  const child=pair==='eyes'?'eye_l':'brow_l';
  setRegionWeight(a,child,'feature',50);
  assert.deepEqual(normalizedWeights(s,[left,right],'feature'),[[.5,1/3],[.5,2/3]]);
 }
});
test('moving a parent clears child overrides for that property only',async()=>{
 const {setRegionWeight,regionValue,clearRegionWeights}=await math,a=state().assets[0];
 a.regions={eyes:{shape:60},eye_l:{shape:20,feature:80},eye_r:{shape:30}};
 setRegionWeight(a,'eyes','shape',90);
 assert.equal(regionValue(a,'eye_l','shape'),90);assert.equal(regionValue(a,'eye_r','shape'),90);
 assert.equal(regionValue(a,'eye_l','feature'),80);
 clearRegionWeights(a,'eye_l');assert.equal(regionValue(a,'eye_l','feature'),50);
 clearRegionWeights(a,'eyes');assert.equal(regionValue(a,'eye_r','shape'),50);
 assert.deepEqual(a.regions,{});
});
test('hierarchy exposes parents before optional left and right controls',async()=>{
 const {REGION_OPTIONS}=await math,keys=REGION_OPTIONS.map(([k])=>k);
 assert.deepEqual(keys.slice(keys.indexOf('eyes'),keys.indexOf('eyes')+3),['eyes','eye_l','eye_r']);
 assert.deepEqual(keys.slice(keys.indexOf('brows'),keys.indexOf('brows')+3),['brows','brow_l','brow_r']);
 assert.ok(keys.every(key=>!key.startsWith('ear')));
});
test('legacy projects retain face edits and discard retired data',async()=>{
 const {upgradeState}=await math,s=state();
 const old=Array.from({length:544},(_,i)=>[i,i]);
 s.assets[0].points=old;s.assets[0].ears=[true,true];s.assets[0].regions={ear_l:{shape:1},eyes:{shape:60}};
 s.assets[0].warnings=['Dashed ear guides are estimates.','2 faces detected.'];
 upgradeState(s);
 assert.equal(s.assets[0].points.length,512);assert.deepEqual(s.assets[0].points[468],old[484]);
 assert.deepEqual(s.assets[0].regions,{eyes:{shape:60}});assert.equal(s.assets[0].ears,undefined);
 assert.deepEqual(s.assets[0].warnings,['2 faces detected.']);
});
test('geometry moves to the weighted target and invalid shape is rejected',async()=>{
 const {updateMesh}=await math,s=state();
 const mesh={groups:[0,0,0],aligned:[[[0,0],[1,0],[0,1]],[[0,0],[.8,0],[0,.8]]],reference:[[0,0],[1,0],[0,1]],triangles:[[0,1,2]],alpha:[1,1,1]};
 assert.deepEqual(updateMesh(s,mesh).target,[[0,0],[.9,0],[0,.9]]);
 mesh.aligned[1][2]=[0,-2];assert.throws(()=>updateMesh(s,mesh),/mesh/);
});
test('hair visibility changes shape confidence but preserves appearance weights',async()=>{
 const {normalizedWeights}=await math,s=state(),visibility=[[1,1,1],[.1,.25,1]];
 assert.deepEqual(normalizedWeights(s,[0,1,2],'feature',visibility),[[.5,.5,.5],[.5,.5,.5]]);
 const w=normalizedWeights(s,[0,1,2],'shape',visibility);
 assert.ok(Math.abs(w[1][0]-1/11)<1e-12);assert.equal(w[1][1],.2);assert.equal(w[1][2],.5);
});
test('occlusion confidence is softened before it can fold a valid mesh',async()=>{
 const {updateMesh}=await math,s=state();
 const mesh={groups:[0,0,0],aligned:[[[0,0],[1,0],[0,1]],[[0,0],[-.8,0],[0,1]]],reference:[[0,0],[1,0],[0,1]],triangles:[[0,1,2]],alpha:[1,1,1],visibility:[[1,.1,1],[1,1,1]]};
 const result=updateMesh(s,mesh);assert.ok(result.target[1][0]>0);
 assert.ok(result.shape_weights[0][1]>.1/1.1);
});
