const {test}=require('node:test');const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const src=fs.readFileSync(path.join(__dirname,'../web/liquify.js'),'utf8');
const modulePromise=import('data:text/javascript;base64,'+Buffer.from(src).toString('base64'));
const points=JSON.parse(fs.readFileSync(path.join(__dirname,'landmarks.json'),'utf8'));
test('neutral and disabled liquify do not add a warp pass',async()=>{
 const {buildWarp}=await modulePromise;
 assert.equal(buildWarp(points,[1024,800],{}),null);
 assert.equal(buildWarp(points,[1024,800],{enabled:false,smile:80}),null);
});
test('eyes are linked by default and individual values override only their property',async()=>{
 const {adjustment,setAdjustment}=await modulePromise,s={eyes:{size:30,height:20}};
 assert.equal(adjustment(s,'size','eye_l'),30);assert.equal(adjustment(s,'size','eye_r'),30);
 setAdjustment(s,'size',-20,'eye_l');assert.equal(adjustment(s,'size','eye_l'),-20);assert.equal(adjustment(s,'size','eye_r'),30);
 setAdjustment(s,'size',60,'eyes');assert.equal(adjustment(s,'size','eye_l'),60);assert.equal(s.eye_l,undefined);
 assert.equal(adjustment(s,'height','eye_l'),20);
});
test('all face controls stay within the shader operation budget',async()=>{
 const {buildWarp,SECTIONS}=await modulePromise;
 const s={eyes:{size:100,height:100,width:100,tilt:100,distance:100}};
 SECTIONS.flatMap(([,items])=>items).forEach(([key])=>s[key]=100);
 const w=buildWarp(points,[800,1024],s);assert.equal(w.ops.length,13);
 for(const op of w.ops)assert.ok([...op.center,...op.radius,...op.matrix,...op.shift].every(Number.isFinite));
});
