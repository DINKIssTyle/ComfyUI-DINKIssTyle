const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const path=require('node:path');
const modulePromise=import('data:text/javascript;base64,'+Buffer.from(fs.readFileSync(path.join(__dirname,'../web/crop.js'),'utf8')).toString('base64'));
test('crop view and pointer mapping preserve original pixel centers',async()=>{
 const {cropPoints,originalPoint}=await modulePromise,p=Array.from({length:512},()=>[40/99,32/79]),b=[20,16,60,48],size=[100,80];
 const view=cropPoints(p,b,size);assert.ok(Math.abs(view[0][0]-20/39)<1e-12);assert.ok(Math.abs(view[0][1]-16/31)<1e-12);
 const restored=originalPoint(view[0],b,size);restored.forEach((v,i)=>assert.ok(Math.abs(v-p[0][i])<1e-12));
 assert.deepEqual(view.at(-1),[0,.5]);assert.deepEqual(p[0],[40/99,32/79]);
});
test('uncropped view keeps geometry and missing source metadata passes through',async()=>{
 const {cropPoints,originalPoint}=await modulePromise,p=Array.from({length:512},()=>[.5,.5]);
 assert.equal(cropPoints(p),p);assert.deepEqual(originalPoint([.4,.6]),[.4,.6]);
 assert.deepEqual(cropPoints(p,[0,0,80,64],[80,64])[0],[.5,.5]);
});
