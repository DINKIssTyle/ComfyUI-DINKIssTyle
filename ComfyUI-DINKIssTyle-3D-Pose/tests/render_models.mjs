// Produce model QA sheets with the actual output renderer, not editor helpers.
import fs from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{}),args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
const output=process.env.POSE_RENDER_DIR||'/private/tmp/dkst-pose-model';await fs.mkdir(output,{recursive:true});
try {
    await page.goto(process.env.POSE_TEST_URL||'http://127.0.0.1:28993');await page.waitForFunction(()=>globalThis.harnessReady);
    const images=await page.evaluate(async()=>{
        const {DEFAULT,clone,applyPreset}=await import('/extensions/pose/state.mjs');
        const {renderPNG}=await import('/extensions/pose/renderer.mjs');
        const {Quaternion,Vector3}=await import('/extensions/pose/vendor/three.module.mjs');
        const definitions=[... [2,3,6,7,8,9].map(ratio=>({name:`${ratio} heads`,file:`front-${ratio}`,ratio,preset:'Relaxed standing',angle:0})),
            {name:'Front',file:'adult-front',ratio:8,preset:'Relaxed standing',angle:0},
            {name:'Side',file:'adult-side',ratio:8,preset:'Relaxed standing',angle:Math.PI/2},
            {name:'Back',file:'adult-back',ratio:8,preset:'Relaxed standing',angle:Math.PI},
            {name:'A-pose',file:'adult-a-pose',ratio:8,preset:'A-pose',angle:0},
            {name:'Run',file:'adult-run',ratio:8,preset:'Run',angle:.30}];
        const result=[];
        for(const def of definitions){
            const s=clone(DEFAULT);s.ratio=def.ratio;s.output={width:480,height:720};s.appearance.body='#abb4bf';s.appearance.background='#edf1f5';s.appearance.light=1.1;
            applyPreset(s,def.preset);s.camera.projection='orthographic';s.camera.ortho=def.preset==='A-pose'?2.75:2.28;
            const rotation=new Quaternion().setFromAxisAngle(new Vector3(0,1,0),def.angle);
            s.camera.quaternion=rotation.toArray();s.camera.target=[0,1,0];s.camera.position=new Vector3(0,0,5).applyQuaternion(rotation).add(new Vector3(0,1,0)).toArray();
            const blob=await renderPNG(s),url=await new Promise(resolve=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.readAsDataURL(blob);});
            result.push({...def,url});
        }
        const sheet=async(defs)=>{
            const cellW=260,canvas=document.createElement('canvas');canvas.width=cellW*defs.length;canvas.height=446;
            const ctx=canvas.getContext('2d');ctx.fillStyle='#edf1f5';ctx.fillRect(0,0,canvas.width,canvas.height);
            for(let i=0;i<defs.length;i++){
                const img=new Image();img.src=defs[i].url;await img.decode();ctx.drawImage(img,i*cellW+6,42,248,372);
                ctx.fillStyle='#263346';ctx.font='600 18px system-ui';ctx.textAlign='center';ctx.fillText(defs[i].name,i*cellW+cellW/2,29);
            }
            return canvas.toDataURL('image/png');
        };
        return {renders:result,proportions:await sheet(result.slice(0,6)),anatomy:await sheet(result.slice(6))};
    });
    for(const item of images.renders)await fs.writeFile(`${output}/${item.file}.png`,Buffer.from(item.url.split(',')[1],'base64'));
    for(const name of ['proportions','anatomy'])await fs.writeFile(`${output}/${name}.png`,Buffer.from(images[name].split(',')[1],'base64'));
    if(errors.length)throw Error(errors.join('\n'));console.log(`Rendered model sheets: ${output}`);
} finally {await browser.close();}
