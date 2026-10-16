import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{}),args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1280,height:950},baseURL:process.env.POSE_TEST_URL||'http://127.0.0.1:28993'}),errors=[];
page.on('pageerror',e=>errors.push(e.message));
try{
    await page.goto('/');await page.waitForFunction(()=>globalThis.harnessReady);
    await page.evaluate(()=>{
        globalThis.cameraNode=createCameraNode();
        poseNode.widgets.find(w=>w.name==='width').value=128;poseNode.widgets.find(w=>w.name==='height').value=192;
    });
    assert.equal(await page.evaluate(()=>cameraNode.dkstCameraPreview.type),'DKST_CAMERA_INFO');
    assert.equal(await page.getByRole('textbox',{name:'Camera description'}).inputValue(),'');
    await page.evaluate(async()=>{
        await queue();const prompt=structuredClone(lastPrompt);
        prompt.output['1001']={class_type:'TestCameraConsumer',inputs:{text:['997',0],json:['997',1]}};
        await testApi.queuePrompt(0,prompt);
    });
    const camera=await page.evaluate(()=>lastResult.results.find(r=>r.type==='camera'));
    const consumer=await page.evaluate(()=>lastResult.results.find(r=>r.type==='camera-consumer'));
    assert.ok(camera.camera_text.includes('50 mm equivalent'));assert.equal(consumer.text,camera.camera_text);assert.equal(consumer.json,camera.camera_json);
    assert.equal(await page.getByRole('textbox',{name:'Camera description'}).inputValue(),consumer.text);
    assert.equal(await page.evaluate(()=>cameraNode.dkstCameraPreview.value),consumer.text,'DOM and ComfyUI widget values must agree');
    assert.equal(await page.evaluate(()=>lastPrompt.workflow.nodes.find(n=>n.id===997).widgets_values.length),0,'Display widget must not enter execution inputs');
    // Vue and promoted DOM widgets access value through getValue/setValue.
    await page.evaluate(()=>{const value=cameraNode.dkstCameraPreview.value;cameraNode.dkstCameraText.value='';cameraNode.dkstCameraPreview.value=value;});
    assert.equal(await page.getByRole('textbox',{name:'Camera description'}).inputValue(),consumer.text);
    // Live websocket results must work even if onExecuted was bypassed.
    await page.evaluate(async()=>{
        suppressNodeExecuted=true;const widget=poseNode.widgets.find(w=>w.name==='scene_json'),s=JSON.parse(widget.value);s.camera.focal_mm=85;widget.value=JSON.stringify(s);await queue();
    });
    const text85=await page.evaluate(()=>lastResult.results.find(r=>r.type==='camera').camera_text);
    assert.ok(text85.includes('85 mm equivalent'));assert.equal(await page.getByRole('textbox',{name:'Camera description'}).inputValue(),text85);
    // History/tab restoration updates the output store without onExecuted.
    await page.evaluate(()=>{cameraNode.dkstCameraText.value='';testApp.extensions[0].onNodeOutputsUpdated({'997':{camera_text:['Restored camera description']}});});
    assert.equal(await page.getByRole('textbox',{name:'Camera description'}).inputValue(),'Restored camera description');
    await page.evaluate(()=>{
        cameraNode.dkstCameraText.value='';testApp.nodeOutputs={'997':{camera_text:['Cached camera description']}};
        testApi.dispatchEvent(new CustomEvent('execution_cached',{detail:{nodes:['997']}}));
    });
    assert.equal(await page.getByRole('textbox',{name:'Camera description'}).inputValue(),'Cached camera description');
    // Recreated nodes restore their last description from workflow properties.
    await page.evaluate(()=>{
        const properties=structuredClone(cameraNode.properties);testApp.nodeOutputs={};testApp.graph._nodes=testApp.graph._nodes.filter(n=>n!==cameraNode);
        globalThis.cameraNode=createCameraNode(997);cameraNode.properties=properties;cameraNode.onConfigure?.();
    });
    assert.equal(await page.getByRole('textbox',{name:'Camera description'}).inputValue(),'Cached camera description');
    // Handle merged output lists and do not clear on unrelated result events.
    await page.evaluate(()=>{cameraNode.onExecuted({camera_text:['Old description','Latest description']});cameraNode.onExecuted({unrelated:[]});});
    assert.equal(await page.getByRole('textbox',{name:'Camera description'}).inputValue(),'Latest description');
    // Nodes inside subgraphs can be targeted by display_node locator IDs.
    await page.evaluate(()=>{
        const nestedGraph={id:'camera-subgraph',_nodes:[],change(){}};cameraNode.graph=nestedGraph;nestedGraph._nodes.push(cameraNode);testApp.graph._nodes=testApp.graph._nodes.filter(n=>n!==cameraNode);testApp.graph._nodes.push({id:10,subgraph:nestedGraph});
        testApi.dispatchEvent(new CustomEvent('executed',{detail:{node:'dynamic-id',display_node:'camera-subgraph:997',output:{camera_text:['Nested camera description']}}}));
    });
    assert.equal(await page.getByRole('textbox',{name:'Camera description'}).inputValue(),'Nested camera description');
    assert.deepEqual(errors,[]);assert.deepEqual(await page.evaluate(()=>testErrors),[]);
    await fs.mkdir('/private/tmp/dkst-camera-debug',{recursive:true});await page.screenshot({path:'/private/tmp/dkst-camera-debug/camera-preview.png'});
    console.log('PASS: actual Camera Info preview widget, both downstream STRING outputs, DOM value synchronization, live websocket, cached outputs, history, workflow restoration, merged outputs and subgraphs.');
}finally{await browser.close();}
