const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join, resolve, dirname } = require('node:path');
const { pathToFileURL } = require('node:url');
const vm = require('node:vm');
const { JSDOM } = require('jsdom');
const settle = () => new Promise(resolve => setImmediate(resolve));
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

const defaults = Object.fromEntries([
    ...['light_exposure','light_contrast','light_highlights','light_shadows','light_whites','light_blacks','color_temperature','color_tint','color_vibrance','color_saturation','effects_texture','effects_clarity','effects_dehaze','effects_vignette','effects_grain','effects_glow','detail_sharpening','detail_noise_reduction','detail_color_noise_reduction','optics_distortion','optics_vignette','lens_blur_bokeh_boost','grain_seed'].map(name => [name, 0]),
    ['color_white_balance','Off'], ['lens_blur_apply',false], ['lens_blur_focus',.5], ['lens_blur_amount',50], ['lens_blur_bokeh',4], ['lens_blur_aperture_blades',9], ['lens_blur_depth_blur_radius',5], ['lens_blur_depth_sigma',2], ['depth_near_is_white',true],
]);
const schema = Object.fromEntries(Object.entries(defaults).map(([name, value]) => [name,
    typeof value === 'boolean' ? {kind:'bool'} : typeof value === 'string' ? {kind:'choice',values:['Off','Auto','Custom']} : { kind: ['grain_seed','lens_blur_aperture_blades','lens_blur_depth_blur_radius'].includes(name) ? 'int' : 'float', min: name === 'light_exposure' ? -5 : name === 'lens_blur_focus' ? 0 : -100, max: name === 'light_exposure' ? 5 : name === 'lens_blur_focus' ? 1 : 100, step: name === 'light_exposure' ? .05 : name === 'lens_blur_focus' ? .01 : 1 },
]));
const tiny = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==';

async function fixture() {
    const dom = new JSDOM('<html><head></head><body><button id="opener">Open</button></body></html>', { runScripts:'outside-only', url:'http://localhost/' });
    const context = dom.getInternalVMContext();
    dom.window.HTMLCanvasElement.prototype.getContext = () => null;
    dom.window.prompt = () => 'My Look';
    const calls = [], queueCalls = [], listeners = new Map();
    const opened = [], downloads = [];
    dom.window.open = url => { const popup={location:{href:url},close(){this.closed=true;}}; opened.push(popup); return popup; };
    dom.window.URL.createObjectURL = () => `blob:photo-${downloads.length + opened.length}`;
    dom.window.URL.revokeObjectURL = () => {};
    dom.window.HTMLAnchorElement.prototype.click = function() { downloads.push({href:this.href,filename:this.download}); };
    dom.window.fetch = async () => ({ok:true,blob:async()=>new dom.window.Blob(['png'],{type:'image/png'})});
    let extension, previewHandler;
    const extensions = new Map();
    let nextId = 1, nextLink = 1, sessionId = 1;
    const graph = { _nodes:[], links:{}, version:0,
        getNodeById(id) { return this._nodes.find(n => n.id === id); },
        add(n) { n.id = nextId++; n.graph = this; this._nodes.push(n); },
        remove(n) { this._nodes.splice(this._nodes.indexOf(n),1); n.onRemoved?.(); },
        incrementVersion() { this.version++; }, beforeChange() {}, afterChange() {},
    };
    class Node {
        constructor(type='DINKI_Photo_Studio') {
            this.type = this.comfyClass = type; this.title = type; this.pos = [0,0]; this.size = [320,700]; this.properties={};
            this.outputs = type === 'PreviewImage' ? [] : [{type:'IMAGE',name:'image'}];
            this.inputs = type === 'DINKI_Photo_Studio' ? [{name:'image',type:'IMAGE',link:null},{name:'depth_image',type:'IMAGE',link:null}] : type === 'PreviewImage' ? [{name:'images',type:'IMAGE',link:null}] : [];
            this.widgets = type === 'DINKI_Photo_Studio' ? [{name:'active',value:true},{name:'preset',value:'Custom'}, ...Object.entries(defaults).map(([name,value])=>({name,value,type:'number',options:{}}))] : [{name:'image',value:'photo.png',options:{}}];
            if (type === 'DINKI_Photo_Studio_Source') this.widgets.push({name:'interpretation',value:'Image'});
            if (type === 'DINKI_Photo_Studio') this.onNodeCreated?.();
        }
        addWidget(type,name,value,callback,options) { const w={type,name,value,callback,options}; this.widgets.push(w); return w; }
        addDOMWidget(name,type,element,options) { const w={name,type,element,options}; this.widgets.push(w); return w; }
        computeSize() { return [320,900]; }
        setSize(value) { this.size=value; }
        setDirtyCanvas() {}
        disconnectInput(slot) { const old=this.inputs[slot].link; if (old != null) delete this.graph.links[old]; this.inputs[slot].link=null; }
        connect(slot,target,input) { target.disconnectInput(input); const link={id:nextLink++,origin_id:this.id,origin_slot:slot,target_id:target.id,target_slot:input}; this.graph.links[link.id]=link; target.inputs[input].link=link.id; return link; }
    }
    const app = { graph, canvas:{graph_mouse:[500,100]}, nodeOutputs:{}, registerExtension(e) {extension=e;extensions.set(e.name,e);}, ui:{dialog:{show(){}}}, queuePrompt(...args){queueCalls.push(args);} };
    const api = { apiURL:path=>path,
        addEventListener(name,handler) { if(!listeners.has(name))listeners.set(name,new Set()); listeners.get(name).add(handler); },
        removeEventListener(name,handler) { listeners.get(name)?.delete(handler); },
        async fetchApi(path, options) {
            const payload = options?.body && typeof options.body === 'string' ? JSON.parse(options.body) : undefined;
            calls.push({path,payload});
            let data;
            if (path.endsWith('/presets')) data = options?.method === 'POST' ? {presets:{[payload.name]:payload.settings}} : {defaults,controls:schema,presets:{Portrait:{...defaults,light_exposure:1}}};
            else if (path.endsWith('/session')) data={session:`session-${sessionId++}`,original:tiny,width:640,height:480,has_depth:!!payload.depth || (!!payload.source_session && payload.use_source_depth),auto:{light_exposure:.75}};
            else if (path.endsWith('/preview')) data=previewHandler ? await previewHandler(payload) : {preview:tiny,proxy:false,histogram:[[],[],[]],depth_preview:tiny};
            else if (path.endsWith('/focus')) data={focus:.712345};
            else if (path.endsWith('/materialize')) data={filename:'persistent.png',subfolder:'dkst_photo_studio',type:'input'};
            else data={ok:true};
            return {ok:true,status:200,json:async()=>data,blob:async()=>new dom.window.Blob(['export png'],{type:'image/png'})};
        },
    };
    context.LiteGraph={createNode:type=>new Node(type)};
    const modules=new Map();
    const synthetic = exports => new vm.SyntheticModule(Object.keys(exports),function(){for(const [key,value] of Object.entries(exports))this.setExport(key,value);},{context});
    const appModule=synthetic({app,ComfyApp:class {}}),apiModule=synthetic({api});
    async function load(path) {
        if(modules.has(path))return modules.get(path);
        const module=new vm.SourceTextModule(readFileSync(path,'utf8'),{context,identifier:path,initializeImportMeta(meta){meta.url=pathToFileURL(path).href;}});
        modules.set(path,module);
        await module.link((specifier, reference)=>specifier.endsWith('/scripts/app.js')?appModule:specifier.endsWith('/scripts/api.js')?apiModule:load(resolve(dirname(reference.identifier),specifier)));
        return module;
    }
    const root=resolve(__dirname,'../ComfyUI-DINKIssTyle/js');
    const entry=await load(join(root,'dinki_photo_studio.js')); await entry.evaluate();
    await extension.beforeRegisterNodeDef(Node,{name:'DINKI_Photo_Studio'});
    const editorModule=modules.get(join(root,'dinki_photo_studio_editor.js')).namespace;
    const add=type=>{const n=new Node(type);graph.add(n);return n;};
    const photo=add('LoadImage'), depth=add('LoadImage'); depth.title='Depth';depth.widgets[0].value='depth.png';
    const node=add('DINKI_Photo_Studio');photo.connect(0,node,0);
    await settle();
    return {dom,graph,app,api,node,photo,depth,add,editorModule,calls,queueCalls,Node,extension,opened,downloads,
        async addViewer(properties = {}) {
            const module = await load(join(root,'dinki_nodes.js')); await module.evaluate();
            class Viewer {
                constructor() { this.type=this.comfyClass='DINKI_Preview_Image';this.title='DKST Image (Viewer)';this.properties=properties;this.outputs=[{type:'IMAGE'}];this.inputs=[{name:'images',link:null}];this.widgets=[];this.pos=[0,0];this.onNodeCreated(); }
                addDOMWidget(...args) { const w=Node.prototype.addDOMWidget.apply(this,args);this.preview=w.element;return w; }
                setDirtyCanvas() {}
                disconnectInput(...args) { return Node.prototype.disconnectInput.apply(this,args); }
                connect(...args) { return Node.prototype.connect.apply(this,args); }
            }
            await extensions.get('DINKIssTyle.PhotoStudio').beforeRegisterNodeDef(Viewer,{name:'DINKI_Preview_Image'});
            await extensions.get('DINKI.PreviewImage.Resolution').beforeRegisterNodeDef(Viewer,{name:'DINKI_Preview_Image'});
            const viewer=new Viewer();graph.add(viewer);viewer.dkstRestorePreview();return viewer;
        },
        setPreview(handler){previewHandler=handler;},
        cleanup(){for(const overlay of dom.window.document.querySelectorAll('.dkst-photo-overlay'))overlay.remove();dom.window.close();},
    };
}

// These exercise the actual extension modules, not a copy of editor logic.
test('Viewer images are discoverable from output cache, saved workflow and live DOM preview',async()=>{
    const f=await fixture();
    try {
        const viewer=f.add('DINKI_Preview_Image');
        f.app.nodeOutputs[viewer.id]={dkst_images:[{filename:'cached.png',type:'temp'}]};
        assert.equal(f.editorModule.selectedImageCandidate(viewer).descriptor.filename,'cached.png');
        viewer.properties.dkstPreview={dkst_images:[],selectedIndex:0};
        assert.equal(f.editorModule.selectedImageCandidate(viewer).descriptor.filename,'cached.png');
        viewer.properties.dkstPreview={dkst_images:[{filename:'restored.png',type:'output'}],selectedIndex:0};
        assert.equal(f.editorModule.selectedImageCandidate(viewer).descriptor.filename,'restored.png');
        viewer.dkstPreviewImages=[{filename:'live.png',type:'temp'}];
        assert.equal(f.editorModule.selectedImageCandidate(viewer).descriptor.filename,'live.png');
        viewer.dkstPreviewImages=[];
        assert.equal(f.editorModule.selectedImageCandidate(viewer),null);
    } finally {f.cleanup();}
});

test('node image context menu opens and downloads the full result, and Apply clears an older result',async()=>{
    const f=await fixture();
    try {
        f.node.onExecuted({photo_studio_preview:[tiny],photo_studio_image:[{filename:'full.png',subfolder:'dkst_photo_studio',type:'temp'}]});
        assert.equal(f.editorModule.workflowImages(f.graph).find(c=>c.nodeId===f.node.id).descriptor.filename,'full.png');
        const image=f.node.dkstPhotoStudioThumbnail;
        const event=new f.dom.window.MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:15,clientY:15});
        image.dispatchEvent(event);
        assert.equal(event.defaultPrevented,true);
        const menu=f.dom.window.document.querySelector('.dkst-photo-context-menu');
        assert.deepEqual([...menu.querySelectorAll('button')].map(b=>b.textContent),['Open Image','Save Image','Open Photo Studio']);
        await menu.querySelector('button').onclick();
        assert.match(f.opened[0].location.href,/filename=full.png/);
        const options=[];f.node.getExtraMenuOptions(null,options);
        await options.find(o=>o.content==='Save Image').callback();
        assert.equal(f.downloads[0].filename,'full.png');
        const editor=await f.editorModule.openPhotoStudio(f.node);
        await editor.apply();
        assert.equal(f.node.dkstPhotoStudioImage,undefined);
        assert.equal(f.node.properties.dkstPhotoStudioPreview,undefined);
        assert.equal(f.node.dkstPhotoStudioImageUrl(),tiny);
        assert.ok(f.node.size[1]<=300);
    } finally {f.cleanup();}
});

test('editor Open/Save exports current adjustments at source resolution without running the workflow',async()=>{
    const f=await fixture();
    try {
        const editor=await f.editorModule.openPhotoStudio(f.node);
        editor.controlChange('light_exposure',1.25,true);
        await editor.exportImage(true);
        const payload=f.calls.find(c=>c.path.endsWith('/export')).payload;
        assert.equal(payload.session,editor.session);assert.equal(payload.settings.light_exposure,1.25);
        assert.equal(payload.original,false);
        assert.equal(f.downloads[0].filename,'DKST_Photo_Studio.png');
        editor.exportTarget.value='original';await editor.exportImage(false);
        assert.equal(f.calls.filter(c=>c.path.endsWith('/export')).at(-1).payload.original,true);
        assert.match(f.opened[0].location.href,/^blob:/);
        assert.equal(f.opened[0].opener,null);
        assert.equal(f.queueCalls.length,0);
        editor.close();
    } finally {f.cleanup();}
});

test('center zoom, minimap navigation and comparison use the same aligned image coordinates',async()=>{
    const f=await fixture();
    try {
        const editor=await f.editorModule.openPhotoStudio(f.node);
        Object.defineProperties(editor.viewport,{clientWidth:{get:()=>600},clientHeight:{get:()=>400}});
        editor.viewport.getBoundingClientRect=()=>({left:0,top:0,width:600,height:400});
        Object.defineProperties(editor.pair,{clientWidth:{get:()=>parseFloat(editor.pair.style.width)||600},clientHeight:{get:()=>parseFloat(editor.pair.style.height)||400}});
        editor.pair.getBoundingClientRect=()=>({left:Math.max(0,(600-editor.pair.clientWidth)/2)-editor.viewport.scrollLeft,top:Math.max(0,(400-editor.pair.clientHeight)/2)-editor.viewport.scrollTop,width:editor.pair.clientWidth,height:editor.pair.clientHeight});
        editor.setZoom('fit');editor.setZoom('2');
        const area=editor.visibleArea();
        assert.ok(Math.abs((area.left+area.right)/2-.5)<.001);
        assert.ok(Math.abs((area.top+area.bottom)/2-.5)<.001);
        assert.equal(editor.minimap.hidden,false);
        editor.map.dispatchEvent(new f.dom.window.KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true,cancelable:true}));
        assert.ok(editor.viewport.scrollLeft>340);
        editor.compare.click();
        assert.equal(editor.after.style.clipPath,'inset(0 50% 0 0)');
        const bounds=editor.pair.getBoundingClientRect();
        editor.pair.dispatchEvent(new f.dom.window.MouseEvent('pointermove',{clientX:bounds.left+bounds.width*.75,bubbles:true}));
        assert.equal(editor.after.style.clipPath,'inset(0 25% 0 0)');
        editor.comparisonMode.value='Difference';editor.comparisonMode.dispatchEvent(new f.dom.window.Event('change'));
        assert.equal(editor.pair.classList.contains('difference'),true);assert.equal(editor.divider.hidden,true);
        editor.setZoom('fit');assert.equal(editor.minimap.hidden,true);
        editor.close();
    } finally {f.cleanup();}
});

test('hidden promoted setting rows do not make a newly created Studio node tall',async()=>{
    const f=await fixture();
    try {
        f.node.inputs.push(...Object.keys(defaults).map(name=>({name,widget:{name},link:null})));
        f.node.dkstPhotoStudioCompact();
        assert.ok(f.node.size[1]<=300);
        const editor=await f.editorModule.openPhotoStudio(null,{descriptor:{filename:'photo.png',type:'input'},nodeId:f.photo.id,slot:0,direct:true});
        // Reproduce native graph.add growing a just-created node after onNodeCreated.
        const add=f.graph.add;
        f.graph.add=function(node){add.call(this,node);node.size=[320,1800];};
        await editor.apply();await settle();
        const created=f.graph._nodes.filter(n=>n.type==='DINKI_Photo_Studio').at(-1);
        assert.ok(created.size[1]<=300);
        assert.equal(f.graph.links[created.inputs[0].link].origin_id,f.photo.id);
    } finally {f.cleanup();}
});

test('Viewer node menu opens its IMAGE output as a new Studio branch',async()=>{
    const f=await fixture();
    try {
        const viewer=await f.addViewer();f.photo.connect(0,viewer,0);
        viewer.onExecuted({dkst_images:[{filename:'viewer.png',type:'temp'}]});
        const options=[];viewer.getExtraMenuOptions(null,options);
        const actions=options.filter(o=>o.content==='Open in Photo Studio');
        assert.equal(actions.length,1);
        const editor=await actions[0].callback();
        assert.equal(editor.imageCandidate.nodeId,viewer.id);
        assert.equal(editor.imageCandidate.direct,true);
        await editor.apply();
        const created=f.graph._nodes.filter(n=>n.type==='DINKI_Photo_Studio').at(-1);
        assert.equal(f.graph.links[created.inputs[0].link].origin_id,viewer.id);
        assert.equal(f.graph.links[viewer.inputs[0].link].origin_id,f.photo.id);
        assert.equal(f.queueCalls.length,0);
    } finally {f.cleanup();}
});

test('Viewer image menu opens the restored selected batch frame and persists a snapshot on Apply',async()=>{
    const f=await fixture();
    try {
        const viewer=await f.addViewer({dkstPreview:{dkst_images:[
            {filename:'first.png',type:'temp'},{filename:'second.png',type:'output'},
        ],selectedIndex:1}});
        assert.match(viewer.preview.querySelector('img').src,/second.png/);
        const image=viewer.preview.querySelector('img');
        const event=new f.dom.window.MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:20,clientY:20});
        image.dispatchEvent(event);
        assert.equal(event.defaultPrevented,true);
        const button=[...f.dom.window.document.querySelectorAll('button')].find(b=>b.textContent==='Open in Photo Studio');
        assert.ok(button);assert.equal(button.disabled,false);
        await button.onclick();
        assert.equal(f.calls.find(c=>c.path.endsWith('/session')).payload.image.filename,'second.png');
        assert.ok(f.dom.window.document.querySelector('.dkst-photo-overlay'));
        // The menu has already opened this exact candidate; reuse the singleton.
        const editor=await f.editorModule.openPhotoStudio();
        assert.equal(editor.imageCandidate.index,1);
        assert.equal(editor.imageCandidate.direct,false);
        await editor.apply();
        assert.equal(f.calls.find(c=>c.path.endsWith('/materialize')).payload.image.filename,'second.png');
        const source=f.graph._nodes.find(n=>n.type==='DINKI_Photo_Studio_Source');
        const created=f.graph._nodes.filter(n=>n.type==='DINKI_Photo_Studio').at(-1);
        assert.equal(f.graph.links[created.inputs[0].link].origin_id,source.id);
        assert.equal(f.queueCalls.length,0);
        const selector=viewer.preview.querySelector('select');
        selector.value='0';selector.onchange();
        const options=[];viewer.getExtraMenuOptions(null,options);
        const firstEditor=await options.find(o=>o.content==='Open in Photo Studio').callback();
        assert.equal(firstEditor.imageCandidate.descriptor.filename,'first.png');
        firstEditor.close();
    } finally {f.cleanup();}
});

test('modal uses detached draft; Cancel leaves values, links and graph unchanged',async()=>{
    const f=await fixture();
    try {
        const editor=await f.editorModule.openPhotoStudio(f.node);
        assert.equal(editor.controls.size,32);
        assert.equal(editor.root.parentElement,f.dom.window.document.body);
        assert.equal(editor.root.getAttribute('aria-modal'),'true');
        editor.controlChange('light_exposure',1.25,true);
        await editor.select({descriptor:{filename:'depth.png',type:'input'},nodeId:f.depth.id,slot:0,direct:true,label:'Depth'},'depth');
        assert.equal(f.node.widgets.find(w=>w.name==='light_exposure').value,0);
        assert.equal(f.node.inputs[1].link,null);
        editor.close();
        assert.equal(f.graph._nodes.length,3);
        assert.equal(f.node.widgets.find(w=>w.name==='light_exposure').value,0);
        assert.equal(f.queueCalls.length,0);
    } finally {f.cleanup();}
});

test('Apply connects depth IMAGE and stores adjustments without queuing workflow',async()=>{
    const f=await fixture();
    try {
        const editor=await f.editorModule.openPhotoStudio(f.node);
        editor.controlChange('light_exposure',.8,true);
        await editor.select({descriptor:{filename:'depth.png',type:'input'},nodeId:f.depth.id,slot:0,direct:true,label:'Depth'},'depth');
        editor.controlChange('lens_blur_apply',true,true);
        await editor.apply();
        assert.equal(f.node.widgets.find(w=>w.name==='light_exposure').value,.8);
        assert.equal(f.node.widgets.find(w=>w.name==='lens_blur_apply').value,true);
        assert.equal(f.graph.links[f.node.inputs[1].link].origin_id,f.depth.id);
        assert.equal(editor.closed,true);
        assert.equal(f.queueCalls.length,0);
        const info={};f.Node.prototype.onSerialize.call(f.node,info);
        assert.equal(info.properties.dkstPhotoStudioSettings.light_exposure,.8);
        assert.equal(JSON.stringify(info).includes('session-'),false);
    } finally {f.cleanup();}
});

test('right-click branch uses real IMAGE origin, and ambiguous batch snapshots persist as input files',async()=>{
    const f=await fixture();
    try {
        const preview=f.add('PreviewImage');f.photo.connect(0,preview,0);
        f.app.nodeOutputs[preview.id]={images:[{filename:'preview.png',type:'temp'}]};
        let candidates=f.editorModule.workflowImages(f.graph);
        const candidate=candidates.find(c=>c.descriptor.filename==='preview.png');
        assert.equal(candidate.nodeId,f.photo.id);assert.equal(candidate.direct,true);
        f.app.nodeOutputs[preview.id].images.push({filename:'second.png',type:'temp'});
        candidates=f.editorModule.workflowImages(f.graph);
        const snapshot=candidates.find(c=>c.descriptor.filename==='second.png');
        assert.equal(snapshot.direct,false);
        const editor=await f.editorModule.openPhotoStudio(null,snapshot);
        await editor.apply();
        const source=f.graph._nodes.find(n=>n.type==='DINKI_Photo_Studio_Source');
        const created=f.graph._nodes.filter(n=>n.type==='DINKI_Photo_Studio').at(-1);
        assert.equal(source.widgets.find(w=>w.name==='image').value,'dkst_photo_studio/persistent.png');
        assert.equal(f.graph.links[created.inputs[0].link].origin_id,source.id);
        assert.equal(f.graph.links[preview.inputs[0].link].origin_id,f.photo.id);
        assert.equal(f.calls.filter(c=>c.path.endsWith('/materialize')).length,1);
    } finally {f.cleanup();}
});

test('Undo/Redo and Auto operate on draft, connected controls remain locked',async()=>{
    const f=await fixture();
    try {
        f.node.inputs.push({name:'light_contrast',widget:{name:'light_contrast'},link:123});
        const editor=await f.editorModule.openPhotoStudio(f.node);
        assert.equal(editor.controls.get('light_contrast').input.disabled,true);
        editor.change({...editor.draft,light_exposure:1,light_contrast:20});
        assert.equal(editor.draft.light_exposure,1);assert.equal(editor.draft.light_contrast,0);
        editor.undo();assert.equal(editor.draft.light_exposure,0);
        editor.redo();assert.equal(editor.draft.light_exposure,1);
        [...editor.root.querySelectorAll('button')].find(b=>b.textContent==='Auto').click();
        assert.equal(editor.draft.light_exposure,.75);
        assert.equal(editor.draft.color_white_balance,'Off');
        assert.equal(f.node.widgets.find(w=>w.name==='light_exposure').value,0);
        editor.close();
    } finally {f.cleanup();}
});

test('Auto preserves custom white balance and creative settings and is undoable',async()=>{
    const f=await fixture();
    try {
        const editor=await f.editorModule.openPhotoStudio(f.node);
        editor.change({...editor.draft,color_white_balance:'Custom',color_temperature:24,color_tint:8,effects_vignette:-20});
        const before={...editor.draft};
        const auto=[...editor.root.querySelectorAll('button')].find(b=>b.textContent==='Auto');
        auto.click();
        assert.equal(editor.draft.light_exposure,.75);
        for(const name of ['color_white_balance','color_temperature','color_tint','effects_vignette']) assert.equal(editor.draft[name],before[name]);
        const after={...editor.draft};
        auto.click();assert.deepEqual({...editor.draft},after);
        editor.undo();editor.undo();assert.deepEqual({...editor.draft},before);
        editor.redo();assert.deepEqual({...editor.draft},after);
        assert.equal(f.node.widgets.find(w=>w.name==='color_white_balance').value,'Off');
        editor.close();
    } finally {f.cleanup();}
});

test('coalesced preview ignores stale responses and renders newest draft',async()=>{
    const f=await fixture();
    try {
        let release;const requests=[];
        f.setPreview(payload=>{requests.push(payload);return requests.length===1?new Promise(resolve=>{release=resolve;}):{preview:'data:image/png;base64,newest',proxy:true};});
        const editor=await f.editorModule.openPhotoStudio(f.node);
        await pause(15);
        assert.equal(requests.length,1);
        editor.controlChange('light_exposure',1,true);
        editor.schedule(0);await pause(10);
        editor.controlChange('light_exposure',2,true);
        editor.schedule(0);await pause(10);
        release({preview:'data:image/png;base64,stale',proxy:false});
        await settle();await settle();
        assert.equal(requests.length,2);
        assert.equal(requests[1].settings.light_exposure,2);
        assert.equal(editor.after.src,'data:image/png;base64,newest');
        assert.match(editor.status.textContent,/Proxy preview/);
        editor.close();
    } finally {f.cleanup();}
});

test('focus picker accounts for image letterboxing and keeps float precision',async()=>{
    const f=await fixture();
    try {
        const editor=await f.editorModule.openPhotoStudio(f.node);
        await editor.select({descriptor:{filename:'depth.png',type:'input'},nodeId:f.depth.id,slot:0,direct:true,label:'Depth'},'depth');
        editor.pickFocus=true;
        editor.after.getBoundingClientRect=()=>({left:100,top:100,width:400,height:400});
        await editor.focusAt({target:editor.after,clientX:200,clientY:100});
        assert.equal(f.calls.filter(c=>c.path.endsWith('/focus')).length,0);
        await editor.focusAt({target:editor.after,clientX:200,clientY:300});
        const sampled=f.calls.find(c=>c.path.endsWith('/focus'));
        assert.deepEqual(Array.from(sampled.payload.point),[.25,.5]);
        assert.equal(editor.draft.lens_blur_focus,.712345);
        editor.close();
    } finally {f.cleanup();}
});

test('expired or absent upstream preview never falls back to arbitrary ancestor image',async()=>{
    const f=await fixture();
    try {
        const crop=f.add('ImageCrop');crop.inputs=[{name:'image',link:null}];f.photo.connect(0,crop,0);crop.connect(0,f.node,0);
        const editor=await f.editorModule.openPhotoStudio(f.node);
        assert.equal(editor.session,null);
        assert.equal(editor.applyButton.disabled,true);
        assert.match(editor.status.textContent,/Select an existing/);
        editor.close();
    } finally {f.cleanup();}
});

test('failed connection rolls back existing image/depth links',async()=>{
    const f=await fixture();
    try {
        const editor=await f.editorModule.openPhotoStudio(f.node);
        await editor.select({descriptor:{filename:'depth.png',type:'input'},nodeId:f.depth.id,slot:99,direct:true,label:'bad port'},'depth');
        await editor.apply();
        assert.equal(editor.closed,false);
        assert.equal(f.node.inputs[1].link,null);
        assert.equal(f.graph.links[f.node.inputs[0].link].origin_id,f.photo.id);
        assert.match(editor.status.textContent,/Cannot connect/);
        editor.close();
    } finally {f.cleanup();}
});

test('number input events update the preview draft immediately and coalesce a typing gesture',async()=>{
    const f=await fixture();
    try {
        const editor=await f.editorModule.openPhotoStudio(f.node);
        const input=editor.controls.get('light_exposure').input;
        input.value='0.5';input.dispatchEvent(new f.dom.window.Event('input',{bubbles:true}));
        input.value='0.75';input.dispatchEvent(new f.dom.window.Event('input',{bubbles:true}));
        assert.equal(editor.draft.light_exposure,.75);
        assert.equal(editor.controls.get('light_exposure').slider.value,'0.75');
        assert.equal(editor.history.length,1);
        editor.undo();assert.equal(editor.draft.light_exposure,0);
        editor.close();
    } finally {f.cleanup();}
});


test('overwriting a preset then applying keeps the newly saved draft and updates compact preset options',async()=>{
    const f=await fixture();
    try {
        const editor=await f.editorModule.openPhotoStudio(f.node);
        editor.controlChange('light_exposure',2,true);
        editor.preset.value='Portrait';
        await editor.savePreset();
        await editor.apply();
        assert.equal(f.node.widgets.find(w=>w.name==='light_exposure').value,2);
        assert.equal(f.node.widgets.find(w=>w.name==='preset').value,'Portrait');
        assert.equal(f.calls.find(c=>c.path.endsWith('/presets')&&c.payload).payload.overwrite,true);
    } finally {f.cleanup();}
});

test('expired cached input falls back to the actual linked image file',async()=>{
    const f=await fixture();
    try {
        f.node.dkstPhotoStudioSource='expired';
        const fetch=f.api.fetchApi;
        f.api.fetchApi=async(path,options)=>{
            const payload=options?.body&&JSON.parse(options.body);
            if(path.endsWith('/session')&&payload.source_session)return {ok:false,status:410,json:async()=>({error:'Expired'})};
            return fetch(path,options);
        };
        const editor=await f.editorModule.openPhotoStudio(f.node);
        assert.notEqual(editor.session,null);
        assert.equal(editor.capture,null);
        assert.equal(f.calls.find(c=>c.path.endsWith('/session')).payload.image.filename,'photo.png');
        editor.close();
    } finally {f.cleanup();}
});
