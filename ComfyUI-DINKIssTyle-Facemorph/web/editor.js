import {editMask} from './mask-editor.js';
import { api } from '../../scripts/api.js';
import { MorphRenderer } from './renderer.js';
import {EYE_CONTROLS,SECTIONS,adjustment,setAdjustment} from './liquify.js';
import { CURVES, REGION_OPTIONS, PAIRS, PARENTS, regionValue, setRegionWeight, clearRegionWeights, upgradeState, updateMesh } from './math.js';
const ROOT='/dinki/facemorph';
const el=(tag,props={},...children)=>{
    const node=document.createElement(tag);
    for(const [key,value] of Object.entries(props)){
        if(key==='text')node.textContent=value;
        else if(key.startsWith('on'))node.addEventListener(key.slice(2),value);
        else if(key==='class')node.className=value;
        else node[key]=value;
    }
    node.append(...children);return node;
};
async function request(path,data){
    const response=await api.fetchApi(ROOT+path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const result=await response.json();if(!response.ok)throw new Error(result.error||response.statusText);return result;
}
const maskUrl=id=>api.apiURL(ROOT+'/mask/'+encodeURIComponent(id));
const imageUrl=id=>api.apiURL(ROOT+'/image/'+encodeURIComponent(id));
async function loadImage(id){const image=new Image();image.src=imageUrl(id);await image.decode();return image;}
function supports(asset){
    const p=asset.points,oval=CURVES[0].map(j=>p[j]);
    const center=[0,1].map(d=>oval.reduce((s,v)=>s+v[d],0)/oval.length);
    oval.forEach((v,j)=>p[468+j]=v.map((x,d)=>center[d]+(x-center[d])*1.12));
}
export async function openEditor(node){
    if(node.dkstMorphClose){node.dkstMorphFocus?.();return;}
    const widget=node.widgets.find(w=>w.name==='project');
    let state;
    try{state=JSON.parse(widget.value);}catch{throw new Error('Project JSON is invalid. Restore a saved workflow.');}
    if(state.version!==1||!Array.isArray(state.assets))throw new Error('Unsupported project format.');
    upgradeState(state);state.liquify??={};state.hair??={enabled:true,protect_base:true};widget.value=JSON.stringify(state);
    let selected=0,region='all',mesh=null,images=[],closed=false,busy=false,raf=0,generation=0,previewEdge=640,modelReady=false,auxiliary=null;
    const undo=[],liquifyExpanded=new Map();
    const dialog=el('dialog',{class:'dkst-fm'});
    const style=el('style',{text:`
        .dkst-fm{box-sizing:border-box;width:min(1760px,98vw);height:92vh;max-width:none;max-height:none;padding:18px;background:#151922;color:#e7edf5;border:1px solid #435062;border-radius:14px;font:14px system-ui;}
        .dkst-fm::backdrop{background:#0009}.dkst-fm *{box-sizing:border-box}.dkst-fm button,.dkst-fm select,.dkst-fm input[type=number]{background:#293343;color:inherit;border:1px solid #4b5b70;border-radius:6px;padding:7px;}
        .dkst-fm button{cursor:pointer}.dkst-fm button:disabled{opacity:.4;cursor:wait}.dkst-fm button:hover{border-color:#69c5d1}.dkst-fm .bar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:12px;}
        .dkst-fm .layout{display:grid;grid-template-columns:260px minmax(0,1fr) 270px;gap:16px;height:calc(100% - 82px);min-height:200px}.dkst-fm .sidebar{overflow:auto;padding-right:5px}.dkst-fm .views{overflow:auto;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;align-content:start;min-width:0;}
        .dkst-fm canvas{display:block;width:100%;height:auto;touch-action:none;background:#080a0f}.dkst-fm .card{padding:10px;margin-bottom:10px;border:1px solid #364457;border-radius:9px}.dkst-fm .active{border-color:#72d0dc;background:#202c39;}
        .dkst-fm .photo{width:58px;height:58px;object-fit:cover;border-radius:6px}.dkst-fm .small{font-size:12px;color:#aab9cc;line-height:1.5}.dkst-fm .status{white-space:pre-wrap;max-height:64px;overflow:auto;color:#a6e1e7;margin:8px 0;min-height:20px}.dkst-fm label{display:block;margin:9px 0}.dkst-fm input[type=range]{width:100%;accent-color:#77d3df}.dkst-fm .actions{margin-left:auto;display:flex;gap:9px;align-items:center}.dkst-fm .help-button{width:32px;height:32px;border-radius:50%;font-weight:700}.dkst-fm .popup{width:min(580px,86vw);max-height:78vh;overflow:auto;padding:22px;background:#1c2431;color:#e7edf5;border:1px solid #4b5b70;border-radius:12px;font:14px/1.6 system-ui}.dkst-fm .popup::backdrop{background:#0009}.dkst-fm .mask-popup{width:min(800px,90vw)}.dkst-fm .mask-popup canvas{max-height:48vh;width:auto;max-width:100%;margin:auto;cursor:crosshair}.dkst-fm .mask-popup .bar{gap:8px}.dkst-fm .mask-popup label{flex:1;font-size:12px}.dkst-fm .mask-popup [aria-pressed=true]{border-color:#77d3df;background:#1b5059}.dkst-fm .photo-actions{flex-wrap:wrap}.dkst-fm .popup p{white-space:pre-wrap}.dkst-fm .popup h3{font-size:17px}.dkst-fm h3{font-size:15px;font-weight:600;margin:0 0 10px}.dkst-fm .warn{color:#ffc77a;}
        .dkst-fm .view-pane{min-width:0}.dkst-fm .view-title{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:20px;height:20px}.dkst-fm .source-title{display:flex}.dkst-fm .title-label{flex:0 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis}.dkst-fm .title-file{flex:1 1 0;min-width:0;overflow:hidden;text-overflow:ellipsis}
        .dkst-fm .sidebar-head{position:sticky;top:0;z-index:1;background:#151922;padding-bottom:10px}.dkst-fm .sidebar-head label{margin:0 0 7px}.dkst-fm .sidebar-head select{width:100%;min-width:0}.dkst-fm .photo-header{display:flex;align-items:center;gap:7px;margin-bottom:9px;min-width:0}.dkst-fm .photo-pick{display:flex;align-items:center;gap:9px;flex:1;min-width:0;text-align:left;border:0;background:transparent;padding:0}.dkst-fm .photo-pick .photo{width:44px;height:44px;flex-shrink:0}.dkst-fm .photo-meta{min-width:0;flex:1}.dkst-fm .photo-name{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:600;line-height:20px}.dkst-fm .photo-index{display:block;font-size:11px;color:#aab9cc}.dkst-fm .remove-photo{flex:0 0 26px;width:26px;height:26px;padding:0;border:0;background:transparent;color:#aab9cc;font-size:21px;line-height:1}.dkst-fm .remove-photo:hover{background:#4c2935;color:#ffc0c0}.dkst-fm .photo-actions{display:flex;align-items:center;justify-content:space-between;gap:6px;margin-bottom:10px}.dkst-fm .photo-actions button{font-size:12px;padding:5px 7px}.dkst-fm .base-badge{font-size:12px;color:#91e1da;background:#1b393e;border-radius:5px;padding:5px 7px}.dkst-fm .photo-weight{display:block;margin:8px 0}.dkst-fm .weight-head{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:12px}.dkst-fm .weight-head input[type=number]{width:62px;padding:3px 5px;font-variant-numeric:tabular-nums}.dkst-fm .photo-weight input[type=range]{display:block;margin:7px 0 0}.dkst-fm .region-reset{width:100%;font-size:12px;padding:5px;margin-top:5px}
        .dkst-fm .liquify-head{position:sticky;top:0;background:#151922;z-index:1;padding-bottom:3px}.dkst-fm .liquify{scroll-padding-top:90px;overflow:auto;border-left:1px solid #364457;padding-left:14px;min-width:0}.dkst-fm .liquify details{border-top:1px solid #364457;padding:10px 0}.dkst-fm .liquify summary{cursor:pointer;font-weight:600;margin-bottom:8px}.dkst-fm .liquify .individual{margin-top:10px;font-size:12px}.dkst-fm .liquify .control{margin:10px 0}.dkst-fm .liquify .control-head{display:flex;justify-content:space-between;align-items:center;gap:8px}.dkst-fm .liquify input[type=number]{width:62px;padding:3px 5px}.dkst-fm .liquify .individual h4{font-size:12px;color:#aab9cc;margin:12px 0 5px}.dkst-fm .liquify .bar{justify-content:space-between}
        @media(max-width:1100px){.dkst-fm .layout{grid-template-columns:210px minmax(0,1fr) 240px}.dkst-fm .views{grid-template-columns:1fr}}
        @media(max-width:760px){.dkst-fm .layout{grid-template-columns:1fr;overflow:auto}.dkst-fm .sidebar,.dkst-fm .views,.dkst-fm .liquify{overflow:visible}.dkst-fm .liquify{border-left:0;padding-left:0}}
    `});
    const status=el('div',{class:'status',text:''});
    const sidebar=el('div',{class:'sidebar'}),source=el('canvas'),preview=el('canvas'),liquifyPanel=el('div',{class:'liquify'});
    const sourceFile=el('span',{class:'title-file'});
    const sourceTitle=el('h3',{class:'view-title source-title'},el('span',{class:'title-label',text:'Edit correspondence points'}),sourceFile),resultTitle=el('h3',{class:'view-title',text:'Live morph preview'});
    let renderer;
    const say=message=>{status.textContent=message;};
    const save=()=>{widget.value=JSON.stringify(state);node.setDirtyCanvas?.(true,true);};
    const remember=()=>{undo.push(JSON.stringify(state));if(undo.length>30)undo.shift();};
    const close=()=>{if(closed)return;closed=true;generation++;cancelAnimationFrame(raf);window.removeEventListener('paste',onPaste,true);auxiliary?.close();auxiliary?.remove();renderer?.dispose();dialog.close();dialog.remove();delete node.dkstMorphClose;delete node.dkstMorphFocus;};
    const button=(text,fn)=>el('button',{text,onclick:()=>Promise.resolve().then(fn).catch(e=>say(e.message))});
    const upload=el('input',{type:'file',accept:'image/*',multiple:true,hidden:true});
    const help=button('?',showHelp);help.className='help-button';help.title='Help';help.setAttribute('aria-label','Help');
    const hairToggle=el('input',{type:'checkbox',checked:state.hair.enabled!==false});
    hairToggle.addEventListener('change',async()=>{if(busy){hairToggle.checked=state.hair.enabled!==false;return;}remember();state.hair.enabled=hairToggle.checked;save();await refresh().catch(e=>say(e.message));});
    const toolbar=el('div',{class:'bar'},el('strong',{text:'DKST Face Morph Studio'}),
        button('Add photos',async()=>{if(!busy&&await ensureModel())upload.click();}),button('Undo',async()=>{if(busy||!undo.length)return;state=JSON.parse(undo.pop());selected=Math.min(selected,state.assets.length-1);save();await refresh();}),
        el('label',{},hairToggle,document.createTextNode(' Hair protection')),el('div',{class:'actions'},help,button('Apply',()=>{save();close();})));
    const views=el('div',{class:'views'},el('div',{class:'view-pane'},sourceTitle,source),el('div',{class:'view-pane'},resultTitle,preview));
    dialog.append(style,toolbar,status,el('div',{class:'layout'},sidebar,views,liquifyPanel),upload);
    document.body.append(dialog);dialog.showModal();
    node.dkstMorphClose=close;node.dkstMorphFocus=()=>dialog.focus();
    dialog.addEventListener('cancel',event=>{if(event.target!==dialog)return;event.preventDefault();save();close();});
    try{renderer=new MorphRenderer(preview);}catch(error){close();throw error;}
    function popup(title,...content){
        auxiliary?.close();auxiliary?.remove();
        const panel=el('dialog',{class:'popup'},el('h3',{text:title}),...content);
        panel.setAttribute('aria-label',title);dialog.append(panel);auxiliary=panel;
        panel.addEventListener('close',()=>{if(auxiliary===panel)auxiliary=null;panel.remove();});
        panel.showModal();return panel;
    }
    function showHelp(){
        const panel=popup('Face Morph Studio help',
            el('p',{text:'Drop 2–10 photos into the editor, click Add photos, or paste clipboard images with Ctrl+V / Cmd+V. Slider weights are relative and normalized automatically.'}),
            el('p',{text:'Select Eyes or Eyebrows to adjust both sides together. Choose a Left or Right child region for individual control. Drag a correspondence point and release to update the mesh.'}),
            el('p',{text:'Photos, correspondence points, and weights are saved in the workflow. Originals are stored in ComfyUI input/DKST_Facemorph. Copy those originals too when moving a workflow to another computer.'}),
            el('p',{text:'Face Liquify adjusts the completed morph. Eye controls are paired by default; expand Individual eyes for separate adjustments. Zero leaves a control unchanged. Enabled toggles this stage for both preview and Run; Reset all restores its defaults.'}),
            el('p',{text:'Hair protection excludes source hair, reduces covered shape contributions, and preserves the base hairstyle in front of the adjusted face. Open Mask on a photo to correct detection with Include face, Exclude hair, and Restore strokes. All corrections are saved in the workflow. Fine strands and heavily covered faces may require manual review.'}),
            el('p',{text:'Preview uses reduced images and linear interpolation. Click Apply to return to the workflow, then use ComfyUI Run to render from originals with high-quality interpolation. Color match applies only to Run.'}));
        panel.append(button('Done',()=>panel.close()));
    }
    async function ensureModel(){
        if(modelReady)return true;
        busy=true;
        try{
            const info=await request('/models');
            if(closed)return false;
            if(info.landmarker&&info.hair){modelReady=true;return true;}
            return await new Promise(resolve=>{
                const message=el('p'),actions=el('div',{class:'bar'});
                const panel=popup('Preparing face models',message,actions);
                panel.addEventListener('cancel',event=>event.preventDefault());
                panel.addEventListener('close',()=>resolve(false),{once:true});
                async function download(){
                    actions.style.display='none';
                    message.textContent='Required face or hair models are missing. Downloading them now. This only needs to run once. Please keep this window open.';
                    try{
                        let result=info;
                        for(const kind of ['landmarker','hair'])if(!result[kind])result=await request('/models',{kind});
                        if(closed)return;
                        if(!result.landmarker||!result.hair)throw new Error('The face model is still unavailable after download.');
                        modelReady=true;resolve(true);panel.close();
                    }catch(error){
                        if(closed)return;
                        message.textContent='Could not download the face model.\n'+error.message;
                        actions.style.display='flex';
                    }
                }
                actions.append(button('Retry',download),button('Cancel',()=>{resolve(false);close();}));
                download();
            });
        }finally{busy=false;}
    }
    async function addFiles(files){
        if(busy||!await ensureModel())return;busy=true;remember();
        try{
            for(const file of files){
                if(closed)break;
                if(state.assets.length>=10){say('You can add up to 10 photos.');break;}
                say(`${file.name}: Uploading and analyzing face…`);
                const form=new FormData();form.append('image',file);
                const res=await api.fetchApi(ROOT+'/upload',{method:'POST',body:form});
                const asset=await res.json();if(!res.ok)throw new Error(asset.error||res.statusText);
                const analysis=await request('/analyze',{id:asset.id});
                if(closed)break;
                analysis.name=file.name;state.assets.push(analysis);selected=state.assets.length-1;save();
            }
            await refresh();
        }catch(error){if(!closed){await refresh().catch(()=>{});say(error.message);}}finally{busy=false;upload.value='';}
    }
    upload.addEventListener('change',()=>addFiles(Array.from(upload.files)));
    dialog.addEventListener('dragover',event=>{event.preventDefault();});
    dialog.addEventListener('drop',event=>{event.preventDefault();addFiles(Array.from(event.dataTransfer.files));});
    function onPaste(event){
        if(closed||!dialog.open)return;
        const data=event.clipboardData;
        let images=Array.from(data?.items??[]).filter(item=>item.kind==='file'&&item.type.startsWith('image/')).map(item=>item.getAsFile()).filter(Boolean);
        if(!images.length)images=Array.from(data?.files??[]).filter(file=>file.type.startsWith('image/'));
        // Text paste remains native, including numeric inputs and editable fields.
        if(!images.length)return;
        event.preventDefault();event.stopImmediatePropagation();
        if(auxiliary?.open)return;
        if(busy){say('Please wait for the current operation to finish, then paste again.');return;}
        if(state.assets.length>=10){say('You can add up to 10 photos.');return;}
        const extensions={'image/jpeg':'jpg','image/webp':'webp','image/gif':'gif','image/bmp':'bmp'};
        const files=images.map((file,i)=>new File([file],`Clipboard-${state.assets.length+i+1}.${extensions[file.type]??'png'}`,{type:file.type}));
        addFiles(files);
    }
    // Capture before graph paste handlers; remove the listener when the editor closes.
    window.addEventListener('paste',onPaste,true);
    function sourceDraw(){
        if(closed)return;
        const asset=state.assets[selected],image=images[selected],ctx=source.getContext('2d');
        if(!asset||!image){sourceFile.textContent='';sourceTitle.removeAttribute('title');source.width=512;source.height=512;ctx.clearRect(0,0,512,512);return;}
        const edge=768;source.width=Math.round(edge*image.width/Math.max(image.width,image.height));source.height=Math.round(edge*image.height/Math.max(image.width,image.height));
        ctx.drawImage(image,0,0,source.width,source.height);
        const pixel=point=>[point[0]*(source.width-1),point[1]*(source.height-1)];
        CURVES.forEach(curve=>{
            ctx.strokeStyle='#69f0df';ctx.lineWidth=1;
            ctx.beginPath();curve.forEach((j,i)=>{const [x,y]=pixel(asset.points[j]);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.closePath();ctx.stroke();
        });
        ctx.setLineDash([]);
        const points=new Set(CURVES.flat());
        for(const j of points){const [x,y]=pixel(asset.points[j]);ctx.beginPath();ctx.arc(x,y,2.5,0,Math.PI*2);ctx.fillStyle='#ecfffd';ctx.fill();}
        sourceFile.textContent=` · ${asset.name||selected+1}`;sourceTitle.title=`Edit correspondence points · ${asset.name||selected+1}`;
    }
    function draw(){
        cancelAnimationFrame(raf);raf=requestAnimationFrame(()=>{
            if(closed||!mesh)return;
            try{const start=performance.now();renderer.draw(state,updateMesh(state,mesh),previewEdge);const elapsed=performance.now()-start;resultTitle.textContent=`Live morph preview · ${previewEdge}px · ${elapsed.toFixed(0)}ms`;if(elapsed>50)previewEdge=Math.max(384,Math.round(previewEdge*.8));}
            catch(error){say(error.message);preview.width=preview.width;}
        });
    }
    async function refresh(){
        const ticket=++generation;mesh=null;
        const loaded=await Promise.all(state.assets.map(a=>loadImage(a.id)));
        if(closed||ticket!==generation)return;
        images=loaded;renderer.setImages(images);sourceDraw();panel();
        if(state.assets.length>=2){
            const prepared=await request('/prepare',state);
            if(closed||ticket!==generation)return;
            await installMasks(prepared,ticket);if(closed||ticket!==generation)return;mesh=prepared;draw();say('Ready.'+(mesh.warnings?.length?'\n'+mesh.warnings.join('\n'):''));
        }else{preview.width=512;preview.height=512;say('Add at least 2 photos.');}
    }
    async function installMasks(prepared,ticket){
        const loaded=await Promise.all((prepared.mask_ids??[]).map(async id=>{const im=new Image();im.src=maskUrl(id);await im.decode();return im;}));
        if(!closed&&ticket===generation)renderer.setMasks(loaded);
    }
    async function prepareEdited(){
        const ticket=++generation;mesh=null;save();sourceDraw();
        if(state.assets.length<2)return;
        try{const prepared=await request('/prepare',state);if(closed||ticket!==generation)return;await installMasks(prepared,ticket);if(closed||ticket!==generation)return;mesh=prepared;draw();say('Correspondence points updated.');}catch(error){say(error.message);}
    }
    function liquifyControls(){
        state.liquify??={};const controls=[];
        const sync=()=>controls.forEach(update=>update());
        const enabled=el('input',{type:'checkbox',checked:state.liquify.enabled!==false});
        enabled.addEventListener('change',()=>{remember();state.liquify.enabled=enabled.checked;save();draw();});
        liquifyPanel.replaceChildren(el('div',{class:'liquify-head'},el('h3',{text:'Face Liquify'}),el('div',{class:'bar'},
            el('label',{},enabled,document.createTextNode(' Enabled')),
            button('Reset all',()=>{remember();state.liquify={enabled:state.liquify.enabled!==false};save();sync();draw();}))));
        function section(title,defaultOpen=true,className=''){
            const details=el('details',{class:className,open:liquifyExpanded.get(title)??defaultOpen},el('summary',{text:title}));
            details.addEventListener('toggle',()=>liquifyExpanded.set(title,details.open));return details;
        }
        function control(key,label,side){
            const name=(side==='eye_l'?'Left ':side==='eye_r'?'Right ':'')+label;
            const slider=el('input',{type:'range',min:-100,max:100,step:1});slider.setAttribute('aria-label','Liquify '+name);
            const number=el('input',{type:'number',min:-100,max:100,step:1});number.setAttribute('aria-label','Liquify '+name+' value');
            const update=()=>{const value=adjustment(state.liquify,key,side);slider.value=value;number.value=value;};
            controls.push(update);update();
            function change(value){
                if(busy||!Number.isFinite(value))return;
                setAdjustment(state.liquify,key,Math.max(-100,Math.min(100,value)),side);save();sync();draw();
            }
            slider.addEventListener('pointerdown',remember);
            slider.addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(event.key))remember();});
            number.addEventListener('focus',remember);
            slider.addEventListener('input',()=>change(Number(slider.value)));
            number.addEventListener('input',()=>change(number.valueAsNumber));
            return el('div',{class:'control'},el('div',{class:'control-head'},el('span',{text:label}),number),slider);
        }
        const eyes=section('Eyes');
        EYE_CONTROLS.forEach(([key,label])=>eyes.append(control(key,label,'eyes')));
        eyes.append(control('distance','Eye distance','eyes'));
        const individual=section('Individual eyes',false,'individual');
        individual.append(button('Link both eyes',()=>{remember();delete state.liquify.eye_l;delete state.liquify.eye_r;save();sync();draw();}));
        for(const [side,title] of [['eye_l','Left eye (screen)'],['eye_r','Right eye (screen)']]){
            individual.append(el('h4',{text:title}));EYE_CONTROLS.forEach(([key,label])=>individual.append(control(key,label,side)));
        }
        eyes.append(individual);liquifyPanel.append(eyes);
        for(const [title,items] of SECTIONS){const group=section(title);items.forEach(([key,label])=>group.append(control(key,label)));liquifyPanel.append(group);}
    }
    function panel(){
        hairToggle.checked=state.hair?.enabled!==false;liquifyControls();
        sidebar.replaceChildren();
        const regionSelect=el('select',{},...REGION_OPTIONS.map(([value,text])=>el('option',{value,text})));
        regionSelect.value=region;regionSelect.addEventListener('change',()=>{region=regionSelect.value;panel();});sidebar.append(el('div',{class:'sidebar-head'},el('label',{text:'Region to adjust'}),regionSelect));
        state.assets.forEach((asset,i)=>{
            const card=el('div',{class:'card'+(i===selected?' active':'')});
            const name=asset.name||`Photo ${i+1}`;
            const thumb=el('img',{class:'photo',src:imageUrl(asset.id),alt:name});
            const pick=button('',()=>{if(busy)return;selected=i;sourceDraw();panel();});
            pick.className='photo-pick';pick.title=name;pick.setAttribute('aria-label',`Select photo ${i+1}: ${name}`);
            pick.append(thumb,el('span',{class:'photo-meta'},el('span',{class:'photo-name',text:name}),el('span',{class:'photo-index',text:`Photo ${i+1}`})));
            const remove=button('×',async()=>{if(busy)return;remember();state.assets.splice(i,1);if(state.base===i)state.base=0;else if(state.base>i)state.base--;selected=Math.max(0,Math.min(selected,state.assets.length-1));save();await refresh();});
            remove.className='remove-photo';remove.title='Remove photo';remove.setAttribute('aria-label',`Remove photo ${i+1}`);
            card.append(el('div',{class:'photo-header'},pick,remove));
            const photoActions=el('div',{class:'photo-actions'});
            if(i===state.base)photoActions.append(el('span',{class:'base-badge',text:'Base photo ✓'}));
            else photoActions.append(button('Use as base',async()=>{if(busy)return;remember();state.base=i;save();await refresh();}));
            if(i===selected){const analyze=button('Reanalyze',()=>reanalyze(asset.face_index||0));analyze.title='Reanalyze photo and reset correspondence points';photoActions.append(analyze);}
            if(i===selected)photoActions.append(button('Mask',async()=>{
                if(busy)return;
                await editMask({asset,image:images[i],popup,button,el,request,url:maskUrl,apply:async edits=>{
                    const previous=asset.mask_edit;remember();asset.mask_edit=edits;
                    try{await refresh();save();}
                    catch(error){if(previous===undefined)delete asset.mask_edit;else asset.mask_edit=previous;undo.pop();await refresh().catch(()=>{});throw error;}
                }});
            }));
            card.append(photoActions);
            for(const [key,label] of [['shape','Shape'],['feature','Appearance']]){
                const isOverride=region!=='all'&&asset.regions?.[region]?.[key]!==undefined;
                const value=regionValue(asset,region,key);
                const independent=(PAIRS[region]??[]).some(child=>asset.regions?.[child]?.[key]!==undefined);
                const caption=el('span',{text:label,title:(key==='feature'?'Appearance and texture':label)+(independent?' (individual overrides)':region!=='all'&&!isOverride?' (inherited)':'')});
                const number=el('input',{type:'number',min:0,max:100,step:.1,value});number.setAttribute('aria-label',`${label} weight value for photo ${i+1}`);
                const slider=el('input',{type:'range',min:0,max:100,step:.1,value});slider.setAttribute('aria-label',`${label} weight for photo ${i+1}`);
                slider.addEventListener('pointerdown',remember);number.addEventListener('focus',remember);
                slider.addEventListener('keydown',event=>{if(event.key.startsWith('Arrow'))remember();});
                const change=value=>{
                    if(busy||!Number.isFinite(value))return;
                    const v=Math.round(Math.max(0,Math.min(100,value))*10)/10;
                    setRegionWeight(asset,region,key,v);slider.value=v;number.value=v;save();draw();
                };
                slider.addEventListener('input',()=>change(Number(slider.value)));
                number.addEventListener('input',()=>change(number.valueAsNumber));
                number.addEventListener('blur',()=>number.value=regionValue(asset,region,key));
                card.append(el('div',{class:'photo-weight'},el('span',{class:'weight-head'},caption,number),slider));
            }
            if(region!=='all'){const reset=button(PARENTS[region]?'Use paired weights':'Use global weights',()=>{remember();clearRegionWeights(asset,region);save();panel();draw();});reset.className='region-reset';reset.title=PAIRS[region]?'Reset both sides to global weights':'Restore inherited region weights';card.append(reset);}
            if(i===selected){
                if(asset.face_count>1){
                    const face=el('select',{},...Array.from({length:asset.face_count},(_,j)=>el('option',{value:j,text:`Face ${j+1}${j===0?' (largest)':''}`})));
                    face.value=asset.face_index;face.addEventListener('change',()=>reanalyze(Number(face.value)).catch(e=>say(e.message)));card.append(face);
                }
                if(asset.warnings?.length)card.append(el('div',{class:'small warn',text:asset.warnings.join('\n')}));
            }
            sidebar.append(card);
        });
    }
    async function reanalyze(faceIndex){
        if(busy)return;busy=true;remember();say('Reanalyzing selected photo…');
        try{const old=state.assets[selected],result=await request('/analyze',{id:old.id,face_index:faceIndex});
            if(closed)return;state.assets[selected]={...old,...result,shape:old.shape,feature:old.feature,regions:old.regions};save();await refresh();
        }finally{busy=false;}
    }
    let dragging=-1;
    const coords=event=>{const r=source.getBoundingClientRect();return [(event.clientX-r.left)/r.width,(event.clientY-r.top)/r.height];};
    source.addEventListener('pointerdown',event=>{
        if(busy||!state.assets[selected])return;
        const [x,y]=coords(event),r=source.getBoundingClientRect();let best=12;
        for(const j of new Set(CURVES.flat())){
            const p=state.assets[selected].points[j],distance=Math.hypot((p[0]-x)*r.width,(p[1]-y)*r.height);
            if(distance<best){best=distance;dragging=j;}
        }
        if(dragging>=0){remember();source.setPointerCapture(event.pointerId);}
    });
    source.addEventListener('pointermove',event=>{
        if(dragging<0)return;
        state.assets[selected].points[dragging]=coords(event).map(v=>Math.max(0,Math.min(1,v)));
        supports(state.assets[selected]);sourceDraw();
    });
    const endDrag=()=>{if(dragging<0)return;dragging=-1;prepareEdited();};
    source.addEventListener('pointerup',endDrag);source.addEventListener('pointercancel',endDrag);
    try{if(await ensureModel()&&!closed)await refresh();}catch(error){say(error.message);}
}
