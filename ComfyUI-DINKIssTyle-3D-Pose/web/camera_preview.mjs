// A custom DOM widget avoids the frontend's built-in text input renderer.
// Keep the DOM, widget value and workflow property in sync, including cached
// execution results that do not invoke the node's onExecuted callback.
const PROPERTY='dkstCameraDescription';

function description(output){
    const value=output?.camera_text??output?.ui?.camera_text;
    if(typeof value==='string')return value;
    if(Array.isArray(value))return [...value].reverse().find(v=>typeof v==='string');
}

export function attachCameraPreview(node){
    if(node.dkstSetCameraDescription)return;
    const root=document.createElement('div'),caption=document.createElement('div'),text=document.createElement('textarea');
    Object.assign(root.style,{width:'100%',height:'100%',minHeight:'0',display:'flex',flexDirection:'column',overflow:'hidden',background:'#182230',color:'#c9daeb',borderRadius:'6px',boxSizing:'border-box'});
    caption.textContent='Last executed camera';Object.assign(caption.style,{padding:'6px 8px',font:'11px system-ui',color:'#9fb5ca',flex:'0 0 auto'});
    text.readOnly=true;text.setAttribute('aria-label','Camera description');text.placeholder='Connect pose and queue to display camera information.';
    Object.assign(text.style,{width:'100%',flex:'1 1 0',minHeight:'0',resize:'none',border:'0',outline:'none',background:'transparent',color:'inherit',font:'12px/1.5 system-ui',padding:'8px',boxSizing:'border-box'});
    for(const name of ['pointerdown','mousedown','dblclick','wheel','keydown'])root.addEventListener(name,event=>event.stopPropagation());
    root.append(caption,text);
    let value='';
    const display=next=>{if(typeof next!=='string')return;value=next;text.value=next;};
    let preview;
    if(node.addDOMWidget){
        preview=node.addDOMWidget('camera_description','DKST_CAMERA_INFO',root,{
            serialize:false,hideOnZoom:false,getMinHeight:()=>160,getMaxHeight:()=>10000,getHeight:()=>200,
            getValue:()=>value,setValue:display,
        });
        preview.serialize=false;preview.options??={};preview.options.serialize=false;
    }
    node.dkstCameraText=text;node.dkstCameraPreview=preview;
    node.dkstSetCameraDescription=next=>{
        if(typeof next!=='string')return;
        const changed=value!==next;
        display(next);if(preview)preview.value=next;
        node.properties??={};node.properties[PROPERTY]=next;
        if(changed)node.graph?.change?.();node.setDirtyCanvas?.(true,true);
    };
    node.dkstRestoreCameraDescription=()=>node.dkstSetCameraDescription(node.properties?.[PROPERTY]);
    node.dkstRestoreCameraDescription();
}

export function updateCameraPreview(node,output){
    const text=description(output);if(text===undefined)return false;
    node.dkstSetCameraDescription?.(text);return true;
}

export function visitCameraNodes(graph,visit,path=[],seen=new Set()){
    if(!graph||seen.has(graph))return;seen.add(graph);
    for(const node of graph.nodes??graph._nodes??[]){
        const id=[...path,String(node.id)].join(':');
        if((node.comfyClass??node.type)==='DKST_3D_CameraInfo'){
            const ids=[id];if(path.length&&node.graph?.id)ids.unshift(`${node.graph.id}:${node.id}`);
            visit(node,ids);
        }
        if(node.subgraph)visitCameraNodes(node.subgraph,visit,[...path,String(node.id)],seen);
    }
}
