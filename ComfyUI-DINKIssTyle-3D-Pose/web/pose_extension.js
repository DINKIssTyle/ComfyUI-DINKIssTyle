import {app} from '../../scripts/app.js';
import {api} from '../../scripts/api.js';
import {clone,DEFAULT,validate,sceneString,hash} from './state.mjs';
import {PoseEditor} from './editor.mjs';
import {renderArtifact} from './renderer.mjs';

export const NODE_CLASS='DKST_3D_Pose';
const widget=(node,name)=>node.widgets?.find(w=>w.name===name);
const imageURL=id=>api.apiURL(`/dkst/3d-pose/render/${id}`);
function report(error) {const message=error?.message||String(error);if(app.ui?.dialog?.show)app.ui.dialog.show(message);else alert(message);}
export function effectiveScene(inputs) {
    for(const name of ['heads','width','height','scene_json','render_id']) if(Array.isArray(inputs[name])) throw Error('Set body proportions, resolution and scene data in the 3D Pose editor. Linked inputs require server rendering, which is not supported.');
    const scene=inputs.scene_json ? JSON.parse(inputs.scene_json) : clone(DEFAULT);
    scene.ratio=Number(inputs.heads);scene.output={width:Number(inputs.width),height:Number(inputs.height)};
    return validate(scene);
}
function nodeScene(node) {
    return effectiveScene(Object.fromEntries(['heads','width','height','scene_json','render_id'].map(n=>[n,widget(node,n)?.value])));
}
function saveScene(node,scene) {
    const raw=sceneString(scene);widget(node,'scene_json').value=raw;
    widget(node,'heads').value=scene.ratio;widget(node,'width').value=scene.output.width;widget(node,'height').value=scene.output.height;
    node.graph?.change?.();node.setDirtyCanvas?.(true,true);return raw;
}
const renders=new Map();
async function requestRender(raw,previousID='') {
    if(previousID && /^[a-f\d]{64}$/.test(previousID)) {
        // The immutable ID is useful only for the exact snapshot recorded on this client.
        // Imported workflow artifacts are verified by the backend during execution.
        const check=await api.fetchApi(`/dkst/3d-pose/render/${previousID}?require_pose=1`,{method:'HEAD'});
        if(check.ok)return {render_id:previousID};
    }
    if(renders.has(raw))return renders.get(raw);
    const pending=(async()=>{
        const scene=validate(JSON.parse(raw)),artifact=await renderArtifact(scene);
        const form=new FormData();form.append('scene',raw);form.append('image',artifact.image,'pose.png');form.append('pose',JSON.stringify(artifact.pose));
        const response=await api.fetchApi('/dkst/3d-pose/render',{method:'POST',body:form});
        const result=await response.json();if(!response.ok)throw Error(result.error||'Unable to save the 3D pose render.');
        const digest=await hash(raw);if(digest&&digest!==result.scene_hash)throw Error('The saved render does not match this scene.');
        return result;
    })();
    renders.set(raw,pending);try{return await pending;}finally{renders.delete(raw);}
}
export async function prepareInputs(inputs,knownRaw=null) {
    const scene=effectiveScene(inputs),raw=sceneString(scene);
    // Unknown/moved workflows re-render once, so a missing or stale render never wins.
    const result=await requestRender(raw,knownRaw===raw?inputs.render_id:'');
    return {...inputs,heads:scene.ratio,width:scene.output.width,height:scene.output.height,scene_json:raw,render_id:result.render_id};
}
function walk(graph,visit,seen=new Set()) {
    if(!graph||seen.has(graph))return;seen.add(graph);
    for(const node of graph._nodes||[]){visit(node);if(node.subgraph)walk(node.subgraph,visit,seen);}
}

app.registerExtension({
    name:'DINKIssTyle.3DPose',
    setup() {
        const original=api.queuePrompt;
        api.queuePrompt=async function(number,prompt,...rest) {
            const entries=Object.entries(prompt?.output||{}).filter(([,n])=>n.class_type===NODE_CLASS);
            if(!entries.length)return original.call(this,number,prompt,...rest);
            // Copy before any await; subsequent edits cannot change this submitted job.
            const frozen=clone(prompt);const nodes=[];
            walk(app.rootGraph||app.graph,n=>{if(n.comfyClass===NODE_CLASS)nodes.push(n);});
            for(const [id] of entries) {
                const input=frozen.output[id].inputs;
                const raw=sceneString(effectiveScene(input));
                const node=nodes.find(n=>String(n.id)===id)||nodes.find(n=>{try{return sceneString(nodeScene(n))===raw;}catch{return false;}});
                const prepared=await prepareInputs(input,node?.dkstPoseRenderedRaw);
                frozen.output[id].inputs=prepared;
                // Reflect only if this node still has the same scene. UI edits made while
                // rendering remain dirty and are used in the next queued job.
                if(node&&sceneString(nodeScene(node))===raw){
                    widget(node,'scene_json').value=prepared.scene_json;widget(node,'render_id').value=prepared.render_id;
                    node.dkstPoseRenderedRaw=raw;node.dkstPosePreview?.(prepared.render_id);
                }
                // Keep the submitted workflow metadata consistent with its API snapshot.
                const syncWorkflow=graph=>{for(const serialized of graph?.nodes||[]){if(String(serialized.id)===id&&serialized.type===NODE_CLASS&&Array.isArray(serialized.widgets_values)){serialized.widgets_values[3]=prepared.scene_json;serialized.widgets_values[4]=prepared.render_id;}}};
                syncWorkflow(frozen.workflow);
            }
            return original.call(this,number,frozen,...rest);
        };
    },
    async beforeRegisterNodeDef(nodeType,nodeData) {
        if(nodeData.name==='DKST_3D_OpenPose'){
            const created=nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated=function(...args){
                const result=created?.apply(this,args);
                for(const [name,label] of Object.entries({include_body:'Include body',include_hands:'Include fingers',include_head:'Include head landmarks',line_width:'Line width',joint_radius:'Joint radius',include_floor:'Include floor grid'})){
                    const w=widget(this,name);if(w)w.label=label;
                }
                return result;
            };
            return;
        }
        if(nodeData.name!==NODE_CLASS)return;
        const created=nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated=function(...args){
            const result=created?.apply(this,args),node=this;
            if(!widget(node,'scene_json').value)saveScene(node,clone(DEFAULT));
            for(const name of ['scene_json','render_id']) {
                const w=widget(node,name);w.type='hidden';w.hidden=true;w.computeSize=()=>[0,-4];w.computeLayoutSize=()=>({minHeight:0,maxHeight:0,minWidth:0});
            }
            for(const name of ['heads','width','height']) {
                const w=widget(node,name),previous=w.callback;
                w.label={heads:'Proportion',width:'Width',height:'Height'}[name];
                w.callback=function(...a){previous?.apply(this,a);try{const scene=nodeScene(node);saveScene(node,scene);if(node.dkstPoseEditor&&!node.dkstPoseEditor.closed){node.dkstPoseEditor.state=scene;node.dkstPoseEditor.updateWorld();node.dkstPoseEditor.history.push(scene);node.dkstPoseEditor.rebuildPanel();}}catch(e){report(e);}};
            }
            node.dkstPoseOpen=()=>{try{
                if(node.dkstPoseEditor&&!node.dkstPoseEditor.closed)return;
                node.dkstPoseEditor=new PoseEditor({state:nodeScene(node),ui:node.properties?.dkstPoseUI,
                    onChange:s=>saveScene(node,s),onUI:ui=>{node.properties??={};node.properties.dkstPoseUI=ui;},
                    onRender:async scene=>{
                        const raw=sceneString(scene),inputs={heads:scene.ratio,width:scene.output.width,height:scene.output.height,scene_json:raw,render_id:widget(node,'render_id').value};
                        const prepared=await prepareInputs(inputs,node.dkstPoseRenderedRaw);
                        if(sceneString(nodeScene(node))===raw){widget(node,'render_id').value=prepared.render_id;node.dkstPoseRenderedRaw=raw;node.dkstPosePreview?.(prepared.render_id);}
                    },onClose:()=>{node.dkstPoseEditor=null;},
                });
            }catch(e){report(e);}};
            const open=node.addWidget('button','Open 3D Pose Editor',null,node.dkstPoseOpen,{serialize:false});open.serialize=false;
            if(node.addDOMWidget) {
                const box=document.createElement('div');Object.assign(box.style,{width:'100%',height:'100%',minHeight:'100px',display:'flex',alignItems:'center',justifyContent:'center',overflow:'hidden',background:'#182230',borderRadius:'7px'});
                const image=document.createElement('img');Object.assign(image.style,{width:'100%',height:'100%',objectFit:'contain',display:'none'});image.alt='3D Pose camera render';
                const hint=document.createElement('span');hint.textContent='Open editor · Queue to render';Object.assign(hint.style,{color:'#a1b9d0',padding:'12px',font:'12px system-ui'});box.append(image,hint);
                node.addDOMWidget('dkst_pose_preview','image',box,{serialize:false,hideOnZoom:false,getMinHeight:()=>180});
                node.dkstPosePreview=id=>{if(!id)return;image.onload=()=>{image.style.display='block';hint.style.display='none';};image.onerror=()=>{image.style.display='none';hint.style.display='block';hint.textContent='Queue to restore missing render';};image.src=imageURL(id);};
            }
            node.setSize?.([340,390]);return result;
        };
        const configured=nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure=function(...args){
            const r=configured?.apply(this,args);
            if(!this.outputs?.some(o=>o.type==='DKST_POSE'))this.addOutput?.('pose','DKST_POSE');
            this.dkstPoseRenderedRaw=null;this.dkstPosePreview?.(widget(this,'render_id')?.value);return r;
        };
        const removed=nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved=function(...args){this.dkstPoseEditor?.close();return removed?.apply(this,args);};
    },
});
