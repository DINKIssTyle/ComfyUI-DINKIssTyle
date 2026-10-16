import * as T from './vendor/three.module.mjs';
import {OrbitControls} from './vendor/OrbitControls.mjs';
import {TransformControls} from './vendor/TransformControls.mjs';
import {World,makeRenderer} from './renderer.mjs';
import {clone,DEFAULT,RATIOS,JOINTS,PRESETS,History,euler,quaternion,validate,applyPreset,mirror,copySide} from './state.mjs';
import {label,solveIK} from './rig.mjs';
import {lensFOV,focalFromFOV} from './camera.mjs';
import {createBox} from './objects.mjs';
import {drawDepthPreview} from './depth.mjs';
import {projectOpenPose} from './openpose.mjs';

let activeEditor;
function el(tag,className='',text='') { const e=document.createElement(tag); e.className=className; e.textContent=text; return e; }
function button(text,fn,className='') { const b=el('button',className,text); b.type='button'; b.onclick=fn; return b; }
function select(values,value,fn) { const s=el('select'); for(const [key,text] of values) { const o=el('option','',text); o.value=key; s.append(o); } s.value=String(value); s.onchange=()=>fn(s.value); return s; }
function number(value,min,max,step,fn) { const i=el('input'); i.type='number'; i.value=value; i.min=min; i.max=max; i.step=step; i.onchange=()=>{ const n=Number(i.value); if(!Number.isFinite(n)||n<min||n>max) {i.value=value;return;} fn(n); }; return i; }
function row(parent,text,input) { const r=el('label','dkst-pose-row'); r.append(el('span','',text),input); parent.append(r); return input; }
function checkbox(parent,text,value,fn) { const i=el('input'); i.type='checkbox'; i.checked=value; i.onchange=()=>fn(i.checked); return row(parent,text,i); }
function section(parent,title) { const d=el('details'); d.open=true; d.append(el('summary','',title)); parent.append(d); return d; }
function triplet(parent,title,values,fn,min=-180,max=180,step=.5) {
    parent.append(el('div','',title)); const r=el('div','dkst-pose-triplet'); parent.append(r);
    const inputs=['X','Y','Z'].map((axis,i)=>{const l=el('label','',axis); const inp=number(Number(values[i].toFixed(2)),min,max,step,()=>fn(inputs.map(input=>Number(input.value)))); l.append(inp);r.append(l);return inp;});
    return inputs;
}
function download(name,data) {
    const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
    const a=el('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

export class PoseEditor {
    constructor(options) {
        activeEditor?.close(); this.options=options;
        this.state=validate(clone(options.state||DEFAULT)); this.history=new History(this.state);
        this.selected='head';this.mode='external';this.previewMode='Mesh';this.constrained=true;this.locked=true;this.closed=false;
        this.abort=new AbortController(); this.world=new World(this.state);
        try {this.renderer=makeRenderer();this.previewRenderer=makeRenderer();}
        catch(error){this.world.dispose();this.renderer?.dispose();this.renderer?.forceContextLoss();throw Error('Unable to start WebGL. Check your browser hardware acceleration settings. '+error.message);}
        this.editCamera=new T.PerspectiveCamera(40,1,.01,200);
        const ui=options.ui;
        this.editCamera.position.fromArray(ui?.position||[3,2.4,4]);
        this.overlay=el('div','dkst-pose-overlay'); this.overlay.setAttribute('role','dialog');this.overlay.setAttribute('aria-label','3D Pose Editor');this.overlay.setAttribute('aria-modal','true');
        this.box=el('div','dkst-pose-editor');this.overlay.append(this.box);document.body.append(this.overlay);
        const cssURL=new URL('./editor.css',import.meta.url).href;
        if(!document.querySelector('link[data-dkst-pose]')) {const css=el('link');css.rel='stylesheet';css.href=cssURL;css.dataset.dkstPose='';document.head.append(css);}
        this.buildUI();
        const canvas=this.renderer.domElement;canvas.tabIndex=0;canvas.setAttribute('aria-label','3D editing canvas');this.canvasHost.append(canvas);
        this.previewHost.append(this.previewRenderer.domElement);
        this.posePreview=document.createElement('canvas');this.posePreview.style.display='none';this.previewHost.append(this.posePreview);
        this.orbit=new OrbitControls(this.editCamera,canvas); this.orbit.target.fromArray(ui?.target||[0,1,0]);this.orbit.enableDamping=false;
        this.orbit.minDistance=.2;this.orbit.maxDistance=50;this.orbit.minZoom=.05;this.orbit.maxZoom=20;this.orbit.update();
        this.orbit.addEventListener('change',()=>{ if(this.mode==='camera'&&!this.locked) {this.world.readCamera(this.orbit.target);this.syncCameraHandle();this.changed(false);} this.draw(); });
        this.orbit.addEventListener('end',()=>{if(this.mode==='camera'&&!this.locked) this.changed();else this.saveUI();});
        this.gizmo=new TransformControls(this.editCamera,canvas); this.gizmo.setSpace('local');this.gizmo.setMode('rotate');this.gizmo.setSize(.85);
        this.gizmoHelper=this.gizmo.getHelper();this.world.scene.add(this.gizmoHelper);
        this.gizmoHelper.userData.helper=true;
        this.gizmo.addEventListener('dragging-changed',event=>{this.orbit.enabled=!event.value&&(this.mode==='external'||!this.locked);if(!event.value) this.changed();});
        this.gizmo.addEventListener('objectChange',()=>this.handleTransform());
        this.gizmo.addEventListener('change',()=>this.draw());
        this.grid=new T.GridHelper(10,50,'#53657a','#293b4f');this.world.scene.add(this.grid);
        this.grid.userData.helper=true;
        this.cameraHandle=new T.Group(); const body=new T.Mesh(new T.BoxGeometry(.16,.11,.22),new T.MeshBasicMaterial({color:'#5ab8e7'}));
        body.userData.camera=true;this.cameraHandle.add(body);this.cameraPick=body;this.world.scene.add(this.cameraHandle);
        this.cameraHandle.userData.helper=true;
        this.cameraHelper=new T.CameraHelper(this.world.camera);this.world.scene.add(this.cameraHelper);
        this.cameraHelper.userData.helper=true;
        this.ikTarget=new T.Group();this.pole=new T.Group();
        for(const [g,color,size] of [[this.ikTarget,'#f5b955',.045],[this.pole,'#b393fa',.035]]) {
            g.add(new T.Mesh(new T.SphereGeometry(size,16,12),new T.MeshBasicMaterial({color})));g.visible=false;this.world.scene.add(g);
            g.userData.helper=true;
        }
        this.syncCameraHandle(); this.selectJoint(this.selected);
        let down;
        canvas.addEventListener('pointerdown',event=>{canvas.focus();down={x:event.clientX,y:event.clientY,gizmo:!!this.gizmo.axis};},{signal:this.abort.signal});
        canvas.addEventListener('pointerup',event=>{
            if(!down||down.gizmo||Math.hypot(event.clientX-down.x,event.clientY-down.y)>4) return;
            const rect=canvas.getBoundingClientRect(),pointer=new T.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);
            const ray=new T.Raycaster();ray.setFromCamera(pointer,this.currentCamera());
            const picks=[...this.world.rig.visiblePickables(),...this.world.objects.picks()];if(this.mode==='external') picks.push(this.cameraPick);
            if(this.ikTarget.visible) picks.push(this.ikTarget.children[0],this.pole.children[0]);
            const intersection=ray.intersectObjects(picks,false)[0],hit=intersection?.object;
            if(hit===this.ikTarget.children[0]) this.attachIK(false);
            else if(hit===this.pole.children[0]) this.attachIK(true);
            else if(hit?.userData.camera) this.selectJoint('camera');
            else if(hit?.userData.objectID)this.selectJoint('object:'+hit.userData.objectID);
            else if(hit?.userData.surface)this.selectJoint(this.world.rig.jointAt(intersection));
            else if(hit?.userData.joint) this.selectJoint(hit.userData.joint);
        },{signal:this.abort.signal});
        canvas.addEventListener('keydown',e=>this.key(e),{signal:this.abort.signal});
        this.overlay.addEventListener('keydown',e=>{if(e.key==='Escape'){e.stopPropagation();this.close();}},{signal:this.abort.signal});
        for(const name of ['pointerdown','pointerup','wheel','keydown','contextmenu']) this.overlay.addEventListener(name,e=>e.stopPropagation(),{signal:this.abort.signal});
        this.observer=new ResizeObserver(()=>this.draw());this.observer.observe(this.canvasHost);this.draw();
        this.previousFocus=document.activeElement;canvas.focus();activeEditor=this;
    }
    buildUI() {
        const header=el('header');header.append(el('strong','','DKST · 3D Pose'));
        header.append(button('Undo',()=>this.restore(this.history.move(-1))),button('Redo',()=>this.restore(this.history.move(1))));
        this.applyButton=button('Apply render',()=>this.render().catch(e=>this.message(e.message,true)),'primary');header.append(this.applyButton,button('Close',()=>this.close()));this.box.append(header);
        const body=el('div','dkst-pose-body');this.panel=el('aside','dkst-pose-panel');const view=el('div','dkst-pose-view');body.append(this.panel,view);this.box.append(body);
        const toolbar=el('div','dkst-pose-toolbar');view.append(toolbar);
        this.viewSelect=select([['external','External view'],['camera','Camera view']],this.mode,v=>this.setView(v));toolbar.append(this.viewSelect);
        const lockLabel=el('div');this.lockInput=checkbox(lockLabel,'Lock camera',this.locked,v=>{this.locked=v;this.setView(this.mode);});toolbar.append(lockLabel);
        toolbar.append(button('View → Camera',()=>this.adoptView()),button('Select camera',()=>this.selectJoint('camera')));
        toolbar.append(button('Rotate',()=>{this.gizmo.setMode('rotate');this.gizmo.setSpace('local');}),button('Move',()=>this.translationMode()),button('Scale',()=>{if(!this.selected.startsWith('object:')){this.message('Select a box to scale.');return;}this.gizmo.setMode('scale');this.gizmo.setSpace('local');}));
        const viewPresets=[['front','Front'],['back','Back'],['left','Left'],['right','Right'],['top','Top']];
        toolbar.append(select([['','View direction'],...viewPresets],'',v=>v&&this.standardView(v)),button('Frame scene',()=>this.fitScene()),button('Frame body',()=>this.fitCamera()));
        this.canvasHost=el('div','dkst-pose-canvas');view.append(this.canvasHost);
        this.help=el('div','dkst-pose-help','Drag: orbit · Right drag: pan · Wheel: zoom · Click body: select joint');this.canvasHost.append(this.help);
        const preview=el('div','dkst-pose-preview');preview.append(select(['Mesh','Depth','OpenPose'].map(n=>[n,n]),this.previewMode,v=>{this.previewMode=v;this.draw();}));this.previewHost=el('div');preview.append(this.previewHost);this.canvasHost.append(preview);
        const footer=el('footer');this.status=el('span','status','Select a joint to rotate it or enter angles. Edits are saved in the workflow.');footer.append(this.status);this.box.append(footer);
        this.rebuildPanel();
    }
    rebuildPanel() {
        this.panel.replaceChildren();const s=this.state;
        const model=section(this.panel,'Body & pose');
        row(model,'Proportion',select(RATIOS.map(n=>[n,n+' heads']),s.ratio,v=>{s.ratio=Number(v);this.updateWorld();this.changed();}));
        row(model,'Preset',select([['','Choose preset'],...Object.keys(PRESETS).map(n=>[n,n])],'',v=>{if(v){applyPreset(s,v);this.world.rig.apply(s);this.ground();this.changed();this.rebuildPanel();}}));
        const actions=el('div','dkst-pose-actions');actions.append(button('Mirror pose',()=>{mirror(s);this.updateWorld();this.changed();}),button('L → R',()=>{copySide(s,'L');this.updateWorld();this.changed();}),button('R → L',()=>{copySide(s,'R');this.updateWorld();this.changed();}));model.append(actions);
        model.append(button('Reset pose',()=>{s.root=[0,0,0];s.joints=clone(DEFAULT.joints);this.updateWorld();this.changed();}));
        const joints=section(this.panel,'Joints');
        const visibleJoints=JOINTS.filter(j=>!(s.appearance.simple_hands&&/_\d$/.test(j))&&!(s.appearance.simple_feet&&j.endsWith('_toe')));
        this.jointSelect=select([['camera','Render camera'],...visibleJoints.map(j=>[j,label(j)]),...s.objects.map(b=>['object:'+b.id,b.name])],this.selected,v=>this.selectJoint(v));row(joints,'Selection',this.jointSelect);
        this.selectedFields=el('div');joints.append(this.selectedFields);
        checkbox(joints,'Joint limits',this.constrained,v=>this.constrained=v);
        const ik=el('div','dkst-pose-actions');ik.append(button('Hand / foot IK',()=>this.startIK()),button('Bend pole',()=>{if(this.ik)this.attachIK(true);else this.message('Select a wrist or ankle, then enable IK.');}));joints.append(ik);
        const camera=section(this.panel,'Render camera');
        row(camera,'Projection',select([['perspective','Perspective'],['orthographic','Orthographic']],s.camera.projection,v=>{s.camera.projection=v;this.updateWorld();this.changed();this.rebuildPanel();}));
        const focal=row(camera,'Focal length (mm)',number(s.camera.focal_mm,8,300,1,n=>{s.camera.focal_mm=n;this.updateWorld();this.changed();this.rebuildPanel();}));
        focal.disabled=s.camera.projection==='orthographic';focal.title='35mm equivalent, 36 x 24mm diagonal angle-of-view convention.';
        row(camera,'Lens preset',select([['','Custom'],...[18,24,35,50,85,135].map(n=>[n,n+' mm'])],'',v=>{if(v){s.camera.focal_mm=Number(v);this.updateWorld();this.changed();this.rebuildPanel();}}));
        camera.append(el('p','',s.camera.projection==='perspective'?`35mm equivalent · Vertical FOV ${lensFOV(s.camera.focal_mm,s.output.width/s.output.height).toFixed(1)}°`:'Orthographic camera · Focal length is inactive.'));
        row(camera,'Ortho size',number(s.camera.ortho,.1,50,.1,n=>{s.camera.ortho=n;this.updateWorld();this.changed();}));
        this.cameraPositionInputs=triplet(camera,'Position',s.camera.position,v=>{s.camera.position=v;this.retarget();this.updateWorld();this.changed();},-100,100,.05);
        this.cameraRotationInputs=triplet(camera,'Rotation (°)',euler(s.camera.quaternion),v=>{s.camera.quaternion=quaternion(...v);this.retarget();this.updateWorld();this.changed();});
        const objects=section(this.panel,'Objects');
        const add=el('div','dkst-pose-actions');for(const color of ['red','green','blue'])add.append(button('Add '+color+' box',()=>{try{const b=createBox(s,color);this.updateWorld();this.selectJoint('object:'+b.id);this.changed();this.rebuildPanel();}catch(e){this.message(e.message,true);}}));objects.append(add);
        row(objects,'Object',select([['','Select object'],...s.objects.map(b=>[b.id,b.name])],this.selected.startsWith('object:')?this.selected.slice(7):'',id=>{if(id)this.selectJoint('object:'+id);}));
        const output=section(this.panel,'Output & appearance');
        for(const [k,text] of [['width','Width'],['height','Height']]) row(output,text,number(s.output[k],64,4096,1,n=>{s.output[k]=Math.round(n);this.updateWorld();this.changed();}));
        for(const [k,text] of [['body','Body color'],['background','Background']]) {const input=el('input');input.type='color';input.value=s.appearance[k];input.onchange=()=>{s.appearance[k]=input.value;this.updateWorld();this.changed();};row(output,text,input);}
        for(const [k,text] of [['transparent','Transparent background'],['face','Face construction guides'],['floor','Render floor grid'],['simple_hands','Simplify hands'],['simple_feet','Simplify feet']]){
            const input=checkbox(output,text,s.appearance[k],v=>{s.appearance[k]=v;this.updateWorld();this.changed();this.rebuildPanel();});
            input.title={simple_hands:'Use a single hand form and omit OpenPose finger landmarks. Stored finger poses are preserved.',simple_feet:'Use one foot form without a toe joint seam. COCO-18 OpenPose ends at the ankle.',face:'Front centerline and curved brow line follow the head surface.',floor:'Include a perspective ground grid in the camera output.'}[k]||'';
        }
        row(output,'Light intensity',number(s.appearance.light,.1,5,.1,n=>{s.appearance.light=n;this.updateWorld();this.changed();}));
        const guides=section(this.panel,'Surface & guides');
        row(guides,'Surface style',select([['smooth','Smooth'],['construction','Construction']],s.appearance.centerlines||s.appearance.contours||s.appearance.joint_markers?'construction':'smooth',v=>{for(const k of ['centerlines','contours','joint_markers','outline','face'])s.appearance[k]=v==='construction';this.updateWorld();this.changed();this.rebuildPanel();}));
        for(const [k,text] of [['centerlines','Centerlines'],['contours','Cross-contours'],['joint_markers','Joint markers'],['outline','Outline']])checkbox(guides,text,s.appearance[k],v=>{s.appearance[k]=v;this.updateWorld();this.changed();});
        const guideColor=el('input');guideColor.type='color';guideColor.value=s.appearance.guide_color;guideColor.onchange=()=>{s.appearance.guide_color=guideColor.value;this.updateWorld();this.changed();};row(guides,'Guide color',guideColor);
        row(guides,'Line width (px)',number(s.appearance.line_width,.5,8,.1,n=>{s.appearance.line_width=n;this.updateWorld();this.changed();}));
        const files=section(this.panel,'Save & load');const fileActions=el('div','dkst-pose-actions');
        fileActions.append(button('Save pose',()=>download('dkst-pose.json',{version:1,kind:'pose',ratio:s.ratio,root:s.root,joints:s.joints})),button('Save scene',()=>download('dkst-scene.json',s)),button('Import JSON',()=>this.importFile()));files.append(fileActions);
        this.rebuildSelected();
    }
    rebuildSelected() {
        if(!this.selectedFields)return;this.selectedFields.replaceChildren();this.jointSelect.value=this.selected;
        const s=this.state,j=this.selected;
        if(j==='camera') { this.selectedFields.append(el('p','','Use the move or rotate gizmo in the external view to position this camera.'));return; }
        if(j.startsWith('object:')){
            const b=s.objects.find(o=>o.id===j.slice(7));if(!b)return;
            const name=el('input');name.value=b.name;name.maxLength=80;name.onchange=()=>{b.name=name.value.trim()||b.name;this.changed();this.rebuildPanel();};row(this.selectedFields,'Name',name);
            const color=el('input');color.type='color';color.value=b.color;color.onchange=()=>{b.color=color.value;this.updateWorld();this.changed();};row(this.selectedFields,'Color',color);
            triplet(this.selectedFields,'Position',b.position,v=>{b.position=v;this.updateWorld();this.changed();},-100,100,.05);
            triplet(this.selectedFields,'Rotation (°)',euler(b.quaternion),v=>{b.quaternion=quaternion(...v);this.updateWorld();this.changed();});
            triplet(this.selectedFields,'Size',b.size,v=>{b.size=v;this.updateWorld();this.changed();},.01,50,.05);
            checkbox(this.selectedFields,'Visible',b.visible,v=>{b.visible=v;this.updateWorld();this.changed();});
            this.selectedFields.append(button('Place on floor',()=>{const m=this.world.objects.meshes.get(b.id);b.position[1]-=new T.Box3().setFromObject(m).min.y;this.updateWorld();this.changed();this.rebuildSelected();}),button('Frame selected',()=>this.frameObject(b.id)),button('Duplicate object',()=>{
                if(s.objects.length>=32){this.message('Use at most 32 objects.',true);return;}
                const duplicate=createBox(s,'red'),id=duplicate.id;Object.assign(duplicate,clone(b),{id,name:(b.name+' copy').slice(0,80),position:[b.position[0]+.3,b.position[1],b.position[2]+.3]});this.updateWorld();this.selectJoint('object:'+id);this.changed();this.rebuildPanel();
            }),button('Delete object',()=>this.deleteObject()));return;
        }
        triplet(this.selectedFields,'Rotation (°)',euler(s.joints[j]),v=>{this.world.rig.joints[j].quaternion.fromArray(quaternion(...v));if(this.constrained)this.world.rig.constrain(j);this.world.rig.read(s);this.clearIK();this.changed();this.rebuildSelected();});
        if(j==='root') triplet(this.selectedFields,'Root position',s.root,v=>{s.root=v;this.updateWorld();this.changed();},-100,100,.05);
        this.selectedFields.append(button('Reset joint',()=>{s.joints[j]=[0,0,0,1];this.updateWorld();this.changed();this.rebuildSelected();}));
    }
    selectJoint(j) {
        if(j.startsWith('object:')&&!this.state.objects.some(b=>b.id===j.slice(7)))j='head';
        if(this.state.appearance.simple_hands&&/_\d$/.test(j))j=j.slice(0,1)+'_wrist';
        if(this.state.appearance.simple_feet&&j.endsWith('_toe'))j=j.slice(0,1)+'_ankle';
        const previous=this.selected;this.clearIK();this.selected=j;this.rebuildSelected();
        if(!this.gizmo)return;
        if(j==='camera') {if(this.mode!=='external')this.setView('external');if(previous!=='camera')this.fitScene();this.gizmo.attach(this.cameraHandle);}
        else if(j.startsWith('object:'))this.gizmo.attach(this.world.objects.meshes.get(j.slice(7)));
        else this.gizmo.attach(this.world.rig.joints[j]);
        this.gizmo.setMode('rotate');this.gizmo.setSpace('local');this.draw();
    }
    translationMode() {
        if(this.selected.startsWith('object:')){this.gizmo.attach(this.world.objects.meshes.get(this.selected.slice(7)));this.gizmo.setMode('translate');this.gizmo.setSpace('world');return;}
        if(this.selected!=='camera' && this.selected!=='root') {this.startIK();return;}
        this.gizmo.attach(this.selected==='root'?this.world.rig.root:this.cameraHandle);this.gizmo.setMode('translate');this.gizmo.setSpace('world');
    }
    clearIK() {this.ik=null;if(this.ikTarget){this.ikTarget.visible=false;this.pole.visible=false;}}
    startIK() {
        const match=/^([LR])_(wrist|ankle)$/.exec(this.selected);
        if(!match){this.message('Select a wrist or ankle to pose the limb with an IK target.');return;}
        this.ik={side:match[1],limb:match[2]==='wrist'?'arm':'leg'};
        this.world.rig.root.updateMatrixWorld(true);const end=this.world.rig.joints[this.selected];
        this.ikTarget.position.copy(end.getWorldPosition(new T.Vector3()));
        const middle=this.world.rig.joints[match[1]+'_'+(this.ik.limb==='arm'?'elbow':'knee')];
        this.pole.position.copy(middle.getWorldPosition(new T.Vector3())).add(new T.Vector3(0,0,this.ik.limb==='arm'?-.7:.7));
        this.ikTarget.visible=true;this.pole.visible=true;this.attachIK(false);
        this.message('Gold target: hand / foot position · Purple pole: elbow / knee bend direction');
    }
    attachIK(pole) {this.gizmo.attach(pole?this.pole:this.ikTarget);this.gizmo.setMode('translate');this.gizmo.setSpace('world');this.draw();}
    handleTransform() {
        const s=this.state;
        if(this.ik && [this.ikTarget,this.pole].includes(this.gizmo.object)) {
            solveIK(this.world.rig,this.ik.side,this.ik.limb,this.ikTarget.position,this.pole.position,this.constrained);
        } else if(this.selected==='camera') {
            this.cameraHandle.position.clampScalar(-100,100);
            s.camera.position=this.cameraHandle.position.toArray();s.camera.quaternion=this.cameraHandle.quaternion.clone().normalize().toArray();this.retarget();
            this.world.setCamera(s.camera,s.output.width/s.output.height);
        } else if(this.selected.startsWith('object:')){
            const b=s.objects.find(b=>b.id===this.selected.slice(7)),m=this.world.objects.meshes.get(b.id);
            m.position.clampScalar(-100,100);m.scale.clampScalar(.01,50);b.position=m.position.toArray();b.quaternion=m.quaternion.clone().normalize().toArray();b.size=m.scale.toArray();
        } else if(this.constrained&&this.gizmo.getMode()==='rotate') this.world.rig.constrain(this.selected);
        this.world.rig.root.position.clampScalar(-100,100);
        this.world.rig.read(s);this.syncCameraHandle();this.changed(false);this.rebuildSelected();this.draw();
    }
    retarget() {
        const c=this.state.camera, distance=Math.max(.2,new T.Vector3().fromArray(c.target).distanceTo(new T.Vector3().fromArray(c.position)));
        c.target=new T.Vector3(0,0,-distance).applyQuaternion(new T.Quaternion().fromArray(c.quaternion)).add(new T.Vector3().fromArray(c.position)).toArray();
    }
    syncCameraHandle() {
        this.cameraHandle?.position.fromArray(this.state.camera.position);this.cameraHandle?.quaternion.fromArray(this.state.camera.quaternion);this.cameraHelper?.update();
        for(const [inputs,values] of [[this.cameraPositionInputs,this.state.camera.position],[this.cameraRotationInputs,euler(this.state.camera.quaternion)]])
            inputs?.forEach((input,i)=>{if(document.activeElement!==input)input.value=Number(values[i].toFixed(2));});
    }
    updateWorld() {
        this.clearIK();this.gizmo?.detach();const previous=this.world.camera;
        this.world.setState(this.state);this.syncCameraHandle();
        if(previous!==this.world.camera && this.cameraHelper) {this.cameraHelper.dispose();this.cameraHelper.removeFromParent();this.cameraHelper=new T.CameraHelper(this.world.camera);this.cameraHelper.userData.helper=true;this.world.scene.add(this.cameraHelper);}
        if(this.orbit){this.orbit.object=this.currentCamera();if(this.mode==='camera')this.orbit.target.fromArray(this.state.camera.target);}
        this.selectJoint(this.selected);this.draw();
    }
    currentCamera() {return this.mode==='camera'?this.world.camera:this.editCamera;}
    setView(mode) {
        if(mode!==this.mode && this.mode==='external')this.saveUI();
        this.mode=mode;this.viewSelect.value=mode;this.lockInput.checked=this.locked;
        this.orbit.object=this.currentCamera();
        this.orbit.target.fromArray(mode==='camera'?this.state.camera.target:this.options.ui?.target||[0,1,0]);
        this.orbit.enabled=mode==='external'||!this.locked;this.gizmo.camera=this.currentCamera();
        if(mode==='camera'&&this.selected==='camera'){this.selected='head';this.selectJoint('head');}
        this.help.textContent=mode==='camera'?(this.locked?'Camera locked · Select a joint to edit':'Camera controls · Drag / pan / wheel · WASD move · Q/E vertical · Shift faster'):'Drag: orbit · Right drag: pan · Wheel: zoom · Click body: select joint';
        this.draw();
    }
    saveUI() {
        this.options.ui={position:this.editCamera.position.toArray(),target:this.orbit.target.toArray()};
        this.options.onUI?.(clone(this.options.ui));
    }
    adoptView() {
        if(this.mode==='camera')return;
        this.state.camera.position=this.editCamera.position.toArray();this.state.camera.quaternion=this.editCamera.quaternion.toArray();
        this.state.camera.target=this.orbit.target.toArray();this.state.camera.focal_mm=T.MathUtils.clamp(focalFromFOV(this.editCamera.fov,this.state.output.width/this.state.output.height),8,300);
        this.updateWorld();this.changed();this.rebuildPanel();
    }
    standardView(name) {
        this.setView('external');const center=this.world.rig.bounds().getCenter(new T.Vector3());
        const dirs={front:[0,0,1],back:[0,0,-1],left:[1,0,0],right:[-1,0,0],top:[0,1,.001]};
        this.editCamera.position.copy(center).add(new T.Vector3(...dirs[name]).multiplyScalar(4));this.orbit.target.copy(center);this.orbit.update();this.saveUI();this.draw();
    }
    fitScene() {
        this.setView('external');const box=this.world.bounds();
        box.expandByPoint(new T.Vector3().fromArray(this.state.camera.position));const center=box.getCenter(new T.Vector3());
        const size=box.getSize(new T.Vector3()),distance=Math.max(size.x,size.y,size.z,2)*1.7;
        this.editCamera.position.copy(center).add(new T.Vector3(1,.55,1).normalize().multiplyScalar(distance));
        this.orbit.target.copy(center);this.orbit.update();this.saveUI();this.draw();
    }
    fitCamera() {
        const box=this.world.rig.bounds(),center=box.getCenter(new T.Vector3()),size=box.getSize(new T.Vector3());
        const c=this.state.camera,aspect=this.state.output.width/this.state.output.height;
        const distance=Math.max(size.y,size.x/aspect)*.65/Math.tan(T.MathUtils.degToRad(lensFOV(c.focal_mm,aspect)/2))+size.z/2;
        const forward=new T.Vector3(0,0,1).applyQuaternion(new T.Quaternion().fromArray(c.quaternion));
        c.position=center.clone().addScaledVector(forward,distance).toArray();c.target=center.toArray();c.ortho=Math.max(size.y,size.x/aspect)*1.25;
        this.updateWorld();this.changed();this.rebuildPanel();
    }
    ground() {
        const bounds=this.world.rig.bounds();this.state.root[1]-=bounds.min.y;this.world.rig.apply(this.state);
    }
    key(event) {
        if(event.key==='Delete'||event.key==='Backspace'){if(this.selected.startsWith('object:')){event.preventDefault();this.deleteObject();}return;}
        if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();this.restore(this.history.move(event.shiftKey?1:-1));return;}
        if(this.mode!=='camera'||this.locked)return;
        const key=event.key.toLowerCase(), step=event.shiftKey ? .2 : .04;
        const moves={w:[0,0,-step],s:[0,0,step],a:[-step,0,0],d:[step,0,0],q:[0,-step,0],e:[0,step,0]};
        if(!moves[key])return;event.preventDefault();
        const delta=new T.Vector3(...moves[key]).applyQuaternion(this.world.camera.quaternion);
        this.world.camera.position.add(delta);this.orbit.target.add(delta);this.world.readCamera(this.orbit.target);this.syncCameraHandle();this.changed();this.draw();
    }
    changed(history=true) {
        if(this.closed)return;
        this.world.rig.read(this.state);
        if(history)this.history.push(this.state);
        this.options.onChange?.(clone(this.state));this.message('Edits saved · Queue to render the latest scene.');this.draw();
    }
    restore(state) {this.state=state;this.updateWorld();this.options.onChange?.(clone(state));this.rebuildPanel();this.message('Pose and camera restored.');}
    deleteObject(){if(!this.selected.startsWith('object:'))return;const id=this.selected.slice(7);this.state.objects=this.state.objects.filter(b=>b.id!==id);this.selected='head';this.updateWorld();this.changed();this.rebuildPanel();}
    frameObject(id){const m=this.world.objects.meshes.get(id);if(!m)return;this.setView('external');const b=new T.Box3().setFromObject(m),center=b.getCenter(new T.Vector3()),size=b.getSize(new T.Vector3());this.editCamera.position.copy(center).add(new T.Vector3(1,.6,1).normalize().multiplyScalar(Math.max(size.x,size.y,size.z,.2)*2.5));this.orbit.target.copy(center);this.orbit.update();this.draw();}
    async importFile() {
        const input=el('input');input.type='file';input.accept='.json,application/json';
        input.onchange=async()=>{try{const file=input.files?.[0];if(!file)return;if(file.size>128*1024)throw Error('JSON files must be no larger than 128 KiB.');
            const data=JSON.parse(await file.text());let state;
            if(data.kind==='pose')state=validate({...clone(this.state),ratio:data.ratio,root:data.root,joints:data.joints});else state=validate(data);
            this.state=state;this.updateWorld();this.changed();this.rebuildPanel();this.message('JSON imported.');
        }catch(e){this.message(e.message,true);}};input.click();
    }
    async render() {
        if(this.busy)return;this.busy=true;this.applyButton.disabled=true;this.message('Rendering and saving at output resolution…');
        try{await this.options.onRender?.(clone(this.state));this.message('Render saved. Queue in ComfyUI to output IMAGE and MASK.');}
        finally{this.busy=false;this.applyButton.disabled=false;}
    }
    message(text,error=false) {this.status.textContent=text;this.status.classList.toggle('error',error);}
    draw() {
        if(this.closed||this.frame)return;
        this.frame=requestAnimationFrame(()=>{this.frame=0;if(this.closed)return;this.drawNow();});
    }
    drawNow() {
        const host=this.canvasHost,width=Math.max(1,host.clientWidth),height=Math.max(1,host.clientHeight),aspect=this.state.output.width/this.state.output.height;
        let w=width,h=height;
        if(this.mode==='camera') {w=Math.min(width,height*aspect);h=w/aspect;}
        this.editCamera.aspect=width/height;this.editCamera.updateProjectionMatrix();
        const canvas=this.renderer.domElement;
        if(canvas.width!==Math.round(w)||canvas.height!==Math.round(h))this.renderer.setSize(Math.round(w),Math.round(h));
        const ext=this.mode==='external';this.grid.visible=!this.state.appearance.floor;this.cameraHandle.visible=ext;this.cameraHelper.visible=ext&&this.selected==='camera';this.cameraHelper.update();
        this.renderer.render(this.world.scene,this.currentCamera());
        const guides=[this.grid,this.cameraHandle,this.cameraHelper,this.gizmoHelper,this.ikTarget,this.pole,...this.world.rig.marks];const visible=guides.map(g=>g.visible);
        for(const g of guides)g.visible=false;
        try {
            const scale=Math.min(180/this.state.output.width,240/this.state.output.height);
            const pw=Math.max(1,Math.round(this.state.output.width*scale)),ph=Math.max(1,Math.round(this.state.output.height*scale));
            if(this.previewRenderer.domElement.width!==pw||this.previewRenderer.domElement.height!==ph)this.previewRenderer.setSize(pw,ph,false);
            this.posePreview.style.display=this.previewMode==='OpenPose'?'block':'none';this.previewRenderer.domElement.style.display=this.previewMode==='OpenPose'?'none':'block';
            if(this.previewMode==='Depth')drawDepthPreview(this.previewRenderer,this.world);
            else if(this.previewMode==='OpenPose')this.drawPosePreview(pw,ph);
            else this.previewRenderer.render(this.world.scene,this.world.camera);
        }
        finally{guides.forEach((g,i)=>g.visible=visible[i]);}
    }
    drawPosePreview(w,h){
        const canvas=this.posePreview;canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d');ctx.fillStyle='black';ctx.fillRect(0,0,w,h);
        const doc=projectOpenPose(this.world),person=doc.people[0],body=person.pose_keypoints_2d;
        const palette=['#ff0000','#ff5500','#ffaa00','#ffff00','#aaff00','#55ff00','#00ff00','#00ff55','#00ffaa','#00ffff','#00aaff','#0055ff','#0000ff','#5500ff','#aa00ff','#ff00ff','#ff00aa','#ff0055'];
        const line=(p,a,b,color)=>{if(!p[a*3+2]||!p[b*3+2])return;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(p[a*3]*w,p[a*3+1]*h);ctx.lineTo(p[b*3]*w,p[b*3+1]*h);ctx.stroke();};
        if(doc.floor_grid)for(const [x,y,u,v,major] of doc.floor_grid){ctx.strokeStyle=major?'#566579':'#303d4d';ctx.lineWidth=.5;ctx.beginPath();ctx.moveTo(x*w,y*h);ctx.lineTo(u*w,v*h);ctx.stroke();}
        const edges=[[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[1,0],[0,14],[14,16],[0,15],[15,17]];
        edges.forEach(([a,b],i)=>line(body,a,b,palette[i]));
        for(let i=0;i<18;i++)if(body[i*3+2]){ctx.fillStyle=palette[i];ctx.beginPath();ctx.arc(body[i*3]*w,body[i*3+1]*h,2,0,Math.PI*2);ctx.fill();}
        for(const p of [person.hand_left_keypoints_2d,person.hand_right_keypoints_2d])for(let finger=0;finger<5;finger++)for(let j=0;j<4;j++)line(p,j?finger*4+j:0,finger*4+j+1,palette[finger*3]);
    }
    close() {
        if(this.closed)return;this.changed();if(this.mode==='external')this.saveUI();this.closed=true;
        cancelAnimationFrame(this.frame);this.abort.abort();this.observer.disconnect();this.orbit.dispose();this.gizmo.dispose();this.cameraHelper.dispose();this.grid.geometry.dispose();this.grid.material.dispose();
        for(const group of [this.cameraHandle,this.ikTarget,this.pole])group.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});
        this.world.dispose();for(const r of [this.renderer,this.previewRenderer]){r.dispose();r.forceContextLoss();}
        this.overlay.remove();this.options.onClose?.();if(activeEditor===this)activeEditor=null;this.previousFocus?.focus?.();
    }
}
