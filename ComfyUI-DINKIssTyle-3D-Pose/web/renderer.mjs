import * as T from './vendor/three.module.mjs';
import {Rig} from './rig.mjs';
import {projectOpenPose} from './openpose.mjs';
import {createFloor} from './floor.mjs';

export function makeRenderer() {
    const renderer=new T.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:true});
    renderer.outputColorSpace=T.SRGBColorSpace;
    renderer.setPixelRatio(1); return renderer;
}
export class World {
    constructor(state) {
        this.scene=new T.Scene(); this.rig=new Rig(state); this.scene.add(this.rig.root);
        this.hemi=new T.HemisphereLight('#ffffff','#566079',1.7); this.scene.add(this.hemi);
        this.key=new T.DirectionalLight('#fff5e9',state.appearance.light*2); this.key.position.set(-3,5,4); this.scene.add(this.key);
        this.fill=new T.DirectionalLight('#bbd8ff',1); this.fill.position.set(3,2,-3); this.scene.add(this.fill);
        this.floor=createFloor();this.scene.add(this.floor);
        this.setState(state);
    }
    setState(state) {
        if(this.ratio!==undefined && this.ratio!==state.ratio) { this.rig.dispose(); this.rig=new Rig(state); this.scene.add(this.rig.root); }
        this.ratio=state.ratio; this.state=state; this.rig.apply(state);
        this.scene.background=state.appearance.transparent?null:new T.Color(state.appearance.background);
        this.key.intensity=state.appearance.light*2; this.floor.visible=state.appearance.floor;
        if(!this.camera || this.projection!==state.camera.projection) {
            this.projection=state.camera.projection;
            this.camera=this.projection==='perspective'?new T.PerspectiveCamera():new T.OrthographicCamera();
        }
        this.setCamera(state.camera,state.output.width/state.output.height);
    }
    setCamera(c,aspect) {
        this.camera.position.fromArray(c.position); this.camera.quaternion.fromArray(c.quaternion);
        this.camera.near=.01; this.camera.far=200;
        if(this.camera.isPerspectiveCamera) { this.camera.fov=c.fov; this.camera.aspect=aspect; }
        else { this.camera.top=c.ortho/2; this.camera.bottom=-c.ortho/2; this.camera.left=-c.ortho*aspect/2; this.camera.right=c.ortho*aspect/2; }
        this.camera.updateProjectionMatrix(); this.camera.updateMatrixWorld(true);
    }
    readCamera(target) {
        this.camera.position.clampScalar(-100,100);
        this.state.camera.position=this.camera.position.toArray();
        this.state.camera.quaternion=this.camera.quaternion.clone().normalize().toArray();
        if(target) this.state.camera.target=target.clampScalar(-100,100).toArray();
        if(this.camera.isOrthographicCamera && this.camera.zoom!==1) {
            this.state.camera.ortho/=this.camera.zoom; this.camera.zoom=1;
            this.setCamera(this.state.camera,this.state.output.width/this.state.output.height);
        }
    }
    dispose() { this.rig.dispose(); this.floor.geometry.dispose(); this.floor.material.dispose(); }
}
export async function renderArtifact(state) {
    const world=new World(state);let renderer;
    try {
        renderer=makeRenderer();
        const {width,height}=state.output;
        const max=renderer.getContext().getParameter(renderer.getContext().MAX_RENDERBUFFER_SIZE);
        if(width>max || height>max) throw Error(`The maximum render size in this browser is ${max}px.`);
        renderer.setSize(width,height,false);
        for(const mark of world.rig.marks) mark.visible=false;
        renderer.render(world.scene,world.camera);
        const blob=await new Promise((resolve,reject)=>renderer.domElement.toBlob(b=>b?resolve(b):reject(Error('PNG rendering failed.')),'image/png'));
        return {image:blob,pose:projectOpenPose(world)};
    } finally { world.dispose(); renderer?.dispose(); renderer?.forceContextLoss(); }
}
export async function renderPNG(state) {return (await renderArtifact(state)).image;}
