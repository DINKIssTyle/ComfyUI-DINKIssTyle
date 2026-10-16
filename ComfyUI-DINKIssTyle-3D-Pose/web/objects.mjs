import * as T from './vendor/three.module.mjs';
export const BOX_COLORS={red:'#ef3038',green:'#22c85a',blue:'#2675f5'};
export function createBox(state,color){
    if(state.objects.length>=32)throw Error('Use at most 32 objects.');
    const base=color+' box',count=state.objects.filter(b=>b.name===base||b.name.startsWith(base+' ')).length;
    const id=globalThis.crypto?.randomUUID?.()??`box-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const b={id,type:'box',name:base+(count?' '+(count+1):''),color:BOX_COLORS[color],position:[.65+(state.objects.length%4)*.6,.25,0],quaternion:[0,0,0,1],size:[.5,.5,.5],visible:true};
    state.objects.push(b);return b;
}
export class SceneObjects{
    constructor(scene){this.scene=scene;this.meshes=new Map();}
    update(objects){
        const wanted=new Set(objects.map(b=>b.id));
        for(const [id,m] of this.meshes)if(!wanted.has(id)){m.removeFromParent();m.geometry.dispose();m.material.dispose();this.meshes.delete(id);}
        for(const b of objects){
            let m=this.meshes.get(b.id);
            if(!m){m=new T.Mesh(new T.BoxGeometry(1,1,1),new T.MeshStandardMaterial({roughness:.8}));m.userData.objectID=b.id;this.meshes.set(b.id,m);this.scene.add(m);}
            m.position.fromArray(b.position);m.quaternion.fromArray(b.quaternion);m.scale.fromArray(b.size);m.visible=b.visible;m.material.color.set(b.color);m.updateMatrixWorld(true);
        }
    }
    picks(){return [...this.meshes.values()].filter(m=>m.visible);}
    bounds(){const b=new T.Box3();for(const m of this.picks())b.expandByObject(m);return b;}
    dispose(){this.update([]);}
}
