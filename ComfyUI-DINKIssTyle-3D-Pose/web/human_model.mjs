import * as T from './vendor/three.module.mjs';
const asset=await (await fetch(new URL('./assets/human/base.json',import.meta.url))).json();

function basis(direction,across=null){
    const y=direction.clone().normalize();
    let x=across?across.clone().addScaledVector(y,-across.dot(y)):new T.Vector3().crossVectors(y,new T.Vector3(0,0,1));
    if(x.lengthSq()<1e-8)x=new T.Vector3(1,0,0);x.normalize();const z=x.clone().cross(y).normalize();
    return new T.Matrix4().makeBasis(x,y,z);
}
function subdivision(points,weights,faces){
    const avg=(items,factors=items.map(()=>1/items.length))=>{
        const p=new T.Vector3(),w=new Map();items.forEach((item,j)=>{p.addScaledVector(item.p,factors[j]);for(const [i,v] of item.w)w.set(i,(w.get(i)||0)+v*factors[j]);});return {p,w};
    };
    const original=points.map((p,i)=>({p,w:weights[i]})),facePoints=faces.map(f=>avg(f.map(i=>original[i]))),edges=new Map(),vertexFaces=points.map(()=>[]),vertexEdges=points.map(()=>[]);
    faces.forEach((f,fi)=>{for(let j=0;j<f.length;j++){const a=f[j],b=f[(j+1)%f.length],key=Math.min(a,b)+':'+Math.max(a,b);let e=edges.get(key);if(!e){e={a,b,faces:[]};edges.set(key,e);vertexEdges[a].push(e);vertexEdges[b].push(e);}e.faces.push(fi);vertexFaces[a].push(fi);}});
    const result=original.map((v,i)=>{
        const incident=vertexEdges[i],boundary=incident.filter(e=>e.faces.length===1);
        if(boundary.length>=2)return avg([v,...boundary.slice(0,2).map(e=>original[e.a===i?e.b:e.a])],[.75,.125,.125]);
        const n=incident.length;if(n<3)return v;
        const f=avg(vertexFaces[i].map(fi=>facePoints[fi])),r=avg(incident.map(e=>avg([original[e.a],original[e.b]])));
        return avg([f,r,v],[1/n,2/n,(n-3)/n]);
    });
    for(const e of edges.values()){e.index=result.length;result.push(e.faces.length===2?avg([original[e.a],original[e.b],...e.faces.map(fi=>facePoints[fi])]):avg([original[e.a],original[e.b]]));}
    const offset=result.length;result.push(...facePoints);const quads=[];
    faces.forEach((f,fi)=>{for(let j=0;j<f.length;j++){const a=f[j],b=f[(j+1)%f.length],c=f[(j+f.length-1)%f.length];quads.push([a,edges.get(Math.min(a,b)+':'+Math.max(a,b)).index,offset+fi,edges.get(Math.min(a,c)+':'+Math.max(a,c)).index]);}});
    return {vertices:result,quads};
}

function modelPoint(rig,values){
    const [x,y,z]=values,head=rig.p.head,bodyScale=.12*rig.p.chest/.191,headScale=head/(8.4913-6.1555);
    const t=T.MathUtils.smoothstep(y,5.8,6.6),xz=T.MathUtils.lerp(bodyScale,headScale,t);
    const py=y<=6.1555?(y+8.1676)*(2-head)/(6.1555+8.1676):2-head+(y-6.1555)*headScale;
    return new T.Vector3(x*xz,py,z*xz);
}
function setWorldPose(rig,goals){
    for(const bone of Object.values(rig.joints)){
        const goal=goals.get(bone.name);rig.root.updateMatrixWorld(true);
        bone.position.copy(goal.position).applyMatrix4(bone.parent.matrixWorld.clone().invert());
        bone.quaternion.copy(bone.parent.getWorldQuaternion(new T.Quaternion()).invert().multiply(goal.rotation));
    }
    rig.root.updateMatrixWorld(true);
}
export function prepareAuthoredRig(rig){
    // Bind directly to the model's actual authored joint positions. Skinning
    // brings its A-pose into our neutral control frame; the mesh is not warped
    // independently around each joint.
    const bind=new Map(),neutral=new Map(),lengths={};
    for(const bone of Object.values(rig.joints)){
        if(bone.parent!==rig.root&&!bone.parent.isBone)bone.parent.quaternion.identity();
        const name=bone.name,a=asset.anchors[name];
        if(!a){bind.set(name,{position:new T.Vector3(),rotation:new T.Quaternion()});neutral.set(name,{position:new T.Vector3(),rotation:new T.Quaternion()});continue;}
        const head=modelPoint(rig,a.head),tail=modelPoint(rig,a.tail),delta=tail.clone().sub(head);lengths[name]=delta.length();
        let rotation=new T.Quaternion();
        if(/_(shoulder|elbow|hip|knee|wrist)$/.test(name)||/_\d$/.test(name)){
            let across;
            if(a.across){const sign=name[0]==='L'?1:-1;across=new T.Vector3(...a.across).multiplyScalar(sign);}
            rotation.setFromRotationMatrix(basis(delta.clone().negate(),across));
        }
        bind.set(name,{position:head,rotation});neutral.set(name,{position:head.clone(),rotation:new T.Quaternion()});
    }
    for(const side of ['L','R']){
        const key=n=>side+'_'+n;
        const shoulder=neutral.get(key('shoulder')).position,elbow=shoulder.clone().add(new T.Vector3(0,-lengths[key('shoulder')],0));
        const wrist=elbow.clone().add(new T.Vector3(0,-lengths[key('elbow')],0));neutral.get(key('elbow')).position=elbow;neutral.get(key('wrist')).position=wrist;
        const hip=neutral.get(key('hip')).position,knee=neutral.get(key('knee')).position,ankle=neutral.get(key('ankle')).position;
        knee.x=ankle.x=hip.x;knee.z=ankle.z=hip.z;
        neutral.get(key('toe')).position=ankle.clone().add(bind.get(key('toe')).position.clone().sub(bind.get(key('ankle')).position));
        const wristInverse=bind.get(key('wrist')).rotation.clone().invert();
        for(const finger of ['thumb','index','middle','ring','pinky'])for(let j=0;j<3;j++){
            const name=key(finger+'_'+j);
            neutral.get(name).position=j===0?wrist.clone().add(bind.get(name).position.clone().sub(bind.get(key('wrist')).position).applyQuaternion(wristInverse)):neutral.get(key(finger+'_'+(j-1))).position.clone().add(new T.Vector3(0,-lengths[key(finger+'_'+(j-1))],0));
            if(j===2)rig.fingerTips[key(finger)].local.set(0,-lengths[name],0);
        }
        // Solid hands/feet remain an optional drawing aid with the same rig.
        rig.feet[side].simple.position.y=rig.p.foot-ankle.y;
    }
    rig.p.upperArm=lengths.L_shoulder;rig.p.lowerArm=lengths.L_elbow;
    rig.authoredLengths=lengths;setWorldPose(rig,bind);
    return ()=>setWorldPose(rig,neutral);
}

export function authoredGeometry(rig,state,bones){
    const p=rig.p,lengths=rig.authoredLengths,points=[],weights=[];
    const boneIndices=new Map(bones.map((b,i)=>[b.name,i]));
    for(let i=0;i<asset.positions.length/3;i++){
        const point=modelPoint(rig,asset.positions.slice(i*3,i*3+3)),influences=new Map();
        for(let j=0;j<4;j++){
            const index=asset.skin_indices[i*4+j],weight=asset.skin_weights[i*4+j];if(weight)influences.set(boneIndices.get(asset.joint_names[index]),weight);
        }
        points.push(point);weights.push(influences);
    }
    const {vertices,quads}=subdivision(points,weights,asset.quads);
    const positions=[],skinIndices=[],skinWeights=[],guides=[],regions=[],indices=[],correctives=bones.filter(b=>/_(elbow|knee)$/.test(b.name)),morphs=correctives.map(()=>[]);
    const centers=correctives.map(b=>b.getWorldPosition(new T.Vector3()));
    const markers=bones.filter(b=>/_(shoulder|elbow|wrist|hip|knee|ankle)$/.test(b.name)).map(b=>({name:b.name,center:b.getWorldPosition(new T.Vector3())}));
    const inverse=bones.map(b=>b.matrixWorld.clone().invert()),main=[];
    for(const v of vertices){
        const items=[...v.w].filter(([,w])=>w>0).sort((a,b)=>b[1]-a[1]).slice(0,4),sum=items.reduce((s,[,w])=>s+w,0),index=items[0][0],name=bones[index].name;
        main.push(name);const point=v.p;point.y=Math.max(0,point.y);positions.push(point.x,point.y,point.z);
        for(let j=0;j<4;j++){skinIndices.push(items[j]?.[0]??0);skinWeights.push((items[j]?.[1]??0)/sum);}
        const local=point.clone().applyMatrix4(inverse[index]),isHead=name==='head',length=lengths[name]||.1;
        let ring=1;for(const marker of markers){const c=marker.center,radius=/shoulder|hip/.test(marker.name)?.045:/wrist|ankle/.test(marker.name)?.023:.033;ring=Math.min(ring,Math.abs(Math.hypot(point.x-c.x,point.y-c.y)-radius)+(Math.abs(point.z-c.z)>.16?1:0));}
        guides.push(local.x,isHead?local.y-p.head*.10:local.y+length*.5,local.z,ring);regions.push(isHead?1:0);
        for(let j=0;j<correctives.length;j++){const d=point.clone().sub(centers[j]),strength=Math.exp(-d.lengthSq()/.002)*.007;d.y=0;d.normalize().multiplyScalar(strength);morphs[j].push(d.x,d.y,d.z);}
    }
    const simple=name=>(state.appearance.simple_hands&&/_(wrist|thumb|index|middle|ring|pinky)/.test(name))||(state.appearance.simple_feet&&/_(ankle|toe)$/.test(name));
    for(const f of quads)if(!f.every(i=>simple(main[i])))indices.push(f[0],f[1],f[2],f[0],f[2],f[3]);
    const geometry=new T.BufferGeometry();geometry.setIndex(indices);geometry.setAttribute('position',new T.Float32BufferAttribute(positions,3));geometry.setAttribute('skinIndex',new T.Uint16BufferAttribute(skinIndices,4));geometry.setAttribute('skinWeight',new T.Float32BufferAttribute(skinWeights,4));geometry.setAttribute('guide',new T.Float32BufferAttribute(guides,4));geometry.setAttribute('guideRegion',new T.Float32BufferAttribute(regions,1));geometry.morphTargetsRelative=true;geometry.morphAttributes.position=morphs.map(m=>new T.Float32BufferAttribute(m,3));
    if(state.appearance.simple_hands||state.appearance.simple_feet){
        // Skinned bounds iterate all attributes, including unreferenced points.
        // Remove the hidden detailed vertices as well as their faces.
        const used=[...new Set(indices)].sort((a,b)=>a-b),remap=new Int32Array(vertices.length);used.forEach((index,i)=>remap[index]=i);
        const compact=attribute=>{const size=attribute.itemSize,data=new attribute.array.constructor(used.length*size);used.forEach((index,i)=>{for(let j=0;j<size;j++)data[i*size+j]=attribute.array[index*size+j];});return new T.BufferAttribute(data,size);};
        for(const [name,attribute] of Object.entries(geometry.attributes))geometry.setAttribute(name,compact(attribute));
        geometry.morphAttributes.position=geometry.morphAttributes.position.map(compact);geometry.setIndex(indices.map(i=>remap[i]));
    }
    geometry.computeVertexNormals();
    geometry.userData.modelSource='MakeHuman CC0 authored base mesh';
    return {geometry,correctiveNames:correctives.map(b=>b.name)};
}
