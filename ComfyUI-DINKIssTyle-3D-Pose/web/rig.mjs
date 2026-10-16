import * as T from './vendor/three.module.mjs';
import {JOINTS, limits, euler, quaternion} from './state.mjs';
import {installSurface,updateSurface} from './surface.mjs';

export const LABELS = {root:'Root',pelvis:'Pelvis',waist:'Waist',chest:'Chest',neck:'Neck',head:'Head',clavicle:'Clavicle',shoulder:'Shoulder',elbow:'Elbow',wrist:'Wrist',hip:'Hip',knee:'Knee',ankle:'Ankle',toe:'Toes',thumb:'Thumb',index:'Index',middle:'Middle',ring:'Ring',pinky:'Pinky'};
export function label(j) {
    const parts=j.split('_');
    return parts.length===1 ? LABELS[j] : `${parts[0]==='L'?'Left':'Right'} ${LABELS[parts[1]]}${parts[2]!==undefined?' '+(Number(parts[2])+1):''}`;
}
// World-space artistic proportion tables. Widths and joint sizes follow the body,
// rather than the head, so 2/3-head figures do not acquire enormous limb joints.
const PROFILES = {
    2: {neck:.050,foot:.055,torso:.470,shoulder:.230,hip:.105,chest:.174,waist:.111,pelvis:.159,upperArm:.205,lowerArm:.165,hand:.105,armRadius:.047,thighRadius:.068,footLength:.160},
    3: {neck:.060,foot:.065,torso:.560,shoulder:.218,hip:.104,chest:.169,waist:.112,pelvis:.159,upperArm:.255,lowerArm:.215,hand:.130,armRadius:.049,thighRadius:.075,footLength:.190},
    6: {neck:.080,foot:.075,torso:.630,shoulder:.245,hip:.116,chest:.197,waist:.131,pelvis:.185,upperArm:.328,lowerArm:.277,hand:.185,armRadius:.057,thighRadius:.088,footLength:.270},
    7: {neck:.074,foot:.075,torso:.635,shoulder:.239,hip:.112,chest:.191,waist:.125,pelvis:.179,upperArm:.326,lowerArm:.275,hand:.183,armRadius:.054,thighRadius:.085,footLength:.265},
    8: {neck:.070,foot:.075,torso:.625,shoulder:.235,hip:.109,chest:.187,waist:.120,pelvis:.175,upperArm:.324,lowerArm:.272,hand:.180,armRadius:.052,thighRadius:.082,footLength:.260},
    9: {neck:.066,foot:.075,torso:.600,shoulder:.229,hip:.106,chest:.181,waist:.115,pelvis:.170,upperArm:.320,lowerArm:.269,hand:.177,armRadius:.049,thighRadius:.079,footLength:.255},
};
export function proportions(ratio) {
    if(!PROFILES[ratio])throw Error('Unsupported body proportion.');
    const p={...PROFILES[ratio],head:2/ratio,chibi:ratio<=3};
    p.leg=2-p.head-p.neck-p.foot-p.torso;
    p.waistY=p.torso*.255;p.chestY=p.torso*.475;
    return p;
}

// Smooth elliptical cross-sections with a variable front/back center. Limited
// Hermite slopes retain the silhouette without overshoot between measured rings.
export function sculptGeometry(profile,radial=40,samples=5) {
    const rings=profile.map(([y,x,z,center=0])=>[y,x,z,center]).sort((a,b)=>a[0]-b[0]),points=[];
    const slope=(i,column)=>{
        if(i===0)return (rings[1][column]-rings[0][column])/(rings[1][0]-rings[0][0]);
        if(i===rings.length-1)return (rings[i][column]-rings[i-1][column])/(rings[i][0]-rings[i-1][0]);
        const a=(rings[i][column]-rings[i-1][column])/(rings[i][0]-rings[i-1][0]);
        const b=(rings[i+1][column]-rings[i][column])/(rings[i+1][0]-rings[i][0]);
        return a*b<=0?0:2*a*b/(a+b);
    };
    for(let i=0;i<rings.length-1;i++)for(let j=0;j<samples;j++){
        const t=j/samples,dy=rings[i+1][0]-rings[i][0],row=[rings[i][0]+t*dy];
        for(let c=1;c<4;c++){
            const a=rings[i][c]||0,b=rings[i+1][c]||0;
            const value=(2*t**3-3*t*t+1)*a+(t**3-2*t*t+t)*dy*slope(i,c)+(-2*t**3+3*t*t)*b+(t**3-t*t)*dy*slope(i+1,c);
            row[c]=c===3?value:Math.max(.00001,value);
        }
        points.push(row);
    }
    points.push(rings.at(-1));
    const vertices=[],indices=[];
    for(const [y,rx,rz,cz=0] of points)for(let j=0;j<=radial;j++){
        const a=j/radial*Math.PI*2;vertices.push(Math.cos(a)*rx,y,Math.sin(a)*rz+cz);
    }
    for(let i=0;i<points.length-1;i++)for(let j=0;j<radial;j++){
        const a=i*(radial+1)+j,b=a+radial+1;indices.push(a,b,a+1,a+1,b,b+1);
    }
    for(const [ring,bottom] of [[0,true],[points.length-1,false]]){
        const center=vertices.length/3;vertices.push(0,points[ring][0],points[ring][3]||0);
        for(let j=0;j<radial;j++){
            const a=ring*(radial+1)+j;indices.push(...(bottom?[center,a,a+1]:[center,a+1,a]));
        }
    }
    const geometry=new T.BufferGeometry();geometry.setAttribute('position',new T.Float32BufferAttribute(vertices,3));geometry.setIndex(indices);geometry.computeVertexNormals();
    // Weld shading normals along the duplicated angular seam.
    const normals=geometry.attributes.normal;
    for(let i=0;i<points.length;i++){
        const a=i*(radial+1),b=a+radial,v=new T.Vector3().fromBufferAttribute(normals,a).add(new T.Vector3().fromBufferAttribute(normals,b)).normalize();
        normals.setXYZ(a,v.x,v.y,v.z);normals.setXYZ(b,v.x,v.y,v.z);
    }
    geometry.computeBoundingBox();geometry.computeBoundingSphere();return geometry;
}
export class Rig {
    constructor(state) {
        this.root=new T.Group();this.joints={};this.fingerTips={};this.hands={};this.feet={};this.pickables=[];this.resources=[];this.marks=[];
        this.material=new T.MeshStandardMaterial({color:state.appearance.body,roughness:.72});
        this.socketMaterial=new T.MeshStandardMaterial({color:state.appearance.body,roughness:.62});
        this.jointMat=new T.MeshBasicMaterial({color:'#7a9ab5',wireframe:true,transparent:true,opacity:.3});
        this.faceMat=new T.MeshBasicMaterial({color:'#243342'});
        this.resources.push(this.material,this.socketMaterial,this.jointMat,this.faceMat);
        this.p=proportions(state.ratio);const p=this.p;
        const group=(name,parent,pos,restZ=0)=>{
            const g=new T.Bone();g.name=name;
            if(restZ){const bind=new T.Group();bind.position.fromArray(pos);bind.rotation.z=restZ;parent.add(bind);bind.add(g);}
            else{g.position.fromArray(pos);parent.add(g);}
            this.joints[name]=g;return g;
        };
        const root=group('root',this.root,[0,0,0]);
        const pelvis=group('pelvis',root,[0,p.leg+p.foot,0]);
        this.form(pelvis,[
            [-p.torso*.14,p.pelvis*.46,.065,0],[-p.torso*.07,p.pelvis*.86,.088,-.008],
            [p.torso*.035,p.pelvis,.105,-.008],[p.torso*.15,p.pelvis*.91,.100,0],
            [p.waistY+.018,p.waist,.083,0],
        ],'pelvis');
        const waist=group('waist',pelvis,[0,p.waistY,0]);
        const abdomen=p.chestY-p.waistY;
        this.form(waist,[[-.025,p.waist,.083,0],[abdomen*.20,p.waist*.96,.081,.002],
            [abdomen*.70,p.waist*1.035,.086,.002],[abdomen+.025,p.chest*.71,.090,0]],'waist');
        const chest=group('chest',waist,[0,abdomen,0]);
        const rib=p.torso-p.chestY,neckWidth=p.chibi?.034:.042;
        this.form(chest,[[-.025,p.chest*.70,.090,0],[rib*.18,p.chest*.78,.102,.003],
            [rib*.46,p.chest*.94,.119,.005],[rib*.70,p.chest,.115,-.004],
            [rib*.84,p.chest*.94,.094,-.009],[rib*.95,p.chest*.48,.065,-.003],
            [rib,neckWidth,.046,0]],'chest');
        const neck=group('neck',chest,[0,rib,0]);
        this.form(neck,[[-.015,neckWidth,.047,0],[p.neck*.25,neckWidth*.91,.042,-.005],
            [p.neck*.8,neckWidth*.90,.042,-.006],[p.neck+.02,neckWidth*.90,.043,0]],'neck');
        const head=group('head',neck,[0,p.neck+p.head/2,0]);
        const h=p.head;
        const headProfile=p.chibi?
            [[-.5,.015,.025,.04],[-.48,.12,.12,.035],[-.40,.28,.25,.015],[-.20,.365,.335,0],
                [0,.39,.37,-.01],[.20,.38,.375,-.03],[.37,.31,.325,-.04],[.46,.18,.20,-.04],[.5,.0001,.0001,-.02]]:
            [[-.5,.035,.05,.05],[-.48,.13,.15,.045],[-.41,.245,.225,.025],[-.26,.315,.305,.015],
                [-.02,.345,.343,0],[.20,.35,.36,-.02],[.36,.29,.315,-.035],[.46,.16,.19,-.035],[.5,.0001,.0001,-.03]];
        const headShape=this.form(head,headProfile.map(r=>r.map(n=>n*h)),'head',56,7);
        // Drawing construction guides: front centerline and wrapped brow line.
        const face=new T.Group();head.add(face);this.face=face;
        const surface=new T.Mesh(headShape.geometry,this.material);
        for(const horizontal of [true,false]){
            const path=[];
            for(let i=0;i<=48;i++){
                const angle=(i/48-.5)*Math.PI*.96;
                const direction=horizontal?new T.Vector3(Math.sin(angle),0,Math.cos(angle)):new T.Vector3(0,0,1);
                const y=horizontal?h*.10:h*(i/48-.5)*.84;
                const origin=direction.clone().multiplyScalar(h);origin.y=y;
                const ray=new T.Raycaster(origin,direction.clone().negate());
                const hit=ray.intersectObject(surface,false)[0];
                if(hit)path.push(hit.point.clone().addScaledVector(direction,.001));
            }
            const geometry=new T.TubeGeometry(new T.CatmullRomCurve3(path),80,h*.008,8,false);this.resources.push(geometry);face.add(new T.Mesh(geometry,this.faceMat));
        }
        const nose=this.ellipsoid(head,[0,-h*.10,h*.335],[h*.037,h*.068,h*.052],'head');nose.rotation.x=-.12;
        for(const side of ['L','R']){
            const sign=side==='L'?1:-1;
            const clavicle=group(side+'_clavicle',chest,[sign*p.shoulder*.32,rib-.052,0]);
            const shoulder=group(side+'_shoulder',clavicle,[sign*p.shoulder*.68,0,0]);
            this.ellipsoid(clavicle,[sign*p.shoulder*.30,0,-.005],[p.shoulder*.35,.024,.037],side+'_clavicle');
            this.ball(shoulder,p.armRadius*1.04,side+'_shoulder');
            this.limb(shoulder,p.upperArm,[[0,.86,.90,0],[.15,1.05,1.02,0],[.43,1,.98,.004],[.76,.73,.81,0],[1,.60,.64,0]],p.armRadius,side+'_shoulder');
            const elbow=group(side+'_elbow',shoulder,[0,-p.upperArm,0]);this.ball(elbow,p.armRadius*.63,side+'_elbow');
            this.limb(elbow,p.lowerArm,[[0,.65,.66,0],[.18,.86,.84,0],[.38,.88,.81,.003],[.73,.60,.56,0],[1,.43,.44,0]],p.armRadius,side+'_elbow');
            const wrist=group(side+'_wrist',elbow,[0,-p.lowerArm,0]);this.ball(wrist,p.hand*.115,side+'_wrist');
            const hand=p.hand;
            const detailedHand=new T.Group(),simpleHand=new T.Group();wrist.add(detailedHand,simpleHand);this.hands[side]={detailed:detailedHand,simple:simpleHand};
            this.form(simpleHand,[[0,hand*.13,hand*.10,0],[-hand*.18,hand*.22,hand*.12,0],
                [-hand*.42,hand*.27,hand*.12,0],[-hand*.72,hand*.26,hand*.11,0],[-hand*.94,hand*.17,hand*.075,0],
                [-hand,hand*.025,hand*.025,0]],side+'_wrist',36);
            this.form(detailedHand,[[0,hand*.13,hand*.10,0],[-hand*.16,hand*.18,hand*.12,.002],
                [-hand*.36,hand*.205,hand*.095,.002],[-hand*.51,hand*.18,hand*.075,0]],side+'_wrist',28);
            const fingers={thumb:{x:-.225,y:-.19,length:.37,angle:-32},index:{x:-.125,y:-.475,length:.44,angle:-3},
                middle:{x:-.015,y:-.49,length:.49,angle:0},ring:{x:.095,y:-.475,length:.455,angle:3},pinky:{x:.19,y:-.44,length:.35,angle:8}};
            for(const [finger,f] of Object.entries(fingers)){
                const lengths=[.46,.31,.23].map(v=>hand*f.length*v);let parent=detailedHand;
                for(let i=0;i<3;i++){
                    const name=`${side}_${finger}_${i}`;
                    const pos=i===0?[sign*hand*f.x,hand*f.y,finger==='thumb'?hand*.025:0]:[0,-lengths[i-1],0];
                    const g=group(name,parent,pos,i===0?T.MathUtils.degToRad(f.angle*sign):0);
                    const radius=hand*(finger==='thumb'?.053:finger==='pinky'?.037:.044)*(1-i*.12);
                    this.ball(g,radius,name);
                    this.limb(g,lengths[i],[[0,.96,.90,0],[.23,1,1,0],[.78,.83,.86,0],[1,.58,.63,0]],radius,name,20);
                    if(i===2){
                        this.ellipsoid(g,[0,-lengths[i],0],[radius*.58,radius*.65,radius*.63],name);
                        this.fingerTips[`${side}_${finger}`]={joint:g,local:new T.Vector3(0,-lengths[i],0)};
                    }
                    parent=g;
                }
            }
            const hip=group(side+'_hip',pelvis,[sign*p.hip,0,0]);this.ball(hip,p.thighRadius*.90,side+'_hip');
            this.limb(hip,p.leg*.51,[[0,.90,.95,0],[.18,1.04,1.09,.001],[.42,.96,1.00,.003],[.78,.68,.72,0],[1,.54,.58,0]],p.thighRadius,side+'_hip');
            const knee=group(side+'_knee',hip,[0,-p.leg*.51,0]);this.ball(knee,p.thighRadius*.56,side+'_knee');
            this.limb(knee,p.leg*.49,[[0,.55,.57,0],[.18,.69,.83,-.009],[.38,.73,.91,-.012],
                [.67,.53,.64,-.007],[1,.34,.39,0]],p.thighRadius,side+'_knee');
            const ankle=group(side+'_ankle',knee,[0,-p.leg*.49,0]);this.ball(ankle,p.thighRadius*.37,side+'_ankle');
            const f=p.foot,fl=p.footLength,fw=fl*.185;
            const detailedFoot=new T.Group(),simpleFoot=new T.Group();ankle.add(detailedFoot,simpleFoot);this.feet[side]={detailed:detailedFoot,simple:simpleFoot};
            this.form(simpleFoot,[[-f,fw*.94,fl*.46,fl*.20],[-f*.8,fw,fl*.47,fl*.20],
                [-f*.45,fw*.95,fl*.43,fl*.18],[-f*.16,fw*.8,fl*.34,fl*.10],
                [-f*.04,fw*.61,fl*.23,0],[f*.23,fw*.43,fl*.12,-fl*.04]],side+'_ankle');
            this.form(detailedFoot,[[-f,fw*.92,fl*.40,fl*.20],[-f*.8,fw,fl*.43,fl*.20],
                [-f*.45,fw*.93,fl*.38,fl*.17],[-f*.04,fw*.61,fl*.23,0],[f*.23,fw*.43,fl*.12,-fl*.04]],side+'_ankle');
            const toe=group(side+'_toe',detailedFoot,[0,-f*.61,fl*.40]);
            this.ellipsoid(toe,[0,-f*.05,fl*.05],[fw*.92,f*.30,fl*.22],side+'_toe');
        }
        for(const j of JOINTS.filter(j=>!/_\d$/.test(j)&&j!=='root')){
            const radius=j.endsWith('shoulder')?p.armRadius*1.10:j.endsWith('hip')?p.thighRadius*.96:j.endsWith('knee')?p.thighRadius*.59:j.endsWith('toe')?p.foot*.25:j.endsWith('wrist')?p.hand*.14:.027;
            const geometry=new T.SphereGeometry(radius,12,8);this.resources.push(geometry);
            const mark=new T.Mesh(geometry,this.jointMat);mark.userData.joint=j;this.joints[j].add(mark);this.pickables.push(mark);this.marks.push(mark);
        }
        this.apply(state);
        installSurface(this,state);
    }
    mesh(parent,geometry,joint,material=this.material){
        this.resources.push(geometry);const mesh=new T.Mesh(geometry,material);mesh.userData.joint=joint;parent.add(mesh);this.pickables.push(mesh);return mesh;
    }
    form(parent,profile,joint,radial=40,samples=5){const m=this.mesh(parent,sculptGeometry(profile,radial,samples),joint);m.userData.form={profile:profile.map(r=>[...r]).sort((a,b)=>a[0]-b[0])};return m;}
    ellipsoid(parent,pos,scale,joint,material=this.material){
        const mesh=this.mesh(parent,new T.SphereGeometry(1,28,20),joint,material);mesh.position.fromArray(pos);mesh.scale.fromArray(scale);mesh.userData.form={};return mesh;
    }
    ball(parent,radius,joint){return this.ellipsoid(parent,[0,0,0],[radius,radius,radius],joint,this.socketMaterial);}
    limb(parent,length,profile,radius,joint,radial=40){
        return this.form(parent,profile.map(([t,x,z,center])=>[-t*length,x*radius,z*radius,center]),joint,radial);
    }
    apply(state) {
        this.root.position.fromArray(state.root);
        for(const j of JOINTS) this.joints[j].quaternion.fromArray(state.joints[j]);
        this.material.color.set(state.appearance.body);this.socketMaterial.color.set(state.appearance.body).multiplyScalar(.86);this.face.visible=state.appearance.face;
        for(const side of ['L','R']){
            this.hands[side].detailed.visible=!state.appearance.simple_hands;this.hands[side].simple.visible=!!state.appearance.simple_hands;
            this.feet[side].detailed.visible=!state.appearance.simple_feet;this.feet[side].simple.visible=!!state.appearance.simple_feet;
        }
        this.root.updateMatrixWorld(true);
        updateSurface(this,state);
    }
    visiblePickables() {
        return this.pickables.filter(mesh=>{for(let o=mesh;o;o=o.parent)if(!o.visible)return false;return true;});
    }
    jointAt(hit){
        const indices=this.surface.geometry.attributes.skinIndex,weights=this.surface.geometry.attributes.skinWeight,influence=new Map();
        for(const index of [hit.face.a,hit.face.b,hit.face.c])for(let j=0;j<4;j++){const bone=indices.array[index*4+j],weight=weights.array[index*4+j];influence.set(bone,(influence.get(bone)||0)+weight);}
        const bone=[...influence].sort((a,b)=>b[1]-a[1])[0][0];return this.surface.skeleton.bones[bone].name;
    }
    bounds() {
        this.root.updateMatrixWorld(true);const bounds=new T.Box3(),markers=new Set(this.marks);
        for(const mesh of this.visiblePickables()){
            if(markers.has(mesh))continue;
            if(mesh.isSkinnedMesh){mesh.computeBoundingBox();bounds.union(mesh.boundingBox.clone().applyMatrix4(mesh.matrixWorld));}
            else {if(!mesh.geometry.boundingBox)mesh.geometry.computeBoundingBox();bounds.union(mesh.geometry.boundingBox.clone().applyMatrix4(mesh.matrixWorld));}
        }
        return bounds;
    }
    read(state) {
        state.root=this.root.position.toArray();
        for(const j of JOINTS) state.joints[j]=this.joints[j].quaternion.clone().normalize().toArray();
        this.root.updateMatrixWorld(true);updateSurface(this,state);
    }
    constrain(joint) {
        const angles=euler(this.joints[joint].quaternion.toArray()), ranges=limits(joint);
        this.joints[joint].quaternion.fromArray(quaternion(...angles.map((a,i)=>T.MathUtils.clamp(a,...ranges[i]))));
    }
    dispose() { for(const r of this.resources) r.dispose(); this.root.removeFromParent(); }
}

// Analytic two-bone IK. Pole controls the bend plane; unreachable goals are clamped.
export function solveIK(rig, side, limb, target, pole, constrained=false) {
    const names=limb==='arm'?['shoulder','elbow','wrist']:['hip','knee','ankle'];
    const [upper,lower,end]=names.map(n=>rig.joints[side+'_'+n]);
    rig.root.updateMatrixWorld(true);
    const start=upper.getWorldPosition(new T.Vector3());
    const a=lower.position.length(),b=end.position.length();
    const dir=target.clone().sub(start); if(dir.lengthSq()<1e-10) dir.set(0,-1,0);
    const min=constrained?Math.sqrt(a*a+b*b+2*a*b*Math.cos(T.MathUtils.degToRad(150))):Math.abs(a-b)+1e-5;
    const d=T.MathUtils.clamp(dir.length(),min,a+b-1e-5); dir.normalize();
    let bend=pole.clone().sub(start).addScaledVector(dir,-pole.clone().sub(start).dot(dir));
    if(bend.lengthSq()<1e-8) { bend.set(0,0,1); bend.addScaledVector(dir,-bend.dot(dir)); if(bend.lengthSq()<1e-8) bend.set(1,0,0); }
    bend.normalize(); const along=(a*a-b*b+d*d)/(2*d);
    const middle=start.clone().addScaledVector(dir,along).addScaledVector(bend,Math.sqrt(Math.max(0,a*a-along*along)));
    const desiredEnd=start.clone().addScaledVector(dir,d);
    const upperDir=middle.clone().sub(start).normalize(),lowerDir=desiredEnd.clone().sub(middle).normalize();
    const sign=limb==='arm'?-1:1;
    const x=upperDir.clone().cross(lowerDir).normalize().multiplyScalar(sign);
    const y=upperDir.clone().negate(),z=x.clone().cross(y).normalize();
    const basis=new T.Matrix4().makeBasis(x,y,z),world=new T.Quaternion().setFromRotationMatrix(basis);
    const parent=upper.parent.getWorldQuaternion(new T.Quaternion()).invert();upper.quaternion.copy(parent.multiply(world));
    const angle=Math.acos(T.MathUtils.clamp(upperDir.dot(lowerDir),-1,1));
    lower.quaternion.setFromAxisAngle(new T.Vector3(1,0,0),sign*angle);
    rig.root.updateMatrixWorld(true);
}
