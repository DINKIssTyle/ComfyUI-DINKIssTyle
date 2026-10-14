import * as THREE from './vendor/three.module.mjs';

export const RATIOS = [2, 3, 6, 7, 8, 9];
export const clone = value => JSON.parse(JSON.stringify(value));
export const DEFAULT = await (await fetch(new URL('./default_scene.json', import.meta.url))).json();
export const JOINTS = Object.keys(DEFAULT.joints);
export const radians = n => THREE.MathUtils.degToRad(n);
export const degrees = n => THREE.MathUtils.radToDeg(n);
export function quaternion(x=0, y=0, z=0) {
    return new THREE.Quaternion().setFromEuler(new THREE.Euler(radians(x), radians(y), radians(z), 'XYZ')).toArray();
}
export function euler(q) {
    const e = new THREE.Euler().setFromQuaternion(new THREE.Quaternion().fromArray(q), 'XYZ');
    return [degrees(e.x), degrees(e.y), degrees(e.z)];
}
export function sceneString(scene) {
    // Stable order and precision stop incidental floating-point drift invalidating renders.
    const stable = v => Array.isArray(v) ? v.map(stable) : v && typeof v === 'object'
        ? Object.fromEntries(Object.keys(v).sort().map(k => [k, stable(v[k])]))
        : typeof v === 'number' ? Math.round(v*1e8)/1e8 : v;
    return JSON.stringify(stable(scene));
}
export async function hash(raw) {
    if (globalThis.crypto?.subtle) {
        const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(raw));
        return Array.from(new Uint8Array(digest), n => n.toString(16).padStart(2, '0')).join('');
    }
    // Non-HTTPS LAN ComfyUI still works: server is authoritative for the hash.
    return null;
}
export function validate(scene) {
    if (!scene || scene.version !== DEFAULT.version || ![DEFAULT.renderer,'dkst-pose-1-three-180','dkst-pose-2-three-180'].includes(scene.renderer)) throw Error('Unsupported scene version.');
    if (!RATIOS.includes(scene.ratio)) throw Error('Invalid body proportion.');
    const vec = (v,n,lo,hi) => Array.isArray(v) && v.length === n && v.every(x => typeof x==='number' && Number.isFinite(x) && x>=lo && x<=hi);
    const quat = q => vec(q,4,-1.001,1.001) && Math.abs(q.reduce((n,x)=>n+x*x,0)-1)<.002;
    if (!vec(scene.root,3,-100,100) || !scene.joints || Object.keys(scene.joints).length!==JOINTS.length || !JOINTS.every(j=>quat(scene.joints[j]))) throw Error('Invalid joint data.');
    const c=scene.camera, a=scene.appearance&&{...scene.appearance,simple_hands:scene.appearance.simple_hands??false,simple_feet:scene.appearance.simple_feet??false}, o=scene.output;
    if (!c || !['perspective','orthographic'].includes(c.projection) || !vec(c.position,3,-100,100) || !vec(c.target,3,-100,100) || !quat(c.quaternion) || !vec([c.fov],1,10,120) || !vec([c.ortho],1,.1,50)) throw Error('Invalid camera data.');
    if (!a || !['body','background'].every(k=>/^#[a-f\d]{6}$/i.test(a[k])) || !['transparent','face','floor','simple_hands','simple_feet'].every(k=>typeof a[k]==='boolean') || !vec([a.light],1,.1,5)) throw Error('Invalid appearance data.');
    if (!o || !['width','height'].every(k=>Number.isInteger(o[k]) && o[k]>=64 && o[k]<=4096)) throw Error('Resolution must be an integer between 64 and 4096.');
    // Discard unexpected fields and rebuild the versioned schema.
    return clone({version:scene.version,renderer:DEFAULT.renderer,ratio:scene.ratio,root:scene.root,joints:Object.fromEntries(JOINTS.map(j=>[j,scene.joints[j]])),camera:{projection:c.projection,position:c.position,quaternion:c.quaternion,target:c.target,fov:c.fov,ortho:c.ortho},appearance:{background:a.background,body:a.body,transparent:a.transparent,face:a.face,floor:a.floor,light:a.light,simple_hands:a.simple_hands,simple_feet:a.simple_feet},output:{width:o.width,height:o.height}});
}
export const PRESETS = {
    'T-pose': {L_shoulder:[0,0,90],R_shoulder:[0,0,-90]},
    'A-pose': {L_shoulder:[0,0,45],R_shoulder:[0,0,-45]},
    'Relaxed standing': {L_shoulder:[0,0,8],R_shoulder:[0,0,-8]},
    'Sit': {L_hip:[-90,0,0],R_hip:[-90,0,0],L_knee:[90,0,0],R_knee:[90,0,0],L_elbow:[-65,0,0],R_elbow:[-65,0,0]},
    'Squat': {waist:[25,0,0],L_hip:[-125,0,8],R_hip:[-125,0,-8],L_knee:[145,0,0],R_knee:[145,0,0],L_ankle:[-20,0,0],R_ankle:[-20,0,0],L_shoulder:[-55,0,8],R_shoulder:[-55,0,-8]},
    'Walk': {L_hip:[-28,0,0],R_hip:[25,0,0],L_knee:[12,0,0],R_knee:[35,0,0],L_shoulder:[22,0,5],R_shoulder:[-22,0,-5],L_elbow:[-15,0,0],R_elbow:[-25,0,0]},
    'Run': {waist:[12,0,0],L_hip:[-65,0,0],R_hip:[30,0,0],L_knee:[75,0,0],R_knee:[115,0,0],L_shoulder:[35,0,5],R_shoulder:[-55,0,-5],L_elbow:[-90,0,0],R_elbow:[-90,0,0]},
    'Raised arm': {L_shoulder:[0,0,165],L_elbow:[-15,0,0],R_shoulder:[0,0,-8]},
};
export function applyPreset(scene, name) {
    if (!PRESETS[name]) throw Error('Unknown preset');
    for (const joint of JOINTS) scene.joints[joint]=quaternion();
    for (const [joint,angles] of Object.entries(PRESETS[name])) scene.joints[joint]=quaternion(...angles);
}
export function mirror(scene) {
    const before=clone(scene.joints);
    for (const j of JOINTS) {
        const source=j.startsWith('L_') ? j.replace(/^L_/,'R_') : j.startsWith('R_') ? j.replace(/^R_/,'L_') : j;
        const [x,y,z,w]=before[source]; scene.joints[j]=[x,-y,-z,w];
    }
}
export function copySide(scene, from) {
    const to=from==='L'?'R':'L';
    for (const j of JOINTS.filter(j=>j.startsWith(from+'_'))) {
        const [x,y,z,w]=scene.joints[j]; scene.joints[j.replace(from+'_',to+'_')]=[x,-y,-z,w];
    }
}
export function limits(name) {
    if (name.endsWith('_elbow')) return [[-150,0],[-10,10],[-10,10]];
    if (name.endsWith('_knee')) return [[0,150],[-8,8],[-8,8]];
    if (/_\d$/.test(name)) return [[-100,20],[-35,35],[-35,35]];
    if (name==='neck' || name==='head') return [[-70,70],[-90,90],[-55,55]];
    if (name==='waist' || name==='chest') return [[-65,65],[-70,70],[-55,55]];
    if (name.endsWith('_ankle')) return [[-65,65],[-40,40],[-40,40]];
    return [[-180,180],[-180,180],[-180,180]];
}
export class History {
    constructor(scene) { this.items=[clone(scene)]; this.index=0; }
    push(scene) {
        if (sceneString(this.items[this.index])===sceneString(scene)) return;
        this.items=this.items.slice(0,this.index+1); this.items.push(clone(scene));
        if(this.items.length>80) this.items.shift(); this.index=this.items.length-1;
    }
    move(delta) { this.index=Math.max(0,Math.min(this.items.length-1,this.index+delta)); return clone(this.items[this.index]); }
}
