// Same per-vertex interpolation as core/geometry.py; sliders never send render requests.
export function normalizedWeights(state, groups, key, visibility=null) {
    const raw = state.assets.map(asset => groups.map(group => regionValue(asset,REGIONS[group],key)));
    if(visibility&&key==='shape')raw.forEach((row,i)=>row.forEach((v,j)=>row[j]=v*visibility[i][j]));
    for (let j=0;j<groups.length;j++) {
        let total = raw.reduce((sum,row)=>sum+row[j],0);
        if (!total) {raw[state.base][j]=1; total=1;}
        raw.forEach(row=>row[j]/=total);
    }
    return raw;
}
export const REGIONS = ['face','eye_l','eye_r','nose','mouth','lips','brow_l','brow_r'];
export const PAIRS = {eyes:['eye_l','eye_r'],brows:['brow_l','brow_r']};
export const PARENTS = Object.fromEntries(Object.entries(PAIRS).flatMap(([parent,children])=>children.map(child=>[child,parent])));
export const REGION_OPTIONS = [
    ['all','Global weights'], ['face','Face outline and skin'],
    ['eyes','Eyes'], ['eye_l','　├─ Left eye (screen)'], ['eye_r','　└─ Right eye (screen)'],
    ['nose','Nose'], ['mouth','Inner mouth'], ['lips','Lips'],
    ['brows','Eyebrows'], ['brow_l','　├─ Left eyebrow (screen)'], ['brow_r','　└─ Right eyebrow (screen)'],
];
export function regionValue(asset,region,key) {
    return Number(asset.regions?.[region]?.[key] ?? asset.regions?.[PARENTS[region]]?.[key] ?? asset[key] ?? 1);
}
export function setRegionWeight(asset,region,key,value) {
    if(region==='all'){asset[key]=value;return;}
    asset.regions??={};asset.regions[region]??={};asset.regions[region][key]=value;
    // Moving a parent slider explicitly links both sides for that property.
    for(const child of PAIRS[region]??[]){
        if(asset.regions[child]){
            delete asset.regions[child][key];
            if(!Object.keys(asset.regions[child]).length)delete asset.regions[child];
        }
    }
}
export function clearRegionWeights(asset,region) {
    for(const item of [region,...(PAIRS[region]??[])])delete asset.regions?.[item];
}
export function upgradeState(state) {
    for(const asset of state.assets){
        if(asset.points?.length===544)asset.points=[...asset.points.slice(0,468),...asset.points.slice(484,520),...asset.points.slice(536,544)];
        delete asset.ears;delete asset.regions?.ear_l;delete asset.regions?.ear_r;
        if(Array.isArray(asset.warnings))asset.warnings=asset.warnings.filter(w=>!(/\bears?\b/i.test(w)));
    }
    return state;
}
export const CURVES = [
    [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109],
    [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],
    [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466],
    [70,63,105,66,107,55,65,52,53,46], [300,293,334,296,336,285,295,282,283,276],
    [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185],
    [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191],
    [1,2,4,5,6,19,94,97,98,115,168,195,197,220,275,278,294,326,327,344,440]
];
export function updateMesh(state, mesh) {
    let shape = normalizedWeights(state,mesh.groups,'shape',mesh.visibility);
    const feature = normalizedWeights(state,mesh.groups,'feature');
    let target = mesh.aligned[0].map((_,j)=>[0,1].map(d=>
        mesh.aligned.reduce((s,p,i)=>s+p[j][d]*shape[i][j],0)));
    function area(points,t) {
        const [a,b,c]=t.map(i=>points[i]);
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
    }
    const folded=points=>mesh.triangles.some(t=>{const a=area(mesh.reference,t),b=area(points,t);return a*b < -1e-12||Math.abs(b)<1e-12;});
    if(folded(target)&&mesh.visibility){
        const plain=normalizedWeights(state,mesh.groups,'shape');
        const coordinates=w=>mesh.aligned[0].map((_,j)=>[0,1].map(d=>mesh.aligned.reduce((s,p,i)=>s+p[j][d]*w[i][j],0)));
        if(!folded(coordinates(plain))){
            const desired=shape;
            for(const amount of [.5,.25,.125,.0625,.03125,.015625,0]){
                shape=plain.map((row,i)=>row.map((v,j)=>v*(1-amount)+desired[i][j]*amount));target=coordinates(shape);
                if(!folded(target))break;
            }
        }
    }
    if(folded(target))throw new Error('The mesh folds over. Reduce shape weights or correct correspondence points.');
    return {...mesh,target,shape_weights:shape,feature_weights:feature};
}

// Same forward mesh constraint as core/geometry.py. Mask blue is cached distance.
export function protectTarget(mesh,base,mask){
    if(!mask?.hair)return mesh;
    const {width:w,height:h,data}=mask;
    const distanceAt=p=>{
        const x=Math.max(0,Math.min(1,p[0]))*(w-1),y=Math.max(0,Math.min(1,p[1]))*(h-1);
        const x0=Math.floor(x),y0=Math.floor(y),x1=Math.min(x0+1,w-1),y1=Math.min(y0+1,h-1),fx=x-x0,fy=y-y0;
        const sample=(x,y)=>data[(y*w+x)*4+2]/255;
        return ((sample(x0,y0)*(1-fx)+sample(x1,y0)*fx)*(1-fy)+(sample(x0,y1)*(1-fx)+sample(x1,y1)*fx)*fy)*Math.max(w,h)*.25;
    };
    const desired=mesh.target.map((target,i)=>{
        const start=base[i],delta=target.map((v,d)=>v-start[d]);
        let distance=Math.max(w,h)*.25;
        for(const t of [0,.25,.5,.75,1])distance=Math.min(distance,distanceAt(start.map((v,d)=>v+delta[d]*t)));
        const travel=Math.hypot(delta[0]*(w-1),delta[1]*(h-1)),x=Math.max(0,Math.min(1,distance/Math.max(3,3*travel))),mobility=x*x*(3-2*x);
        return start.map((v,d)=>v+delta[d]*mobility);
    });
    const area=(points,t)=>{const [a,b,c]=t.map(i=>points[i]);return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);};
    for(const amount of [1,.5,.25,.125,.0625,.03125,.015625,0]){
        const target=mesh.target.map((p,i)=>p.map((v,d)=>v+(desired[i][d]-v)*amount));
        if(mesh.triangles.every(t=>{const a=area(mesh.reference,t),b=area(target,t);return a*b>=-1e-12&&Math.abs(b)>=1e-12;}))return {...mesh,target,hair_pin_amount:amount};
    }
    throw new Error('The mesh folds over. Reduce shape weights or correct correspondence points.');
}
