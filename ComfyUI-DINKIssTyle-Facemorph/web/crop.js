// Coordinates in saved workflows always refer to the unchanged original photo.
const FRAME=[[0,0],[.5,0],[1,0],[1,.5],[1,1],[.5,1],[0,1],[0,.5]];
export function cropPoints(points,bounds,size){
    if(!bounds||!size)return points;
    const [l,t,r,b]=bounds;
    const result=points.map(p=>[(p[0]*(size[0]-1)-l)/(r-l-1),(p[1]*(size[1]-1)-t)/(b-t-1)]);
    result.splice(result.length-8,8,...FRAME.map(p=>[...p]));return result;
}
export function originalPoint(point,bounds,size){
    if(!bounds||!size)return point;
    const [l,t,r,b]=bounds;
    return [(point[0]*(r-l-1)+l)/(size[0]-1),(point[1]*(b-t-1)+t)/(size[1]-1)];
}
