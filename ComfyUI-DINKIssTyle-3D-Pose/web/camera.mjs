import * as T from './vendor/three.module.mjs';
export const FILM_DIAGONAL=Math.hypot(36,24);
export function lensFOV(focal,aspect){return T.MathUtils.radToDeg(2*Math.atan(FILM_DIAGONAL/Math.sqrt(1+aspect*aspect)/(2*focal)));}
export function focalFromFOV(fov,aspect){return FILM_DIAGONAL/Math.sqrt(1+aspect*aspect)/(2*Math.tan(T.MathUtils.degToRad(fov/2)));}
