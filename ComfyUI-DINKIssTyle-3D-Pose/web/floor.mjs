import * as T from './vendor/three.module.mjs';

export const GRID_SPACING=.25;
export const GRID_MAJOR_EVERY=4;
export const FLOOR_Y=-.004;
// Shader grid stays readable as it recedes, without coplanar line flicker.
export function createFloor() {
    const material=new T.ShaderMaterial({
        uniforms:{spacing:{value:GRID_SPACING},base:{value:new T.Color('#35404b')},minor:{value:new T.Color('#718194')},major:{value:new T.Color('#acbdcf')}},
        vertexShader:`varying vec3 worldPosition;
            void main(){vec4 world=modelMatrix*vec4(position,1.0);worldPosition=world.xyz;gl_Position=projectionMatrix*viewMatrix*world;}`,
        fragmentShader:`varying vec3 worldPosition;uniform float spacing;uniform vec3 base;uniform vec3 minor;uniform vec3 major;
            float grid(vec2 p){vec2 d=max(fwidth(p),vec2(0.0001));vec2 g=abs(fract(p-0.5)-0.5)/d;
                float fade=1.0-smoothstep(0.25,0.75,max(d.x,d.y));return (1.0-min(min(g.x,g.y),1.0))*fade;}
            void main(){float fade=1.0-smoothstep(10.0,35.0,length(worldPosition.xz));
                vec3 color=mix(base,minor,grid(worldPosition.xz/spacing)*0.7*fade);
                color=mix(color,major,grid(worldPosition.xz/(spacing*4.0))*0.85*fade);
                gl_FragColor=vec4(color,1.0);
                #include <tonemapping_fragment>
                #include <colorspace_fragment>
            }`,
    });
    const floor=new T.Mesh(new T.PlaneGeometry(200,200),material);floor.rotation.x=-Math.PI/2;floor.position.y=FLOOR_Y;return floor;
}
