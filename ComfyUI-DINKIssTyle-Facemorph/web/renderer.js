"use strict";
import {buildWarp,WARP_FRAGMENT,HAIR_PIN} from './liquify.js';
const QUAD=[0,0,1,0,0,1,0,1,1,0,1,1];
const VERT=`attribute vec2 position;varying vec2 vUV;void main(){gl_Position=vec4(position.x*2.-1.,1.-position.y*2.,0.,1.);vUV=position;}`;
export class MorphRenderer {
    constructor(canvas){
        this.canvas=canvas;
        const gl=this.gl=canvas.getContext('webgl',{alpha:false,preserveDrawingBuffer:true,antialias:false});
        if(!gl)throw new Error('WebGL is unavailable. Enable hardware acceleration in your browser.');
        const compile=(type,src)=>{const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;};
        const program=(vert,frag)=>{const p=gl.createProgram(),v=compile(gl.VERTEX_SHADER,vert),f=compile(gl.FRAGMENT_SHADER,frag);gl.attachShader(p,v);gl.attachShader(p,f);gl.linkProgram(p);gl.deleteShader(v);gl.deleteShader(f);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p;};
        this.morph=program(`attribute vec2 position;attribute vec2 uv;attribute float weight;varying vec2 vUV;varying float vW;
            void main(){gl_Position=vec4(position.x*2.-1.,1.-position.y*2.,0.,1.);vUV=uv;vW=weight;}`,
            `precision highp float;uniform sampler2D photo;uniform sampler2D mask;uniform vec2 photoSize;uniform vec2 maskSize;uniform float coverage;varying vec2 vUV;varying float vW;
            void main(){vec2 uv=(vUV*(photoSize-1.)+.5)/photoSize;vec2 m=(vUV*(maskSize-1.)+.5)/maskSize;
            float valid=texture2D(mask,m).r;float weight=vW*valid;
            gl_FragColor=coverage>.5?vec4(vec3(1.),vW>0.00000001?valid:0.):vec4(texture2D(photo,uv).rgb*weight,weight);}`);
        this.normalize=program(`attribute vec2 position;attribute float alpha;varying float vA;void main(){gl_Position=vec4(position.x*2.-1.,1.-position.y*2.,0.,1.);vA=alpha;}`,
            `precision highp float;uniform sampler2D mixed;uniform sampler2D coverage;uniform vec2 outputSize;varying float vA;
            void main(){vec2 uv=gl_FragCoord.xy/outputSize;vec4 m=texture2D(mixed,uv);float a=texture2D(coverage,uv).a*vA*step(.00000001,m.a);gl_FragColor=vec4(m.rgb/max(m.a,.00000001)*a,a);}`);
        this.copy=program(VERT,`precision highp float;uniform sampler2D photo;uniform vec2 photoSize;uniform vec2 outputSize;varying vec2 vUV;
            void main(){vec2 p=(vUV*outputSize-.5)/(outputSize-1.);gl_FragColor=vec4(texture2D(photo,(p*(photoSize-1.)+.5)/photoSize).rgb,1.);}`);
        this.context=program(`attribute vec2 position;attribute vec2 uv;varying vec2 vSource;void main(){gl_Position=vec4(position.x*2.-1.,1.-position.y*2.,0.,1.);vSource=uv;}`,
            `precision highp float;uniform sampler2D photo;uniform vec2 photoSize;uniform vec2 outputSize;uniform float active;uniform vec4 contextBounds;varying vec2 vSource;
            ${HAIR_PIN}
            void main(){vec2 dst=(vec2(gl_FragCoord.x,outputSize.y-gl_FragCoord.y)-.5)/(outputSize-1.);vec2 outside=max(max(contextBounds.xy-dst,dst-contextBounds.zw),0.)/max(contextBounds.zw-contextBounds.xy,vec2(.00001));float influence=1.-smoothstep(0.,.25,max(outside.x,outside.y));vec2 src=pinHair(dst,mix(dst,clamp(vSource,0.,1.),influence));src=mix(dst,src,active);
            gl_FragColor=vec4(texture2D(photo,(src*(photoSize-1.)+.5)/photoSize).rgb,1.);}`);
        this.finish=program(VERT,`precision highp float;uniform sampler2D photo;uniform sampler2D face;uniform sampler2D context;uniform sampler2D mask;uniform float protect;uniform vec2 maskSize;uniform vec2 photoSize;uniform vec2 outputSize;varying vec2 vUV;
            void main(){vec2 point=(vUV*outputSize-.5)/(outputSize-1.);vec2 frame=vec2(vUV.x,1.-vUV.y);vec4 f=texture2D(face,frame);float hair=texture2D(mask,(point*(maskSize-1.)+.5)/maskSize).g*protect;
            vec3 backing=texture2D(context,frame).rgb;vec3 mixed=min(f.rgb,vec3(f.a))+backing*(1.-f.a);
            gl_FragColor=vec4(mix(mixed,texture2D(photo,(point*(photoSize-1.)+.5)/photoSize).rgb,hair),1.);}`);
        this.warp=program(VERT,WARP_FRAGMENT);
        this.buffer=gl.createBuffer();this.frame=gl.createFramebuffer();this.photos=[];this.masks=[];this.targets=Array.from({length:6},()=>gl.createTexture());
        this.defaultMask=gl.createTexture();this.bind(0,this.defaultMask);this.parameters();gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,1,1,0,gl.RGB,gl.UNSIGNED_BYTE,new Uint8Array([255,0,0]));
    }
    bind(unit,texture){const gl=this.gl;gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,texture);}
    parameters(){const gl=this.gl;for(const p of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,p,gl.LINEAR);for(const p of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,p,gl.CLAMP_TO_EDGE);}
    texture(image){const gl=this.gl,t=gl.createTexture();this.bind(0,t);this.parameters();gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,image);return t;}
    setImages(images){this.photos.forEach(t=>this.gl.deleteTexture(t));this.photos=images.map(i=>this.texture(i));this.images=images;}
    setMasks(images=[]){this.masks.forEach(t=>this.gl.deleteTexture(t));this.masks=images.map(i=>this.texture(i));this.maskImages=images;
        this.maskPixels=images.map(image=>{const c=document.createElement('canvas');c.width=image.width;c.height=image.height;const x=c.getContext('2d');x.drawImage(image,0,0);const data=x.getImageData(0,0,c.width,c.height).data;let hair=false;for(let j=1;j<data.length;j+=4)if(data[j]>5){hair=true;break;}return {data,width:c.width,height:c.height,hair};});
    }
    hasFace(state,mesh){
        // Inspect actual mesh vertices and triangle centers, without a GPU readback on slider changes.
        const ids=[...new Set(mesh.triangles.flat())];
        return state.assets.some((asset,i)=>{
            const pixels=this.maskPixels[i],weights=mesh.feature_weights[i];
            const sample=p=>{const x=Math.max(0,Math.min(pixels.width-1,Math.round(p[0]*(pixels.width-1)))),y=Math.max(0,Math.min(pixels.height-1,Math.round(p[1]*(pixels.height-1))));return pixels.data[(y*pixels.width+x)*4]>0;};
            if(ids.some(j=>mesh.alpha[j]>0&&weights[j]>1e-8&&sample(asset.points[j])))return true;
            return mesh.triangles.some(t=>t.some(j=>mesh.alpha[j]>0&&weights[j]>1e-8)&&sample(t.reduce((p,j)=>[p[0]+asset.points[j][0]/3,p[1]+asset.points[j][1]/3],[0,0])));
        });
    }
    attributes(program,data,fields){
        const gl=this.gl;for(let i=0;i<gl.getParameter(gl.MAX_VERTEX_ATTRIBS);i++)gl.disableVertexAttribArray(i);
        gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.DYNAMIC_DRAW);
        const stride=fields.reduce((n,[,s])=>n+s,0)*4;let offset=0;
        for(const [name,size] of fields){const a=gl.getAttribLocation(program,name);if(a>=0){gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,size,gl.FLOAT,false,stride,offset);}offset+=size*4;}
    }
    target(index){const gl=this.gl;gl.bindFramebuffer(gl.FRAMEBUFFER,this.frame);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,this.targets[index],0);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);}
    draw(state,mesh,edge=640){
        const gl=this.gl,[bw,bh]=mesh.sizes[state.base],warp=buildWarp(mesh.target,[bw,bh],state.liquify??{});
        const w=Math.max(2,Math.round(edge*bw/Math.max(bw,bh))),h=Math.max(2,Math.round(edge*bh/Math.max(bw,bh)));
        if(this.canvas.width!==w||this.canvas.height!==h||!this.allocated){
            this.canvas.width=w;this.canvas.height=h;
            this.targets.forEach(t=>{this.bind(0,t);this.parameters();gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,null);});this.allocated=true;
        }
        gl.viewport(0,0,w,h);
        const screen=p=>[(p[0]*(w-1)+.5)/w,(p[1]*(h-1)+.5)/h];
        const masked=state.hair?.enabled!==false&&this.masks.length===state.assets.length;
        for(let pass=0;pass<2;pass++){
            this.target(pass);gl.enable(gl.BLEND);
            if(pass)gl.blendFuncSeparate(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA,gl.ONE,gl.ONE_MINUS_SRC_ALPHA);else gl.blendFunc(gl.ONE,gl.ONE);
            gl.useProgram(this.morph);const loc=n=>gl.getUniformLocation(this.morph,n);
            gl.uniform1i(loc('photo'),0);gl.uniform1i(loc('mask'),1);gl.uniform1f(loc('coverage'),pass);
            for(let i=0;i<state.assets.length;i++){
                const data=[];for(const tri of mesh.triangles)for(const j of tri)data.push(...screen(mesh.target[j]),...state.assets[i].points[j],mesh.feature_weights[i][j]);
                this.attributes(this.morph,data,[['position',2],['uv',2],['weight',1]]);
                this.bind(0,this.photos[i]);this.bind(1,masked?this.masks[i]:this.defaultMask);
                gl.uniform2f(loc('photoSize'),this.images[i].width,this.images[i].height);
                gl.uniform2f(loc('maskSize'),masked?this.maskImages[i].width:1,masked?this.maskImages[i].height:1);
                gl.drawArrays(gl.TRIANGLES,0,data.length/5);
            }
        }
        gl.disable(gl.BLEND);this.target(2);gl.useProgram(this.normalize);
        const data=[];for(const tri of mesh.triangles)for(const j of tri)data.push(...screen(mesh.target[j]),mesh.alpha[j]);
        this.attributes(this.normalize,data,[['position',2],['alpha',1]]);this.bind(0,this.targets[0]);this.bind(1,this.targets[1]);
        gl.uniform1i(gl.getUniformLocation(this.normalize,'mixed'),0);gl.uniform1i(gl.getUniformLocation(this.normalize,'coverage'),1);gl.uniform2f(gl.getUniformLocation(this.normalize,'outputSize'),w,h);gl.drawArrays(gl.TRIANGLES,0,data.length/3);
        const active=!masked||this.hasFace(state,mesh),protect=masked&&state.hair?.protect_base!==false;
        const hairPresent=protect&&this.maskPixels[state.base].hair;
        const protectUniforms=program=>{
            this.bind(1,masked?this.masks[state.base]:this.defaultMask);
            gl.uniform1i(gl.getUniformLocation(program,'protectionMask'),1);
            gl.uniform2f(gl.getUniformLocation(program,'protectionSize'),masked?this.maskImages[state.base].width:1,masked?this.maskImages[state.base].height:1);
            gl.uniform1f(gl.getUniformLocation(program,'protectHair'),hairPresent?1:0);
        };
        // The backing must follow the base mesh as well as Liquify; otherwise its old outline reappears.
        this.target(4);gl.useProgram(this.copy);this.attributes(this.copy,QUAD,[['position',2]]);this.bind(0,this.photos[state.base]);
        gl.uniform1i(gl.getUniformLocation(this.copy,'photo'),0);gl.uniform2f(gl.getUniformLocation(this.copy,'photoSize'),this.images[state.base].width,this.images[state.base].height);gl.uniform2f(gl.getUniformLocation(this.copy,'outputSize'),w,h);gl.drawArrays(gl.TRIANGLES,0,6);
        gl.useProgram(this.context);const contextData=[];
        const contextPoints=[...state.assets[state.base].points.slice(0,468),...mesh.target.slice(0,468)];
        const bounds=[0,1].map(d=>Math.min(...contextPoints.map(p=>p[d]))).concat([0,1].map(d=>Math.max(...contextPoints.map(p=>p[d]))));
        gl.uniform4fv(gl.getUniformLocation(this.context,'contextBounds'),bounds);
        for(const tri of mesh.triangles)for(const j of tri)contextData.push(...screen(mesh.target[j]),...state.assets[state.base].points[j]);
        this.attributes(this.context,contextData,[['position',2],['uv',2]]);this.bind(0,this.photos[state.base]);protectUniforms(this.context);
        gl.uniform1i(gl.getUniformLocation(this.context,'photo'),0);gl.uniform2f(gl.getUniformLocation(this.context,'photoSize'),this.images[state.base].width,this.images[state.base].height);gl.uniform2f(gl.getUniformLocation(this.context,'outputSize'),w,h);gl.uniform1f(gl.getUniformLocation(this.context,'active'),active?1:0);gl.drawArrays(gl.TRIANGLES,0,contextData.length/4);
        if(warp){
            for(const [source,destination] of [[2,3],...(active?[[4,5]]:[])]){
                this.target(destination);gl.useProgram(this.warp);this.attributes(this.warp,QUAD,[['position',2]]);this.bind(0,this.targets[source]);protectUniforms(this.warp);
                const loc=n=>gl.getUniformLocation(this.warp,n);
                gl.uniform1i(loc('photo'),0);gl.uniform2f(loc('outputSize'),w,h);gl.uniform2f(loc('referenceSize'),bw,bh);
                gl.uniform2fv(loc('origin'),warp.origin);gl.uniform2fv(loc('axis'),warp.axis);gl.uniform1f(loc('scale'),warp.scale);gl.uniform1i(loc('opCount'),warp.ops.length);
                const regions=new Float32Array(64),matrices=new Float32Array(64),shifts=new Float32Array(32);
                warp.ops.forEach((op,i)=>{regions.set([...op.center,...op.radius],i*4);matrices.set(op.matrix,i*4);shifts.set(op.shift,i*2);});
                gl.uniform4fv(loc('regions[0]'),regions);gl.uniform4fv(loc('matrices[0]'),matrices);gl.uniform2fv(loc('shifts[0]'),shifts);gl.drawArrays(gl.TRIANGLES,0,6);
            }
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.useProgram(this.finish);this.attributes(this.finish,QUAD,[['position',2]]);
        this.bind(0,this.photos[state.base]);this.bind(1,this.targets[warp?3:2]);this.bind(2,masked?this.masks[state.base]:this.defaultMask);this.bind(3,this.targets[warp&&active?5:4]);
        const loc=n=>gl.getUniformLocation(this.finish,n);
        gl.uniform2f(loc('photoSize'),this.images[state.base].width,this.images[state.base].height);gl.uniform2f(loc('outputSize'),w,h);
        gl.uniform1i(loc('photo'),0);gl.uniform1i(loc('face'),1);gl.uniform1i(loc('mask'),2);gl.uniform1i(loc('context'),3);gl.uniform1f(loc('protect'),masked&&state.hair?.protect_base!==false?1:0);
        gl.uniform2f(loc('maskSize'),masked?this.maskImages[state.base].width:1,masked?this.maskImages[state.base].height:1);gl.drawArrays(gl.TRIANGLES,0,6);
    }
    dispose(){const gl=this.gl;[...this.photos,...this.masks,...this.targets,this.defaultMask].forEach(t=>gl.deleteTexture(t));gl.deleteFramebuffer(this.frame);gl.deleteBuffer(this.buffer);[this.morph,this.normalize,this.finish,this.warp,this.context,this.copy].forEach(p=>gl.deleteProgram(p));gl.getExtension('WEBGL_lose_context')?.loseContext();}
}
