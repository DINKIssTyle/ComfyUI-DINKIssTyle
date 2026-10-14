"""Real backend/editor smoke harness without an installed ComfyUI.

python tests/browser_harness.py --port 28993
"""
import argparse
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from aiohttp import web

ROOT=Path(__file__).resolve().parents[1]
TEMP=tempfile.TemporaryDirectory(prefix="dkst-pose-browser-")
routes=web.RouteTableDef()
sys.modules["server"]=types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes)))
sys.modules["folder_paths"]=types.SimpleNamespace(get_input_directory=lambda:TEMP.name)
spec=importlib.util.spec_from_file_location("pose_pack",ROOT/"__init__.py",submodule_search_locations=[str(ROOT)])
pack=importlib.util.module_from_spec(spec);sys.modules["pose_pack"]=pack;spec.loader.exec_module(pack)

@routes.get("/")
async def index(request):
    return web.Response(text='''<!doctype html><html><head><meta charset="utf-8"><title>DKST 3D Pose test</title><style>body{margin:0;background:#101722;color:#c9daeb;font:14px system-ui}main{padding:25px}button{padding:9px;margin:5px;background:#31465e;color:white;border:1px solid #6585a3;border-radius:5px}#node{width:340px;padding:15px;background:#1e2b3e;border-radius:10px}#preview{height:210px}pre{white-space:pre-wrap}</style></head><body><main><h2>Independent pack · real WebGL and IMAGE/MASK backend</h2><div id="node"><h3>DKST 3D Pose</h3><div id="buttons"></div><div id="preview"></div></div><button id="queue">Queue</button><button id="openpose">Queue OpenPose</button><button id="reload">Save / reload workflow</button><button id="clone">Clone node</button><pre id="result"></pre></main><script type="module" src="/harness.mjs"></script></body></html>''',content_type="text/html")

@routes.get("/scripts/app.js")
async def app_js(request):
    return web.Response(text='''export const app=globalThis.testApp={graph:{_nodes:[],change(){}},extensions:[],registerExtension(e){this.extensions.push(e)},ui:{dialog:{show:m=>{document.querySelector('#result').textContent='ERROR: '+m;globalThis.testErrors.push(m);}}}};''',content_type="text/javascript")

@routes.get("/scripts/api.js")
async def api_js(request):
    return web.Response(text='''export const api=globalThis.testApi={apiURL:p=>p,fetchApi:(p,o)=>fetch(p,o),async queuePrompt(n,p){const r=await fetch('/prompt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});const data=await r.json();if(!r.ok)throw Error(data.error);globalThis.lastPrompt=p;globalThis.lastResult=data;document.querySelector('#result').textContent=JSON.stringify(data,null,2);return data;}};''',content_type="text/javascript")

@routes.get("/extensions/pose/{file:.*}")
async def asset(request):
    path=(ROOT/"web"/request.match_info['file']).resolve()
    if not path.is_relative_to((ROOT/"web").resolve()) or not path.is_file():raise web.HTTPNotFound()
    return web.FileResponse(path)

@routes.post("/prompt")
async def prompt(request):
    try:
        data=await request.json(); results=[];outputs={}
        for key,node in data['output'].items():
            if node['class_type']!='DKST_3D_Pose':continue
            image,mask,pose=pack.nodes.PoseImage().render(**node['inputs']);outputs[key]=(image,mask,pose)
            results.append({'node':key,'image_shape':list(image.shape),'mask_shape':list(mask.shape),'mask_min':mask.min().item(),'mask_max':mask.max().item(),'rgb_mean':image.mean().item(),'render_id':node['inputs']['render_id']})
        for key,node in data['output'].items():
            if node['class_type']!='DKST_3D_OpenPose':continue
            inputs=dict(node['inputs']);source,slot=inputs['pose'];inputs['pose']=outputs[source][slot]
            image,keypoints,raw=pack.openpose.OpenPoseImage().render(**inputs)
            from PIL import Image
            pixels=(image[0].numpy()*255).round().astype('uint8');Image.fromarray(pixels).save(Path(TEMP.name)/'openpose.png')
            results.append({'node':key,'type':'openpose','image_shape':list(image.shape),'keypoints':keypoints,'openpose_json':raw,'rgb_mean':image.mean().item()})
        return web.json_response({'prompt_id':'test','results':results})
    except (ValueError,TypeError) as exc:return web.json_response({'error':str(exc)},status=400)

@routes.delete("/test/render/{render_id}")
async def delete_test_render(request):
    # Test-only route for missing-render recovery. Never installed by the pack.
    path=pack.storage.artifact_path(request.match_info['render_id'])
    path.unlink(missing_ok=True);path.with_suffix('.json').unlink(missing_ok=True)
    return web.json_response({'removed':True})

@routes.post("/test/legacy/{render_id}")
async def legacy_test_render(request):
    path=pack.storage.artifact_path(request.match_info['render_id']);metadata=json.loads(path.with_suffix('.json').read_text())
    # Simulate a valid render made before projection metadata was supported.
    return web.json_response(pack.storage.save_render(metadata['scene'],path.read_bytes()))

@routes.get("/test/openpose.png")
async def openpose_test_image(request):
    return web.FileResponse(Path(TEMP.name)/'openpose.png')

@routes.get("/harness.mjs")
async def harness(request):
    return web.Response(text='''globalThis.testErrors=[];import {app} from '/scripts/app.js';import {api} from '/scripts/api.js';import '/extensions/pose/pose_extension.js';
    let nextID=1;
    class Node{constructor(){this.id=nextID++;this.type=this.comfyClass='DKST_3D_Pose';this.outputs=[{type:'IMAGE'},{type:'MASK'},{type:'DKST_POSE'}];this.widgets=[{name:'heads',value:7},{name:'width',value:768},{name:'height',value:1024},{name:'scene_json',value:''},{name:'render_id',value:''}];this.properties={};this.graph=app.graph;this.onNodeCreated?.()}addOutput(name,type){this.outputs.push({name,type});}setSize(){}setDirtyCanvas(){}addWidget(type,name,value,callback,options){const w={type,name,value,callback,options};this.widgets.push(w);if(type==='button'){const b=document.createElement('button');b.textContent=name;b.onclick=callback;document.querySelector('#buttons').append(b);}return w;}addDOMWidget(name,type,element,options){const w={name,type,element,options,serialize:false};this.widgets.push(w);document.querySelector('#preview').replaceChildren(element);return w;}}
    for(const e of app.extensions){await e.setup?.();await e.beforeRegisterNodeDef?.(Node,{name:'DKST_3D_Pose'});}
    globalThis.poseNode=new Node();app.graph._nodes.push(poseNode);
    function serialize(n){return {id:n.id,type:n.type,widgets_values:n.widgets.filter(w=>w.serialize!==false).map(w=>w.value),properties:JSON.parse(JSON.stringify(n.properties))};}
    globalThis.queue=async(withOpenPose=false,options={})=>{const workflow={nodes:app.graph._nodes.map(serialize)},output={};for(const n of app.graph._nodes)output[n.id]={class_type:'DKST_3D_Pose',inputs:Object.fromEntries(n.widgets.filter(w=>w.serialize!==false).map(w=>[w.name,w.value]))};if(withOpenPose)output['999']={class_type:'DKST_3D_OpenPose',inputs:{pose:[String(poseNode.id),2],include_body:true,include_hands:true,include_head:true,line_width:4,joint_radius:4,include_floor:true,...options}};return api.queuePrompt(0,{workflow,output});};
    document.querySelector('#queue').onclick=()=>queue().catch(e=>app.ui.dialog.show(e.message));
    document.querySelector('#openpose').onclick=()=>queue(true).catch(e=>app.ui.dialog.show(e.message));
    document.querySelector('#reload').onclick=()=>{const saved=serialize(poseNode);poseNode.onRemoved?.();app.graph._nodes=[];document.querySelector('#buttons').replaceChildren();globalThis.poseNode=new Node();poseNode.properties=saved.properties;poseNode.widgets.filter(w=>w.serialize!==false).forEach((w,i)=>w.value=saved.widgets_values[i]);poseNode.onConfigure?.(saved);app.graph._nodes.push(poseNode);};
    document.querySelector('#clone').onclick=()=>{const saved=serialize(poseNode),n=new Node();n.widgets.filter(w=>w.serialize!==false).forEach((w,i)=>w.value=saved.widgets_values[i]);n.onConfigure?.(saved);app.graph._nodes.push(n);globalThis.clonedNode=n;};
    globalThis.harnessReady=true;
    ''',content_type="text/javascript")

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=28993);args=parser.parse_args()
    app=web.Application(client_max_size=34*1024*1024);app.add_routes(routes);web.run_app(app,host='127.0.0.1',port=args.port)
