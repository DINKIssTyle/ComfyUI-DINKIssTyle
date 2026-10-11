"""Local UI smoke harness using the real editor and Python preview routes.

Run with a Python environment containing torch, pillow, numpy and aiohttp:
  python tests/photo_studio_browser.py
Open http://127.0.0.1:28991. This is an isolated test graph, not ComfyUI.
"""
import runpy
import sys
import tempfile
import types
from pathlib import Path
from aiohttp import web
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
TEMP = tempfile.TemporaryDirectory(prefix="dkst-photo-smoke-")
BASE = Path(TEMP.name)
for kind in ("input", "output", "temp"):
    (BASE / kind).mkdir()

routes = web.RouteTableDef()
server = types.ModuleType("server")
server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(
    routes=routes, user_manager=types.SimpleNamespace(get_request_user_filepath=lambda request, file, create_dir=True: str(BASE / "userdata" / file))))
folders = types.ModuleType("folder_paths")
folders.get_directory_by_type = lambda kind: str(BASE / kind)
sys.modules["server"] = server
sys.modules["folder_paths"] = folders
module = runpy.run_path(str(ROOT / "ComfyUI-DINKIssTyle/dinki_photo_studio.py"))
core = types.ModuleType("nodes")
class TestLoadImage:
    def load_image(self, filename):
        return (module["_read_image_descriptor"]({"filename": filename, "type": "input"}, depth=filename == "depth.png"),)
core.LoadImage = TestLoadImage
sys.modules["nodes"] = core

# Deterministic scene with fine edges, soft tones and highlight spots.
w, h = 1280, 800
x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
pixels = np.stack((.2 + .5*x, .25 + .35*(1-y), .3 + .45*(1-x)), -1)
image = Image.fromarray((pixels*255).astype(np.uint8))
draw = ImageDraw.Draw(image)
draw.polygon([(0,560),(270,160),(650,560)],fill=(45,73,98))
draw.polygon([(400,580),(850,240),(1280,580)],fill=(58,89,92))
draw.polygon([(0,620),(640,520),(1280,640),(1280,800),(0,800)],fill=(95,110,68))
for i in range(9):
    draw.ellipse((800+i*30,400-i*12,815+i*30,415-i*12),fill=(255,230,170))
draw.rounded_rectangle((350,340,480,630),radius=30,fill=(201,91,60))
draw.text((35,40),"PHOTO STUDIO / PREVIEW TEST",fill=(220,230,240))
image.save(BASE / "input/scene.png")
depth = (np.repeat(np.linspace(.1,.9,w)[None,:],h,axis=0)*65535).astype(np.uint16)
depth[340:630,350:480] = 60000
Image.fromarray(depth).resize((640,400)).save(BASE / "input/depth.png")
image.save(BASE / "temp/batch-frame.png")

@routes.get("/")
async def index(request):
    return web.Response(text='''<!doctype html><html><head><title>DKST Photo Studio · isolated UI test</title><style>
    body{margin:0;background:#11151b;color:#ced8e5;font:14px system-ui}header{padding:20px;background:#202832}button,select{padding:8px;margin:4px;background:#3a495b;color:white;border:1px solid #63748a;border-radius:5px}#graph{transform-origin:0 0;margin:40px;width:310px;background:#292f38;border:2px solid #717f91;border-radius:12px;padding:12px;}#graph h3{margin:0 0 15px}#graph button{width:100%}#graph .preview{height:120px}#state{white-space:pre-wrap;padding:20px;color:#85b3d8}
    </style></head><body><header><strong>Isolated editor test · live Python preview engine</strong><label> Graph zoom <select id="zoom"><option>.8</option><option>.4</option><option>1.25</option></select></label><button id="new">Edit selected workflow image</button></header><div id="graph"><h3>DKST Color (Photo Studio)</h3></div><pre id="state">No workflow changes.</pre><script type="module" src="/harness.js"></script></body></html>''', content_type="text/html")

@routes.get("/scripts/app.js")
async def app_js(request):
    return web.Response(text='''export const app=globalThis.testApp={graph:null,canvas:{graph_mouse:[600,100]},nodeOutputs:{},extensions:[],registerExtension(e){this.extensions.push(e)},ui:{dialog:{show:alert}}};''',content_type="text/javascript")

@routes.get("/scripts/api.js")
async def api_js(request):
    return web.Response(text='''const events=new EventTarget();export const api={apiURL:p=>p,fetchApi:(p,o)=>fetch(p,o),addEventListener:(...a)=>events.addEventListener(...a),removeEventListener:(...a)=>events.removeEventListener(...a)};''', content_type="text/javascript")

@routes.get("/extensions/dkst/{file}")
async def extension(request):
    filename = request.match_info["file"]
    if filename not in ("dinki_photo_studio.js", "dinki_photo_studio_editor.js", "dinki_photo_studio.css"):
        raise web.HTTPNotFound()
    return web.FileResponse(ROOT / "ComfyUI-DINKIssTyle/js" / filename)

@routes.get("/view")
async def view(request):
    return web.FileResponse(module["_descriptor_path"](dict(request.query)))

@routes.post("/upload/image")
async def upload(request):
    reader = await request.multipart()
    field = await reader.next()
    name = Path(field.filename).name
    data = await field.read()
    (BASE / "input" / name).write_bytes(data)
    return web.json_response({"name":name,"type":"input","subfolder":""})

@routes.get("/harness.js")
async def harness(request):
    return web.Response(text='''import {app} from '/scripts/app.js';import {api} from '/scripts/api.js';import '/extensions/dkst/dinki_photo_studio.js';import {openPhotoStudio,workflowImages} from '/extensions/dkst/dinki_photo_studio_editor.js';
    const data=await (await fetch('/dinki/photo-studio/presets')).json();let id=1,linkId=1;
    const graph=app.graph={_nodes:[],links:{},getNodeById(id){return this._nodes.find(n=>n.id===id)},add(n){n.id=id++;n.graph=this;this._nodes.push(n)},remove(n){this._nodes.splice(this._nodes.indexOf(n),1)},beforeChange(){},afterChange(){document.querySelector('#state').textContent=JSON.stringify(this._nodes.map(n=>({id:n.id,type:n.type,inputs:n.inputs,settings:Object.fromEntries(n.widgets.filter(w=>w.serialize!==false).map(w=>[w.name,w.value]))})),null,2)},incrementVersion(){}};
    class Node{constructor(type){this.type=this.comfyClass=type;this.title=type==='LoadImage'?'Original image':type;this.pos=[100,100];this.size=[300,700];this.widgets=type==='DINKI_Photo_Studio'?[{name:'active',value:true},{name:'preset',value:'Custom'},...Object.entries(data.defaults).map(([name,value])=>({name,value,type:'number',options:{}}))]:[{name:'image',value:'scene.png'},{name:'interpretation',value:'Image'}];this.inputs=type==='DINKI_Photo_Studio'?[{name:'image',link:null},{name:'depth_image',link:null}]:[];this.outputs=[{type:'IMAGE'}];if(type==='DINKI_Photo_Studio')this.onNodeCreated?.()}computeSize(){return [300,700]}setSize(s){this.size=s}setDirtyCanvas(){}addWidget(type,name,value,callback,options){const w={type,name,value,callback,options};this.widgets.push(w);if(type==='button'){const b=document.createElement('button');b.textContent=name;b.onclick=callback;document.querySelector('#graph').append(b)}return w}addDOMWidget(name,type,element,options){const w={name,type,element,options};this.widgets.push(w);const box=document.createElement('div');box.className='preview';box.append(element);document.querySelector('#graph').append(box);return w}disconnectInput(i){const l=this.inputs[i].link;if(l)delete graph.links[l];this.inputs[i].link=null}connect(slot,target,input){target.disconnectInput(input);const l={id:linkId++,origin_id:this.id,origin_slot:slot,target_id:target.id,target_slot:input};graph.links[l.id]=l;target.inputs[input].link=l.id;return l}}
    globalThis.LiteGraph={createNode:type=>new Node(type)};
    for(const e of app.extensions)await e.beforeRegisterNodeDef(Node,{name:'DINKI_Photo_Studio'});
    const photo=new Node('LoadImage');graph.add(photo);const depth=new Node('LoadImage');depth.title='16-bit depth map';depth.widgets[0].value='depth.png';graph.add(depth);const node=new Node('DINKI_Photo_Studio');graph.add(node);photo.connect(0,node,0);graph.afterChange();
    document.querySelector('#zoom').onchange=e=>document.querySelector('#graph').style.transform=`scale(${e.target.value})`;
    document.querySelector('#zoom').dispatchEvent(new Event('change'));
    document.querySelector('#new').onclick=()=>openPhotoStudio(null,workflowImages(graph)[0]);
    ''',content_type="text/javascript")

if __name__ == "__main__":
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app, host="127.0.0.1", port=28991, print=print)
