"""API and node integration with temporary ComfyUI host adapters."""
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from aiohttp import web
from aiohttp.test_utils import TestClient,TestServer

ROOT=Path(__file__).resolve().parents[1]


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.routes=web.RouteTableDef()
        self.host_patch=patch.dict(sys.modules,{
            'server':types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=self.routes))),
            'folder_paths':types.SimpleNamespace(get_input_directory=lambda:self.temp.name,models_dir=self.temp.name),
        })
        self.host_patch.start()
        name='fm_integration'
        for key in list(sys.modules):
            if key==name or key.startswith(name+'.'):sys.modules.pop(key)
        spec=importlib.util.spec_from_file_location(name,ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
        self.pack=importlib.util.module_from_spec(spec);sys.modules[name]=self.pack;spec.loader.exec_module(self.pack)
        self.svc=sys.modules[name+'.core.service'].service()
        app=web.Application();app.add_routes(self.routes)
        self.client=TestClient(TestServer(app));await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close();self.host_patch.stop();self.temp.cleanup()

    async def test_api_rejects_cross_origin_and_bad_schema(self):
        response=await self.client.post('/dinki/facemorph/prepare',json={'version':1,'assets':[1,2],'base':0})
        self.assertEqual(response.status,400)
        response=await self.client.post('/dinki/facemorph/prepare',json={},headers={'Origin':'https://other.example'})
        self.assertEqual(response.status,403)
        response=await self.client.post('/dinki/facemorph/prepare',data='{}',headers={'Content-Type':'text/plain'})
        self.assertEqual(response.status,415)
        response=await self.client.get('/dinki/facemorph/image/not-a-hash')
        self.assertEqual(response.status,400)

    async def test_model_status_and_removed_model_rejection(self):
        response=await self.client.get('/dinki/facemorph/models')
        self.assertFalse((await response.json())['landmarker'])
        response=await self.client.post('/dinki/facemorph/models',json={'kind':'parsing'})
        self.assertEqual(response.status,400)
        self.assertIn('Unknown model',(await response.json())['error'])

    async def test_upload_preview_prepare_and_node_output(self):
        from aiohttp import FormData
        points=json.loads((ROOT/'tests/landmarks.json').read_text())
        assets=[]
        for color in [(200,40,20),(20,80,200)]:
            buffer=io.BytesIO();Image.new('RGB',(80,64),color).save(buffer,format='PNG')
            form=FormData();form.add_field('image',buffer.getvalue(),filename='photo.png',content_type='image/png')
            response=await self.client.post('/dinki/facemorph/upload',data=form)
            self.assertEqual(response.status,200)
            asset=await response.json();asset.update(points=points,shape=50,feature=50,regions={})
            assets.append(asset)
        project={'version':1,'assets':assets,'base':0,'hair':{'enabled':False}}
        response=await self.client.get('/dinki/facemorph/image/'+assets[0]['id'])
        self.assertEqual(response.status,200);self.assertEqual(response.content_type,'image/jpeg')
        response=await self.client.post('/dinki/facemorph/prepare',json=project)
        self.assertEqual(response.status,200,await response.text());self.assertTrue((await response.json())['triangles'])
        try:import torch
        except ImportError:self.skipTest('PyTorch required for IMAGE/MASK output test')
        management=types.ModuleType('comfy.model_management')
        management.throw_exception_if_processing_interrupted=lambda:None
        comfy=types.ModuleType('comfy');comfy.model_management=management
        with patch.dict(sys.modules,{'comfy':comfy,'comfy.model_management':management}):
            image,mask=self.pack.nodes.FaceMorphStudio().execute(json.dumps(project))
        self.assertEqual(tuple(image.shape),(1,64,80,3));self.assertEqual(tuple(mask.shape),(1,64,80))
        self.assertEqual(image.dtype,torch.float32);self.assertTrue(torch.isfinite(image).all())
        self.assertGreater(mask.max().item(),.99)
        project['liquify']={'face_width':60,'eyes':{'size':40}}
        with patch.dict(sys.modules,{'comfy':comfy,'comfy.model_management':management}):
            adjusted,adjusted_mask=self.pack.nodes.FaceMorphStudio().execute(json.dumps(project))
        self.assertGreater((adjusted-image).abs().max().item(),.001)
        self.assertGreater((adjusted_mask-mask).abs().max().item(),.001)
        project['assets'][0]['crop']={'enabled':True,'margin':0}
        response=await self.client.post('/dinki/facemorph/sources',json=project)
        self.assertEqual(response.status,200)
        sources=await response.json();w,h=sources['sizes'][0]
        self.assertLess(w,80);self.assertLess(h,64)
        bounds=sources['crop_bounds'][0]
        response=await self.client.get('/dinki/facemorph/image/'+assets[0]['id']+'?crop='+','.join(map(str,bounds)))
        self.assertEqual(response.status,200)
        with Image.open(io.BytesIO(await response.read())) as preview:
            self.assertEqual(preview.size,(w,h))
        with patch.dict(sys.modules,{'comfy':comfy,'comfy.model_management':management}):
            cropped,cropped_mask=self.pack.nodes.FaceMorphStudio().execute(json.dumps(project))
        self.assertEqual(tuple(cropped.shape),(1,h,w,3));self.assertEqual(tuple(cropped_mask.shape),(1,h,w))
        for bad in ['-1,0,80,64','0,0,81,64','0,0,1,1','0,nan,20,20','0,0,20']:
            response=await self.client.get('/dinki/facemorph/image/'+assets[0]['id']+'?crop='+bad)
            self.assertEqual(response.status,400)


    async def test_hair_mask_api_and_prepare_share_corrected_cache(self):
        import numpy as np
        hair=importlib.import_module('fm_integration.core.hair')
        buffer=io.BytesIO();Image.new('RGB',(80,64),(180,100,80)).save(buffer,format='PNG')
        asset=self.svc.upload(buffer.getvalue())
        asset.update(points=json.loads((ROOT/'tests/landmarks.json').read_text()),shape=50,feature=50,regions={})
        raw=np.zeros((64,80,3),np.float32);raw[:,:,0]=1
        identifier=hair.token(['synthetic-api']);hair.store(self.svc,identifier,raw)
        asset['mask_edit']={'feather':0,'strokes':[{'mode':'exclude','radius':.06,'points':[[.5,.5]]}]}
        with patch.object(hair,'automatic',return_value=identifier):
            response=await self.client.post('/dinki/facemorph/mask',json={'asset':asset})
            self.assertEqual(response.status,200)
            correction=await response.json()
            response=await self.client.get('/dinki/facemorph/mask/'+correction['mask_ids'][0])
            self.assertEqual(response.status,200);self.assertEqual(response.content_type,'image/png')
            self.assertIn('immutable',response.headers['Cache-Control'])
            response=await self.client.post('/dinki/facemorph/prepare',json={'version':1,'assets':[asset,asset],'base':0})
            self.assertEqual(response.status,200)
            mesh=await response.json()
            self.assertEqual(mesh['mask_ids'],correction['mask_ids']*2)
            self.assertLess(min(mesh['visibility'][0]),1)
        response=await self.client.get('/dinki/facemorph/mask/not-a-hash')
        self.assertEqual(response.status,400)
        asset['mask_edit']['strokes'][0]['points']=[[float('nan'),0]]
        response=await self.client.post('/dinki/facemorph/mask',json={'asset':asset})
        self.assertEqual(response.status,400)



if __name__=='__main__':unittest.main()
