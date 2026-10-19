"""Source crop coordinates, padding, mask alignment and saved-state invariants."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import crop,hair
from core.service import Service
from core.geometry import add_support


class CropTests(unittest.TestCase):
    def test_margin_matches_face_crop_export_and_shifts_at_edges(self):
        p=np.tile([30/99,45/99],(512,1));p[0]=[20/99,30/99];p[1]=[39/99,59/99]
        a={'points':p.tolist(),'crop':{'enabled':True,'margin':50}}
        self.assertEqual(crop.bounds(a,(100,100)),[10,15,50,75])
        p[0]=[0,0];p[1]=[19/99,29/99];p[2:468]=[.1,.1];a['points']=p.tolist()
        self.assertEqual(crop.bounds(a,(100,100)),[0,0,40,60])
        a['crop']['margin']=200
        b=crop.bounds(a,(100,100));self.assertTrue(0<=b[0]<b[2]<=100);self.assertTrue(0<=b[1]<b[3]<=100)

    def test_disabled_identity_and_invalid_settings(self):
        self.assertEqual(crop.bounds({'crop':{},'points':[]},(120,80)),[0,0,120,80])
        for value in [None,{'enabled':1},{'margin':True},{'margin':-1},{'margin':201},{'margin':float('nan')},{'other':1}]:
            with self.assertRaises(ValueError):crop.validate(value)

    def test_pixel_center_transform_and_frame_anchors(self):
        p=np.tile([40/99,32/79],(512,1)).tolist();b=[20,16,60,48]
        result=crop.points(p,b,(100,80))
        np.testing.assert_allclose(result[0],[20/39,16/31])
        np.testing.assert_array_equal(result[-8:],[[0,0],[.5,0],[1,0],[1,.5],[1,1],[.5,1],[0,1],[0,.5]])
        self.assertEqual(p[0],[40/99,32/79])

    def test_mask_crop_uses_original_pixel_centers(self):
        with tempfile.TemporaryDirectory() as d:
            service=Service(d,d)
            raw=np.zeros((64,80,3),np.float32);raw[:,:,0]=np.arange(80)[None,:]/79;raw[16:48,20:60,1]=1
            identifier=hair.token(['gradient']);hair.store(service,identifier,raw)
            result=crop.mask(service,identifier,[20,16,60,48],[80,64]);mask=hair.read(service,result)
            self.assertEqual(mask.shape,(32,40,3));self.assertEqual(mask[:,:,1].min(),1)
            np.testing.assert_allclose(mask[0,0,0],20/79,atol=1/255)
            np.testing.assert_allclose(mask[-1,-1,0],59/79,atol=1/255)
            self.assertEqual(crop.mask(service,identifier,[0,0,80,64],[80,64]),identifier)

    def test_source_and_base_crops_leave_saved_points_and_strokes_intact(self):
        with tempfile.TemporaryDirectory() as d:
            service=Service(d,d);image=Image.new('RGB',(100,80));buffer=__import__('io').BytesIO();image.save(buffer,format='PNG')
            a=service.upload(buffer.getvalue());a.update(points=json.loads(Path(__file__).with_name('landmarks.json').read_text()),shape=50,feature=50,regions={})
            a['crop']={'enabled':True,'margin':30};a['mask_edit']={'strokes':[{'mode':'include','radius':.02,'points':[[.5,.5]]}]}
            state={'version':1,'base':0,'assets':[a,copy.deepcopy(a)],'hair':{'enabled':False}}
            saved=copy.deepcopy(state);mesh=service.prepare(state)
            self.assertEqual(state,saved)
            self.assertLess(mesh['sizes'][0][0],100);self.assertLess(mesh['sizes'][0][1],80)
            self.assertEqual(mesh['source_assets'][0]['mask_edit'],a['mask_edit'])
            self.assertNotEqual(mesh['source_assets'][0]['points'],a['points'])
            state['assets'][0]['crop']['enabled']=False
            self.assertEqual(service.prepare(state)['sizes'][0],[100,80])
            single={**state,'assets':[a]};self.assertEqual(len(service.sources(single)['sizes']),1)


if __name__=='__main__':unittest.main()
