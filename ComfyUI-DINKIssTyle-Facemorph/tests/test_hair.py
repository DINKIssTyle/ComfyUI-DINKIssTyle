"""Protection invariants use synthetic masks; no model download is required."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import hair
from core.geometry import plan,weights,validate_state
from core.render import render
from core.service import Service


def fixture():
    points=json.loads(Path(__file__).with_name('landmarks.json').read_text())
    a={'id':'a'*64,'points':points,'shape':50,'feature':50,'regions':{}}
    state={'version':1,'assets':[a,copy.deepcopy(a)],'base':0}
    mesh=plan(state,[(80,64)]*2);mesh['sizes']=[(80,64)]*2
    red=np.full((64,80,3),[240,20,10],np.uint8);blue=np.full_like(red,[10,40,230])
    masks=[np.zeros((64,80,3),np.float32) for _ in range(2)]
    for m in masks:m[:,:,0]=1
    return state,mesh,[red,blue],masks


class HairTests(unittest.TestCase):
    def test_excluded_source_has_no_color_contribution(self):
        s,m,images,masks=fixture();masks[1][:]=[0,1,0]
        result,mask=render(images,s['assets'],0,m,(80,64),masks=masks)
        np.testing.assert_allclose(result[32,40],images[0][32,40]/255,atol=1e-6)
        self.assertGreater(mask[32,40],.99)

    def test_soft_validity_renormalizes_color(self):
        s,m,images,masks=fixture();masks[1][:,:,0]=.25
        result,mask=render(images,s['assets'],0,m,(80,64),masks=masks)
        np.testing.assert_allclose(result[32,40],(images[0][32,40]*.8+images[1][32,40]*.2)/255,atol=1e-6)
        self.assertGreater(mask[32,40],.99)

    def test_all_invalid_retains_base_and_zero_mask(self):
        s,m,images,masks=fixture()
        for a in masks:a[:]=[0,1,0]
        result,mask=render(images,s['assets'],0,m,(80,64),masks=masks,liquify={'forehead':80,'face_width':80})
        np.testing.assert_allclose(result,images[0]/255,atol=1e-6)
        self.assertEqual(mask.max(),0)
        self.assertTrue(np.isfinite(result).all())

    def test_base_hair_and_background_fixed_after_liquify(self):
        s,m,images,masks=fixture();masks[0][20:40,30:50]=[0,1,0]
        # A nonuniform base exposes unintended background deformation.
        images[0]=np.random.default_rng(42).integers(0,255,(64,80,3),dtype=np.uint8)
        result,mask=render(images,s['assets'],0,m,(80,64),masks=masks,liquify={'forehead':90,'face_width':90,'eyes':{'size':70}})
        np.testing.assert_allclose(result[20:40,30:50],images[0][20:40,30:50]/255,atol=1e-6)
        np.testing.assert_allclose(result[0],images[0][0]/255,atol=1e-6)
        self.assertEqual(mask[20:40,30:50].max(),0)
        self.assertGreater(mask.max(),.5)

    def test_occlusion_reduces_shape_only(self):
        s,_,_,_=fixture();v=np.ones((2,512));v[1,:468]=.1
        shape=weights(s['assets'],0,'shape',v)
        self.assertAlmostEqual(shape[1,33],1/11)
        self.assertEqual(shape[1,-1],.5)
        np.testing.assert_array_equal(weights(s['assets'],0,'feature',v),np.full((2,512),.5))

    def test_confidence_relaxes_before_mesh_folds(self):
        s,_,_,_=fixture()
        aligned=np.zeros((2,512,2));aligned[0,1]=[1,0];aligned[1,1]=[-.8,0];aligned[:,2]=[0,1]
        v=np.ones((2,512));v[0,1]=.1
        with patch('core.geometry.aligned_points',return_value=aligned),patch('core.triangulation.triangulate',return_value=np.array([[0,1,2]])):
            mesh=plan(s,[(80,64)]*2,v)
        self.assertGreater(mesh['target'][1][0],0)
        self.assertIn('Hair occlusion',mesh['warnings'][0])

    def test_strokes_restore_and_boundary(self):
        s,_,_,_=fixture();raw=np.zeros((64,80,3),np.float32);raw[:]=[0,1,0]
        stroke={'mode':'include','radius':.05,'points':[[.4,.5],[.6,.5]]}
        edit={'feather':0,'strokes':[stroke]}
        corrected=hair.corrected(raw,s['assets'][0]['points'],edit)
        np.testing.assert_array_equal(corrected[32,40],[1,0,0])
        np.testing.assert_array_equal(corrected[0,0],[0,1,0])
        edit['strokes'].append({**stroke,'mode':'restore'})
        np.testing.assert_array_equal(hair.corrected(raw,s['assets'][0]['points'],edit),raw)
        raw[:]=[1,0,0]
        tightened=hair.corrected(raw,s['assets'][0]['points'],{'forehead':50,'temple':50,'feather':0})
        self.assertEqual(tightened[0,0,0],0);self.assertEqual(tightened[32,40,0],1)

    def test_validation_and_safe_cache_paths(self):
        for edits in [{'strokes':[{'mode':'oops','radius':.02,'points':[[.5,.5]]}]},{'feather':float('nan')},{'strokes':[{'mode':'exclude','radius':.02,'points':[[2,0]]}]},{'strokes':[{}]*201}]:
            with self.assertRaises(ValueError):hair.validate({},edits)
        with self.assertRaises(ValueError):hair.validate({'enabled':'yes'})
        with tempfile.TemporaryDirectory() as d:
            svc=Service(d,d)
            with self.assertRaises(ValueError):hair.mask_path(svc,'../secret')
            raw=np.ones((64,80,3),np.float32)
            identifier=hair.token(['raw']);hair.store(svc,identifier,raw)
            hair.store(svc,identifier,np.zeros_like(raw))
            np.testing.assert_array_equal(hair.read(svc,identifier),raw)

    def test_cache_manual_edit_revision_and_workflow_replay(self):
        s,_,_,_=fixture()
        with tempfile.TemporaryDirectory() as d:
            svc=Service(d,d);raw=np.zeros((64,80,3),np.float32);raw[:,:,0]=1
            raw_id=hair.token(['raw']);hair.store(svc,raw_id,raw)
            with patch.object(hair,'automatic',return_value=raw_id):
                ids,_,visibility=hair.prepare(svc,s['assets']);original=ids[0]
                self.assertEqual(ids[0],ids[1]);self.assertEqual(min(visibility[0]),1)
                s['assets'][0]['mask_edit']={'feather':0,'strokes':[{'mode':'exclude','radius':.1,'points':[[.5,.5]]}]}
                ids,_,_=hair.prepare(svc,s['assets']);self.assertNotEqual(ids[0],original)
                replay=json.loads(json.dumps(s));self.assertEqual(hair.prepare(svc,replay['assets'])[0],ids)
                np.testing.assert_array_equal(hair.read(svc,original),raw)


if __name__=='__main__':unittest.main()
