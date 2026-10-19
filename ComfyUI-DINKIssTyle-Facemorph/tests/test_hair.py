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

    def test_shrinking_jaw_does_not_reveal_original_outline(self):
        import cv2
        from core.topology import OVAL
        from core.liquify import apply_liquify
        s,m,_,masks=fixture()
        m['sizes']=[(320,256)]*2
        polygon=np.round(np.asarray(s['assets'][0]['points'])[OVAL]*[319,255]).astype(np.int32)
        face=np.zeros((256,320),np.uint8);cv2.fillPoly(face,[polygon],255)
        image=np.full((256,320,3),[30,100,150],np.uint8);image[face>0]=[220,170,140]
        cv2.polylines(image,[polygon],True,(20,10,5),2)
        masks=[np.zeros((256,320,3),np.float32) for _ in range(2)]
        for mask in masks:mask[:,:,0]=face/255
        for settings in [{'jawline':-81},{'face_width':-80},{'chin_height':-80},{'jawline':-81,'face_width':-50}]:
            result,alpha=render([image,image],s['assets'],0,m,(320,256),masks=masks,liquify=settings)
            expected,_=apply_liquify(image.astype(np.float32)/255,None,m['target'],(320,256),settings)
            revealed=(np.abs(expected-image/255).max(-1)>.1)&(alpha<1e-6)&(face>0)
            self.assertGreater(revealed.sum(),0)
            np.testing.assert_allclose(result[revealed],expected[revealed],atol=1e-6)

    def test_hair_pins_swept_path_to_avoid_displaced_hair_copies(self):
        from core.liquify import pin_hair
        protection=np.zeros((32,32),np.float32);protection[:,15:17]=1
        destination=np.array([[[.2,.5],[.1,.1],[.5,.5]]],np.float32)
        source=np.array([[[.8,.5],[.12,.12],[.8,.5]]],np.float32)
        result=pin_hair(destination,source,hair.distance_field(protection))
        np.testing.assert_allclose(result[0,0],destination[0,0])
        np.testing.assert_allclose(result[0,1],source[0,1])
        np.testing.assert_allclose(result[0,2],destination[0,2])

    def test_distance_halo_keeps_a_pinned_translation_monotonic(self):
        from core.liquify import pin_hair
        protection=np.zeros((256,256),np.float32);protection[:,180:190]=1
        field=np.floor(hair.distance_field(protection)*255+.5)/255
        x=np.linspace(.02,.98,1000)
        destination=np.stack([x,np.full_like(x,.5)],-1)[None]
        source=destination.copy();source[:,:,0]=np.minimum(1,source[:,:,0]+.1)
        result=pin_hair(destination,source,field)
        self.assertGreaterEqual(np.diff(result[0,:,0]).min(),-1e-6)

    def test_premultiplied_warp_preserves_color_at_alpha_edges(self):
        from core.liquify import apply_liquify
        s,m,_,_=fixture();mask=np.zeros((64,80),np.float32);mask[15:45,25:55]=1
        color=np.array([.6,.4,.2],np.float32);image=mask[:,:,None]*color
        image,alpha=apply_liquify(image,mask,m['target'],(80,64),{'jawline':-81,'face_width':-50},premultiplied=True)
        np.testing.assert_allclose(image,alpha[:,:,None]*color,atol=1e-6)

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
        np.testing.assert_array_equal(corrected[32,40,:2],[1,0])
        np.testing.assert_array_equal(corrected[0,0],[0,1,0])
        edit['strokes'].append({**stroke,'mode':'restore'})
        restored=hair.corrected(raw,s['assets'][0]['points'],edit)
        np.testing.assert_array_equal(restored[:,:,:2],raw[:,:,:2])
        np.testing.assert_array_equal(restored[:,:,2],hair.distance_field(raw[:,:,1]))
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
                snapshot=hair.read(svc,original).copy()
                s['assets'][0]['mask_edit']={'feather':0,'strokes':[{'mode':'exclude','radius':.1,'points':[[.5,.5]]}]}
                ids,_,_=hair.prepare(svc,s['assets']);self.assertNotEqual(ids[0],original)
                replay=json.loads(json.dumps(s));self.assertEqual(hair.prepare(svc,replay['assets'])[0],ids)
                np.testing.assert_array_equal(hair.read(svc,original),snapshot)


if __name__=='__main__':unittest.main()
