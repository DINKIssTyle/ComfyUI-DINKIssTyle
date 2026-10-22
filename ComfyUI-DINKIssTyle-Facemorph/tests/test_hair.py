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
    def test_guided_confidence_follows_photo_edges(self):
        import cv2
        guide=np.zeros((64,96,3),np.float32);guide[:,48:]=[.8,.6,.4]
        truth=np.zeros((64,96),np.float32);truth[:,:48]=1
        coarse=cv2.GaussianBlur(truth,(0,0),6)
        refined=hair.guided_mask(guide,coarse,radius=12)
        self.assertLess(np.mean((refined-truth)**2),np.mean((coarse-truth)**2)*.5)
        self.assertTrue(np.isfinite(refined).all())
        self.assertTrue(((refined>=0)&(refined<=1)).all())

    def test_morph_backing_uses_the_same_hair_constrained_mesh(self):
        import cv2
        from core.geometry import protect_target
        from core.topology import OVAL
        s,m,_,_=fixture();w,h=320,256
        original=np.asarray(s['assets'][0]['points'])
        target=original.copy();target[:-8,0]=.5+(target[:-8,0]-.5)*.8
        m['target']=target.tolist();m['sizes']=[(w,h)]*2
        m['color_corrections']=[{'gain':[1,1,1],'offset':[0,0,0]}]*2
        image=np.full((h,w,3),[35,80,130],np.uint8)
        polygon=np.round(original[OVAL]*[w-1,h-1]).astype(np.int32)
        cv2.fillPoly(image,[polygon],(220,170,140));cv2.polylines(image,[polygon],True,(15,10,5),2)
        masks=[np.zeros((h,w,3),np.float32) for _ in range(2)]
        for mask in masks:mask[:,:,0]=1
        masks[0][:,245:]=[0,1,0]
        protection=np.floor(hair.distance_field(masks[0][:,:,1])*255+.5)/255
        constrained=protect_target(m,original,protection)
        self.assertGreater(np.max(abs(np.asarray(constrained['target'])-target)),.005)
        expected,alpha=render([image,image],s['assets'],0,{**constrained,'protect_base':False},(w,h),masks=masks)
        actual,_=render([image,image],s['assets'],0,m,(w,h),masks=masks)
        visible=masks[0][:,:,1]==0
        np.testing.assert_allclose(actual[visible],expected[visible],atol=1e-6)
        self.assertTrue(np.any((alpha>0)&(alpha<.9)&visible))
        np.testing.assert_allclose(actual[:,245:],image[:,245:]/255,atol=1e-6)

    def test_target_pinning_matches_browser_and_preserves_triangles(self):
        import subprocess
        from core.geometry import protect_target
        s,m,_,_=fixture()
        original=np.asarray(s['assets'][0]['points'])
        target=original.copy();target[:-8,0]=.5+(target[:-8,0]-.5)*.85
        m['target']=target.tolist()
        mask=np.zeros((64,80,4),np.uint8);mask[:,:,0]=255;mask[:,:16,1]=255
        mask[:,:,2]=np.uint8(hair.distance_field(mask[:,:,1]/255)*255+.5)
        expected=protect_target(m,original,mask[:,:,2].astype(float)/255)
        source=(Path(__file__).resolve().parents[1]/'web/math.js').read_text()
        code="""import fs from 'node:fs';const input=JSON.parse(fs.readFileSync(0,'utf8'));
const {protectTarget}=await import('data:text/javascript;base64,'+Buffer.from(input.source).toString('base64'));
console.log(JSON.stringify(protectTarget(input.mesh,input.base,input.mask)));"""
        payload={'source':source,'mesh':m,'base':original.tolist(),'mask':{'width':80,'height':64,'hair':True,'data':mask.ravel().tolist()}}
        result=subprocess.run(['node','--input-type=module','-e',code],input=json.dumps(payload),text=True,capture_output=True,check=True)
        actual=json.loads(result.stdout)
        np.testing.assert_allclose(actual['target'],expected['target'],atol=1e-12)
        self.assertEqual(actual['hair_pin_amount'],expected['hair_pin_amount'])
        np.testing.assert_array_equal(np.asarray(actual['target'])[-8:],target[-8:])
        self.assertIs(protect_target(m,original,None),m)

    def test_boundary_color_transition_is_local_to_protected_hair(self):
        s,m,images,masks=fixture()
        m['triangles']=[[504,506,508],[504,508,510]];m['alpha']=[1]*512
        masks[0][:,48:]=[0,1,0]
        protected,_=render(images,s['assets'],0,m,(80,64),masks=masks)
        ordinary,_=render(images,s['assets'],0,{**m,'protect_base':False},(80,64),masks=masks)
        base=images[0][32,47]/255
        self.assertLess(np.linalg.norm(protected[32,47]-base),np.linalg.norm(ordinary[32,47]-base)*.5)
        np.testing.assert_allclose(protected[32,32],ordinary[32,32],atol=1e-6)
        np.testing.assert_allclose(protected[:,48:],images[0][:,48:]/255,atol=1e-6)

    def test_target_protection_relaxes_instead_of_folding(self):
        from core.geometry import protect_target
        target=[[0,0],[1,0],[0,1]]
        mesh={'target':target,'reference':target,'triangles':[[0,1,2]]}
        base=[[0,0],[-.8,0],[0,1]]
        result=protect_target(mesh,base,np.zeros((16,16)))
        self.assertEqual(result['hair_pin_amount'],.5)
        self.assertGreater(result['target'][1][0],0)

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
