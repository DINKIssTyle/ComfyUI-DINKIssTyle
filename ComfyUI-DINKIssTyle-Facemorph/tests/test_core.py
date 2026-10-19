"""Run: python -m unittest discover -s ComfyUI-DINKIssTyle-Facemorph/tests -v"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.geometry import add_support, plan, validate_state, weights, upgrade_state
from core.topology import OVAL, COUNT, GROUPS
from core.render import render
from core.service import Service


def fixture():
    points=json.loads(Path(__file__).with_name('landmarks.json').read_text())
    a={'id':'a'*64,'points':points,'shape':1,'feature':1,'regions':{}}
    return {'version':1,'assets':[a,copy.deepcopy(a)],'base':0}


class GeometryTests(unittest.TestCase):
    def test_topology_and_identity(self):
        s=fixture();self.assertEqual(len(s['assets'][0]['points']),COUNT)
        mesh=plan(s,[(80,64)]*2)
        np.testing.assert_allclose(mesh['target'],s['assets'][0]['points'],atol=1e-10)
        self.assertTrue(mesh['triangles'])

    def test_zero_weights_use_base(self):
        s=fixture()
        for a in s['assets']:
            a['feature']=0
        w=weights(s['assets'],0,'feature')
        np.testing.assert_array_equal(w[0],np.ones(COUNT))
        np.testing.assert_array_equal(w[1],np.zeros(COUNT))
        mesh=plan(s,[(64,64)]*2)
        self.assertEqual(mesh['alpha'][468],0)

    def test_region_override(self):
        s=fixture();s['assets'][1]['regions']={'eye_l':{'feature':3}}
        w=weights(s['assets'],0,'feature')
        self.assertEqual(w[1,33],.75)
        self.assertEqual(w[1,263],.5)

    def test_translation_alignment(self):
        s=fixture();s['assets'][1]['points']=(np.array(s['assets'][1]['points'])+[.04,-.03]).tolist()
        mesh=plan(s,[(80,64)]*2)
        np.testing.assert_allclose(np.array(mesh['aligned'])[0,:468],np.array(mesh['aligned'])[1,:468],atol=1e-12)

    def test_bad_states(self):
        for value in (-1,float('nan'),float('inf'),101):
            s=fixture();s['assets'][0]['feature']=value
            with self.assertRaises(ValueError):validate_state(s)
        s=fixture();s['assets'][0]['points'][0]=[float('nan'),0]
        with self.assertRaises(ValueError):validate_state(s)
        s=fixture();s['base']=-1
        with self.assertRaises(ValueError):validate_state(s)

    def test_fold_rejected(self):
        s=fixture();s['assets'][1]['points'][61]=[.99,.99];s['assets'][0]['shape']=0
        with self.assertRaises(ValueError):plan(s,[(64,64)]*2)

    def test_render_identity_and_mask(self):
        s=fixture();mesh=plan(s,[(80,64)]*2)
        image=np.random.default_rng(4).integers(0,256,(64,80,3),dtype=np.uint8)
        result,mask=render([image,image],s['assets'],0,mesh,(80,64))
        np.testing.assert_allclose(result,image/255,atol=1/255)
        self.assertGreater(mask[32,40],.99)
        self.assertEqual(mask[0,0],0)

    def test_render_weighted_color_and_multiple_tiles(self):
        s=fixture();mesh=plan(s,[(256,256)]*2)
        a=np.full((256,256,3),[200,40,20],np.uint8);b=np.full_like(a,[20,80,200])
        s['assets'][0]['feature']=1;s['assets'][1]['feature']=3
        mesh=plan(s,[(256,256)]*2)
        result,mask=render([a,b],s['assets'],0,mesh,(256,256))
        np.testing.assert_allclose(result[128,128],np.array([65,70,155])/255,atol=1/255)
        np.testing.assert_allclose(result[0,0],a[0,0]/255,atol=1/255)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(((mask>=0)&(mask<=1)).all())

    def test_cancel(self):
        s=fixture();mesh=plan(s,[(64,64)]*2)
        def cancel():raise InterruptedError('cancel')
        with self.assertRaises(InterruptedError):render([np.zeros((64,64,3),np.uint8)]*2,s['assets'],0,mesh,(64,64),check_cancel=cancel)

    def test_paired_regions_and_individual_override(self):
        s=fixture()
        s['assets'][1]['regions']={'eyes':{'feature':3},'brows':{'shape':3}}
        w=weights(s['assets'],0,'feature')
        self.assertEqual(w[1,33],.75)
        self.assertEqual(w[1,263],.75)
        s['assets'][1]['regions']['eye_l']={'feature':1}
        w=weights(s['assets'],0,'feature')
        self.assertEqual(w[1,33],.5)
        self.assertEqual(w[1,263],.75)
        shape=weights(s['assets'],0,'shape')
        self.assertEqual(shape[1,70],.75)
        self.assertEqual(shape[1,300],.75)

    def test_legacy_workflows_migrate_without_reanalysis(self):
        s=fixture()
        expected=copy.deepcopy(s['assets'][0]['points'])
        for a in s['assets']:
            p=a['points']
            a['points']=p[:468]+[[0,0]]*16+p[468:504]+[[0,0]]*16+p[504:]
            a['ears']=[True,False]
            a['regions']={'ear_l':{'shape':100},'eye_l':{'shape':2}}
            a['warnings']=['Dashed ear guides are estimates.','2 faces detected.']
        mesh=plan(s,[(80,64)]*2)
        self.assertEqual(len(mesh['target']),512)
        self.assertEqual(s['assets'][0]['points'],expected)
        self.assertNotIn('ears',s['assets'][0])
        self.assertEqual(s['assets'][0]['regions'],{'eye_l':{'shape':2}})
        self.assertEqual(s['assets'][0]['warnings'],['2 faces detected.'])

    def test_safe_asset_paths(self):
        with tempfile.TemporaryDirectory() as root:
            svc=Service(root,root)
            for name in ('../secret','a'*63,'a'*64+'.png',None):
                with self.assertRaises(ValueError):svc.path(name)


if __name__=='__main__':unittest.main()
