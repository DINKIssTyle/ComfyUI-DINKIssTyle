import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.liquify import build_warp,source_coordinates,apply_liquify,validate

ROOT=Path(__file__).resolve().parents[1]
POINTS=json.loads((ROOT/'tests/landmarks.json').read_text())


class LiquifyTests(unittest.TestCase):
    def test_zero_and_disabled_are_exact_identity(self):
        image=np.random.default_rng(4).random((64,80,3),dtype=np.float32)
        mask=np.ones((64,80),np.float32)
        for settings in ({},{'eyes':{'size':0}},{'enabled':False,'face_width':100}):
            output,out_mask=apply_liquify(image,mask,POINTS,(80,64),settings)
            self.assertIs(output,image);self.assertIs(out_mask,mask)

    def test_linked_eyes_and_optional_override(self):
        warp=build_warp(POINTS,(1024,800),{'eyes':{'size':60,'tilt':20}})
        a,b=warp['ops'][:2]
        self.assertAlmostEqual(a['matrix'][0],b['matrix'][0])
        self.assertAlmostEqual(a['matrix'][1],-b['matrix'][1])
        warp=build_warp(POINTS,(1024,800),{'eyes':{'size':60},'eye_l':{'size':-40}})
        self.assertNotEqual(warp['ops'][0]['matrix'],warp['ops'][1]['matrix'])

    def test_edges_pinned_and_extreme_controls_finite(self):
        settings={'eyes':{'size':100,'height':-100,'width':100,'tilt':100,'distance':-100},
                  **{k:100 for k in ('nose_height','nose_width','smile','upper_lip','lower_lip','mouth_width','mouth_height','forehead','chin_height','jawline','face_width')}}
        warp=build_warp(POINTS,(800,1024),settings)
        self.assertLessEqual(len(warp['ops']),16)
        edges=np.array([[0,.5],[1,.5],[.5,0],[.5,1],[0,0],[1,1]])
        np.testing.assert_allclose(source_coordinates(edges,warp),edges,atol=1e-12)
        q=source_coordinates(np.random.default_rng(4).random((1000,2)),warp)
        self.assertTrue(np.isfinite(q).all());self.assertTrue(((q>=0)&(q<=1)).all())

    def test_image_and_mask_warp_and_restore(self):
        h,w=192,160
        x,y=np.meshgrid(np.linspace(0,1,w),np.linspace(0,1,h))
        image=np.stack((x,y,x*y),-1).astype(np.float32)
        mask=(((x-.5)**2+(y-.5)**2)<.08).astype(np.float32)
        out,alpha=apply_liquify(image,mask,POINTS,(80,64),{'face_width':80,'chin_height':60})
        self.assertGreater(float(abs(out-image).max()),.001)
        self.assertGreater(float(abs(alpha-mask).max()),.1)
        self.assertEqual(out.dtype,np.float32);self.assertEqual(alpha.dtype,np.float32)
        np.testing.assert_array_equal(out[0],image[0])
        restored,_=apply_liquify(image,mask,POINTS,(80,64),{})
        np.testing.assert_array_equal(restored,image)

    def test_invalid_settings_rejected(self):
        for bad in ({'nose_width':float('nan')},{'smile':101},{'eyes':{'distance':'5'}},
                    {'eye_l':{'distance':5}},{'enabled':'yes'},{'unknown':4}):
            with self.assertRaises(ValueError):validate(bad)

    def test_cancel(self):
        def cancel():raise InterruptedError('cancel')
        with self.assertRaises(InterruptedError):
            apply_liquify(np.zeros((64,80,3),np.float32),np.zeros((64,80),np.float32),POINTS,(80,64),{'smile':20},cancel)

    def test_javascript_python_numerical_parity(self):
        node=shutil.which('node')
        bundled=Path('/Users/dinki/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if bundled.exists():node=str(bundled)
        if not node:self.skipTest('Node.js is needed for cross-runtime parity')
        settings={'eyes':{'size':45,'tilt':-30,'distance':20},'eye_l':{'height':-20},'nose_width':25,'smile':40,'upper_lip':30,'jawline':-35,'chin_height':18,'face_width':12}
        data={'points':POINTS,'size':[1024,800],'settings':settings}
        code="""const fs=require('fs');(async()=>{const src=fs.readFileSync(process.argv[1],'utf8');const {buildWarp}=await import('data:text/javascript;base64,'+Buffer.from(src).toString('base64'));const d=JSON.parse(fs.readFileSync(0,'utf8'));console.log(JSON.stringify(buildWarp(d.points,d.size,d.settings)))})().catch(e=>{console.error(e);process.exitCode=1});"""
        result=subprocess.run([node,'-e',code,str(ROOT/'web/liquify.js')],input=json.dumps(data),capture_output=True,text=True,check=True)
        js=json.loads(result.stdout);py=build_warp(POINTS,data['size'],settings)
        for key in ('origin','axis','scale','size'):np.testing.assert_allclose(js[key],py[key],atol=1e-10)
        self.assertEqual(len(js['ops']),len(py['ops']))
        for a,b in zip(js['ops'],py['ops']):
            for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-10)


if __name__=='__main__':unittest.main()
