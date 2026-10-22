import json


class FaceMorphStudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'project':('STRING',{'default':'{"version":1,"assets":[],"base":0}','multiline':True}),
            'output_size':(['Base photo','1024','2048','4096'],),
            'color_match':('BOOLEAN',{'default':False,'tooltip':'Match source color statistics to the base. Also shown in the editor preview; strength is adjustable in the editor.'}),
        }}
    RETURN_TYPES = ('IMAGE','MASK')
    RETURN_NAMES = ('image','face_mask')
    FUNCTION = 'execute'
    CATEGORY = 'DINKIssTyle/Facemorph'
    DESCRIPTION = 'Open Face Morph Studio. Upload 1–10 photos, edit automatic correspondence points and blend in real time. Face Liquify also works with a single photo. Run renders from original photos. The standalone folder must be installed in custom_nodes.'

    def execute(self,project,output_size='Base photo',color_match=False):
        import numpy as np
        import torch
        import comfy.model_management
        from .core.service import service
        from .core.render import render
        from .core.hair import read as read_mask
        state = json.loads(project)
        svc = service()
        mesh = svc.prepare(state)
        images = [svc.image(a['id'])[b[1]:b[3],b[0]:b[2]] for a,b in zip(state['assets'],mesh['crop_bounds'])]
        base = state['base']
        h,w = images[base].shape[:2]
        edge = max(w,h) if output_size == 'Base photo' else int(output_size)
        if edge>4096:
            raise ValueError('Base photo output is limited to 4096px. Select a smaller output size.')
        size = (max(2,round(w*edge/max(w,h))),max(2,round(h*edge/max(w,h))))
        masks=[read_mask(svc,identifier) for identifier in mesh['mask_ids']] or None
        mesh['protect_base']=state.get('hair',{}).get('protect_base',True)
        result,mask = render(images,mesh['source_assets'],base,mesh,size,((state.get('blend',{}).get('color_match',100) or 100)/100 if color_match else 0),
                           comfy.model_management.throw_exception_if_processing_interrupted,
                           masks=masks,liquify=state.get('liquify',{}),blend=state.get('blend',{}))
        return torch.from_numpy(np.ascontiguousarray(result))[None],torch.from_numpy(np.ascontiguousarray(mask))[None]
