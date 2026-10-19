"""Persistent local assets and lazy, serialized analysis models."""
import hashlib
import io
from pathlib import Path
import re
import threading
import urllib.request
import uuid
import zipfile

import numpy as np
from PIL import Image, ImageOps
from .geometry import add_support, plan

MODEL_SHA256 = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
ID = re.compile(r'^[0-9a-f]{64}$')
MAX_PIXELS = 40_000_000


class Service:
    def __init__(self, input_dir, model_dir):
        self.assets = Path(input_dir)/'DKST_Facemorph'
        self.models = Path(model_dir)/'facemorph'
        self.lock = threading.RLock()
        self.detector = None
        self.segmenter = None

    def path(self, asset_id, suffix='.png'):
        if not isinstance(asset_id,str) or not ID.fullmatch(asset_id):
            raise ValueError('Invalid photo identifier.')
        path = self.assets/(asset_id+suffix)
        if path.is_symlink():
            raise ValueError('Linked assets are not supported.')
        return path

    def status(self):
        return {'landmarker':(self.models/'face_landmarker.task').is_file(),
                'hair':(self.models/'selfie_multiclass.tflite').is_file(),
                'model_directory':str(self.models)}

    def install(self, kind):
        from . import hair
        choices={'landmarker':('face_landmarker.task',MODEL_URL,MODEL_SHA256),
                 'hair':(hair.MODEL_NAME,hair.MODEL_URL,hair.MODEL_SHA256)}
        if kind not in choices:
            raise ValueError('Unknown model.')
        with self.lock:
            self.models.mkdir(parents=True,exist_ok=True)
            name,url,checksum=choices[kind];dest=self.models/name
            if dest.exists():
                return self.status()
            tmp=self.models/(uuid.uuid4().hex+'.part')
            try:
                with urllib.request.urlopen(url,timeout=60) as response,tmp.open('wb') as out:
                    count=0
                    while chunk:=response.read(1024*1024):
                        count+=len(chunk)
                        if count>64*1024*1024:
                            raise ValueError('Unexpected model size.')
                        out.write(chunk)
                if hashlib.sha256(tmp.read_bytes()).hexdigest()!=checksum:
                    raise ValueError('Model checksum mismatch.')
                if kind=='landmarker' and not zipfile.is_zipfile(tmp):
                    raise ValueError('Invalid face model archive.')
                if kind=='hair' and tmp.read_bytes()[4:8]!=b'TFL3':
                    raise ValueError('Invalid hair model.')
                tmp.replace(dest)
            finally:
                tmp.unlink(missing_ok=True)
            return self.status()

    def _hair_segmenter(self):
        if self.segmenter is None:
            from .hair import MODEL_NAME
            if not self.status()['hair']:
                raise ValueError('Open Face Morph Studio to download the missing hair model automatically.')
            mp=self._mediapipe()
            options=mp.tasks.vision.ImageSegmenterOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.models/MODEL_NAME)),
                output_confidence_masks=True,output_category_mask=False)
            self.segmenter=mp.tasks.vision.ImageSegmenter.create_from_options(options)
        return self.segmenter

    def upload(self, raw):
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.width*opened.height > MAX_PIXELS or min(opened.size) < 32:
                raise ValueError('Photos must be at least 32px and at most 40 megapixels.')
            image = ImageOps.exif_transpose(opened).convert('RGB')
            buffer = io.BytesIO()
            image.save(buffer,format='PNG')
        content = buffer.getvalue()
        asset_id = hashlib.sha256(content).hexdigest()
        self.assets.mkdir(parents=True,exist_ok=True)
        dest = self.path(asset_id)
        if not dest.exists():
            tmp = dest.with_suffix('.'+uuid.uuid4().hex+'.part')
            try:
                tmp.write_bytes(content)
                tmp.replace(dest)
            finally:
                tmp.unlink(missing_ok=True)
        return {'id':asset_id,'width':image.width,'height':image.height}

    def image(self, asset_id):
        with Image.open(self.path(asset_id)) as image:
            return np.array(image.convert('RGB'))

    @staticmethod
    def _mediapipe():
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ValueError(
                'MediaPipe is unavailable. Restart ComfyUI to run automatic dependency installation. '
                'If installation fails, check the DKST Face Morph Studio startup log.'
            ) from exc
        return mp

    def _detector(self):
        if self.detector is None:
            if not self.status()['landmarker']:
                raise ValueError('Open Face Morph Studio to download the missing face model automatically.')
            mp = self._mediapipe()
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.models/'face_landmarker.task')),
                num_faces=10, min_face_detection_confidence=.35, min_face_presence_confidence=.35)
            self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        return self.detector

    def analyze(self, asset_id, face_index=0):
        with self.lock:
            mp = self._mediapipe()
            import cv2
            raw = self.image(asset_id)
            h,w = raw.shape[:2]
            scale = min(1,1280/max(w,h))
            small = cv2.resize(raw,(round(w*scale),round(h*scale)))
            detector = self._detector()
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(small)))
            faces = sorted(result.face_landmarks,key=lambda f:(max(p.x for p in f)-min(p.x for p in f))*(max(p.y for p in f)-min(p.y for p in f)),reverse=True)
            if not faces:
                raise ValueError('No face detected. Try a front-facing, well-lit photograph.')
            if not isinstance(face_index,int) or not 0 <= face_index < len(faces):
                raise ValueError('Invalid face selection.')
            face = np.array([[p.x,p.y] for p in faces[face_index][:468]])
            # MediaPipe normalized coordinates divide by width, not width-1.
            face *= np.array([small.shape[1]/(small.shape[1]-1),small.shape[0]/(small.shape[0]-1)])
            warnings = []
            if len(faces)>1:
                warnings.append(f'{len(faces)} faces detected; selected face {face_index+1} (largest first).')
            asset = {'id':asset_id,'width':w,'height':h,'points':add_support(face),
                    'face_count':len(faces),'face_index':face_index,
                    'warnings':warnings,'shape':50,'feature':50,'regions':{}}
            from . import hair
            hair.prepare(self,[asset])
            return asset

    def prepare(self,state):
        from .geometry import validate_state
        assets,base = validate_state(state)
        sizes = []
        for a in assets:
            with Image.open(self.path(a['id'])) as image:
                sizes.append(image.size)
        from . import hair
        ids,raw_ids,visibility=[],[],None
        if state.get('hair',{}).get('enabled',True):
            ids,raw_ids,visibility=hair.prepare(self,assets)
        mesh = plan(state,sizes,visibility)
        return {**mesh,'sizes':sizes,'base':base,'mask_ids':ids,'raw_mask_ids':raw_ids,'visibility':visibility}



_instance = None
_instance_lock = threading.Lock()

def service():
    global _instance
    with _instance_lock:
        if _instance is None:
            import folder_paths
            _instance = Service(folder_paths.get_input_directory(),folder_paths.models_dir)
        return _instance
