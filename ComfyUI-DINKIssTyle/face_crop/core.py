"""YuNet detection, original-resolution crops and persistent background exports."""
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile
import threading
import time
import uuid

import numpy as np
from PIL import Image, ImageOps
from .archive import Archives

MODELS = Path(__file__).parent / "models"
ACTIVE = {"queued", "processing", "cancel_requested"}


def safe_component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[\w-]{1,64}", value):
        raise ValueError("Folder names and prefix must use 1–64 letters, numbers, underscores or hyphens.")
    if value.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                         *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("This name is reserved by Windows.")
    return value


def options(data):
    folder = data.get("output_folder", "DKST_FaceCrops/my_person")
    if not isinstance(folder, str) or not folder or len(folder) > 240:
        raise ValueError("Enter a folder relative to ComfyUI's output directory.")
    folder = folder.replace("\\", "/")
    if folder.startswith("/") or PureWindowsPath(folder).drive:
        raise ValueError("The export folder must be relative to ComfyUI's output directory.")
    folder = "/".join(safe_component(part) for part in folder.split("/"))
    choice = data.get("face_selection", "All faces")
    shape = data.get("crop_shape", "BBox ratio")
    grid = data.get("grid_split", "Off")
    if choice not in {"All faces", "Largest face"} or shape not in {"BBox ratio", "Square"} or grid not in {"Off", "4x4"}:
        raise ValueError("Unknown face selection, crop shape or grid mode.")
    limit = data.get("limit_size", True)
    size = data.get("max_size", 1024)
    padding = data.get("padding_percent", 30.0)
    confidence = data.get("confidence", 0.8)
    if not isinstance(limit, bool) or isinstance(size, bool) or not isinstance(size, int) or not 64 <= size <= 8192:
        raise ValueError("Maximum size must be an integer from 64 to 8192 pixels.")
    for value, low, high, name in [(padding, 0, 200, "Padding"), (confidence, 0.1, 0.99, "Confidence")]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} is out of range.")
    return dict(output_folder=folder, filename_prefix=safe_component(data.get("filename_prefix", "face")),
                face_selection=choice, crop_shape=shape, grid_split=grid, limit_size=limit,
                max_size=size, padding_percent=float(padding), confidence=float(confidence))


def crop_bounds(width, height, bbox, padding_percent, square=False):
    x, y, w, h = map(float, bbox)
    if not all(math.isfinite(n) for n in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError("Invalid face bounding box.")
    # Clip the detection, then shift the padded window inward before clipping.
    x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Face bounding box is outside the image.")
    w, h = x1 - x0, y1 - y0
    crop_w, crop_h = w * (1 + padding_percent / 50), h * (1 + padding_percent / 50)
    if square:
        side = min(width, height, math.ceil(max(crop_w, crop_h)))
        left = math.floor(min(max(0, (x0 + x1 - side) / 2), width - side))
        top = math.floor(min(max(0, (y0 + y1 - side) / 2), height - side))
        return left, top, left + side, top + side
    crop_w, crop_h = min(width, crop_w), min(height, crop_h)
    left = min(max(0, (x0 + x1 - crop_w) / 2), width - crop_w)
    top = min(max(0, (y0 + y1 - crop_h) / 2), height - crop_h)
    return (max(0, math.floor(left)), max(0, math.floor(top)),
            min(width, math.ceil(left + crop_w)), min(height, math.ceil(top + crop_h)))


def regions(width, height, split):
    n = 4 if split == "4x4" else 1
    if width < n or height < n:
        raise ValueError("Image is too small for the selected grid split.")
    return [(col * width // n, row * height // n, (col + 1) * width // n, (row + 1) * height // n)
            for row in range(n) for col in range(n)]


def resize_crop(image, settings):
    if settings["limit_size"] and max(image.size) > settings["max_size"]:
        scale = settings["max_size"] / max(image.size)
        image = image.resize(tuple(max(1, round(n * scale)) for n in image.size), Image.Resampling.LANCZOS)
    return image


class YuNet:
    def __init__(self):
        try:
            import cv2
        except ImportError as error:
            raise ValueError("Face Crop needs OpenCV in ComfyUI's Python. Install requirements-face-crop.txt there, then restart.") from error
        if not hasattr(cv2, "FaceDetectorYN_create"):
            raise ValueError("This OpenCV build has no YuNet face detector. Update OpenCV in ComfyUI's Python.")
        self.cv2 = cv2
        self.model = "face_detection_yunet_2026may.onnx" if int(cv2.__version__.split(".")[0]) >= 5 else "face_detection_yunet_2023mar.onnx"
        manifest = json.loads((MODELS / "UPSTREAM.json").read_text(encoding="utf-8"))
        model = MODELS / self.model
        if not model.is_file() or hashlib.sha256(model.read_bytes()).hexdigest() != manifest["files"][self.model]:
            raise ValueError("The bundled YuNet model is missing or damaged. Reinstall the DKST node files.")
        try:
            self.detector = cv2.FaceDetectorYN_create(str(model), "", (320, 320), 0.8, 0.3, 5000,
                                                   cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        except cv2.error as error:
            raise ValueError(f"YuNet could not initialize with OpenCV {cv2.__version__}. "
                             "Update OpenCV in ComfyUI's Python using requirements-face-crop.txt. "
                             f"Details: {error}") from error
        self.lock = threading.Lock()

    def detect(self, image, confidence):
        scale = min(1.0, 1024 / max(image.size))
        size = tuple(max(1, round(n * scale)) for n in image.size)
        preview = image.resize(size, Image.Resampling.BILINEAR) if size != image.size else image
        bgr = np.ascontiguousarray(np.asarray(preview.convert("RGB"))[:, :, ::-1])
        with self.lock:
            self.detector.setInputSize(size)
            self.detector.setScoreThreshold(confidence)
            _, faces = self.detector.detect(bgr)
        if faces is None:
            return []
        sx, sy = image.width / size[0], image.height / size[1]
        result = []
        for face in faces:
            values = [float(v) for v in face]
            if not all(math.isfinite(v) for v in values) or values[-1] < confidence:
                continue
            x, y, w, h = values[:4]
            if w > 0 and h > 0:
                result.append({"bbox": [x * sx, y * sy, w * sx, h * sy], "score": values[-1]})
        return sorted(result, key=lambda face: (face["bbox"][1], face["bbox"][0]))


def write_json(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = Path(file.name)
    try:
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.03)
    finally:
        temporary.unlink(missing_ok=True)


class FaceCropService:
    def __init__(self, output_root, image_roots, state_root, detector_factory=YuNet):
        self.output_root = Path(output_root).resolve()
        self.image_roots = {name: Path(path).resolve() for name, path in image_roots.items()}
        self.state_root = Path(state_root)
        self.detector_factory = detector_factory
        self.detector = None
        self.lock = threading.RLock()
        self.thread = None
        self.current_id = None
        self.cancel_event = threading.Event()
        self.archives = Archives(self)

    def source_path(self, item):
        if not isinstance(item, dict):
            raise ValueError("Invalid gallery item.")
        root = self.image_roots.get(item.get("type", "input"))
        filename, subfolder = item.get("filename"), item.get("subfolder", "")
        if root is None or not isinstance(filename, str) or not filename or not isinstance(subfolder, str):
            raise ValueError("Invalid gallery image path.")
        path = (root / subfolder / filename).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Gallery images must exist inside ComfyUI's input, output or temp directory.")
        return path

    def job_path(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("Click Execute in Face Crop Export first, then Queue/Run to output its crops.")
        path = self.state_root / (job_id + ".json")
        if not path.is_file():
            raise ValueError("Face-crop export was not found on this server.")
        return path

    def status(self, job_id):
        with self.lock:
            state = json.loads(self.job_path(job_id).read_text(encoding="utf-8"))
            if state["status"] in ACTIVE and (self.current_id != job_id or not self.thread or not self.thread.is_alive()):
                state.update(status="interrupted", error="The export worker stopped. Existing crops are retained; start a new export.")
            elif state["status"] in ACTIVE and self.cancel_event.is_set():
                state["status"] = "cancel_requested"
            return state

    def output_dir(self, state):
        path = (self.output_root / state["output_subfolder"]).resolve()
        if not path.is_relative_to(self.output_root):
            raise ValueError("Invalid export folder.")
        return path

    def result_path(self, state, filename):
        root = self.output_dir(state)
        known = {result["filename"] for result in state["results"]}
        if filename not in known:
            raise ValueError("Crop not found in this export.")
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Crop file is missing.")
        return path

    def start(self, data):
        settings = options(data)
        gallery = data.get("image_gallery", "[]")
        try:
            items = json.loads(gallery) if isinstance(gallery, str) else gallery
        except ValueError as error:
            raise ValueError("Invalid photo gallery.") from error
        if not isinstance(items, list) or not 1 <= len(items) <= 500:
            raise ValueError("Add between 1 and 500 photos to the gallery.")
        sources = []
        for item in items:
            self.source_path(item)
            sources.append({key: item.get(key, default) for key, default in
                            [("filename", ""), ("subfolder", ""), ("type", "input")]})
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("Another face-crop export is active. Finish or cancel it first.")
            if self.detector is None:
                self.detector = self.detector_factory()
            job_id = uuid.uuid4().hex
            subfolder = settings["output_folder"] + "/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + job_id
            state = dict(id=job_id, status="queued", options=settings, sources=sources,
                         output_subfolder=subfolder, results=[], skipped=[], processed=0,
                         total=len(sources) * (16 if settings["grid_split"] == "4x4" else 1),
                         current_file="", error="", model=getattr(self.detector, "model", "YuNet"))
            self.output_dir(state).mkdir(parents=True, exist_ok=False)
            self.state_root.mkdir(parents=True, exist_ok=True)
            self.cancel_event.clear()
            self.current_id = job_id
            self.save(state)
            self.thread = threading.Thread(target=self.process, args=(state,), daemon=True, name="DKST-FaceCrop")
            self.thread.start()
            return copy.deepcopy(state)

    def cancel(self, job_id):
        with self.lock:
            state = self.status(job_id)
            if self.current_id != job_id or state["status"] not in ACTIVE:
                raise ValueError("This face-crop export is not running.")
            self.cancel_event.set()
            state["status"] = "cancel_requested"
            # The worker owns persisted state, preventing stale result overwrites.
            return state

    def save(self, state):
        with self.lock:
            write_json(self.state_root / (state["id"] + ".json"), state)
            write_json(self.output_dir(state) / "manifest.json", state)

    def process(self, state):
        settings = state["options"]
        try:
            state["status"] = "processing"
            self.save(state)
            for source_index, item in enumerate(state["sources"], 1):
                if self.cancel_event.is_set():
                    break
                state["current_file"] = item["filename"]
                try:
                    with Image.open(self.source_path(item)) as opened:
                        image = ImageOps.exif_transpose(opened)
                        if "A" in image.getbands() or "transparency" in image.info:
                            rgba = image.convert("RGBA")
                            image = Image.new("RGB", rgba.size, "white")
                            image.paste(rgba, mask=rgba.getchannel("A"))
                        else:
                            image = image.convert("RGB")
                    face_index = 0
                    for cell_index, region in enumerate(regions(*image.size, settings["grid_split"]), 1):
                        if self.cancel_event.is_set():
                            break
                        cell = image.crop(region)
                        faces = self.detector.detect(cell, settings["confidence"])
                        if settings["face_selection"] == "Largest face" and faces:
                            faces = [max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])]
                        if not faces:
                            state["skipped"].append(dict(source_index=source_index, cell=cell_index, reason="No face detected"))
                        for face in faces:
                            if self.cancel_event.is_set():
                                break
                            bounds = crop_bounds(*cell.size, face["bbox"], settings["padding_percent"], settings["crop_shape"] == "Square")
                            crop = resize_crop(cell.crop(bounds), settings)
                            crop.info.clear()
                            face_index += 1
                            filename = f'{settings["filename_prefix"]}_{source_index:05d}_{face_index:02d}.png'
                            crop.save(self.output_dir(state) / filename, format="PNG")
                            x, y, w, h = face["bbox"]
                            state["results"].append(dict(filename=filename, source_index=source_index, cell=cell_index,
                                bbox=[x + region[0], y + region[1], w, h], score=float(face["score"]),
                                crop_box=[bounds[0] + region[0], bounds[1] + region[1], bounds[2] + region[0], bounds[3] + region[1]],
                                width=crop.width, height=crop.height))
                        state["processed"] += 1
                        self.save(state)
                except (OSError, ValueError) as error:
                    state["skipped"].append(dict(source_index=source_index, reason=str(error)))
                    state["processed"] = min(state["total"], source_index * (16 if settings["grid_split"] == "4x4" else 1))
                    self.save(state)
            state["status"] = "cancelled" if self.cancel_event.is_set() else "completed"
        except Exception as error:
            state.update(status="failed", error=str(error))
        finally:
            state["current_file"] = ""
            self.save(state)
