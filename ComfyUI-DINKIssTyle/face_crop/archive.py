"""Disk-backed, atomic ZIP bundles of completed face exports."""
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile
import threading
import zipfile


def leaf_name(value):
    return PureWindowsPath(str(value).replace("/", "\\")).name


def download_manifest(state, settings):
    # Rebuild the public document rather than copy worker errors or server paths.
    safe_reasons = {"No face detected", "Invalid face bounding box.",
                    "Face bounding box is outside the image.",
                    "Image is too small for the selected grid split."}
    return {
        "schema_version": 1, "id": state["id"], "status": state["status"],
        "model": leaf_name(state.get("model", "YuNet")), "options": settings,
        "processed": state["processed"], "total": state["total"],
        "sources": [{"source_index": index, "filename": leaf_name(source["filename"])}
                    for index, source in enumerate(state["sources"], 1)],
        "results": [{key: result[key] for key in ("filename", "source_index", "cell", "bbox", "score", "crop_box", "width", "height")}
                    for result in state["results"]],
        "skipped": [{**{key: item[key] for key in ("source_index", "cell") if key in item},
                     "reason": item["reason"] if item.get("reason") in safe_reasons else "Image could not be processed."}
                    for item in state["skipped"]],
    }


class Archives:
    def __init__(self, service):
        self.service = service
        self.lock = threading.Lock()

    def snapshot(self, job_id):
        from .core import options
        state = self.service.status(job_id)
        if state["status"] != "completed":
            raise ValueError("Finish face extraction before downloading a ZIP.")
        if not state["results"]:
            raise ValueError("This export has no crops to download.")
        settings = options(state["options"])
        folder = Path(settings["output_folder"]).name
        names, files = set(), []
        for result in state["results"]:
            name = result["filename"]
            if not isinstance(name, str) or name != leaf_name(name) or not name.lower().endswith(".png") or name in names:
                raise ValueError("The export has an invalid or duplicate crop filename.")
            path = self.service.result_path(state, name)
            info = path.stat()
            files.append((name, path, info.st_size, info.st_mtime_ns))
            names.add(name)
        manifest = download_manifest(state, settings)
        fingerprint = json.dumps({"manifest": manifest, "files": [(n, size, stamp) for n, _, size, stamp in files]},
                                 ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
        token = hashlib.sha256(fingerprint).hexdigest()
        run = Path(state["output_subfolder"]).name
        date = re.fullmatch(r"(\d{8}-\d{6})_[a-f0-9]{32}", run)
        filename = f'{folder}_{date[1] + "_" if date else ""}{job_id[:8]}.zip'
        return folder, files, manifest, token, filename

    def cache_path(self, job_id, token):
        # IDs have already been validated by service.status; tokens are URL input.
        if not isinstance(token, str) or not re.fullmatch(r"[a-f0-9]{64}", token):
            raise ValueError("Invalid ZIP download token. Click Download ZIP again.")
        temp_root = self.service.image_roots["temp"]
        root = (temp_root / "dkst_face_crop_zips").resolve()
        path = (root / f"{job_id}_{token}.zip").resolve()
        if not root.is_relative_to(temp_root) or not path.is_relative_to(root):
            raise ValueError("Invalid ZIP cache location.")
        return path

    def prepare(self, job_id):
        # Separate from the detector/state lock: ZIP I/O never blocks crop progress.
        with self.lock:
            folder, files, manifest, token, filename = self.snapshot(job_id)
            path = self.cache_path(job_id, token)
            if not path.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix="bundle-", suffix=".tmp", delete=False) as temporary:
                    temporary_path = Path(temporary.name)
                try:
                    # PNGs are already compressed. ZipFile.write copies incrementally;
                    # ZIP64 handles large bundles without loading images into RAM.
                    with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                        for name, crop, _, _ in files:
                            archive.write(crop, arcname=f"{folder}/{name}")
                        archive.writestr(f"{folder}/manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
                    if self.snapshot(job_id)[3] != token:
                        raise ValueError("Crop files changed while preparing the ZIP. Click Download ZIP again.")
                    os.replace(temporary_path, path)
                finally:
                    temporary_path.unlink(missing_ok=True)
            return {"id": job_id, "token": token, "filename": filename, "size": path.stat().st_size}

    def download(self, job_id, token):
        _, _, _, current_token, filename = self.snapshot(job_id)
        path = self.cache_path(job_id, token)
        if token != current_token or not path.is_file():
            raise ValueError("The ZIP is missing or crops have changed. Click Download ZIP again.")
        return path, filename
