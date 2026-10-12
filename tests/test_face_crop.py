"""Real crop/export geometry and YuNet inference; ComfyUI route/schema isolation."""
import asyncio
import importlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from PIL import Image

NODE_DIR = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"
sys.path.insert(0, str(NODE_DIR))
from face_crop.core import FaceCropService, YuNet, crop_bounds, options, regions, resize_crop


class Detector:
    model = "test"
    def detect(self, image, confidence):
        if image.getpixel((0, 0)) == (0, 0, 0):
            return []
        return [{"bbox": [10, 20, 30, 40], "score": .95}, {"bbox": [50, 5, 10, 15], "score": .9}]


class TestCrops(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in ("input", "output", "temp"):
            (self.root / name).mkdir()
        self.service = FaceCropService(self.root / "output", {n: self.root / n for n in ("input", "output", "temp")},
                                       self.root / "state", Detector)
        self.addCleanup(lambda: self.service.thread and self.service.thread.join(5))
        Image.new("RGB", (100, 100), "white").save(self.root / "input/photo.png")

    def export(self, **data):
        state = self.service.start({"image_gallery": [{"filename": "photo.png"}], **data})
        self.service.thread.join(5)
        self.assertFalse(self.service.thread.is_alive())
        return self.service.status(state["id"])

    def test_padding_bounds_square_clipped_and_no_upscale(self):
        self.assertEqual(crop_bounds(1000, 1000, [100, 100, 100, 200], 30), (70, 40, 230, 360))
        self.assertEqual(crop_bounds(100, 100, [0, 0, 30, 40], 30), (0, 0, 48, 64))
        bounds = crop_bounds(101, 83, [50.1, 45.5, 20.1, 30.3], 30, True)
        self.assertEqual(bounds[2] - bounds[0], bounds[3] - bounds[1])
        bounds = crop_bounds(20, 100, [1, 10, 18, 60], 200, True)
        self.assertEqual(bounds, (0, 30, 20, 50))
        image = Image.new("RGB", (2000, 1000))
        self.assertEqual(resize_crop(image, options({})).size, (1024, 512))
        self.assertIs(resize_crop(image, options({"limit_size": False})), image)
        small = Image.new("RGB", (20, 40))
        self.assertIs(resize_crop(small, options({})), small)

    def test_export_keeps_variable_dimensions_unique_runs_and_manifest(self):
        first = self.export()
        second = self.export(face_selection="Largest face", filename_prefix="person")
        self.assertEqual(first["status"], "completed")
        self.assertEqual(len(first["results"]), 2)
        self.assertEqual(len(second["results"]), 1)
        self.assertNotEqual(first["output_subfolder"], second["output_subfolder"])
        self.assertEqual(second["results"][0]["filename"], "person_00001_01.png")
        self.assertNotEqual((first["results"][0]["width"], first["results"][0]["height"]),
                            (first["results"][1]["width"], first["results"][1]["height"]))
        for state in (first, second):
            manifest = json.loads((self.service.output_dir(state) / "manifest.json").read_text())
            self.assertEqual(manifest, state)
        with Image.open(self.root / "input/photo.png") as original:
            self.assertEqual(original.size, (100, 100))

    def test_grid_cells_do_not_include_neighbor_and_missing_faces_are_skipped(self):
        Image.new("RGB", (401, 403), "white").save(self.root / "input/photo.png")
        state = self.export(grid_split="4x4", crop_shape="Square", padding_percent=200)
        self.assertEqual(len(state["results"]), 32)
        self.assertEqual(state["processed"], 16)
        for result in state["results"]:
            region = regions(401, 403, "4x4")[result["cell"] - 1]
            x0, y0, x1, y1 = result["crop_box"]
            self.assertTrue(region[0] <= x0 < x1 <= region[2])
            self.assertTrue(region[1] <= y0 < y1 <= region[3])
            self.assertEqual(result["width"], result["height"])
        Image.new("RGB", (100, 100), "black").save(self.root / "input/photo.png")
        state = self.export()
        self.assertEqual(state["results"], [])
        self.assertEqual(state["skipped"][0]["reason"], "No face detected")

    def test_metadata_removed_and_exif_orientation_applied(self):
        image = Image.new("RGB", (150, 100), "white")
        exif = Image.Exif(); exif[274] = 6; exif[270] = "private caption"
        image.save(self.root / "input/photo.png", exif=exif)
        state = self.export(padding_percent=200)
        with Image.open(self.service.result_path(state, state["results"][0]["filename"])) as crop:
            self.assertEqual(crop.info, {})
        # Rotation produces a 100 x 150 source, so padding can extend past y=100.
        self.assertEqual(state["results"][0]["crop_box"][3], 150)

    def test_bad_paths_options_and_corrupt_photos(self):
        for folder in ("../escape", "/tmp/export", "C:\\export", "a//b", "CON", "a/.."):
            with self.subTest(folder=folder), self.assertRaises(ValueError):
                options({"output_folder": folder})
        for data in ({"max_size": True}, {"padding_percent": float("nan")}, {"confidence": 2}):
            with self.assertRaises(ValueError): options(data)
        for item in ({"filename": "../input/photo.png", "type": "temp"}, {"filename": "../../secret"}, {"filename": "photo.png", "type": "else"}):
            with self.assertRaises(ValueError): self.service.source_path(item)
        (self.root / "temp/escape.png").symlink_to(self.root / "input/photo.png")
        with self.assertRaises(ValueError): self.service.source_path({"filename": "escape.png", "type": "temp"})
        (self.root / "input/photo.png").write_bytes(b"not an image")
        state = self.export()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(state["skipped"]), 1)
        with self.assertRaises(ValueError): self.service.result_path(state, "../../secret")

    def test_cancel_exclusive_and_restart_retains_results(self):
        entered, release = threading.Event(), threading.Event()
        class Blocking(Detector):
            def detect(self, image, confidence):
                entered.set(); release.wait(3); return super().detect(image, confidence)
        self.service.detector_factory = Blocking
        state = self.service.start({"image_gallery": [{"filename": "photo.png"}]})
        self.assertTrue(entered.wait(3))
        with self.assertRaises(ValueError): self.service.start({"image_gallery": [{"filename": "photo.png"}]})
        self.assertEqual(self.service.cancel(state["id"])["status"], "cancel_requested")
        self.assertEqual(self.service.status(state["id"])["status"], "cancel_requested")
        release.set(); self.service.thread.join(5)
        self.assertEqual(self.service.status(state["id"])["status"], "cancelled")
        self.service.detector = Detector()
        done = self.export()
        restarted = FaceCropService(self.root / "output", self.service.image_roots, self.root / "state", Detector)
        self.assertEqual(restarted.status(done["id"])["results"], done["results"])
        saved = self.root / "state" / (done["id"] + ".json")
        done["status"] = "processing"; saved.write_text(json.dumps(done))
        self.assertEqual(restarted.status(done["id"])["status"], "interrupted")

    def test_zip_contains_only_crops_and_portable_manifest_and_reuses_cache(self):
        state = self.export(output_folder="DKST_FaceCrops/인물")
        state["sources"][0].update(filename="C:\\PrivateServer\\photos\\photo.png", subfolder="/private/server/input")
        state["skipped"] = [{"source_index": 2, "reason": "Cannot open /private/server/input/bad.png"}]
        self.service.save(state)
        (self.service.output_dir(state) / "unrelated.txt").write_text("do not include")
        bundle = self.service.archives.prepare(state["id"])
        path, filename = self.service.archives.download(state["id"], bundle["token"])
        self.assertTrue(filename.startswith("인물_"))
        self.assertEqual(path.stat().st_size, bundle["size"])
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(set(archive.namelist()), {"인물/manifest.json", *["인물/" + r["filename"] for r in state["results"]]})
            self.assertIsNone(archive.testzip())
            manifest = json.loads(archive.read("인물/manifest.json"))
            self.assertEqual(manifest["sources"], [{"source_index": 1, "filename": "photo.png"}])
            self.assertNotIn("PrivateServer", json.dumps(manifest))
            self.assertNotIn("/private/server", json.dumps(manifest))
            self.assertNotIn("output_dir", manifest)
            self.assertTrue(all(i.compress_type == zipfile.ZIP_STORED for i in archive.infolist()))
            for result in state["results"]:
                self.assertEqual(archive.read("인물/" + result["filename"]), self.service.result_path(state, result["filename"]).read_bytes())
        stamp = path.stat().st_mtime_ns
        with patch("face_crop.archive.zipfile.ZipFile", side_effect=AssertionError("cached ZIP was rebuilt")):
            self.assertEqual(self.service.archives.prepare(state["id"]), bundle)
        self.assertEqual(path.stat().st_mtime_ns, stamp)

    def test_zip_changes_invalidate_cache_and_missing_crops_are_rejected(self):
        state = self.export()
        first = self.service.archives.prepare(state["id"])
        crop = self.service.result_path(state, state["results"][0]["filename"])
        stamp = crop.stat().st_mtime_ns + 1_000_000_000
        os.utime(crop, ns=(stamp, stamp))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.service.archives.download(state["id"], first["token"])
        second = self.service.archives.prepare(state["id"])
        self.assertNotEqual(second["token"], first["token"])
        crop.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.service.archives.prepare(state["id"])

    def test_zip_rejects_unfinished_empty_traversal_and_duplicate_files(self):
        state = self.export()
        for status in ("queued", "processing", "cancelled", "interrupted", "failed"):
            state["status"] = status; self.service.save(state)
            with self.assertRaises(ValueError): self.service.archives.prepare(state["id"])
        state["status"] = "completed"
        original = state["results"][:]
        for results in ([], [original[0], original[0]], [{**original[0], "filename": "../photo.png"}],
                        [{**original[0], "filename": "C:\\photo.png"}]):
            state["results"] = results; self.service.save(state)
            with self.assertRaises(ValueError): self.service.archives.prepare(state["id"])
        state["results"] = original; self.service.save(state)
        with self.assertRaises(ValueError): self.service.archives.download(state["id"], "../secret")

    def test_zip_failure_leaves_no_partial_file_and_parallel_requests_share_bundle(self):
        state = self.export()
        with patch("face_crop.archive.zipfile.ZipFile.write", side_effect=OSError("Disk full")):
            with self.assertRaisesRegex(OSError, "Disk full"): self.service.archives.prepare(state["id"])
        cache = self.root / "temp/dkst_face_crop_zips"
        self.assertEqual(list(cache.iterdir()), [])
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(self.service.archives.prepare, [state["id"], state["id"]]))
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(list(cache.glob("*.zip"))), 1)
        self.assertEqual(list(cache.glob("*.tmp")), [])


class TestRealYuNet(unittest.TestCase):
    def test_bundled_model_detects_each_cell(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("Install OpenCV to test actual bundled YuNet inference")
        detector = YuNet()
        grid = NODE_DIR.parent / "output/imagegen/identity-portraits-4x4.png"
        if not grid.is_file(): self.skipTest("Local 4x4 photo fixture unavailable")
        with Image.open(grid) as image:
            counts = [len(detector.detect(image.crop(cell), .8)) for cell in regions(*image.size, "4x4")]
            self.assertEqual(counts, [1] * 16)
            self.assertEqual(detector.detect(Image.new("RGB", (320, 320), "white"), .8), [])


class TestComfyNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = types.ModuleType("dkst_face_test"); package.__path__ = [str(NODE_DIR)]
        cls.routes = {}
        class Routes:
            def register(self, method, path):
                def decorator(fn): cls.routes[method, path] = fn; return fn
                return decorator
            def get(self, path): return self.register("get", path)
            def post(self, path): return self.register("post", path)
        server = types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=Routes())))
        web = types.SimpleNamespace(json_response=lambda data, status=200, **kw: {"body": data, "status": status, **kw}, FileResponse=lambda path, **kw: {"path": path, **kw})
        with patch.dict(sys.modules, {"dkst_face_test": package, "folder_paths": types.ModuleType("folder_paths"),
                                      "server": server, "aiohttp": types.SimpleNamespace(web=web)}):
            cls.module = importlib.import_module("dkst_face_test.dinki_face_crop")

    def test_fragmented_json_and_oversize_request(self):
        class Request:
            query = {}
            def __init__(self, data): self.data = data; self.content = self
            async def iter_chunked(self, size):
                for i in range(0, len(self.data), 7): yield self.data[i:i + 7]
        fake = types.SimpleNamespace(start=lambda data: {"id": "a" * 32, "payload": data}, output_dir=lambda state: Path("/output/faces"))
        with patch.object(self.module, "service", lambda: fake):
            result = asyncio.run(self.module.mutate_job(Request(b'{"action":"start","image_gallery":"[]"}')))
            self.assertEqual(result["status"], 200)
            self.assertEqual(result["body"]["payload"]["image_gallery"], "[]")
            result = asyncio.run(self.module.mutate_job(Request(b' ' * (2 * 1024 * 1024 + 1))))
            self.assertEqual(result["status"], 400)
            self.assertIn("too large", result["body"]["error"])

    def test_outputs_keep_sizes_and_loading_does_not_export_again(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, size in (("one.png", (64, 100)), ("two.png", (100, 64))):
                Image.new("RGB", size, "white").save(root / filename)
            state = dict(status="completed", processed=1, total=1, skipped=[], results=[{"filename": n} for n in ("one.png", "two.png")])
            fake = types.SimpleNamespace(status=lambda id: state, result_path=lambda s, n: root / n, output_dir=lambda s: root)
            # Tensor construction is exercised against numpy here; no GPU or Torch import needed.
            class Tensor:
                def __init__(self, array): self.array = array
                def unsqueeze(self, axis): return np.expand_dims(self.array, axis)
            with patch.object(self.module, "service", lambda: fake), patch.dict(sys.modules, {"torch": types.SimpleNamespace(from_numpy=Tensor)}):
                images, folder, info = self.module.DINKI_Face_Crop_Export().report("a" * 32)
                self.assertEqual([a.shape for a in images], [(1, 100, 64, 3), (1, 64, 100, 3)])
                self.assertEqual(float(images[0].max()), 1)
                self.assertEqual(folder, str(root))
                state["status"] = "processing"
                with self.assertRaises(ValueError): self.module.DINKI_Face_Crop_Export().report("a" * 32)
            self.assertEqual(self.module.DINKI_Face_Crop_Export.OUTPUT_IS_LIST, (True, False, False))

    def test_schema_starter_and_crop_access(self):
        schema = self.module.DINKI_Face_Crop_Export.INPUT_TYPES()
        starter = json.loads((NODE_DIR.parent / "sample_workflows/Face_Crop_Export.json").read_text())
        self.assertEqual(len(starter["nodes"][0]["widgets_values"]), len(schema["required"]) + len(schema["optional"]))
        self.assertEqual(schema["required"]["image_gallery"][0], "DKST_IMAGE_REF_GRID")
        self.assertIn(("get", "/dinki/face_crop/manifest"), self.routes)
        def reject(*args): raise ValueError("Crop not found in this export.")
        fake = types.SimpleNamespace(status=lambda id: {}, result_path=reject)
        request = types.SimpleNamespace(query={"id": "a" * 32, "file": "../other.png"})
        with patch.object(self.module, "service", lambda: fake):
            result = asyncio.run(self.module.get_crop(request))
        self.assertEqual(result["status"], 404)

    def test_zip_routes_fragmented_request_file_response_and_errors(self):
        bundle = {"id": "a" * 32, "token": "b" * 64, "filename": "인물_20261012.zip", "size": 1000}
        class Request:
            query = {"id": bundle["id"], "token": bundle["token"]}
            def __init__(self, data): self.data = data; self.content = self
            async def iter_chunked(self, size):
                for i in range(0, len(self.data), 3): yield self.data[i:i + 3]
        archives = types.SimpleNamespace(prepare=lambda id: bundle, download=lambda id, token: (Path("/temp/test.zip"), bundle["filename"]))
        with patch.object(self.module, "service", lambda: types.SimpleNamespace(archives=archives)):
            response = asyncio.run(self.module.prepare_zip(Request(json.dumps({"id": bundle["id"]}).encode())))
            self.assertEqual(response["status"], 200)
            self.assertIn(bundle["token"], response["body"]["url"])
            response = asyncio.run(self.module.download_zip(Request(b"")))
            self.assertEqual(response["path"], Path("/temp/test.zip"))
            self.assertEqual(response["headers"]["Content-Type"], "application/zip")
            self.assertIn("filename*=UTF-8''%EC%9D%B8%EB%AC%BC", response["headers"]["Content-Disposition"])
            for payload in (b"null", b"[1]", b"{bad", b" " * 4097):
                self.assertEqual(asyncio.run(self.module.prepare_zip(Request(payload)))["status"], 400)
            def reject(*args): raise ValueError("Finish face extraction")
            archives.prepare = archives.download = reject
            self.assertEqual(asyncio.run(self.module.prepare_zip(Request(b'{"id":"bad"}')))["status"], 400)
            self.assertEqual(asyncio.run(self.module.download_zip(Request(b"")))["status"], 404)

if __name__ == "__main__": unittest.main()
