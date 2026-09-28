"""Reference audio upload and transcript endpoints without a ComfyUI server."""

import asyncio
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class Routes:
    def __init__(self):
        self.registered = []

    def get(self, route):
        def decorate(function):
            self.registered.append(("GET", route))
            return function
        return decorate

    def post(self, route):
        def decorate(function):
            self.registered.append(("POST", route))
            return function
        return decorate


server = types.ModuleType("server")
server.PromptServer = type("PromptServer", (), {"instance": types.SimpleNamespace(routes=Routes())})
package_path = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle_VoxCPM2" / "__init__.py"
spec = importlib.util.spec_from_file_location(
    "dkst_voxcpm2_api_test", package_path, submodule_search_locations=[str(package_path.parent)]
)
package = importlib.util.module_from_spec(spec)
bootstrap = types.ModuleType(f"{spec.name}.bootstrap")
bootstrap.ensure_dependencies = lambda: None  # Package import must not install packages in a unit test.
with patch.dict(sys.modules, {"server": server, spec.name: package,
                              bootstrap.__name__: bootstrap}):
    spec.loader.exec_module(package)
    api = sys.modules[f"{spec.name}.api"]
    nodes = sys.modules[f"{spec.name}.nodes"]


class UploadPart:
    name = "file"

    def __init__(self, filename, content=b"audio bytes"):
        self.filename = filename
        self.content = content

    async def read_chunk(self, size):
        content, self.content = self.content[:size], self.content[size:]
        return content


class UploadRequest:
    def __init__(self, part):
        self.part = part

    async def multipart(self):
        return self

    async def next(self):
        return self.part


class VoiceApiTests(unittest.TestCase):
    def test_routes_and_repeated_upload_name(self):
        self.assertEqual(len(server.PromptServer.instance.routes.registered), 3)
        self.assertEqual(tuple(package.NODE_DISPLAY_NAME_MAPPINGS.values()), (
            "DKST VoxCPM2 (Downloader)",
            "DKST VoxCPM2 (Reference Audio)",
            "DKST VoxCPM2 (TTS & Cloning)",
        ))
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(api, "VOICE_ROOT", Path(directory)), \
             patch.object(nodes, "VOICE_ROOT", Path(directory)):
            first = asyncio.run(api.upload_voice(UploadRequest(UploadPart("voice.wav"))))
            second = asyncio.run(api.upload_voice(UploadRequest(UploadPart("voice.wav"))))
            self.assertEqual(json.loads(first.text)["name"], "voice.wav")
            self.assertEqual(json.loads(second.text)["name"], "voice_2.wav")
            self.assertEqual((Path(directory) / "voice.wav").read_bytes(), b"audio bytes")
            self.assertEqual(nodes._voice_names(), ["voice.wav", "voice_2.wav"])
            (Path(directory) / "voice.txt").write_text("spoken words", encoding="utf-8")
            request = types.SimpleNamespace(query={"name": "voice.wav"})
            transcript = asyncio.run(api.get_transcript(request))
            self.assertEqual(json.loads(transcript.text)["transcript"], "spoken words")

    def test_rejects_path_traversal_and_oversized_upload(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(api, "VOICE_ROOT", Path(directory)):
            rejected = asyncio.run(api.upload_voice(UploadRequest(UploadPart("../voice.wav"))))
            self.assertEqual(rejected.status, 400)
            rejected_windows = asyncio.run(api.upload_voice(UploadRequest(UploadPart("..\\voice.wav"))))
            self.assertEqual(rejected_windows.status, 400)
            with patch.object(api, "MAX_UPLOAD_BYTES", 2):
                oversized = asyncio.run(api.upload_voice(UploadRequest(UploadPart("voice.wav"))))
            self.assertEqual(oversized.status, 413)
            self.assertFalse((Path(directory) / "voice.wav").exists())


if __name__ == "__main__":
    unittest.main()
