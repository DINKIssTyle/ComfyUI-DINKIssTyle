"""Comfy route/schema checks without importing a GPU trainer or ComfyUI."""
import asyncio
import importlib
import json
from pathlib import Path
import sys
import types
import unittest
import tempfile
from unittest.mock import patch

NODE_DIR = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"
package = types.ModuleType("dkst_qwen_route_tests")
package.__path__ = [str(NODE_DIR)]
folder_paths = types.ModuleType("folder_paths")
registered = {}
class Routes:
    def route(self, method, path):
        def decorator(fn):
            registered[method, path] = fn
            return fn
        return decorator
    def get(self, path): return self.route("get", path)
    def post(self, path): return self.route("post", path)

server = types.ModuleType("server")
server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=Routes(),
    prompt_queue=types.SimpleNamespace(get_current_queue=lambda: ([], []))))
aiohttp = types.ModuleType("aiohttp")
aiohttp.web = types.SimpleNamespace(json_response=lambda data, status=200: {"body": data, "status": status},
                                   FileResponse=lambda path, **kwargs: {"path": path, **kwargs})
with patch.dict(sys.modules, {"dkst_qwen_route_tests": package, "folder_paths": folder_paths,
                              "server": server, "aiohttp": aiohttp}):
    module = importlib.import_module("dkst_qwen_route_tests.dinki_qwen_train")
    config_module = importlib.import_module("dkst_qwen_route_tests.qwen21_trainer.config")


class Request:
    def __init__(self, body, query=None, fragment=None):
        self.data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.content = self
        self.query = query or {}
        self.offset = 0
        self.fragment = fragment
    async def read(self, size):
        size = min(size, self.fragment or size)
        result = self.data[self.offset:self.offset + size]
        self.offset += len(result)
        return result


class TestTrainingRoutes(unittest.TestCase):
    def test_model_dropdown_uses_registered_paths_aliases_and_skips_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("model.safetensors", "vae.safetensors", "encoder.safetensors", "adapter.safetensors"):
                (root / name).touch()
            runtime = root / "qwen21_trainer/.venv/Scripts/python.exe"
            runtime.parent.mkdir(parents=True)
            runtime.touch()
            names = {"diffusion_models": ["sub/model.safetensors", "missing.safetensors"],
                     "unet": ["sub/model.safetensors"], "vae": ["vae.safetensors"],
                     "clip": ["encoder.safetensors"], "loras": ["adapter.safetensors"]}
            def filenames(category):
                if category not in names: raise KeyError(category)
                return names[category]
            with patch.object(module, "__file__", str(root / "dinki_qwen_train.py")), \
                 patch.object(folder_paths, "get_filename_list", side_effect=filenames, create=True), \
                 patch.object(folder_paths, "get_full_path", side_effect=lambda cat, name: str(root / Path(name).name), create=True):
                response = asyncio.run(module.get_models(Request({})))
            choices = response["body"]["choices"]
            self.assertEqual(len(choices["dit"]), 1)
            self.assertEqual(choices["dit"][0]["path"], str((root / "model.safetensors").resolve()))
            self.assertIn("sub/model", choices["dit"][0]["label"])
            self.assertEqual(choices["text_encoder"][0]["path"], str((root / "encoder.safetensors").resolve()))
            self.assertEqual(choices["training_adapter"], choices["turbo_lora"])
            self.assertEqual(choices["python"][0]["path"], str(runtime))

    def test_runtime_validation_preserves_venv_python_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "system-python"
            executable.touch()
            python = root / "venv-python"
            python.symlink_to(executable)
            model = root / "model.safetensors"
            model.touch()
            result = config_module.validate_runtime(dict(python=str(python), dit=str(model), vae=str(model),
                                           text_encoder=str(model), training_adapter=str(model)))
            self.assertEqual(result["python"], str(python))

    def test_gallery_caption_schema_and_queue_reports_without_starting(self):
        with patch.object(module, "caption_aliases", return_value=[module.CAPTION_NONE]):
            inputs = module.DINKI_Qwen21_LoRA_Trainer.INPUT_TYPES()
        schema = inputs["required"]
        self.assertTrue(schema["image_gallery"][1]["captions"])
        self.assertEqual(inputs["optional"]["caption_model"][1]["default"], "")
        self.assertEqual(inputs["optional"]["caption_connection_alias"][1]["default"], module.CAPTION_NONE)
        self.assertTrue(module.DINKI_Qwen21_LoRA_Trainer.VALIDATE_INPUTS(caption_connection_alias="removed"))
        # New optional controls are appended, preserving old positional workflows.
        self.assertEqual(list(schema)[-1], "seed")
        workflow = json.loads((NODE_DIR.parent / "sample_workflows/Qwen21_LoRA_Trainer.json").read_text())
        self.assertEqual(len(workflow["nodes"][0]["widgets_values"]), len(schema) + len(inputs["optional"]))
        service = types.SimpleNamespace(history=lambda project: [{"status": "completed", "lora_path": "/loras/person.safetensors"}])
        with patch.object(module, "manager", return_value=service):
            result = module.DINKI_Qwen21_LoRA_Trainer().report("person")
        self.assertEqual(result["result"][0], "/loras/person.safetensors")
        self.assertTrue(module.DINKI_Qwen21_LoRA_Trainer.OUTPUT_NODE)

    def test_caption_route_runs_in_thread_and_does_not_need_training_runtime(self):
        service = types.SimpleNamespace(active=lambda: None)
        with patch.object(module, "manager", return_value=service), \
             patch.object(module, "generate_caption", return_value={"caption": "a person waving"}) as generate:
            result = asyncio.run(module.caption(Request({"caption_connection_alias": "Home", "caption_model": "vlm"})))
        self.assertEqual(result["status"], 200)
        self.assertEqual(result["body"], {"caption": "a person waving"})
        self.assertIs(generate.call_args.args[0], service)

    def test_caption_route_errors_leave_training_and_gallery_untouched(self):
        service = types.SimpleNamespace(active=lambda: "a" * 32)
        with patch.object(module, "manager", return_value=service), patch.object(module, "generate_caption") as generate:
            result = asyncio.run(module.caption(Request({})))
            self.assertEqual(result["status"], 400)
            generate.assert_not_called()
        service.active = lambda: None
        with patch.object(module, "manager", return_value=service), \
             patch.object(module, "generate_caption", side_effect=ValueError("Select an LLM alias")):
            result = asyncio.run(module.caption(Request({})))
            self.assertEqual(result["status"], 400)

    def test_busy_generation_rejects_before_preparing_or_unloading(self):
        queue = server.PromptServer.instance.prompt_queue
        with patch.object(queue, "get_current_queue", return_value=(["running"], [])), \
             patch.object(module, "manager") as manager:
            with self.assertRaisesRegex(ValueError, "generation queue"):
                module.start_job({})
        manager.assert_not_called()

    def test_start_route_uses_background_thread_and_returns_job(self):
        with patch.object(module, "start_job", return_value={"id": "a" * 32, "status": "starting"}) as start:
            response = asyncio.run(module.change_training(Request({"action": "start", "epochs": 30})))
        self.assertEqual(response["status"], 200)
        self.assertEqual(response["body"]["status"], "starting")
        self.assertEqual(start.call_args.args[0]["epochs"], 30)

    def test_errors_and_large_requests_return_400(self):
        for data in ({"action": "unknown"}, [], b"broken JSON", b"x" * (2 * 1024 * 1024 + 1)):
            with self.subTest(data=type(data)):
                response = asyncio.run(module.change_training(Request(data)))
                self.assertEqual(response["status"], 400)

    def test_fragmented_json_request_waits_for_eof(self):
        data = {"action": "start", "image_gallery": [{"filename": "portrait.png", "caption": "인물 사진"}]}
        for fragment in (1, 43, 65536):
            self.assertEqual(asyncio.run(module.payload(Request(data, fragment=fragment))), data)

    def test_complete_log_is_not_truncated_and_cannot_escape_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "job.json"
            job.touch()
            text = "Traceback: first error\n" + "decoder.weight\n" * 3000
            (root / "train.log").write_text(text)
            service = types.SimpleNamespace(job_file=lambda _: job)
            with patch.object(module, "manager", return_value=service):
                result = asyncio.run(module.full_log(Request({}, {"id": "a" * 32, "download": "1"})))
                self.assertEqual(result["path"].read_text(), text)
                self.assertIn("attachment", result["headers"]["Content-Disposition"])
                self.assertEqual(result["headers"]["Cache-Control"], "no-store")
                for kind in ("../secret", "worker"):
                    result = asyncio.run(module.full_log(Request({}, {"id": "a" * 32, "kind": kind})))
                    self.assertEqual(result["status"], 404)

    def test_sample_path_cannot_escape_output_directory(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            (root / "secret.png").write_bytes(b"private")
            job = root / "job.json"
            job.write_text(json.dumps({"output_dir": str(output)}))
            service = types.SimpleNamespace(job_file=lambda _: job)
            with patch.object(module, "manager", return_value=service):
                result = asyncio.run(module.sample(Request({}, {"id": "a" * 32, "file": "../secret.png"})))
            self.assertEqual(result["status"], 404)


if __name__ == "__main__":
    unittest.main()
