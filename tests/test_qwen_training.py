"""Real tiny-Qwen LoRA gradients/storage plus managed dataset and worker tests."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from safetensors import safe_open
from safetensors.torch import save_file

NODE_DIR = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"
pkg = types.ModuleType("dkst_qwen_tests")
pkg.__path__ = [str(NODE_DIR)]
with patch.dict(sys.modules, {"dkst_qwen_tests": pkg}):
    config = importlib.import_module("dkst_qwen_tests.lora_trainer.config")
    jobs_module = importlib.import_module("dkst_qwen_tests.lora_trainer.jobs")
    runner = importlib.import_module("dkst_qwen_tests.lora_trainer.runner")

sys.path.insert(0, str(NODE_DIR / "lora_trainer/vendor/src"))
from fizgig.families.registry import FAMILIES
from fizgig.families.lora import FamilyLoRA
from fizgig.qwen_image21.model import QwenImage21DiT


class TestQwenWeights(unittest.TestCase):
    def test_actual_qwen_flow_loss_updates_only_lora_and_roundtrips(self):
        torch.manual_seed(3)
        def make_model():
            return QwenImage21DiT(in_channels=4, out_channels=4, num_layers=1,
                attention_head_dim=8, num_attention_heads=2, context_in_dim=16,
                axes_dims_rope=(2, 2, 4)).to(torch.bfloat16).requires_grad_(False)
        model = make_model()
        base = {key: value.detach().clone() for key, value in model.state_dict().items()}
        driver = FAMILIES["qwen_image21"].load_driver()
        model.enable_gradient_checkpointing()
        net = FamilyLoRA(model, driver, device="cpu")
        net.add_trainable(2, 2)
        before = [p.detach().clone() for p in net.parameters()]
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2)
        latent = torch.randn(1, 4, 4, 4)
        cond = {"hidden_states": torch.randn(1, 3, 16)}
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            loss, info = driver.training_loss(model, latent, cond, torch.Generator().manual_seed(7))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))
            optimizer.step()
            losses.append(loss.item())
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, net.parameters())))
        self.assertLess(losses[-1], losses[0])
        for key, value in base.items():
            full, _, leaf = key.rpartition(".")
            module = model.get_submodule(full)
            value_now = getattr(module.base if full in net.wrapped else module, leaf)
            self.assertTrue(torch.equal(value, value_now), key)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "person.safetensors"
            net.save(str(path), metadata={"modelspec.architecture": "qwenimage21"}, dtype=torch.float32)
            runner.validate_lora(path)
            frozen_path = Path(directory) / "frozen.safetensors"
            net.save(str(frozen_path), dtype=torch.float32)
            self.assertGreater(net.add_file(str(frozen_path), "training_adapter"), 0)
            frozen = [p for wrapped in net.wrapped.values() for p in wrapped.adapters["training_adapter"].parameters()]
            self.assertTrue(all(not p.requires_grad for p in frozen))
            net.set_enabled("training_adapter", False)
            net.save(str(path), dtype=torch.float32)
            with safe_open(str(path), framework="pt") as saved:
                self.assertFalse(any("training_adapter" in k for k in saved.keys()))
            restored_model = make_model()
            restored_model.load_state_dict(base)
            restored = FamilyLoRA(restored_model, driver, device="cpu")
            restored.add_trainable(2, 2)
            self.assertGreater(restored.load_trainable(str(path)), 0)
            for key, tensor in net.state_dict(dtype=torch.float32).items():
                self.assertTrue(torch.equal(tensor, restored.state_dict(dtype=torch.float32)[key]))
            first, _ = driver.training_loss(model, latent, cond, torch.Generator().manual_seed(7))
            second, _ = driver.training_loss(restored_model, latent, cond, torch.Generator().manual_seed(7))
            self.assertTrue(torch.allclose(first, second))

    def test_only_qwen_registered(self):
        self.assertEqual(set(FAMILIES), {"qwen_image21"})


class TestProjects(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.input = self.root / "input"
        self.input.mkdir()
        self.service = jobs_module.Jobs(self.root / "projects", self.root / "loras", {"input": self.input})
        settings = {"python": sys.executable, "turbo_lora": ""}
        for key in ("dit", "vae", "text_encoder", "training_adapter"):
            path = self.root / (key + ".safetensors")
            path.write_bytes(b"model fixture")
            settings[key] = str(path)
        self.service.save_settings(settings)
        Image.new("RGB", (96, 64), "red").save(self.input / "portrait.png")
        self.data = dict(project_name="person", trigger_word="person_xyz", sample_every=0,
                         image_gallery=[dict(filename="portrait.png", type="input", caption="a person in a red coat")])

    def prepare(self):
        path = self.service.prepare(self.data)
        return path, json.loads(path.read_text())

    def test_dataset_preserves_aspect_and_captions_and_argv(self):
        path, job = self.prepare()
        files = list(Path(job["image_dir"]).glob("*.png"))
        with Image.open(files[0]) as photo:
            self.assertEqual(photo.size, (96, 64))
        self.assertEqual(files[0].with_suffix(".txt").read_text(), "person_xyz, a person in a red coat")
        self.assertIn("enable_bucket = true", Path(job["dataset_config"]).read_text())
        commands = config.commands(job, runner.VENDOR)
        self.assertIn("--training_adapter", commands["training"])
        self.assertEqual(commands["training"][commands["training"].index("--network_dim")+1], "8")
        self.assertNotIn("--finetune", commands["training"])
        self.assertNotIn("--sample_prompts", commands["training"])
        self.assertEqual(commands["latents"][commands["latents"].index("--family")+1], "qwen_image21")
        self.assertTrue(all(isinstance(arg, str) for argv in commands.values() for arg in argv))

    def test_caption_changes_reuse_latents_but_image_resolution_vae_changes_do_not(self):
        _, first = self.prepare()
        self.data["image_gallery"][0]["caption"] = "side profile"
        _, second = self.prepare()
        self.assertEqual(first["cache_dir"], second["cache_dir"])
        # Earlier job captions remain immutable.
        self.assertIn("red coat", next(Path(first["image_dir"]).glob("*.txt")).read_text())
        self.data["resolution"] = 1024
        _, changed = self.prepare()
        self.assertNotEqual(first["cache_dir"], changed["cache_dir"])
        self.data.pop("resolution")
        Image.new("RGB", (96, 64), "blue").save(self.input / "portrait.png")
        _, changed = self.prepare()
        self.assertNotEqual(first["cache_dir"], changed["cache_dir"])
        Path(first["runtime"]["vae"]).write_bytes(b"different VAE")
        _, last = self.prepare()
        self.assertNotEqual(changed["cache_dir"], last["cache_dir"])

    def test_caption_and_encoder_cache_invalidation_preserves_latent_files(self):
        _, job = self.prepare()
        stem = next(Path(job["image_dir"]).glob("*.png")).stem
        cache = Path(job["cache_dir"])
        latent = cache / "latent.safetensors"
        latent.write_bytes(b"leave alone")
        text = cache / (stem + "_qwenimage21_te.safetensors")
        caption = (Path(job["image_dir"]) / (stem + ".txt")).read_text()
        save_file({"cond__hidden_states": torch.zeros(1, 4)}, str(text), metadata={"caption1": caption})
        signature = config.model_signature(job["runtime"]["text_encoder"])
        runner.write_json(cache / "text_encoder.json", signature)
        runner.invalidate_text_cache(job)
        self.assertTrue(text.exists())
        (Path(job["image_dir"]) / (stem + ".txt")).write_text("changed")
        runner.invalidate_text_cache(job)
        self.assertFalse(text.exists())
        save_file({"cond__hidden_states": torch.zeros(1, 4)}, str(text), metadata={"caption1": "changed"})
        Path(job["runtime"]["text_encoder"]).write_bytes(b"changed encoder")
        runner.invalidate_text_cache(job)
        self.assertFalse(text.exists())
        self.assertEqual(latent.read_bytes(), b"leave alone")

    def test_paths_and_invalid_settings_rejected(self):
        for name in ("../secret", "CON", "a/b", "", "a" * 65):
            with self.subTest(name=name), self.assertRaises(ValueError):
                config.safe_name(name)
        with self.assertRaises(ValueError):
            self.service.resolve_image({"filename": "../dit.safetensors"})
        with self.assertRaises(ValueError):
            self.service.resolve_image({"filename": str(self.root / "dit.safetensors")})
        for bad in ({"epochs": True}, {"epochs": -1}, {"preset": "H3"}, {"precision": "oops"}):
            with self.subTest(data=bad), self.assertRaises(ValueError):
                config.training_options(bad)

    def test_single_job_and_pause_cancel_controls(self):
        process = types.SimpleNamespace(poll=lambda: None)
        with patch.object(jobs_module.subprocess, "Popen", return_value=process) as popen:
            state = self.service.start(self.data)
            self.assertNotIn("shell", popen.call_args.kwargs)
            self.assertEqual(popen.call_args.args[0][0], self.service.settings()["python"])
            with self.assertRaisesRegex(ValueError, "already active"):
                self.service.start(self.data)
            self.assertEqual(self.service.control(state["id"], "pause")["status"], "pause_requested")
            self.assertEqual(self.service.control(state["id"], "cancel")["status"], "cancel_requested")
        self.assertEqual(self.service.history("person")[0]["id"], state["id"])
        self.assertEqual(self.service.history("different"), [])

    def test_launch_failure_releases_reservation_and_reports_error(self):
        with patch.object(jobs_module.subprocess, "Popen", side_effect=OSError("not executable")):
            with self.assertRaises(ValueError):
                self.service.start(self.data)
        self.assertFalse(self.service.preparing)
        self.assertIsNone(self.service.active())
        self.assertEqual(self.service.history()[0]["status"], "failed")

    def test_complete_state_chosen_for_resume_and_model_change_rejected(self):
        path, job = self.prepare()
        runner.write_json(path.parent / "status.json", {"id": job["id"], "status": "paused"})
        output = Path(job["output_dir"])
        complete = output / "person-000002-state"
        complete.mkdir()
        for name in ("training_state.json", "lora.safetensors", "optimizer.pt", "rng.pt"):
            (complete / name).write_text("fixture")
        incomplete = output / "person-000003-state"
        incomplete.mkdir()
        (incomplete / "training_state.json").write_text("fixture")
        with patch.object(self.service, "launch", return_value={}):
            self.service.resume(job["id"])
        self.assertEqual(json.loads(path.read_text())["resume_state"], str(complete))
        Path(job["runtime"]["dit"]).write_bytes(b"model changed")
        with self.assertRaisesRegex(ValueError, "Model files changed"):
            self.service.resume(job["id"])

    def test_dead_known_worker_not_considered_active(self):
        path, job = self.prepare()
        runner.write_json(path.parent / "status.json", {"id": job["id"], "status": "training", "heartbeat": time.time()})
        self.service.processes[job["id"]] = types.SimpleNamespace(poll=lambda: 9)
        self.assertIsNone(self.service.active())
        self.assertEqual(self.service.status(job["id"])["status"], "interrupted")

    def test_real_subprocess_failure_is_failed_not_completed(self):
        path, job = self.prepare()
        scripts = {stage: [sys.executable, "-c", f"print('{stage}'); raise SystemExit({9 if stage == 'text' else 0})"]
                   for stage in ("latents", "text", "training")}
        with patch.object(runner, "check_environment", return_value={"ok": True}), \
             patch.object(runner, "commands", return_value=scripts), \
             patch.object(runner, "invalidate_text_cache", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "code 9"):
                runner.run_job(path)
        state = self.service.status(job["id"])
        self.assertEqual(state["status"], "failed")
        self.assertIn("text process exited", state["error"])
        self.assertNotIn("lora_path", state)

    def test_pause_after_cache_and_resume_before_training_state(self):
        path, job = self.prepare()
        pause = Path(job["output_dir"]) / ".pause_requested"
        scripts = {stage: [sys.executable, "-c", f"from pathlib import Path; Path({str(pause)!r}).touch()"]
                   for stage in ("latents", "text", "training")}
        with patch.object(runner, "check_environment", return_value={"ok": True}), \
             patch.object(runner, "commands", return_value=scripts):
            runner.run_job(path)
        self.assertEqual(self.service.status(job["id"])["status"], "paused")
        with patch.object(self.service, "launch", return_value={}):
            self.service.resume(job["id"])
        self.assertFalse(pause.exists())
        self.assertNotIn("resume_state", json.loads(path.read_text()))

    def test_completed_only_after_valid_final_file_and_progress_is_reported(self):
        path, job = self.prepare()
        fixture = self.root / "valid_lora.safetensors"
        save_file({"transformer.test.lora_A.weight": torch.ones(2, 4),
                   "transformer.test.lora_B.weight": torch.ones(4, 2)}, str(fixture))
        final = Path(job["output_dir"]) / "person.safetensors"
        scripts = {stage: [sys.executable, "-c", "print('cache done')"] for stage in ("latents", "text")}
        scripts["training"] = [sys.executable, "-c",
            f"import shutil; shutil.copyfile({str(fixture)!r}, {str(final)!r}); print('epoch 30/30  avr_loss=0.123  step=240')"]
        with patch.object(runner, "check_environment", return_value={"ok": True}), \
             patch.object(runner, "commands", return_value=scripts), \
             patch.object(runner, "invalidate_text_cache", return_value={}):
            runner.run_job(path)
        state = self.service.status(job["id"])
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["lora_path"], str(final))
        self.assertEqual(state["epoch"], 30)
        self.assertEqual(state["step"], 240)
        self.assertIn("person.safetensors", state["checkpoints"])

    def test_invalid_final_artifact_is_not_a_completed_training_job(self):
        path, job = self.prepare()
        final = Path(job["output_dir"]) / "person.safetensors"
        scripts = {stage: [sys.executable, "-c", "print('cache done')"] for stage in ("latents", "text")}
        scripts["training"] = [sys.executable, "-c", f"from pathlib import Path; Path({str(final)!r}).touch()"]
        with patch.object(runner, "check_environment", return_value={"ok": True}), \
             patch.object(runner, "commands", return_value=scripts), \
             patch.object(runner, "invalidate_text_cache", return_value={}):
            with self.assertRaises(Exception):
                runner.run_job(path)
        self.assertEqual(self.service.status(job["id"])["status"], "failed")

    def test_cancel_terminates_running_stage(self):
        path, job = self.prepare()
        started = path.parent / "stage_started"
        script = [sys.executable, "-u", "-c",
                  f"from pathlib import Path; import time; Path({str(started)!r}).touch(); time.sleep(60)"]
        scripts = {stage: script for stage in ("latents", "text", "training")}
        errors = []
        def work():
            try:
                runner.run_job(path)
            except BaseException as error:
                errors.append(error)
        with patch.object(runner, "check_environment", return_value={"ok": True}), \
             patch.object(runner, "commands", return_value=scripts):
            thread = threading.Thread(target=work)
            thread.start()
            deadline = time.monotonic() + 10
            while not started.is_file() and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.02)
            (path.parent / ".cancel_requested").touch()
            thread.join(timeout=15)
            self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertEqual(self.service.status(job["id"])["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
