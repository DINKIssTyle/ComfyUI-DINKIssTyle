"""Installation and partial-loading regressions without pip/network/GPU access."""
import ast
import importlib
import importlib.util
from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
NODE_DIR = ROOT / "ComfyUI-DINKIssTyle"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = load_file("dkst_bootstrap_tests", NODE_DIR / "bootstrap.py")
loader = load_file("dkst_loader_tests", NODE_DIR / "node_loader.py")
setup = load_file("dkst_setup_tests", NODE_DIR / "runtime_setup.py")
dependencies = load_file("dkst_worker_dependencies_tests", NODE_DIR / "qwen21_trainer/dependencies.py")


class TestHostDependencies(unittest.TestCase):
    def test_ready_environment_does_not_run_pip(self):
        with patch.object(bootstrap, "dependency_error", return_value=None), \
             patch.object(bootstrap.subprocess, "run") as run:
            self.assertEqual(bootstrap.ensure_dependencies(), {})
        run.assert_not_called()

    def test_missing_package_uses_comfy_python_then_rechecks(self):
        with patch.object(bootstrap, "DEPENDENCIES", (("imageio", "imageio"),)), \
             patch.object(bootstrap, "dependency_error", side_effect=["missing", None]), \
             patch.object(bootstrap.subprocess, "run") as run, self.assertLogs(bootstrap.LOGGER):
            self.assertEqual(bootstrap.ensure_dependencies(), {})
        self.assertEqual(run.call_args.args[0][:4], [sys.executable, "-m", "pip", "install"])
        self.assertEqual(run.call_args.args[0][-1], "imageio")

    def test_install_failure_does_not_abort_remaining_dependencies(self):
        with patch.object(bootstrap, "DEPENDENCIES", (("imageio", "imageio"), ("av", "av"))), \
             patch.object(bootstrap, "dependency_error", side_effect=["missing", "missing", None]), \
             patch.object(bootstrap.subprocess, "run", side_effect=[subprocess.CalledProcessError(1, "pip"), None]) as run, \
             self.assertLogs(bootstrap.LOGGER):
            self.assertEqual(set(bootstrap.ensure_dependencies()), {"imageio"})
        self.assertEqual(run.call_count, 2)

    def test_offline_setting_never_installs(self):
        with patch.object(bootstrap, "DEPENDENCIES", (("imageio", "imageio"),)), \
             patch.object(bootstrap, "dependency_error", return_value="missing"), \
             patch.dict(os.environ, {"DKST_AUTO_INSTALL": "0"}), \
             patch.object(bootstrap.subprocess, "run") as run, self.assertLogs(bootstrap.LOGGER):
            self.assertIn("imageio", bootstrap.ensure_dependencies())
        run.assert_not_called()

    def test_opencv_checks_yunet_and_reuses_installed_contrib_variant(self):
        with patch.object(bootstrap.importlib, "import_module", return_value=types.SimpleNamespace(__version__="4.10.0")):
            self.assertIn("YuNet", bootstrap.dependency_error("cv2"))
        with patch.object(bootstrap.importlib, "import_module", return_value=types.SimpleNamespace(
                __version__="4.10.0", FaceDetectorYN_create=object())):
            self.assertIsNone(bootstrap.dependency_error("cv2"))
        def version(name):
            if name == "opencv-contrib-python":
                return "4.7.0"
            raise metadata.PackageNotFoundError(name)
        with patch.object(bootstrap.metadata, "version", side_effect=version):
            self.assertEqual(bootstrap.opencv_requirement(), "opencv-contrib-python>=4.8.0")

    def test_module_failure_keeps_other_nodes_and_side_effect_modules(self):
        good = type("Good", (), {})
        def imported(name, package):
            if name == ".bad":
                raise ModuleNotFoundError("missing imageio")
            return types.SimpleNamespace(Good=good)
        with patch.object(loader.importlib, "import_module", side_effect=imported), self.assertLogs(loader.LOGGER):
            nodes, modules, failures = loader.load_nodes("pack", (("bad", ("Bad",)), ("good", ("Good",)), ("routes", ())))
        self.assertEqual(nodes, {"Good": good})
        self.assertEqual(set(modules), {"good", "routes"})
        self.assertEqual(set(failures), {"bad"})

    def test_repository_entrypoint_registers_surviving_nodes_and_correct_web_path(self):
        tree = ast.parse((NODE_DIR / "__init__.py").read_text())
        modules = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                       and n.targets[0].id == "NODE_MODULES"))
        name = "dkst_repository_install_tests"
        package = name + ".ComfyUI-DINKIssTyle"
        stubs = {}
        for module, classes in modules:
            stub = types.ModuleType(package + "." + module)
            for cls in classes:
                setattr(stub, cls, type(cls, (), {}))
            if module == "dinki_execution_report":
                stub.install_execution_report_hooks = Mock()
            stubs[stub.__name__] = stub
        boot = types.ModuleType(package + ".bootstrap")
        boot.ensure_dependencies = lambda: {"imageio": "offline"}
        stubs[boot.__name__] = boot
        spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        stubs[name] = module
        real_import = importlib.import_module
        def imported(path, package=None):
            if path == ".dinki_comparer":
                raise ModuleNotFoundError("missing imageio")
            return real_import(path, package)
        with patch.dict(sys.modules, stubs), patch.object(importlib, "import_module", side_effect=imported), \
             self.assertLogs():
            spec.loader.exec_module(module)
        expected = {cls for _, classes in modules for cls in classes} - {"DINKI_Image_Comparer_MOV"}
        self.assertEqual(set(module.NODE_CLASS_MAPPINGS), expected)
        self.assertEqual(set(module.NODE_DISPLAY_NAME_MAPPINGS), expected)
        self.assertEqual(module.WEB_DIRECTORY, "./ComfyUI-DINKIssTyle/js")


class TestTrainingDependencies(unittest.TestCase):
    def test_both_trainers_validate_the_same_pins_and_report_all_missing_packages(self):
        qwen = list(dependencies.requirement_lines(NODE_DIR / "qwen21_trainer/requirements.txt"))
        h3 = list(dependencies.requirement_lines(NODE_DIR / "minimax_trainer/requirements.txt"))
        self.assertEqual(qwen, h3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text("torch==2.10.0\naccelerate==1.6.0\n")
            def version(name):
                if name == "torch":
                    return "2.9.0"
                raise metadata.PackageNotFoundError(name)
            with patch.object(dependencies.metadata, "version", side_effect=version):
                with self.assertRaisesRegex(ValueError, "torch.*2.9.0.*accelerate.*missing"):
                    dependencies.check_requirements(path)
            with patch.object(dependencies.metadata, "version", side_effect=lambda n: {"torch": "2.10.0+cu128", "accelerate": "1.6.0"}[n]):
                dependencies.check_requirements(path)

    def test_setup_is_single_flight_and_h3_reuses_qwen_environment(self):
        service = setup.RuntimeSetup()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "qwen21_trainer/.venv/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.object(setup.sys, "platform", "linux"), patch.object(setup.threading, "Thread") as thread:
                state = service.start("minimax_h3", root / "minimax_trainer/runner.py")
                self.assertEqual(state["python"], str(python))
                self.assertEqual(state["status"], "installing")
                service.start("minimax_h3", root / "minimax_trainer/runner.py")
                self.assertEqual(thread.call_count, 1)
                with self.assertRaisesRegex(ValueError, "Another"):
                    service.start("qwen21", root / "qwen21_trainer/runner.py")

    def test_background_setup_reports_log_success_and_failure(self):
        for code, expected in ((0, "ready"), (1, "failed")):
            service = setup.RuntimeSetup()
            state = {"status": "installing", "lines": setup.deque(maxlen=120)}
            service.states["qwen21"] = state
            process = Mock(stdout=iter(["Installing torch\n"]), wait=Mock(return_value=code))
            context = Mock(__enter__=Mock(return_value=process), __exit__=Mock(return_value=False))
            with patch.object(setup, "installer_python", return_value=["/server/python3.12"]), \
                 patch.object(setup.subprocess, "Popen", return_value=context) as run:
                service._install(state, Path("/pack/qwen21_trainer"), Path("/pack/qwen21_trainer/.venv"))
            self.assertEqual(service.status("qwen21")["status"], expected)
            self.assertIn("Installing torch", service.status("qwen21")["log"])
            self.assertEqual(run.call_args.args[0], ["/server/python3.12", "-u", "/pack/qwen21_trainer/setup_runtime.py", "--venv", "/pack/qwen21_trainer/.venv"])

    def test_portable_python_without_venv_gets_actionable_error(self):
        with patch.object(setup.shutil, "which", return_value=None), \
             patch.object(setup.subprocess, "run", return_value=types.SimpleNamespace(returncode=1)):
            with self.assertRaisesRegex(ValueError, "Python 3.12.*venv.*pip"):
                setup.installer_python()


if __name__ == "__main__":
    unittest.main()
