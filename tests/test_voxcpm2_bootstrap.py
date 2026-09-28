"""Startup dependency installation uses ComfyUI's Python only when needed."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle_VoxCPM2" / "bootstrap.py"
SPEC = importlib.util.spec_from_file_location("dkst_voxcpm2_bootstrap", MODULE_PATH)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def test_installed_environment_does_not_run_pip(self):
        with patch.object(bootstrap, "_missing_modules", return_value=[]), \
             patch.object(bootstrap.subprocess, "check_call") as install:
            bootstrap.ensure_dependencies()
        install.assert_not_called()

    def test_missing_module_installs_requirements_with_current_python(self):
        with patch.object(bootstrap, "_missing_modules", side_effect=[["voxcpm"], []]), \
             patch.object(bootstrap.subprocess, "check_call") as install, \
             patch.object(bootstrap.importlib, "invalidate_caches"):
            bootstrap.ensure_dependencies()
        command = install.call_args.args[0]
        self.assertEqual(command[:4], [sys.executable, "-m", "pip", "install"])
        self.assertEqual(command[-2:], ["-r", str(bootstrap.REQUIREMENTS)])

    def test_install_failure_explains_manual_recovery(self):
        with patch.object(bootstrap, "_missing_modules", return_value=["voxcpm"]), \
             patch.object(bootstrap.subprocess, "check_call",
                          side_effect=subprocess.CalledProcessError(1, "pip")):
            with self.assertRaisesRegex(RuntimeError, "restart ComfyUI"):
                bootstrap.ensure_dependencies()


if __name__ == "__main__":
    unittest.main()
