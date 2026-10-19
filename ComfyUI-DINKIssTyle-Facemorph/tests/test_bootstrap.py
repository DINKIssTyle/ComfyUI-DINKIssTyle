"""Dependency setup must use the host interpreter and report failures clearly."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('fm_bootstrap',Path(__file__).resolve().parents[1]/'bootstrap.py')
bootstrap=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        metadata_patch=patch.object(bootstrap.metadata,'requires',return_value=[])
        metadata_patch.start()
        self.addCleanup(metadata_patch.stop)

    def test_working_dependencies_do_not_invoke_pip(self):
        with patch.object(bootstrap,'dependency_errors',return_value={}),patch.object(bootstrap.subprocess,'run') as pip:
            bootstrap.ensure_dependencies()
            pip.assert_not_called()

    def test_missing_mediapipe_installs_with_host_python(self):
        with patch.object(bootstrap,'dependency_errors',side_effect=[{'mediapipe':'missing'},{}]),patch.object(bootstrap.subprocess,'run') as pip:
            bootstrap.ensure_dependencies(install=True)
            command=pip.call_args.args[0]
            self.assertEqual(command[:3],[sys.executable,'-m','pip'])
            self.assertIn('mediapipe>=0.10.14',command)
            self.assertNotIn('opencv-contrib-python>=4.8',command)
            self.assertTrue(pip.call_args.kwargs['check'])

    def test_automatic_install_is_enabled_by_default(self):
        with patch.dict(os.environ,{},clear=True),patch.object(bootstrap,'dependency_errors',side_effect=[{'mediapipe':'missing'},{}]),patch.object(bootstrap.subprocess,'run') as pip:
            bootstrap.ensure_dependencies()
            pip.assert_called_once()

    def test_opt_out_reports_repair_command(self):
        with patch.dict(os.environ,{'DKST_AUTO_INSTALL':'0'}),patch.object(bootstrap,'dependency_errors',return_value={'mediapipe':'missing'}),patch.object(bootstrap.subprocess,'run') as pip:
            with self.assertRaisesRegex(RuntimeError,'Automatic installation is disabled') as error:
                bootstrap.ensure_dependencies()
            self.assertIn(sys.executable,str(error.exception))
            self.assertIn('install.py',str(error.exception))
            pip.assert_not_called()

    def test_failed_install_preserves_cause_and_repair_guidance(self):
        with patch.object(bootstrap,'dependency_errors',return_value={'mediapipe':'missing'}),patch.object(bootstrap.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['pip'])):
            with self.assertRaisesRegex(RuntimeError,'Dependency installation failed') as error:
                bootstrap.ensure_dependencies(install=True)
            self.assertIsInstance(error.exception.__cause__,subprocess.CalledProcessError)
            self.assertIn('restart ComfyUI',str(error.exception))

    def test_unusable_package_after_install_requires_restart(self):
        with patch.object(bootstrap,'dependency_errors',return_value={'mediapipe':'invalid Tasks API'}),patch.object(bootstrap.subprocess,'run'):
            with self.assertRaisesRegex(RuntimeError,'Restart ComfyUI to load newly installed packages'):
                bootstrap.ensure_dependencies(install=True)

    def test_pip_output_is_included_in_failure(self):
        failure=subprocess.CalledProcessError(1,['pip'],output='Collecting mediapipe',stderr='ERROR: No matching distribution found')
        with patch.object(bootstrap,'dependency_errors',return_value={'mediapipe':'missing'}),patch.object(bootstrap.subprocess,'run',side_effect=failure) as pip:
            with self.assertRaises(RuntimeError) as error:
                bootstrap.ensure_dependencies(install=True)
            self.assertIn('No matching distribution found',str(error.exception))
            self.assertIn('Collecting mediapipe',str(error.exception))
            pip.assert_called_once()

    def test_missing_pip_runs_ensurepip_then_retries(self):
        missing=subprocess.CalledProcessError(1,['pip'],stderr='python.exe: No module named pip')
        success=subprocess.CompletedProcess([],0,stdout='Installed',stderr='')
        with patch.object(bootstrap.subprocess,'run',side_effect=[missing,success,success]) as pip:
            bootstrap.install_packages(['mediapipe>=0.10.14'])
            calls=pip.call_args_list
            self.assertEqual(calls[1].args[0],[sys.executable,'-m','ensurepip','--upgrade'])
            self.assertEqual(calls[0].args[0],calls[2].args[0])
            for call in calls:
                self.assertEqual(call.kwargs['env']['PYTHONUTF8'],'1')
                self.assertTrue(call.kwargs['capture_output'])
                self.assertEqual(call.kwargs['encoding'],'utf-8')

    def test_uv_fallback_targets_comfyui_python(self):
        missing=subprocess.CalledProcessError(1,['pip'],stderr='No module named pip')
        no_ensurepip=subprocess.CalledProcessError(1,['ensurepip'],stderr='No module named ensurepip')
        success=subprocess.CompletedProcess([],0,stdout='',stderr='')
        with patch.object(bootstrap.subprocess,'run',side_effect=[missing,no_ensurepip,success]) as pip,patch.object(bootstrap.shutil,'which',return_value='/tools/uv'):
            bootstrap.install_packages(['mediapipe>=0.10.14'])
            self.assertEqual(pip.call_args.args[0],['/tools/uv','pip','install','--python',sys.executable,'mediapipe>=0.10.14'])

    def test_missing_installers_fail_with_actual_diagnostic(self):
        missing=subprocess.CalledProcessError(1,['pip'],stderr='No module named pip')
        no_ensurepip=subprocess.CalledProcessError(1,['ensurepip'],stderr='No module named ensurepip')
        with patch.object(bootstrap,'dependency_errors',return_value={'mediapipe':'missing'}),patch.object(bootstrap.subprocess,'run',side_effect=[missing,no_ensurepip]),patch.object(bootstrap.shutil,'which',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'No module named ensurepip'):
                bootstrap.ensure_dependencies(install=True)

    def test_timeout_preserves_partial_output(self):
        failure=subprocess.TimeoutExpired(['pip'],600,output=b'Downloading mediapipe',stderr=b'Connection stalled')
        with patch.object(bootstrap,'dependency_errors',return_value={'mediapipe':'missing'}),patch.object(bootstrap.subprocess,'run',side_effect=failure):
            with self.assertRaises(RuntimeError) as error:
                bootstrap.ensure_dependencies(install=True)
            self.assertIn('Connection stalled',str(error.exception))
            self.assertIn('Downloading mediapipe',str(error.exception))

    def test_mediapipe_installs_without_transitive_opencv(self):
        with patch.object(bootstrap,'install_packages') as install,patch.object(bootstrap,'mediapipe_requirements',return_value=['sounddevice~=0.5']):
            bootstrap.install_runtime({'mediapipe':'missing'})
            self.assertEqual(install.call_args_list[0].args[0],['mediapipe>=0.10.14'])
            self.assertTrue(install.call_args_list[0].kwargs['no_deps'])
            self.assertEqual(install.call_args_list[1].args[0],['sounddevice~=0.5'])

    def test_metadata_keeps_working_opencv_and_respects_markers(self):
        declared=['opencv-contrib-python','sounddevice~=0.5','certifi', 'unused-package; sys_platform == "never"']
        cv=SimpleNamespace(remap=object(),INTER_CUBIC=2)
        def version(name):
            if name=='sounddevice':return '0.4.0'
            if name=='certifi':return '2026.10.18'
            raise AssertionError('Unexpected dependency lookup: '+name)
        with patch.object(bootstrap.metadata,'requires',return_value=declared),patch.object(bootstrap.metadata,'version',side_effect=version),patch.object(bootstrap.importlib,'import_module',return_value=cv):
            self.assertEqual(bootstrap.mediapipe_requirements(),['sounddevice~=0.5'])

    def test_opencv_is_not_skipped_if_cv2_is_unavailable(self):
        with patch.object(bootstrap.metadata,'requires',return_value=['opencv-contrib-python']),patch.object(bootstrap.metadata,'version',side_effect=bootstrap.metadata.PackageNotFoundError('opencv-contrib-python')),patch.object(bootstrap.importlib,'import_module',side_effect=ImportError('cv2 unavailable')):
            self.assertEqual(bootstrap.mediapipe_requirements(),['opencv-contrib-python'])

    def test_uv_fallback_preserves_no_deps_flag(self):
        missing=subprocess.CalledProcessError(1,['pip'],stderr='No module named pip')
        no_ensurepip=subprocess.CalledProcessError(1,['ensurepip'],stderr='No module named ensurepip')
        success=subprocess.CompletedProcess([],0,stdout='',stderr='')
        with patch.object(bootstrap.subprocess,'run',side_effect=[missing,no_ensurepip,success]) as pip,patch.object(bootstrap.shutil,'which',return_value='/tools/uv'):
            bootstrap.install_packages(['mediapipe>=0.10.14'],no_deps=True)
            self.assertIn('--no-deps',pip.call_args.args[0])



if __name__=='__main__':unittest.main()
