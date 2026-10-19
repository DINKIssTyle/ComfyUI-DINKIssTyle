"""Prepare required packages in the Python interpreter running ComfyUI."""
import importlib
from importlib import metadata
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

DEPENDENCIES = (
    ('numpy', 'numpy>=1.26'),
    ('PIL', 'Pillow>=10'),
    ('scipy.spatial', 'scipy>=1.11'),
    ('packaging.requirements', 'packaging>=20'),
    ('cv2', 'opencv-contrib-python>=4.8'),
    ('mediapipe', 'mediapipe>=0.10.14'),
)
INSTALLER = Path(__file__).resolve().with_name('install.py')
PREFIX = '[DKST Face Morph Studio]'


def dependency_errors():
    errors = {}
    for module, _ in DEPENDENCIES:
        try:
            imported = importlib.import_module(module)
            if module == 'mediapipe':
                # Require the Tasks API rather than the retired Solutions API.
                imported.tasks.vision.FaceLandmarker
            if module == 'cv2':
                imported.remap
                imported.INTER_CUBIC
        except Exception as exc:
            errors[module] = str(exc)
    return errors


def repair_command():
    command = [sys.executable, str(INSTALLER)]
    return subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command)


def process_output(process):
    parts = []
    for field in ('stdout', 'stderr'):
        value = getattr(process, field, None)
        if isinstance(value, bytes):
            value = value.decode('utf-8', errors='replace')
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return '\n'.join(parts)


def run_installer(command):
    # Capture child streams explicitly: some launchers omit inherited pip output
    # from comfyui.log. UTF-8 mode also applies to pip build subprocesses on Windows.
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    try:
        result = subprocess.run(command, check=True, timeout=600, capture_output=True,
                                text=True, encoding='utf-8', errors='replace', env=env)
    except subprocess.SubprocessError as exc:
        output = process_output(exc)
        if output:
            print(f'{PREFIX} Installer output:\n{output}', flush=True)
        raise
    output = process_output(result)
    if output:
        print(f'{PREFIX} Installer output:\n{output}', flush=True)


def install_packages(requirements, no_deps=False):
    flags = ['--no-deps'] if no_deps else []
    command = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-input', *flags, *requirements]
    try:
        run_installer(command)
    except subprocess.CalledProcessError as exc:
        output = process_output(exc).lower()
        if 'no module named pip' not in output and "no module named 'pip'" not in output:
            raise
        print(f'{PREFIX} pip is missing from the ComfyUI environment. Preparing pip with ensurepip.', flush=True)
        try:
            run_installer([sys.executable, '-m', 'ensurepip', '--upgrade'])
        except (OSError, subprocess.SubprocessError):
            # uv-managed environments may provide neither pip nor ensurepip.
            # Always target ComfyUI Python rather than uv's default interpreter.
            uv = shutil.which('uv')
            if uv is None:
                raise
            print(f'{PREFIX} ensurepip is unavailable. Installing with uv into {sys.executable}.', flush=True)
            run_installer([uv, 'pip', 'install', '--python', sys.executable, *flags, *requirements])
            return
        run_installer(command)


OPENCV_DISTRIBUTIONS = {
    'opencv-python', 'opencv-python-headless',
    'opencv-contrib-python', 'opencv-contrib-python-headless',
}


def mediapipe_requirements():
    # Read the installed wheel's metadata before importing MediaPipe, since its
    # own dependencies may not be installed yet. Do not hardcode version-specific
    # MediaPipe dependency lists.
    declared = metadata.requires('mediapipe') or []
    if not declared:
        return []
    from packaging.requirements import Requirement
    try:
        cv2 = importlib.import_module('cv2')
        reuse_opencv = hasattr(cv2, 'remap') and hasattr(cv2, 'INTER_CUBIC')
    except Exception:
        reuse_opencv = False
    needed = []
    for declaration in declared:
        requirement = Requirement(declaration)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        normalized = requirement.name.lower().replace('_','-')
        if normalized in OPENCV_DISTRIBUTIONS and reuse_opencv:
            print(f'{PREFIX} Reusing working OpenCV; skipping {requirement.name} to avoid replacing loaded cv2 binaries.', flush=True)
            continue
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            needed.append(str(requirement))
        else:
            if requirement.url or not requirement.specifier.contains(version, prereleases=True):
                needed.append(str(requirement))
    return needed


def install_runtime(errors):
    other = [requirement for module, requirement in DEPENDENCIES if module in errors and module != 'mediapipe']
    if other:
        install_packages(other)
    if 'mediapipe' in errors:
        requirement = next(req for module, req in DEPENDENCIES if module == 'mediapipe')
        # pip otherwise adds opencv-contrib-python even when a working headless
        # or standard build already provides cv2. On Windows the loaded pyd is locked.
        install_packages([requirement], no_deps=True)
        needed = mediapipe_requirements()
        if needed:
            install_packages(needed)


def ensure_dependencies(install=None):
    if install is None:
        install = os.environ.get('DKST_AUTO_INSTALL', '1').lower() not in {'0', 'false', 'no'}
    errors = dependency_errors()
    if not errors:
        return
    details = ', '.join(errors)
    if not install:
        raise RuntimeError(
            f'{PREFIX} Required dependencies are unavailable: {details}. '
            f'Automatic installation is disabled by DKST_AUTO_INSTALL. Run {repair_command()} '
            'and restart ComfyUI, or enable automatic installation.'
        )
    requirements = [requirement for module, requirement in DEPENDENCIES if module in errors]
    print(f'{PREFIX} Installing missing or unusable dependencies with {sys.executable} '
          f'(Python {sys.version.split()[0]}, {sys.platform}): {", ".join(requirements)}', flush=True)
    for module, error in errors.items():
        print(f'{PREFIX} Dependency check failed for {module}: {error}', flush=True)
    try:
        install_runtime(errors)
    except (OSError, subprocess.SubprocessError) as exc:
        output = process_output(exc)
        diagnostic = f'\nInstaller details:\n{output[-12000:]}' if output else ''
        raise RuntimeError(
            f'{PREFIX} Dependency installation failed: {exc}.{diagnostic}\n'
            f'Run {repair_command()} using ComfyUI Python, then restart ComfyUI.'
        ) from exc
    importlib.invalidate_caches()
    remaining = dependency_errors()
    if remaining:
        detail = '; '.join(f'{module}: {error}' for module, error in remaining.items())
        raise RuntimeError(
            f'{PREFIX} Dependencies remain unavailable after installation: {detail}. '
            f'Restart ComfyUI to load newly installed packages. If the error persists, run {repair_command()}.'
        )
    print(f'{PREFIX} Required Python dependencies are ready.', flush=True)


if __name__ == '__main__':
    ensure_dependencies(install=True)
