"""Background setup of the isolated CUDA runtime used by both trainers."""
from collections import deque
from pathlib import Path
import shutil
import subprocess
import sys
import threading


def venv_python(directory):
    return Path(directory) / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def installer_python():
    """Portable Windows Python may lack venv/ensurepip; try an installed Python."""
    candidates = []
    if sys.platform == "win32" and shutil.which("py"):
        candidates.append([shutil.which("py"), "-3.12"])
    if shutil.which("python3.12"):
        candidates.append([shutil.which("python3.12")])
    candidates.append([sys.executable])
    for command in candidates:
        try:
            result = subprocess.run(command + ["-c", "import sys, venv, ensurepip; "
                "assert sys.version_info[:2] >= (3, 10)"],
                capture_output=True, timeout=15)
            if result.returncode == 0:
                return command
        except (OSError, subprocess.SubprocessError):
            continue
    raise ValueError("Install Python 3.12 on the ComfyUI server (including venv and pip), "
                     "then click Prepare Runtime again, or run the trainer's setup_runtime.py with it.")


class RuntimeSetup:
    def __init__(self):
        self.lock = threading.RLock()
        self.states = {}
        self.thread = None

    def active(self):
        with self.lock:
            return any(state["status"] == "installing" for state in self.states.values())

    def status(self, name):
        with self.lock:
            state = self.states.get(name)
            if state is None:
                return {"status": "idle", "message": "Prepare the separate CUDA training environment."}
            return {k: v for k, v in state.items() if k != "lines"} | {"log": "".join(state["lines"])}

    def start(self, name, runner):
        with self.lock:
            if self.active():
                if self.states.get(name, {}).get("status") == "installing":
                    return self.status(name)
                raise ValueError("Another trainer runtime is being prepared. Wait for it to finish.")
            if sys.platform not in {"win32", "linux"}:
                raise ValueError("The training runtime requires Windows/Linux and an NVIDIA CUDA GPU.")
            root = Path(runner).parent
            target = root / ".venv"
            # H3 uses the same pins. Reuse the Qwen venv when an H3 venv does not exist.
            shared = root.parent / "qwen21_trainer/.venv"
            if name == "minimax_h3" and not venv_python(target).is_file() and venv_python(shared).is_file():
                target = shared
            python = venv_python(target).absolute()
            state = {"status": "installing", "python": str(python),
                     "message": "Installing the separate CUDA training environment…", "lines": deque(maxlen=120)}
            self.states[name] = state
            self.thread = threading.Thread(target=self._install, args=(state, root, target), daemon=True)
            self.thread.start()
            return self.status(name)

    def _install(self, state, root, target):
        try:
            command = installer_python() + ["-u", str(root / "setup_runtime.py"), "--venv", str(target)]
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    with self.lock:
                        state["lines"].append(line)
                code = process.wait()
            if code:
                raise ValueError("Runtime setup failed. Review the installation log below and retry.")
            with self.lock:
                state.update(status="ready", message="Runtime ready. Select this Training Python and save your models.")
        except Exception as error:
            with self.lock:
                state.update(status="failed", message=str(error))


SETUP = RuntimeSetup()
