"""Read-only host telemetry; collection never runs on the HTTP event loop."""

import asyncio
import csv
import io
import math
import os
import shutil
import subprocess
import threading
import time

from aiohttp import web
from server import PromptServer

try:
    import psutil
except ImportError:
    psutil = None


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError):
        return None


def _parse_gpus(output):
    gpus = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) != 6:
            continue
        index, name, utilization, temperature, used, total = (s.strip() for s in row)
        if not index.isdigit():
            continue
        gpus.append({
            "index": int(index), "name": name,
            "utilization": _number(utilization), "temperature": _number(temperature),
            "memory_used_mib": _number(used), "memory_total_mib": _number(total),
        })
    return gpus


def _nvidia_smi_path():
    executable = shutil.which("nvidia-smi")
    if executable:
        return executable
    if os.name == "nt":
        for path in (
            os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"),
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
        ):
            if os.path.isfile(path):
                return path
    return None


def _collect():
    result = {"cpu_percent": None, "ram": None, "gpus": [], "errors": []}
    if psutil is None:
        result["errors"].append("CPU/RAM unavailable: install psutil in ComfyUI's Python environment.")
    else:
        try:
            # A short explicit sample also works when the worker thread changes.
            result["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            result["ram"] = {"used_bytes": memory.total - memory.available, "total_bytes": memory.total}
        except Exception:
            result["errors"].append("CPU/RAM sampling failed.")

    executable = _nvidia_smi_path()
    if not executable:
        result["errors"].append("NVIDIA telemetry unavailable: nvidia-smi was not found.")
    else:
        try:
            completed = subprocess.run(
                [executable, "--query-gpu=index,name,utilization.gpu,temperature.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=3, check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
            result["gpus"] = _parse_gpus(completed.stdout)
            if not result["gpus"]:
                result["errors"].append("No NVIDIA GPU telemetry was returned.")
        except subprocess.TimeoutExpired:
            result["errors"].append("NVIDIA telemetry timed out.")
        except (OSError, subprocess.SubprocessError):
            result["errors"].append("NVIDIA telemetry unavailable: check the NVIDIA driver.")
    result["sampled_at"] = time.time()
    return result


class MonitorCache:
    def __init__(self):
        self.lock = threading.Lock()
        self.value = None
        self.updated = 0

    def read(self):
        # Share samples between browser tabs/users, including failed samples.
        with self.lock:
            if self.value is None or time.monotonic() - self.updated >= 1:
                self.value = _collect()
                self.updated = time.monotonic()
            return self.value


_monitor = MonitorCache()


@PromptServer.instance.routes.get("/dinki/monitor")
async def monitor_status(request):
    return web.json_response(
        await asyncio.to_thread(_monitor.read),
        headers={"Cache-Control": "no-store"},
    )
