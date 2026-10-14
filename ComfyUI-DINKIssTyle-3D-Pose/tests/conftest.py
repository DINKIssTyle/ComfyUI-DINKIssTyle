import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
BASE = tempfile.TemporaryDirectory(prefix="dkst-pose-tests-")
routes = web.RouteTableDef()
sys.modules["server"] = types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes)))
sys.modules["folder_paths"] = types.SimpleNamespace(get_input_directory=lambda: BASE.name)
spec = importlib.util.spec_from_file_location("pose_pack", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
pack = importlib.util.module_from_spec(spec)
sys.modules["pose_pack"] = pack
spec.loader.exec_module(pack)
