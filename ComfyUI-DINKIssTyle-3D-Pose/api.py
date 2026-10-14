import asyncio
import json
from aiohttp import web
from server import PromptServer
from .schema import MAX_BYTES, validate_scene
from .storage import save_render, artifact_path, load_render, load_pose

routes = PromptServer.instance.routes


@routes.post("/dkst/3d-pose/render")
async def upload_render(request):
    try:
        reader = await request.multipart()
        raw = None
        data = None
        pose = None
        total = 0
        while (part := await reader.next()) is not None:
            limits={"scene":128*1024,"pose":32*1024,"image":MAX_BYTES}
            if part.name not in limits:raise ValueError("Unexpected upload field")
            limit=limits[part.name]
            chunks = bytearray()
            while chunk := await part.read_chunk():
                chunks.extend(chunk)
                total += len(chunk)
                if len(chunks) > limit or total > MAX_BYTES + 160*1024:
                    raise ValueError("Pose upload too large")
            if part.name == "scene" and raw is None:
                raw = chunks.decode("utf-8")
            elif part.name == "image" and data is None:
                data = bytes(chunks)
            elif part.name == "pose" and pose is None:
                pose=json.loads(chunks.decode("utf-8"))
                if pose is None:raise ValueError("Invalid OpenPose document")
            else:
                raise ValueError("Unexpected upload field")
        validate_scene(raw)
        result = await asyncio.to_thread(save_render, raw, data, pose)
        return web.json_response(result)
    except (ValueError, TypeError, UnicodeError, OSError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


@routes.get("/dkst/3d-pose/render/{render_id}")
async def view_render(request):
    try:
        path = artifact_path(request.match_info["render_id"])
        if not path.exists() or not path.with_suffix(".json").exists():
            raise web.HTTPNotFound()
        # HEAD is also used before reusing a client-side render. Damaged cache files
        # must cause a fresh browser render instead of failing every subsequent job.
        metadata = json.loads(path.with_suffix(".json").read_text())
        if request.query.get("require_pose")=="1":
            await asyncio.to_thread(load_pose,metadata["scene"],request.match_info["render_id"])
        image = await asyncio.to_thread(load_render, metadata["scene"], request.match_info["render_id"])
        image.close()
        return web.FileResponse(path, headers={"Cache-Control": "private, max-age=31536000, immutable"})
    except (ValueError, KeyError, OSError) as exc:
        raise web.HTTPNotFound(text=str(exc)) from exc
