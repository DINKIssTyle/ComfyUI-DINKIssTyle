"""Voice file listing and upload endpoints for the reference audio node."""

from __future__ import annotations

import unicodedata
from pathlib import Path

from aiohttp import web
from server import PromptServer

from .nodes import AUDIO_EXTENSIONS, VOICE_ROOT, _read_voice_transcript, _voice_names


MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def _upload_filename(filename: str) -> str:
    if not isinstance(filename, str):
        raise ValueError("Choose an audio file to upload.")
    name = unicodedata.normalize("NFC", filename)
    if (not name or len(name) > 200 or name.startswith(".") or Path(name).name != name
            or any(char in name for char in '\\/:*?"<>|\x00')
            or Path(name).suffix.lower() not in AUDIO_EXTENSIONS):
        raise ValueError("Upload a supported audio file with a valid filename.")
    return name


def _upload_destination(name: str) -> Path:
    VOICE_ROOT.mkdir(parents=True, exist_ok=True)
    original = Path(name)
    existing_stems = {path.stem.casefold() for path in VOICE_ROOT.iterdir()
                      if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS | {".txt"}}
    index = 1
    while True:
        stem = original.stem if index == 1 else f"{original.stem}_{index}"
        path = VOICE_ROOT / f"{stem}{original.suffix}"
        if stem.casefold() not in existing_stems and not path.exists():
            return path
        index += 1


@PromptServer.instance.routes.get("/dkst/voxcpm2/voices")
async def list_voices(request):
    return web.json_response({"files": _voice_names()})


@PromptServer.instance.routes.get("/dkst/voxcpm2/transcript")
async def get_transcript(request):
    name = request.query.get("name", "")
    if not name:
        return web.json_response({"transcript": ""})
    try:
        return web.json_response({"transcript": _read_voice_transcript(name)})
    except (OSError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


@PromptServer.instance.routes.post("/dkst/voxcpm2/upload-voice")
async def upload_voice(request):
    try:
        reader = await request.multipart()
        part = await reader.next()
        if part is None or part.name != "file":
            raise ValueError("Choose an audio file to upload.")
        name = _upload_filename(part.filename)
        destination = _upload_destination(name)
        written = 0
        created = False
        try:
            with destination.open("xb") as output:
                created = True
                while chunk := await part.read_chunk(size=1024 * 1024):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise OverflowError("Audio uploads must be 100 MB or smaller.")
                    output.write(chunk)
            if written == 0:
                raise ValueError("The uploaded audio file is empty.")
        except Exception:
            if created:
                destination.unlink(missing_ok=True)
            raise
        return web.json_response({"name": destination.name, "files": _voice_names()})
    except OverflowError as exc:
        return web.json_response({"error": str(exc)}, status=413)
    except (OSError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
