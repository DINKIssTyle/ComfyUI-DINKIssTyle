"""Browse three-column CSV prompt libraries in a ComfyUI node."""

import csv
from pathlib import Path

from aiohttp import web
from server import PromptServer


CSV_DIRECTORY = Path(__file__).resolve().parent / "csv"
DEFAULT_FILE = "Cinema_Prompt.csv"
NONE = "-- None --"


def _csv_path(filename):
    if not filename or filename == NONE:
        return None
    if not isinstance(filename, str) or Path(filename).name != filename or "\\" in filename:
        raise ValueError("Invalid CSV filename.")
    if Path(filename).suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")
    root = CSV_DIRECTORY.resolve()
    path = (root / filename).resolve()
    if path.parent != root or not path.is_file():
        raise ValueError(f"CSV file not found: {filename}")
    return path


def _csv_files():
    if not CSV_DIRECTORY.is_dir():
        return []
    files = []
    for candidate in CSV_DIRECTORY.iterdir():
        if candidate.name.startswith(".") or candidate.suffix.lower() != ".csv":
            continue
        try:
            if _csv_path(candidate.name):
                files.append(candidate.name)
        except ValueError:
            continue
    return sorted(files, key=str.casefold)


def _csv_sections(filename):
    path = _csv_path(filename)
    sections = {}
    if path is None:
        return sections
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.reader(handle)):
            if len(row) != 3:
                continue
            section, title, prompt = (value.strip() for value in row)
            if index == 0 and section.casefold() in {"category", "section"} and title.casefold() in {"technique name", "title"}:
                continue
            if not section or not title or not prompt:
                continue
            entries = sections.setdefault(section, {})
            entries.setdefault(title, prompt)
    return sections


@PromptServer.instance.routes.get("/dinki/prompt-library/files")
async def get_prompt_library_files(request):
    files = _csv_files()
    return web.json_response({"files": files, "default": DEFAULT_FILE if DEFAULT_FILE in files else (files[0] if files else NONE)})


@PromptServer.instance.routes.get("/dinki/prompt-library/entries")
async def get_prompt_library_entries(request):
    filename = request.query.get("file", NONE)
    try:
        sections = _csv_sections(filename)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        return web.json_response({"error": str(error)}, status=400)
    return web.json_response({"file": filename, "sections": sections})


class DINKI_PromptCsvLibrary:
    @classmethod
    def INPUT_TYPES(cls):
        files = _csv_files()
        default = DEFAULT_FILE if DEFAULT_FILE in files else (files[0] if files else NONE)
        try:
            sections = _csv_sections(default)
        except (OSError, UnicodeError, csv.Error, ValueError):
            sections = {}
        titles = list(dict.fromkeys(title for entries in sections.values() for title in entries))
        return {
            "required": {
                "csv_file": ([NONE, *files], {"default": default, "advanced": True}),
                "section": ([NONE, *sections], {"default": NONE}),
                "title": ([NONE, *titles], {"default": NONE}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {"text_input": ("STRING", {"forceInput": True})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_string",)
    FUNCTION = "compose_prompt"
    CATEGORY = "DINKIssTyle/Prompt"

    @classmethod
    def VALIDATE_INPUTS(cls, csv_file, **kwargs):
        try:
            _csv_path(csv_file)
        except ValueError as error:
            return str(error)
        return True

    def compose_prompt(self, csv_file, section, title, prompt, text_input=""):
        prefix = text_input or ""
        selected = prompt or ""
        if prefix.strip() and selected.strip():
            return (f"{prefix}, {selected}",)
        return (prefix if prefix.strip() else selected,)
