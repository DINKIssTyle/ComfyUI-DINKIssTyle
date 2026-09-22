import hashlib
import os

import folder_paths
from aiohttp import web
from server import PromptServer


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".avif",
}
TEMP_PASTE_PREFIX = "DKST_Paste_"


def _input_root():
    return os.path.realpath(folder_paths.get_input_directory())


def _storage_root(source_type):
    if source_type == "input":
        return _input_root()
    if source_type == "temp":
        return os.path.realpath(folder_paths.get_temp_directory())
    raise ValueError("Invalid image source type.")


def _normalize_category(category):
    category = (category or "").replace("\\", "/").strip("/")
    if not category:
        return ""
    parts = category.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Invalid image category.")
    return "/".join(parts)


def _category_directory(category):
    root = _input_root()
    normalized = _normalize_category(category)
    directory = os.path.realpath(os.path.join(root, *normalized.split("/"))) if normalized else root
    if os.path.commonpath((root, directory)) != root:
        raise ValueError("Image category must stay inside the ComfyUI input folder.")
    return directory


def _is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def _image_files(category=""):
    directory = _category_directory(category)
    if not os.path.isdir(directory):
        return []
    return sorted(
        (
            filename
            for filename in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, filename))
            and not filename.startswith("._")
            and _is_image_file(filename)
        ),
        key=str.casefold,
    )


def _image_categories():
    root = _input_root()
    categories = []
    if not os.path.isdir(root):
        return categories

    for current_root, directories, _ in os.walk(root):
        directories[:] = sorted(
            (directory for directory in directories if not directory.startswith(".")),
            key=str.casefold,
        )
        if current_root == root:
            continue
        relative = os.path.relpath(current_root, root).replace(os.sep, "/")
        if _image_files(relative):
            categories.append(relative)
    return sorted(categories, key=str.casefold)


def _image_path(category, filename, source_type="input"):
    if not filename or os.path.basename(filename) != filename:
        raise ValueError("Invalid image filename.")
    if source_type == "temp" and not filename.startswith(TEMP_PASTE_PREFIX):
        raise ValueError("Only images pasted by DKST Image (Load) may use temp storage.")
    root = _storage_root(source_type)
    directory = _category_directory(category) if source_type == "input" else root
    path = os.path.realpath(os.path.join(directory, filename))
    if os.path.commonpath((root, path)) != root:
        raise ValueError("Image file must stay inside its managed ComfyUI folder.")
    if not os.path.isfile(path) or not _is_image_file(filename):
        raise ValueError(f"Image file not found: {filename}")
    return path


def _cleanup_pasted_temp_images():
    try:
        temp_dir = _storage_root("temp")
        if not os.path.isdir(temp_dir):
            return
        for filename in os.listdir(temp_dir):
            if not filename.startswith(TEMP_PASTE_PREFIX):
                continue
            path = os.path.join(temp_dir, filename)
            if os.path.isfile(path):
                os.remove(path)
    except OSError as error:
        print(f"[DKST Image Load] Unable to clean pasted temp images: {error}")


def _delete_pasted_temp_image(filename):
    if not filename or os.path.basename(filename) != filename:
        raise ValueError("Invalid temporary image filename.")
    if not filename.startswith(TEMP_PASTE_PREFIX):
        raise ValueError("Refusing to delete a temp file not created by DKST Image (Load).")
    path = os.path.realpath(os.path.join(_storage_root("temp"), filename))
    if os.path.isfile(path):
        os.remove(path)


_cleanup_pasted_temp_images()


@PromptServer.instance.routes.get("/dinki/image-load/categories")
async def get_dinki_image_categories(request):
    return web.json_response({"categories": [""] + _image_categories()})


@PromptServer.instance.routes.get("/dinki/image-load/files")
async def get_dinki_image_files(request):
    try:
        category = _normalize_category(request.query.get("category", ""))
        return web.json_response({"category": category, "files": _image_files(category)})
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.post("/dinki/image-load/delete-temp")
async def delete_dinki_temp_image(request):
    try:
        data = await request.json()
        _delete_pasted_temp_image(data.get("filename", ""))
        return web.json_response({"deleted": True})
    except (OSError, ValueError) as error:
        return web.json_response({"error": str(error)}, status=400)


class DINKI_Image_Load:
    @classmethod
    def INPUT_TYPES(cls):
        categories = [""] + _image_categories()
        root_files = _image_files("")
        return {
            "required": {
                "category": (categories,),
                "filename": (root_files or [""],),
            },
            "optional": {
                "source_type": (["input", "temp"], {"default": "input"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK", "ALPHA")
    FUNCTION = "load_image"
    CATEGORY = "DINKIssTyle/Image"

    def load_image(self, category, filename, source_type="input"):
        import numpy as np
        import torch
        from PIL import Image, ImageOps, ImageSequence

        path = _image_path(category, filename, source_type)
        output_images = []
        output_masks = []
        output_alphas = []
        expected_size = None

        with Image.open(path) as source:
            for frame in ImageSequence.Iterator(source):
                frame = ImageOps.exif_transpose(frame)
                rgba = frame.convert("RGBA")
                if expected_size is None:
                    expected_size = rgba.size
                elif rgba.size != expected_size:
                    continue

                pixels = np.asarray(rgba, dtype=np.float32) / 255.0
                image = torch.from_numpy(pixels[..., :3].copy()).unsqueeze(0)
                alpha = torch.from_numpy(pixels[..., 3].copy()).unsqueeze(0)
                output_images.append(image)
                output_alphas.append(alpha)
                output_masks.append(1.0 - alpha)

        if not output_images:
            raise ValueError(f"No readable image frames found: {filename}")

        images = torch.cat(output_images, dim=0)
        masks = torch.cat(output_masks, dim=0)
        alphas = torch.cat(output_alphas, dim=0)
        height, width = int(images.shape[1]), int(images.shape[2])
        return {
            "ui": {"resolution": [f"{width} × {height}"]},
            "result": (images, masks, alphas),
        }

    @classmethod
    def IS_CHANGED(cls, category, filename, source_type="input"):
        try:
            path = _image_path(category, filename, source_type)
            digest = hashlib.sha256()
            with open(path, "rb") as image_file:
                for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except (OSError, ValueError):
            return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, category, filename, source_type="input"):
        try:
            _image_path(category, filename, source_type)
        except ValueError as error:
            return str(error)
        return True


NODE_CLASS_MAPPINGS = {
    "DINKI_Image_Load": DINKI_Image_Load,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_Image_Load": "DKST Image (Load)",
}
