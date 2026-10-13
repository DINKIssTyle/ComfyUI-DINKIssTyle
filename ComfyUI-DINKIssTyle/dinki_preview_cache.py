import os
import re
import uuid

import folder_paths
import numpy as np
from comfy_execution.graph_utils import ExecutionBlocker, is_link

from .dinki_preview import DINKI_Preview_Image


class DINKI_Preview_Image_Cache(DINKI_Preview_Image):
    """A lossless IMAGE snapshot whose producer is evaluated only when needed."""

    CACHE_FOLDER = "dkst_preview_cache"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_mode": (["Capture once", "Refresh", "Use cache"], {
                    "default": "Capture once",
                    "tooltip": "Capture once generates the input only until an image is cached. Refresh captures the current input. Use cache never requests the input.",
                }),
            },
            "optional": {
                # rawLink prevents an ExecutionBlocker from bypassing fallback;
                # lazy prevents even an enabled depth model from running on reuse.
                "images": ("IMAGE", {"lazy": True, "rawLink": True}),
                "cache_file": ("STRING", {
                    "default": "", "dynamicPrompts": False, "advanced": True,
                    "forceWidget": True,
                    "tooltip": "Managed cache reference, saved with the workflow. Image data remains on this ComfyUI server.",
                }),
            },
            "hidden": {"execution_list": "EXECUTION_LIST", "unique_id": "UNIQUE_ID"},
        }

    FUNCTION = "preview_cached_image"
    DESCRIPTION = (
        "Preview and output a lossless saved image batch. Capture once (default) requests "
        "the producer only when there is no cache, so depth generation runs once. "
        "Refresh replaces the cache from the current input; empty or blocked input keeps "
        "the previous image. Use cache never requests the producer. Leave this node enabled. "
        "Bypass passes the producer's input through: use Capture once or Use cache to keep "
        "a depth map instead of replacing it with the original image. Save the workflow "
        "after capture. Image data is stored in output/dkst_preview_cache on this server."
    )

    @classmethod
    def _cache_path(cls, cache_file):
        if not isinstance(cache_file, str) or not re.fullmatch(r"[0-9a-f]{32}\.npy", cache_file):
            if cache_file:
                raise ValueError("Invalid Preview Cache reference. Use Refresh with an active image source.")
            return None
        root = os.path.realpath(os.path.join(folder_paths.get_output_directory(), cls.CACHE_FOLDER))
        path = os.path.realpath(os.path.join(root, cache_file))
        if os.path.dirname(path) != root:
            raise ValueError("Preview Cache reference must remain inside its cache folder.")
        return path

    @classmethod
    def IS_CHANGED(cls, cache_file="", **kwargs):
        path = cls._cache_path(cache_file)
        if path and os.path.isfile(path):
            stat = os.stat(path)
            return (stat.st_mtime_ns, stat.st_size)
        return "missing"

    @classmethod
    def check_lazy_status(cls, cache_mode="Capture once", cache_file="", images=None,
                          execution_list=None, unique_id=None):
        path = cls._cache_path(cache_file)
        reuse = cache_mode == "Use cache" or (
            cache_mode == "Capture once" and path and os.path.isfile(path)
        )
        if not reuse and is_link(images) and execution_list is not None and unique_id is not None:
            # A raw link is already a value in get_input_data, so returning its
            # input name alone would not promote this lazy dependency. Register
            # the dependency explicitly; ComfyUI waits while it is blocked.
            execution_list.make_input_strong_link(unique_id, "images")
        return []

    @staticmethod
    def _incoming_images(images, execution_list, unique_id):
        if is_link(images):
            if execution_list is None or unique_id is None:
                return None
            entry = execution_list.get_cache(images[0], unique_id)
            index = images[1]
            if entry is None or entry.outputs is None or not 0 <= index < len(entry.outputs):
                return None
            values = entry.outputs[index]
        else:
            values = images
        if not isinstance(values, (list, tuple)):
            values = (values,)
        batches = [value for value in values
                   if not isinstance(value, ExecutionBlocker)
                   and getattr(value, "ndim", None) == 4 and all(int(n) > 0 for n in value.shape)]
        if not batches:
            return None
        import torch
        return batches[0] if len(batches) == 1 else torch.cat(batches, dim=0)

    @staticmethod
    def _validate_array(array):
        if (array.ndim != 4 or not all(n > 0 for n in array.shape)
                or array.shape[-1] not in (1, 3, 4) or array.dtype.kind != "f"):
            raise ValueError("Preview Cache requires a non-empty floating-point IMAGE batch [B, H, W, C].")

    def _capture(self, images):
        array = np.ascontiguousarray(images.detach().cpu().numpy())
        self._validate_array(array)
        # Immutable snapshots keep copies of a workflow independent when either
        # copy refreshes its cache. The tensor data is not quantized to a PNG.
        cache_file = f"{uuid.uuid4().hex}.npy"
        path = self._cache_path(cache_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pending = path + ".pending"
        written = []
        try:
            with open(pending, "xb") as handle:
                np.save(handle, array, allow_pickle=False)
            for index, image in enumerate(array):
                preview = self._preview_path(path, index)
                written.append(preview)
                self._save_image(preview, image, "png", "8bit", "sRGB")
            os.replace(pending, path)
        except Exception:
            for filename in [pending, *written]:
                if os.path.isfile(filename):
                    os.unlink(filename)
            raise
        return cache_file, array

    @staticmethod
    def _preview_path(path, index):
        return f"{os.path.splitext(path)[0]}_{index:05}.png"

    def preview_cached_image(self, cache_mode="Capture once", images=None, cache_file="",
                             execution_list=None, unique_id=None):
        if cache_mode not in ("Capture once", "Refresh", "Use cache"):
            raise ValueError("Unknown Preview Cache mode.")
        path = self._cache_path(cache_file)
        has_cache = bool(path and os.path.isfile(path))
        incoming = None
        if cache_mode == "Refresh" or (cache_mode == "Capture once" and not has_cache):
            incoming = self._incoming_images(images, execution_list, unique_id)
        if incoming is not None:
            cache_file, array = self._capture(incoming)
            path = self._cache_path(cache_file)
        elif has_cache:
            try:
                array = np.load(path, allow_pickle=False)
                self._validate_array(array)
            except Exception as error:
                raise ValueError("Unable to read the saved image. Use Refresh with an active image source.") from error
        else:
            raise ValueError(
                "Preview Cache has no saved image. Connect an active image source and run with Capture once or Refresh."
            )
        import torch
        ui_images = []
        for index, image in enumerate(array):
            preview = self._preview_path(path, index)
            if not os.path.isfile(preview):
                self._save_image(preview, image, "png", "8bit", "sRGB")
            ui_images.append({"filename": os.path.basename(preview),
                              "subfolder": self.CACHE_FOLDER, "type": "output"})
        ui = {"dkst_images": ui_images, "resolution": [f"{array.shape[2]} × {array.shape[1]}"]}
        if incoming is not None:
            ui["dkst_cache_file"] = [cache_file]
        # A CPU tensor keeps a retained depth map from occupying GPU memory.
        return {"ui": ui, "result": (torch.from_numpy(array.copy()),)}
