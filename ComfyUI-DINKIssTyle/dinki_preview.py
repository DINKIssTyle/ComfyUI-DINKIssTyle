import os
import struct
import zlib

import folder_paths
import numpy as np


class DINKI_Preview_Image:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "DKST_Image "}),
                "format": (["png", "exr", "avif", "webp"],),
                "bit_depth": (["8bit", "16bit"],),
                "input_color_space": (["sRGB"],),
                "always_save": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "preview_image"
    OUTPUT_NODE = True
    CATEGORY = "DINKIssTyle/Image"

    @staticmethod
    def _png_chunk(chunk_type, data):
        payload = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    @classmethod
    def _write_png(cls, path, image, bit_depth):
        height, width, channels = image.shape
        color_type = {1: 0, 3: 2, 4: 6}.get(channels)
        if color_type is None:
            raise ValueError(f"PNG does not support {channels} channels in this node.")

        if bit_depth == "16bit":
            pixels = np.rint(image * 65535.0).astype(">u2")
            depth = 16
        else:
            pixels = np.rint(image * 255.0).astype(np.uint8)
            depth = 8

        scanlines = b"".join(
            b"\x00" + np.ascontiguousarray(pixels[row]).tobytes()
            for row in range(height)
        )
        header = struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, 0)

        with open(path, "wb") as png_file:
            png_file.write(b"\x89PNG\r\n\x1a\n")
            png_file.write(cls._png_chunk(b"IHDR", header))
            png_file.write(cls._png_chunk(b"sRGB", b"\x00"))
            png_file.write(cls._png_chunk(b"IDAT", zlib.compress(scanlines, 6)))
            png_file.write(cls._png_chunk(b"IEND", b""))

    @staticmethod
    def _exr_attribute(name, attribute_type, value):
        return (
            name.encode("ascii")
            + b"\x00"
            + attribute_type.encode("ascii")
            + b"\x00"
            + struct.pack("<I", len(value))
            + value
        )

    @classmethod
    def _write_exr(cls, path, image):
        height, width, channels = image.shape
        if channels == 1:
            channel_map = [("Y", 0)]
        elif channels == 3:
            channel_map = [("B", 2), ("G", 1), ("R", 0)]
        elif channels == 4:
            channel_map = [("A", 3), ("B", 2), ("G", 1), ("R", 0)]
        else:
            raise ValueError(f"EXR does not support {channels} channels in this node.")

        channel_list = b"".join(
            name.encode("ascii")
            + b"\x00"
            + struct.pack("<iB3xii", 1, 0, 1, 1)
            for name, _ in channel_map
        ) + b"\x00"

        data_window = struct.pack("<iiii", 0, 0, width - 1, height - 1)
        header = b"".join(
            (
                cls._exr_attribute("channels", "chlist", channel_list),
                cls._exr_attribute("compression", "compression", b"\x00"),
                cls._exr_attribute("dataWindow", "box2i", data_window),
                cls._exr_attribute("displayWindow", "box2i", data_window),
                cls._exr_attribute("lineOrder", "lineOrder", b"\x00"),
                cls._exr_attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
                cls._exr_attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
                cls._exr_attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
            )
        ) + b"\x00"

        prefix = struct.pack("<II", 20000630, 2) + header
        bytes_per_row = width * len(channel_map) * 2
        first_chunk_offset = len(prefix) + (height * 8)
        chunk_size = 8 + bytes_per_row
        offsets = b"".join(
            struct.pack("<Q", first_chunk_offset + row * chunk_size)
            for row in range(height)
        )

        with open(path, "wb") as exr_file:
            exr_file.write(prefix)
            exr_file.write(offsets)
            for row in range(height):
                row_data = b"".join(
                    np.ascontiguousarray(image[row, :, index]).astype("<f2").tobytes()
                    for _, index in channel_map
                )
                exr_file.write(struct.pack("<iI", row, len(row_data)))
                exr_file.write(row_data)

    @staticmethod
    def _srgb_to_linear(image):
        converted = image.copy()
        color_channels = min(3, converted.shape[2])
        color = converted[..., :color_channels]
        mask = color <= 0.04045
        linear = np.empty_like(color, dtype=np.float32)
        linear[mask] = color[mask] / 12.92
        linear[~mask] = np.power((color[~mask] + 0.055) / 1.055, 2.4)
        converted[..., :color_channels] = linear
        return converted

    @staticmethod
    def _write_pillow(path, image, image_format):
        try:
            from PIL import Image

            pixels = np.rint(image * 255.0).astype(np.uint8)
            mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(pixels.shape[2])
            if mode is None:
                raise ValueError(
                    f"{image_format.upper()} does not support {pixels.shape[2]} channels in this node."
                )
            if mode == "L":
                pixels = pixels[..., 0]

            save_options = {"quality": 100}
            if image_format == "webp":
                save_options["lossless"] = True
            Image.fromarray(pixels, mode=mode).save(
                path, format=image_format.upper(), **save_options
            )
        except Exception as error:
            raise RuntimeError(
                f"Unable to save {image_format.upper()}. "
                "Check that the installed Pillow build supports this format."
            ) from error

    @classmethod
    def _save_image(cls, path, image, image_format, bit_depth, input_color_space):
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)

        if image_format == "png":
            cls._write_png(path, image, bit_depth)
            return

        if image_format == "exr":
            if bit_depth != "16bit":
                raise ValueError("EXR requires bit_depth to be set to 16bit.")
            if input_color_space == "sRGB":
                image = cls._srgb_to_linear(image)
            cls._write_exr(path, image)
            return

        if bit_depth != "8bit":
            raise ValueError(f"{image_format.upper()} supports only 8bit in this node.")
        cls._write_pillow(path, image, image_format)

    @staticmethod
    def _tensor_to_numpy(image):
        array = image.detach().cpu().numpy()
        if array.ndim != 3:
            raise ValueError("Each IMAGE batch item must have shape [height, width, channels].")
        if array.shape[2] > 4:
            array = array[..., :4]
        return array

    def preview_image(
        self,
        images,
        filename_prefix="DKST_Image ",
        format="png",
        bit_depth="8bit",
        input_color_space="sRGB",
        always_save=False,
    ):
        height, width = int(images.shape[1]), int(images.shape[2])
        target_dir = self.output_dir if always_save else self.temp_dir
        image_type = "output" if always_save else "temp"
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, target_dir, width, height
        )

        ui_images = []
        for batch_index, tensor_image in enumerate(images):
            image = self._tensor_to_numpy(tensor_image)
            current_counter = counter + batch_index
            saved_name = f"{filename}_{current_counter:05}_.{format}"
            saved_path = os.path.join(full_output_folder, saved_name)
            self._save_image(saved_path, image, format, bit_depth, input_color_space)

            if format == "exr":
                preview_folder, preview_name, preview_counter, preview_subfolder, _ = (
                    folder_paths.get_save_image_path(
                        f"preview_{filename_prefix}", self.temp_dir, width, height
                    )
                )
                preview_filename = f"{preview_name}_{preview_counter + batch_index:05}_.png"
                preview_path = os.path.join(preview_folder, preview_filename)
                preview_image = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
                self._write_png(preview_path, preview_image, "8bit")
                ui_images.append(
                    {
                        "filename": preview_filename,
                        "subfolder": preview_subfolder,
                        "type": "temp",
                    }
                )
            else:
                ui_images.append(
                    {"filename": saved_name, "subfolder": subfolder, "type": image_type}
                )

        return {
            "ui": {
                "images": ui_images,
                "resolution": [f"{width} × {height}"],
            },
            "result": (images,),
        }


NODE_CLASS_MAPPINGS = {
    "DINKI_Preview_Image": DINKI_Preview_Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_Preview_Image": "DKST Preview (Image)",
}
