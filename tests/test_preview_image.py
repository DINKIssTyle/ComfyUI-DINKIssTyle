import runpy
import struct
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"


class PreviewImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder_paths = MagicMock()
        folder_paths.get_output_directory.return_value = "/output"
        folder_paths.get_temp_directory.return_value = "/temp"
        with patch.dict(sys.modules, {"folder_paths": folder_paths, "numpy": MagicMock()}):
            module = runpy.run_path(str(ROOT / "dinki_preview.py"))
        cls.folder_paths = folder_paths
        cls.node_class = module["DINKI_Preview_Image"]

    def test_declares_requested_inputs_and_defaults(self):
        required = self.node_class.INPUT_TYPES()["required"]
        self.assertEqual(required["filename_prefix"][1]["default"], "DKST_Image ")
        self.assertEqual(required["format"][0], ["png", "exr", "avif", "webp"])
        self.assertEqual(required["bit_depth"][0], ["8bit", "16bit"])
        self.assertEqual(required["input_color_space"][0], ["sRGB"])
        self.assertFalse(required["always_save"][1]["default"])
        self.assertFalse(required["meta_save"][1]["default"])
        for name in ("filename_prefix", "format", "bit_depth", "input_color_space"):
            self.assertTrue(required[name][1]["advanced"])
        text = self.node_class.INPUT_TYPES()["optional"]["metadata_text"][1]
        self.assertTrue(text["forceInput"])
        self.assertNotIn("multiline", text)

    def test_is_image_output_node(self):
        self.assertEqual(self.node_class.RETURN_TYPES, ("IMAGE",))
        self.assertTrue(self.node_class.OUTPUT_NODE)

    def test_display_name_registration(self):
        with patch.dict(sys.modules, {"folder_paths": MagicMock(), "numpy": MagicMock()}):
            module = runpy.run_path(str(ROOT / "dinki_preview.py"))
        self.assertEqual(
            module["NODE_DISPLAY_NAME_MAPPINGS"]["DINKI_Preview_Image"],
            "DKST Image (Viewer)",
        )

    def test_uses_temp_or_output_directory_and_reports_resolution(self):
        class FakeImages(list):
            shape = (1, 720, 1280, 3)

        images = FakeImages([object()])
        for always_save, expected_directory, expected_type in (
            (False, "/temp", "temp"),
            (True, "/output", "output"),
        ):
            with self.subTest(always_save=always_save):
                self.folder_paths.get_save_image_path.reset_mock()
                self.folder_paths.get_save_image_path.return_value = (
                    "/target",
                    "DKST_Image",
                    1,
                    "",
                    "DKST_Image",
                )
                node = self.node_class()
                with (
                    patch.object(node, "_tensor_to_numpy", return_value=object()),
                    patch.object(node, "_save_image"),
                ):
                    result = node.preview_image(images, always_save=always_save)

                call = self.folder_paths.get_save_image_path.call_args
                self.assertEqual(call.args[1], expected_directory)
                self.assertEqual(result["ui"]["dkst_images"][0]["type"], expected_type)
                self.assertEqual(result["ui"]["resolution"], ["1280 × 720"])
                self.assertIs(result["result"][0], images)


class Tensor:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape

    def __iter__(self):
        return (Tensor(image) for image in self.array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class PreviewMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("temp", "output"):
            (self.root / name).mkdir()
        self.folder_paths = types.SimpleNamespace(
            get_output_directory=lambda: str(self.root / "output"),
            get_temp_directory=lambda: str(self.root / "temp"),
            get_save_image_path=MagicMock(side_effect=lambda prefix, directory, width, height:
                                         (directory, prefix, 1, "", prefix)),
        )
        self.args = types.SimpleNamespace(disable_metadata=False)
        modules = patch.dict(sys.modules, {
            "folder_paths": self.folder_paths,
            "comfy.cli_args": types.SimpleNamespace(args=self.args),
        })
        modules.start()
        self.addCleanup(modules.stop)
        self.node = runpy.run_path(str(ROOT / "dinki_preview.py"))["DINKI_Preview_Image"]()
        self.images = Tensor(np.linspace(0, 1, 2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4))
        self.text = "  한글 프롬프트 🌅\n{sunrise|sunset}\r\n두 번째 줄\t  "

    @staticmethod
    def chunks(path):
        data = path.read_bytes()
        offset, chunks = 8, {}
        while offset < len(data):
            size = struct.unpack_from(">I", data, offset)[0]
            kind = data[offset + 4:offset + 8]
            payload = data[offset + 8:offset + 8 + size]
            crc = struct.unpack_from(">I", data, offset + 8 + size)[0]
            if crc != zlib.crc32(kind + payload) & 0xFFFFFFFF:
                raise AssertionError("Invalid PNG CRC")
            chunks.setdefault(kind, []).append(payload)
            offset += size + 12
        return chunks

    def test_unicode_text_roundtrip_in_temp_and_output_batches_at_both_depths(self):
        for always_save in (False, True):
            for depth in ("8bit", "16bit"):
                with self.subTest(always_save=always_save, depth=depth):
                    result = self.node.preview_image(
                        self.images, bit_depth=depth, always_save=always_save,
                        meta_save=True, metadata_text=self.text,
                    )
                    self.assertIs(result["result"][0], self.images)
                    self.assertEqual(len(result["ui"]["dkst_images"]), 2)
                    for index, descriptor in enumerate(result["ui"]["dkst_images"]):
                        directory = "output" if always_save else "temp"
                        self.assertEqual(descriptor["type"], directory)
                        path = self.root / directory / descriptor["filename"]
                        with Image.open(path) as image:
                            image.load()
                            self.assertEqual(image.info["dkst_text"], self.text)
                            self.assertEqual(image.info["Description"], self.text)
                            self.assertEqual(image.info["Comment"], self.text)
                            self.assert_xmp_text(image.info["XML:com.adobe.xmp"], self.text)
                            self.assertNotIn("prompt", image.info)
                            self.assertNotIn("workflow", image.info)
                        chunks = self.chunks(path)
                        self.assertTrue(chunks[b"iTXt"][0].startswith(b"XML:com.adobe.xmp\0\0\0\0\0"))
                        # The XMP chunk starts immediately after the 25-byte
                        # IHDR chunk and 8-byte PNG signature.
                        self.assertEqual(path.read_bytes()[37:41], b"iTXt")
                        header = struct.unpack(">IIBBBBB", chunks[b"IHDR"][0])
                        self.assertEqual(header[:4], (4, 3, 16 if depth == "16bit" else 8, 6))
                        scale, dtype = (65535, ">u2") if depth == "16bit" else (255, "u1")
                        pixels = np.rint(self.images.array[index] * scale).astype(dtype)
                        expected = b"".join(b"\x00" + row.tobytes() for row in pixels)
                        self.assertEqual(zlib.decompress(b"".join(chunks[b"IDAT"])), expected)

    def test_off_or_empty_save_no_text_with_either_host_setting(self):
        baseline = self.node.preview_image(self.images)
        path = self.root / "temp" / baseline["ui"]["dkst_images"][0]["filename"]
        original = path.read_bytes()
        for switch, text, disabled in (
            (False, self.text, False),
            (True, "", False),
            (False, self.text, True),
            (True, "", True),
        ):
            with self.subTest(meta_save=switch, empty=not text, disabled=disabled):
                self.args.disable_metadata = disabled
                self.node.preview_image(self.images, meta_save=switch, metadata_text=text)
                self.assertEqual(path.read_bytes(), original)
                self.assertNotIn(b"iTXt", self.chunks(path))
                with Image.open(path) as image:
                    self.assertNotIn("dkst_text", image.info)
                    self.assertNotIn("Description", image.info)
                    self.assertNotIn("Comment", image.info)
                    self.assertNotIn("XML:com.adobe.xmp", image.info)

    def test_explicit_text_is_saved_when_host_disables_automatic_metadata(self):
        # Production runs with --disable-metadata to omit automatic workflow
        # metadata. The Viewer switch must still save the supplied prompt.
        self.args.disable_metadata = True
        for always_save in (False, True):
            with self.subTest(always_save=always_save):
                result = self.node.preview_image(self.images, always_save=always_save,
                                                 meta_save=True, metadata_text=self.text)
                directory = "output" if always_save else "temp"
                for descriptor in result["ui"]["dkst_images"]:
                    path = self.root / directory / descriptor["filename"]
                    with Image.open(path) as image:
                        self.assert_xmp_text(image.info["XML:com.adobe.xmp"], self.text)
                        for key in ("dkst_text", "Description", "Comment"):
                            self.assertEqual(image.info[key], self.text)
                        self.assertNotIn("prompt", image.info)
                        self.assertNotIn("workflow", image.info)

    def assert_xmp_text(self, packet, text):
        root = ET.fromstring(packet)
        namespaces = {
            "x": "adobe:ns:meta/",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "dc": "http://purl.org/dc/elements/1.1/",
        }
        self.assertEqual(root.tag, "{adobe:ns:meta/}xmpmeta")
        values = root.findall("rdf:RDF/rdf:Description/dc:description/rdf:Alt/rdf:li", namespaces)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0].get("{http://www.w3.org/XML/1998/namespace}lang"), "x-default")
        self.assertEqual(values[0].text, text)

    def test_xmp_preserves_special_characters_and_literal_prompt_structure(self):
        text = '  한글 🌅 <rdf:li> & "quotes" \'apostrophe\'\r\n\r\n{red|blue}\t\r끝  '
        for depth in ("8bit", "16bit"):
            with self.subTest(depth=depth):
                result = self.node.preview_image(self.images, bit_depth=depth,
                                                 meta_save=True, metadata_text=text)
                path = self.root / "temp" / result["ui"]["dkst_images"][0]["filename"]
                with Image.open(path) as image:
                    self.assert_xmp_text(image.info["XML:com.adobe.xmp"], text)
                    self.assertEqual(image.info["dkst_text"], text)

    def test_non_png_metadata_reports_error_before_any_files_are_written(self):
        for image_format in ("exr", "avif", "webp"):
            with self.subTest(format=image_format):
                with self.assertRaisesRegex(ValueError, "PNG only"):
                    self.node.preview_image(self.images, format=image_format,
                                            meta_save=True, metadata_text=self.text)
        self.folder_paths.get_save_image_path.assert_not_called()
        self.assertEqual(list((self.root / "temp").iterdir()), [])

    def test_rgb_and_grayscale_png_metadata_preserves_pixels(self):
        for channels in (1, 3):
            with self.subTest(channels=channels):
                array = self.images.array[0, :, :, :channels]
                path = self.root / "temp" / f"channels_{channels}.png"
                self.node._save_image(path, array, "png", "8bit", "sRGB", metadata_text=self.text)
                with Image.open(path) as image:
                    expected = np.rint(array * 255).astype(np.uint8)
                    if channels == 1:
                        expected = expected[:, :, 0]
                    np.testing.assert_array_equal(np.asarray(image), expected)
                    self.assertEqual(image.info["dkst_text"], self.text)


if __name__ == "__main__":
    unittest.main()
