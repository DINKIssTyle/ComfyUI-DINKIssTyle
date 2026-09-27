import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"


class TinyTensor:
    shape = (1, 2, 2, 3)

    def __getitem__(self, index):
        return self

    def detach(self):
        return self

    def to(self, **kwargs):
        return self

    def clone(self):
        return self


class ColorNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.xmp_dir = Path(cls.temp.name) / "adobe_xmp"
        folder_paths = MagicMock()
        folder_paths.get_input_directory.return_value = cls.temp.name
        folder_paths.get_filename_list.side_effect = lambda kind: [
            path.name for path in (cls.xmp_dir if kind == "adobe_xmp" else Path(cls.temp.name) / "luts").iterdir()
        ]
        folder_paths.get_folder_paths.return_value = [str(cls.xmp_dir)]
        folder_paths.get_full_path.side_effect = lambda kind, name: str(cls.xmp_dir / name)
        server = MagicMock()
        server.PromptServer.instance.routes.post.side_effect = lambda path: lambda function: function
        torch = MagicMock()
        with patch.dict(sys.modules, {
            "folder_paths": folder_paths,
            "numpy": MagicMock(),
            "torch": torch,
            "torch.nn": MagicMock(),
            "torch.nn.functional": MagicMock(),
            "PIL": MagicMock(),
            "server": server,
            "aiohttp": MagicMock(),
        }):
            module = runpy.run_path(str(ROOT / "dinki_color.py"))
        cls.node_class = module["DINKI_adobe_xmp"]
        cls.preview_class = module["DINKI_Adobe_XMP_Preview"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def write_xmp(self, name, body):
        path = self.xmp_dir / name
        path.write_text(body, encoding="utf-8")
        return path

    def test_xmp_parses_attributes_child_values_and_curves(self):
        path = self.write_xmp("valid.xmp", '''<x:xmpmeta xmlns:x="adobe:ns:meta/"
            xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
            xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
            <rdf:RDF><rdf:Description crs:Exposure2012="0.75" crs:HueAdjustmentBlue="-12">
            <crs:Vibrance>24</crs:Vibrance>
            <crs:ToneCurvePV2012><rdf:Seq><rdf:li>0, 0</rdf:li>
            <rdf:li>255, 240</rdf:li></rdf:Seq></crs:ToneCurvePV2012>
            </rdf:Description></rdf:RDF></x:xmpmeta>''')
        values = self.node_class().parse_xmp(path)
        self.assertEqual(values["Exposure"], 0.75)
        self.assertEqual(values["Vibrance"], 24)
        self.assertEqual(values["HSL_Hue"]["Blue"], -12)
        self.assertEqual(values["ToneCurve"], [(0, 0), (255, 240)])

    def test_invalid_curve_and_dtd_are_rejected(self):
        duplicate = self.write_xmp("duplicate.xmp", '''<rdf:RDF
            xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
            xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
            <rdf:Description><crs:ToneCurve><rdf:Seq><rdf:li>10, 0</rdf:li>
            <rdf:li>10, 100</rdf:li></rdf:Seq></crs:ToneCurve></rdf:Description>
            </rdf:RDF>''')
        with self.assertRaisesRegex(ValueError, "distinct X"):
            self.node_class().parse_xmp(duplicate)
        dtd = self.write_xmp("entity.xmp", '''<!DOCTYPE x [<!ENTITY bomb "test">]>
            <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description/></rdf:RDF>''')
        with self.assertRaisesRegex(ValueError, "DTD"):
            self.node_class().parse_xmp(dtd)

    def test_utf16_preset_is_accepted_and_utf16_dtd_is_rejected(self):
        path = self.xmp_dir / "utf16.xmp"
        valid = '''<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
            xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
            <rdf:Description crs:Exposure2012="1"/></rdf:RDF>'''
        path.write_bytes(valid.encode("utf-16"))
        self.assertEqual(self.node_class().parse_xmp(path)["Exposure"], 1)
        path.write_bytes(("<!DOCTYPE x>" + valid).encode("utf-16"))
        with self.assertRaisesRegex(ValueError, "DTD"):
            self.node_class().parse_xmp(path)

    def test_file_changes_invalidate_comfy_cache_and_escape_is_rejected(self):
        path = self.write_xmp("changed.xmp", '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description/></rdf:RDF>')
        before = self.node_class.IS_CHANGED("changed.xmp")
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        self.assertNotEqual(before, self.node_class.IS_CHANGED("changed.xmp"))
        self.assertNotEqual(self.node_class.VALIDATE_INPUTS("../changed.xmp"), True)

    def test_parser_cache_refreshes_when_contents_change_at_same_length(self):
        prefix = '''<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
            xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
            <rdf:Description crs:Exposure2012="'''
        suffix = '''"/></rdf:RDF>'''
        path = self.write_xmp("cached.xmp", prefix + "1" + suffix)
        self.assertEqual(self.node_class().parse_xmp(path)["Exposure"], 1)
        path.write_text(prefix + "2" + suffix, encoding="utf-8")
        self.assertEqual(self.node_class().parse_xmp(path)["Exposure"], 2)

    def test_preview_cache_uses_distinct_tokens(self):
        first = TinyTensor()
        second = TinyTensor()
        with patch.object(self.preview_class, "apply_preset", return_value=("processed",)):
            first_token = self.preview_class().apply_preset_preview(first, "-- None --", 1.0)["ui"]["preview_token"][0]
            second_token = self.preview_class().apply_preset_preview(second, "-- None --", 1.0)["ui"]["preview_token"][0]
        self.assertNotEqual(first_token, second_token)
        self.assertIs(self.preview_class._preview_inputs[first_token], first)
        self.assertIs(self.preview_class._preview_inputs[second_token], second)


if __name__ == "__main__":
    unittest.main()
