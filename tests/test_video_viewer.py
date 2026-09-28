import ast
import runpy
import sys
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"


class VideoContainer(str, Enum):
    MP4 = "mp4"
    MKV = "mkv"
    WEBM = "webm"

    @classmethod
    def get_extension(cls, value):
        return value


class VideoCodec(str, Enum):
    AUTO = "auto"
    H264 = "h264"
    AV1 = "av1"


class FakeVideo:
    def __init__(self):
        self.saves = []

    def get_dimensions(self):
        return 1920, 1080

    def save_to(self, path, **options):
        self.saves.append((path, options))


class VideoViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = MagicMock()
        cls.paths.get_output_directory.return_value = "/output"
        cls.paths.get_temp_directory.return_value = "/temp"
        with patch.dict(sys.modules, {"folder_paths": cls.paths}):
            module = runpy.run_path(str(ROOT / "dinki_viewer.py"))
        cls.sequence_class = module["DINKI_Video_Player"]
        cls.viewer_class = module["DINKI_Video_Viewer"]

    def setUp(self):
        self.paths.get_save_image_path.reset_mock()

        def save_path(prefix, directory, width, height):
            return directory, prefix, 3, "", prefix

        self.paths.get_save_image_path.side_effect = save_path
        latest = SimpleNamespace(Types=SimpleNamespace(
            VideoContainer=VideoContainer, VideoCodec=VideoCodec
        ))
        class Container:
            streams = SimpleNamespace(video=[SimpleNamespace(codec=SimpleNamespace(name="hevc"))])

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        self.av = SimpleNamespace(open=MagicMock(return_value=Container()))
        self.modules = patch.dict(sys.modules, {
            "folder_paths": self.paths,
            "av": self.av,
            "comfy_api": SimpleNamespace(latest=latest),
            "comfy_api.latest": latest,
            "comfy": SimpleNamespace(cli_args=SimpleNamespace(args=SimpleNamespace(disable_metadata=False))),
            "comfy.cli_args": SimpleNamespace(args=SimpleNamespace(disable_metadata=False)),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_existing_sequence_player_keeps_its_filename_input(self):
        self.assertEqual(self.sequence_class.INPUT_TYPES()["required"]["filename"][0], "STRING")
        self.assertEqual(self.sequence_class.RETURN_TYPES, ())

    def test_old_type_is_renamed_and_new_type_is_registered(self):
        tree = ast.parse((ROOT / "__init__.py").read_text())
        dictionaries = {}
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or not isinstance(statement.targets[0], ast.Name):
                continue
            if statement.targets[0].id not in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"):
                continue
            dictionaries[statement.targets[0].id] = {
                key.value: value.value if isinstance(value, ast.Constant) else value.id
                for key, value in zip(statement.value.keys, statement.value.values)
            }
        self.assertEqual(dictionaries["NODE_CLASS_MAPPINGS"]["DINKI_Video_Player"],
                         "DINKI_Video_Player")
        self.assertEqual(dictionaries["NODE_CLASS_MAPPINGS"]["DINKI_Video_Viewer"],
                         "DINKI_Video_Viewer")
        self.assertEqual(dictionaries["NODE_DISPLAY_NAME_MAPPINGS"]["DINKI_Video_Player"],
                         "DKST Video (Sequence Player)")
        self.assertEqual(dictionaries["NODE_DISPLAY_NAME_MAPPINGS"]["DINKI_Video_Viewer"],
                         "DKST Video (Video Player)")

    def test_native_video_is_an_output_node_with_requested_controls(self):
        required = self.viewer_class.INPUT_TYPES()["required"]
        self.assertEqual(required["video"][0], "VIDEO")
        self.assertEqual(required["filename_prefix"][1]["default"], "DKST_Video")
        self.assertEqual(required["format"][0], ["auto", "mp4", "mkv", "webm"])
        self.assertEqual(required["codec"][0], ["auto", "h264", "av1"])
        self.assertFalse(required["always_save"][1]["default"])
        self.assertEqual(self.viewer_class.RETURN_TYPES, ("VIDEO",))
        self.assertTrue(self.viewer_class.OUTPUT_NODE)

    def test_default_writes_temp_video_and_separate_browser_preview(self):
        video = FakeVideo()
        result = self.viewer_class().preview_video(video, prompt={"a": 1})
        self.assertIs(result["result"][0], video)
        self.assertEqual(result["ui"]["resolution"], ["1920 × 1080"])
        self.assertEqual(result["ui"]["dkst_video"], [{
            "filename": "DKST_Video_00003_.mp4", "subfolder": "", "type": "temp",
        }])
        self.assertEqual(len(video.saves), 2)
        self.assertEqual(video.saves[0][0], "/temp/DKST_Video_00003_.mp4")
        self.assertEqual(video.saves[0][1]["metadata"], {"prompt": {"a": 1}})
        self.assertEqual(video.saves[1][1]["codec"], VideoCodec.H264)
        self.assertEqual(result["ui"]["dkst_video_preview"][0]["type"], "temp")

    def test_output_h264_mp4_uses_saved_file_for_playback(self):
        video = FakeVideo()
        result = self.viewer_class().preview_video(
            video, format="mp4", codec="h264", always_save=True,
        )
        self.assertEqual(len(video.saves), 1)
        self.assertEqual(video.saves[0][0], "/output/DKST_Video_00003_.mp4")
        self.assertEqual(result["ui"]["dkst_video_preview"], result["ui"]["dkst_video"])
        self.assertEqual(result["ui"]["dkst_video"][0]["type"], "output")

    def test_auto_h264_mp4_reuses_the_saved_file_without_reencoding(self):
        self.av.open.return_value.streams.video[0].codec.name = "h264"
        video = FakeVideo()
        result = self.viewer_class().preview_video(video)
        self.assertEqual(len(video.saves), 1)
        self.assertEqual(result["ui"]["dkst_video_preview"], result["ui"]["dkst_video"])

    def test_auto_av1_uses_webm_original_and_mp4_playback(self):
        video = FakeVideo()
        result = self.viewer_class().preview_video(video, codec="av1", always_save=True)
        self.assertEqual(video.saves[0][0], "/output/DKST_Video_00003_.webm")
        self.assertEqual(video.saves[0][1]["format"], VideoContainer.WEBM)
        self.assertEqual(video.saves[0][1]["codec"], VideoCodec.AV1)
        self.assertEqual(result["ui"]["dkst_video_preview"][0]["filename"],
                         "preview_DKST_Video_00003_.mp4")

    def test_webm_h264_is_rejected_before_writing(self):
        video = FakeVideo()
        self.assertIsInstance(self.viewer_class.VALIDATE_INPUTS("webm", "h264"), str)
        with self.assertRaisesRegex(ValueError, "WebM"):
            self.viewer_class().preview_video(video, format="webm", codec="h264")
        self.assertEqual(video.saves, [])


if __name__ == "__main__":
    unittest.main()
