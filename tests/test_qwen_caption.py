"""Real photo encoding/profile persistence with a mocked LLM HTTP transport."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import runpy
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"
spec = importlib.util.spec_from_file_location("qwen_caption_tests", ROOT / "dinki_qwen_caption.py")
caption = importlib.util.module_from_spec(spec)
spec.loader.exec_module(caption)


class RequestFailure(Exception):
    pass


class TestQwenCaptions(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.photo = self.root / "portrait.png"
        Image.new("RGBA", (1600, 3200), (100, 120, 140, 255)).save(self.photo)
        server = MagicMock()
        for method in ("get", "post"):
            getattr(server.PromptServer.instance.routes, method).side_effect = lambda path: lambda fn: fn
        self.http = types.SimpleNamespace(post=MagicMock(), RequestException=RequestFailure)
        with patch.dict("sys.modules", {"server": server,
                "folder_paths": types.SimpleNamespace(get_user_directory=lambda: self.root),
                "requests": self.http, "aiohttp": types.SimpleNamespace(web=MagicMock())}):
            self.connections = types.SimpleNamespace(**runpy.run_path(str(ROOT / "dinki_llm_connections.py")))
        self.connections.save({"alias": "Home", "ip_address": "192.0.2.5", "port": 1234, "api_key": "private-test-key"})
        profile_patch = patch.object(caption, "profiles", return_value=self.connections)
        profile_patch.start()
        self.addCleanup(profile_patch.stop)
        self.service = types.SimpleNamespace(resolve_image=MagicMock(return_value=self.photo))
        self.data = {"caption_connection_alias": "Home", "caption_model": "my-vlm", "image": {"filename": "portrait.png"}}
        self.response = self.http.post.return_value.__enter__.return_value
        self.response.status_code = 200
        self.response.json.return_value = {"choices": [{"message": {"content": "a person in a blue coat, side profile"}}]}

    def generate(self, data=None):
        with patch.dict("sys.modules", {"requests": self.http}):
            return caption.generate_caption(self.service, self.data if data is None else data)

    def test_uses_saved_alias_auth_and_manual_model_with_resized_photo(self):
        result = self.generate()
        self.assertEqual(result["caption"], "a person in a blue coat, side profile")
        args = self.http.post.call_args
        self.assertEqual(args.args[0], "http://192.0.2.5:1234/v1/chat/completions")
        self.assertEqual(args.kwargs["headers"], {"Authorization": "Bearer private-test-key"})
        self.assertFalse(args.kwargs["allow_redirects"])
        body = args.kwargs["json"]
        self.assertEqual(body["model"], "my-vlm")
        self.assertFalse(body["stream"])
        self.assertNotIn("private-test-key", json.dumps(body))
        self.assertNotIn("192.0.2.5", json.dumps(result))
        image_url = body["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/jpeg;base64,"))
        with Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))) as photo:
            self.assertEqual(photo.size, (512, 1024))
            self.assertEqual(photo.mode, "RGB")
        self.assertEqual(self.connections.aliases(), ["Home"])

    def test_optional_unconfigured_or_invalid_model_never_sends_a_request(self):
        for changes in ({"caption_connection_alias": caption.NONE}, {"caption_connection_alias": "Removed"},
                        {"caption_model": ""}, {"caption_model": "model\nheader"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.generate({**self.data, **changes})
        self.http.post.assert_not_called()
        self.service.resolve_image.assert_not_called()

    def test_http_failures_and_exceptions_do_not_expose_url_or_credentials(self):
        self.response.status_code = 401
        self.response.text = "private-test-key at 192.0.2.5"
        with self.assertRaisesRegex(ValueError, "HTTP 401") as error:
            self.generate()
        self.assertNotIn("private-test-key", str(error.exception))
        self.http.post.side_effect = RequestFailure("http://192.0.2.5 private-test-key")
        with self.assertRaises(ValueError) as error:
            self.generate()
        self.assertNotIn("192.0.2.5", str(error.exception))
        self.assertNotIn("private-test-key", str(error.exception))

    def test_invalid_responses_fail_and_reasoning_is_removed(self):
        for text in ("", None, "<think>unfinished reasoning", "x" * 4001):
            self.response.json.return_value = {"choices": [{"message": {"content": text}}]}
            with self.subTest(text=type(text)), self.assertRaises(ValueError):
                self.generate()
        self.response.json.return_value = {"error": "private-test-key"}
        with self.assertRaises(ValueError) as error:
            self.generate()
        self.assertNotIn("private-test-key", str(error.exception))
        self.response.json.return_value = {"choices": [{"message": {"content": '<think>reasoning</think>\n"a person waving"'}}]}
        self.assertEqual(self.generate(), {"caption": "a person waving"})

    def test_invalid_or_missing_photo_is_rejected_before_http(self):
        self.service.resolve_image.side_effect = ValueError("Gallery images must exist inside ComfyUI directories.")
        with self.assertRaises(ValueError):
            self.generate()
        self.service.resolve_image.side_effect = None
        self.photo.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "Cannot read"):
            self.generate()
        self.http.post.assert_not_called()

    def test_broken_profile_configuration_does_not_break_optional_schema(self):
        self.connections.profile_path().write_text("broken JSON")
        self.assertEqual(caption.caption_aliases(), [caption.NONE])


if __name__ == "__main__":
    unittest.main()
