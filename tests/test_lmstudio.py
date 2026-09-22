"""Request-boundary tests without a running ComfyUI or LM Studio server."""
import runpy
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"
Switch = runpy.run_path(str(ROOT / "dinki_switch.py"))["DINKI_String_Switch_RT"]


class HTTPError(Exception):
    pass


class LMStudioTests(unittest.TestCase):
    def setUp(self):
        self.requests = MagicMock()
        self.requests.HTTPError = HTTPError
        with patch.dict("sys.modules", {
            "requests": self.requests, "numpy": MagicMock(), "PIL": MagicMock()
        }):
            self.node = runpy.run_path(str(ROOT / "dinki_lmstudio.py"))["DINKI_LMStudio"]()
        self.requests.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "answer"}}]
        }
        self.requests.post.return_value.headers = {"Content-Type": "application/json"}

    def test_slash_model_survives_switch_and_request(self):
        options = "gemma\nqwen/qwen3.8-27b"
        for selection in ("qwen/qwen3.8-27b", "qwen3.8-27b"):
            model, = Switch().switch_and_combine(selection, options)
            self.assertEqual(self.node.run(model_key=model, max_tokens=131072), ("answer",))
            body = self.requests.post.call_args.kwargs["json"]
            self.assertEqual(body["model"], "qwen/qwen3.8-27b")
            self.assertEqual(body["max_tokens"], 131072)

    def test_duplicate_leaf_is_not_guessed_and_literal_match_wins(self):
        self.assertEqual(Switch().switch_and_combine("model", "a/model\nb/model"), ("model",))
        self.assertEqual(Switch().switch_and_combine("model", "model\na/model"), ("model",))

    def test_zero_uses_server_default(self):
        self.node.run(max_tokens=0)
        self.assertNotIn("max_tokens", self.requests.post.call_args.kwargs["json"])
        config = self.node.INPUT_TYPES()["optional"]["max_tokens"][1]
        self.assertEqual(config["min"], 0)
        self.assertGreaterEqual(config["max"], 131072)

    def test_server_error_details_are_returned(self):
        response = self.requests.post.return_value
        response.raise_for_status.side_effect = HTTPError("400 Bad Request")
        response.text = '{"error":"model not found"}'
        result, = self.node.run()
        self.assertIn("400 Bad Request", result)
        self.assertIn("model not found", result)

    def test_api_key_authenticates_generation_and_unload(self):
        with patch("time.sleep"):
            self.node.run(api_key=" test-key ", auto_unload=True, unload_delay=1)
        self.assertEqual(self.requests.post.call_count, 3)
        for call in self.requests.post.call_args_list:
            self.assertEqual(call.kwargs["headers"], {"Authorization": "Bearer test-key"})
            self.assertNotIn("api_key", call.kwargs["json"])
        self.assertIn("api_key", self.node.INPUT_TYPES()["optional"])

    def test_empty_key_omits_authorization(self):
        for key in ("", "   "):
            self.node.run(api_key=key)
            self.assertEqual(self.requests.post.call_args.kwargs["headers"], {})

    def test_stream_finishes_without_waiting_for_connection_close(self):
        response = self.requests.post.return_value
        response.headers = {"Content-Type": "text/event-stream; charset=utf-8"}
        def lines():
            yield b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"private"}}]}'
            yield 'data: {"choices":[{"index":0,"delta":{"content":"가사"}}]}'.encode()
            yield b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            raise AssertionError("Must not wait beyond completion")
        response.iter_lines.return_value = lines()
        self.assertEqual(self.node.run(), ("가사",))
        response.close.assert_called_once()
        self.assertTrue(self.requests.post.call_args.kwargs["json"]["stream"])

    def test_stream_done_marker_and_truncated_stream(self):
        response = self.requests.post.return_value
        response.headers = {"Content-Type": "text/event-stream"}
        response.iter_lines.return_value = iter([b'data: [DONE]'])
        self.assertEqual(self.node.run(), ("",))
        response.iter_lines.return_value = iter([])
        self.assertIn("without a completion marker", self.node.run()[0])


if __name__ == "__main__":
    unittest.main()
