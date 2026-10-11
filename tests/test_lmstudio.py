"""Request-boundary tests without a running ComfyUI or LM Studio server."""
import runpy
import unittest
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"
with patch.dict("sys.modules", {"comfy_execution": types.ModuleType("comfy_execution"),
                               "comfy_execution.graph_utils": MagicMock()}):
    Switch = runpy.run_path(str(ROOT / "dinki_switch.py"))["DINKI_String_Switch_RT"]


class HTTPError(Exception):
    pass


class LMStudioTests(unittest.TestCase):
    def setUp(self):
        self.requests = MagicMock()
        self.requests.HTTPError = HTTPError
        self.connections = types.SimpleNamespace(
            NONE="-- Select connection --", aliases=lambda: ["Local"],
            resolve=MagicMock(return_value={"ip_address": "127.0.0.1", "port": 1234, "protocol": "http", "api_key": "", "revision": "r1"}),
            revision=lambda alias: "r1",
            connection_url=lambda profile: f'http://{profile["ip_address"]}:{profile["port"]}',
            connection_headers=lambda profile: {"Authorization": f'Bearer {profile["api_key"]}'} if profile["api_key"] else {},
        )
        with patch.dict("sys.modules", {
            "requests": self.requests, "numpy": MagicMock(), "PIL": MagicMock(),
            "dkst_test": types.ModuleType("dkst_test"),
            "dkst_test.dinki_llm_connections": self.connections
        }):
            self.node = runpy.run_path(str(ROOT / "dinki_lmstudio.py"), run_name="dkst_test.dinki_lmstudio")["DINKI_LMStudio"]()
        self.requests.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "answer"}}]
        }
        self.requests.post.return_value.status_code = 200
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

    def test_server_error_details_and_address_are_not_returned(self):
        response = self.requests.post.return_value
        response.raise_for_status.side_effect = HTTPError("400 Bad Request")
        response.status_code = 400
        response.text = '{"error":"secret-key 192.168.1.10 model not found"}'
        result, = self.node.run()
        self.assertIn("HTTP 400", result)
        self.assertNotIn("secret-key", result)
        self.assertNotIn("192.168.1.10", result)

    def test_api_key_authenticates_generation_and_unload(self):
        self.connections.resolve.return_value["api_key"] = "test-key"
        with patch("time.sleep"):
            self.node.run(connection_alias="Local", auto_unload=True, unload_delay=1)
        self.assertEqual(self.requests.post.call_count, 3)
        for call in self.requests.post.call_args_list:
            self.assertEqual(call.kwargs["headers"], {"Authorization": "Bearer test-key"})
            self.assertNotIn("api_key", call.kwargs["json"])
        inputs = self.node.INPUT_TYPES()["optional"]
        self.assertEqual(inputs["connection_alias"][0], ["-- Select connection --", "Local"])
        for name in ("api_key", "ip_address", "port"):
            self.assertNotIn(name, inputs)

    def test_empty_key_omits_authorization(self):
        self.connections.resolve.return_value["api_key"] = ""
        self.node.run(connection_alias="Local")
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
        self.assertIn("incomplete response", self.node.run()[0])

    def test_missing_connection_and_passthrough_never_send_requests(self):
        self.connections.resolve.side_effect = ValueError("Select a registered connection.")
        self.assertIn("Select a registered connection", self.node.run()[0])
        self.assertEqual(self.node.run(assistant_enabled=False, user_prompt="hello"), ("hello",))
        self.requests.post.assert_not_called()

    def test_request_errors_do_not_disclose_url_or_key(self):
        self.requests.post.side_effect = RuntimeError("http://private-host:1234 secret-key")
        result = self.node.run()[0]
        self.assertNotIn("private-host", result)
        self.assertNotIn("secret-key", result)

    def test_redirect_is_rejected_and_auth_is_never_forwarded(self):
        self.requests.post.return_value.status_code = 302
        self.assertIn("redirected", self.node.run()[0])
        self.assertFalse(self.requests.post.call_args.kwargs["allow_redirects"])
        self.requests.post.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
