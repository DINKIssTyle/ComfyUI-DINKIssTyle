"""Request-boundary tests without a running ComfyUI or LM Studio server."""
import runpy
import json
import math
import unittest
import types
import numpy as np
from PIL import Image
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
            "requests": self.requests,
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
            self.assertEqual(self.node.run(model_key=model, max_tokens=131072)["result"], ("answer",))
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
        self.assertEqual(self.node.run()["result"], ("가사",))
        response.close.assert_called_once()
        self.assertTrue(self.requests.post.call_args.kwargs["json"]["stream"])

    def test_stream_done_marker_and_truncated_stream(self):
        response = self.requests.post.return_value
        response.headers = {"Content-Type": "text/event-stream"}
        response.iter_lines.return_value = iter([b'data: [DONE]'])
        self.assertEqual(self.node.run()["result"], ("",))
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

    def test_generation_modes_default_to_always_and_execution_is_never_skipped(self):
        schema = self.node.INPUT_TYPES()
        modes, options = schema["optional"]["generation_mode"]
        self.assertEqual(modes, ["Always Generate", "Generate on Change"])
        self.assertEqual(options["default"], "Always Generate")
        self.assertEqual(list(schema["optional"])[-2:], ["generation_mode", "cached_response"])
        self.assertEqual(schema["hidden"]["unique_id"], "UNIQUE_ID")
        self.assertTrue(math.isnan(self.node.IS_CHANGED(seed=42)))
        for _ in range(3):
            self.node.run(user_prompt="same", seed=42)
        self.assertEqual(self.requests.post.call_count, 3)

    def test_change_mode_ignores_fixed_and_random_seed_changes(self):
        for seed in (42, 42, -1, -1, 1234):
            result = self.node.run(generation_mode="Generate on Change", user_prompt="same", seed=seed)
            self.assertEqual(result["result"], ("answer",))
        self.assertEqual(self.requests.post.call_count, 1)

    def test_each_prompt_and_generation_setting_invalidates_the_last_answer(self):
        settings = dict(generation_mode="Generate on Change", user_prompt="A", system_prompt="system")
        self.node.run(**settings)
        changes = [("user_prompt", "B"), ("user_prompt", "A"), ("system_prompt", "changed"),
                   ("user_prompt", "A "), ("model_key", "other/model"),
                   ("temperature", 0.5), ("max_tokens", 500), ("connection_alias", "Other")]
        for index, (name, value) in enumerate(changes, start=2):
            settings[name] = value
            self.node.run(**settings)
            self.node.run(**settings)
            self.assertEqual(self.requests.post.call_count, index, name)
        self.connections.resolve.return_value["revision"] = "r2"
        self.node.run(**settings)
        self.assertEqual(self.requests.post.call_count, len(changes) + 2)

    def test_image_content_not_object_identity_controls_regeneration(self):
        image = np.zeros((2, 4, 4, 3), dtype=np.float32)
        image[1] = 0.5
        settings = dict(generation_mode="Generate on Change", user_prompt="caption")
        self.node.run(image=image, **settings)
        content = self.requests.post.call_args.kwargs["json"]["messages"][-1]["content"]
        self.assertEqual(len(content), 3)
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        with patch.object(self.node, "_convert_single_image_to_base64", side_effect=AssertionError("Must skip PNG encoding")):
            self.node.run(image=image.copy(), seed=99, **settings)
            self.assertEqual(self.requests.post.call_count, 1)
        changed = image.copy()
        changed[0, 0, 0, 0] = 0.0001  # Even changes below PNG's 8-bit quantization count.
        for index, value in enumerate([changed, changed[::-1], changed[:1],
                                        np.zeros((1, 2, 8, 3), dtype=np.float32), None, image], start=2):
            self.node.run(image=value, **settings)
            self.assertEqual(self.requests.post.call_count, index)

    def test_cache_hit_skips_unloading_and_delay_changes(self):
        self.node.run(generation_mode="Generate on Change")
        with patch("time.sleep", side_effect=AssertionError("Must not wait on a cache hit")):
            self.node.run(generation_mode="Generate on Change", seed=87, timeout_seconds=1,
                          auto_unload=True, unload_delay=60)
        self.assertEqual(self.requests.post.call_count, 1)

    def test_saved_answer_survives_recreation_without_serializing_inputs_or_credentials(self):
        result = self.node.run(generation_mode="Generate on Change", user_prompt="private prompt", unique_id="10:2")
        saved = result["ui"]["dkst_llm_response"][0]
        for secret in ("private prompt", "127.0.0.1", "api_key"):
            self.assertNotIn(secret, saved)
        restored = type(self.node)()
        self.assertEqual(restored.run(generation_mode="Generate on Change", user_prompt="private prompt",
                                     unique_id="10:2", seed=45, cached_response=saved)["result"], ("answer",))
        self.assertEqual(self.requests.post.call_count, 1)
        # Repeated local IDs in different subgraphs do not share their answers.
        restored.run(generation_mode="Generate on Change", user_prompt="private prompt",
                     unique_id="11:2", cached_response=saved)
        self.assertEqual(self.requests.post.call_count, 2)

    def test_stale_queued_snapshots_do_not_replay_an_older_cached_answer(self):
        mode = dict(generation_mode="Generate on Change")
        first = self.node.run(user_prompt="A", **mode)
        snapshot = first["ui"]["dkst_llm_response"][0]
        self.requests.post.return_value.json.return_value = {"choices": [{"message": {"content": "B answer"}}]}
        self.node.run(user_prompt="B", cached_response=snapshot, **mode)
        result = self.node.run(user_prompt="B", cached_response=snapshot, seed=32, **mode)
        self.assertEqual(result["result"], ("B answer",))
        self.assertEqual(self.requests.post.call_count, 2)
        self.node.run(user_prompt="A", cached_response=snapshot, **mode)
        self.assertEqual(self.requests.post.call_count, 3, "only the last successful input is retained")

    def test_empty_successful_answer_is_cached_but_errors_are_retried(self):
        self.requests.post.return_value.json.return_value = {"choices": [{"message": {"content": ""}}]}
        for _ in range(2):
            self.assertEqual(self.node.run(generation_mode="Generate on Change")["result"], ("",))
        self.assertEqual(self.requests.post.call_count, 1)
        previous = dict(self.node._last_response)
        self.requests.post.side_effect = RuntimeError("failed")
        for _ in range(2):
            self.assertIn("Error:", self.node.run(generation_mode="Generate on Change", user_prompt="new")[0])
        self.assertEqual(self.requests.post.call_count, 3)
        self.assertEqual(self.node._last_response, previous)
        self.assertEqual(self.node.run(generation_mode="Generate on Change")["result"], ("",))
        self.assertEqual(self.requests.post.call_count, 3)

    def test_passthrough_preserves_cache_and_missing_connection_does_not_reuse_it(self):
        self.node.run(generation_mode="Generate on Change", user_prompt="same")
        self.assertEqual(self.node.run(assistant_enabled=False, user_prompt="raw"), ("raw",))
        self.node.run(generation_mode="Generate on Change", user_prompt="same")
        self.assertEqual(self.requests.post.call_count, 1)
        self.connections.resolve.side_effect = ValueError("Select a registered connection.")
        self.assertIn("Select a registered connection", self.node.run(generation_mode="Generate on Change", user_prompt="same")[0])
        self.assertEqual(self.requests.post.call_count, 1)

    def test_switching_modes_uses_the_latest_success_and_nodes_have_separate_runtime_caches(self):
        self.node.run(user_prompt="same")
        self.node.run(user_prompt="same", generation_mode="Generate on Change", seed=91)
        self.assertEqual(self.requests.post.call_count, 1)
        self.node.run(user_prompt="same")
        self.node.run(user_prompt="same", generation_mode="Generate on Change")
        self.assertEqual(self.requests.post.call_count, 2)
        type(self.node)().run(user_prompt="same", generation_mode="Generate on Change")
        self.assertEqual(self.requests.post.call_count, 3)

    def test_invalid_or_incomplete_saved_cache_is_ignored(self):
        for saved in ("", "bad JSON", "null", "[]", '{"version":1,"scope":"bad"}',
                      json.dumps({"version": 1, "scope": "a" * 32, "text": "answer"})):
            with self.subTest(saved=saved):
                result = type(self.node)().run(generation_mode="Generate on Change", cached_response=saved)
                self.assertEqual(result["result"], ("answer",))
        self.assertEqual(self.requests.post.call_count, 6)

    def test_image_encoding_errors_preserve_the_previous_cache_and_are_retried(self):
        settings = dict(generation_mode="Generate on Change")
        self.node.run(**settings)
        previous = dict(self.node._last_response)
        image = np.zeros((1, 2, 2, 3), dtype=np.float32)
        with patch.object(self.node, "_convert_single_image_to_base64", return_value=None) as encode:
            for _ in range(2):
                self.assertIn("Cannot encode", self.node.run(image=image, **settings)[0])
            self.assertEqual(encode.call_count, 2)
        self.assertEqual(self.requests.post.call_count, 1)
        self.assertEqual(self.node._last_response, previous)

    def test_empty_image_batch_can_be_compared_and_reused(self):
        image = np.zeros((0, 2, 2, 3), dtype=np.float32)
        for _ in range(2):
            self.assertEqual(self.node.run(image=image, generation_mode="Generate on Change")["result"], ("answer",))
        self.assertEqual(self.requests.post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
