"""Real profile persistence with stubbed ComfyUI services and HTTP transport."""
import asyncio
import json
import os
from pathlib import Path
import runpy
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle"


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.server = MagicMock()
        for method in ("get", "post"):
            getattr(self.server.PromptServer.instance.routes, method).side_effect = lambda path: lambda fn: fn
        self.requests = MagicMock()
        folders = types.SimpleNamespace(get_user_directory=lambda: self.directory.name)
        web = types.SimpleNamespace(json_response=lambda data, status=200: types.SimpleNamespace(data=data, status=status))
        with patch.dict("sys.modules", {"server": self.server, "folder_paths": folders,
                                       "requests": self.requests, "aiohttp": types.SimpleNamespace(web=web)}):
            self.module = runpy.run_path(str(ROOT / "dinki_llm_connections.py"))
        self.profile = {"alias": "Work", "ip_address": "192.168.0.9", "port": 1234, "api_key": "private-test-key"}

    def call(self, name, *args):
        return self.module[name](*args)

    def route(self, data):
        body = json.dumps(data).encode()
        async def read(size):
            return body[:size]
        request = types.SimpleNamespace(content_length=len(body), content=types.SimpleNamespace(read=read))
        return asyncio.run(self.call("manage_connections", request))

    def test_persisted_profiles_are_private_and_lists_never_return_keys(self):
        self.call("save", self.profile)
        path = self.call("profile_path")
        stored = json.loads(path.read_text())["connections"]["Work"]
        self.assertEqual(stored["api_key"], "private-test-key")
        if os.name != "nt":
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.call("aliases"), ["Work"])
        summary = self.call("summaries")
        self.assertTrue(summary[0]["has_api_key"])
        self.assertNotIn("api_key", summary[0])
        self.assertNotIn("private-test-key", json.dumps(summary))
        self.assertNotIn("private-test-key", self.call("revision", "Work"))
        self.assertEqual(self.call("resolve", "Work")["api_key"], "private-test-key")
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_duplicate_edit_preserves_key_and_revision_changes(self):
        self.call("save", self.profile)
        before = self.call("revision", "Work")
        with self.assertRaises(ValueError):
            self.call("save", self.profile)
        edit = {**self.profile, "overwrite": True, "port": 2345}
        del edit["api_key"]
        self.call("save", edit)
        self.assertEqual(self.call("resolve", "Work")["api_key"], "private-test-key")
        self.assertNotEqual(before, self.call("revision", "Work"))
        self.call("save", {**edit, "api_key": ""})
        self.assertEqual(self.call("resolve", "Work")["api_key"], "")
        self.call("delete", "Work")
        self.assertEqual(self.call("aliases"), [])
        with self.assertRaises(ValueError):
            self.call("resolve", "Work")

    def test_migration_deduplicates_without_sensitive_aliases(self):
        first = self.call("migrate", self.profile)
        self.assertEqual(first, "Migrated LLM 1")
        self.assertEqual(self.call("migrate", self.profile), first)
        self.assertEqual(self.call("migrate", {**self.profile, "port": 2345}), "Migrated LLM 2")
        self.assertNotIn(self.profile["ip_address"], json.dumps(self.call("aliases")))

    def test_invalid_values_never_echo_submitted_secrets(self):
        for changes in ({"port": 0}, {"port": True}, {"port": "1234"},
                        {"ip_address": "http://private-host"}, {"ip_address": "host/key"},
                        {"api_key": "secret\nheader"}, {"protocol": "ftp"},
                        {"alias": self.module["NONE"]}):
            response = self.route({**self.profile, **changes})
            self.assertEqual(response.status, 400)
            self.assertNotIn("private", json.dumps(response.data))
            self.assertNotIn("secret", json.dumps(response.data))
        self.assertFalse(self.call("profile_path").exists())

    def test_ipv6_and_https(self):
        self.call("save", {**self.profile, "ip_address": "[::1]", "protocol": "https"})
        self.assertEqual(self.call("connection_url", self.call("resolve", "Work")), "https://[::1]:1234")

    def test_test_connection_uses_stored_auth_without_redirects(self):
        self.call("save", self.profile)
        response = self.requests.get.return_value.__enter__.return_value
        response.status_code = 200
        response.json.return_value = {"data": []}
        self.assertTrue(self.call("test_connection", "Work")["ok"])
        kwargs = self.requests.get.call_args.kwargs
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer private-test-key"})
        self.assertFalse(kwargs["allow_redirects"])
        response.status_code = 401
        self.assertIn("401", self.call("test_connection", "Work")["message"])
        self.requests.get.side_effect = RuntimeError("private-host private-test-key")
        self.assertNotIn("private", json.dumps(self.call("test_connection", "Work")))

    def test_routes_only_return_non_secret_data(self):
        self.assertEqual(self.route(self.profile).data, {"alias": "Work"})
        for name in ("list_connections", "list_aliases"):
            response = asyncio.run(self.call(name, None))
            self.assertEqual(response.status, 200)
            self.assertNotIn("private-test-key", json.dumps(response.data))
        self.assertEqual(self.route({"action": "delete", "alias": "Work"}).status, 200)

    def test_real_registry_resolves_node_execution_and_cache_revision(self):
        self.call("save", self.profile)
        registry = types.ModuleType("dkst_integration.dinki_llm_connections")
        registry.__dict__.update(self.module)
        with patch.dict("sys.modules", {
            "dkst_integration": types.ModuleType("dkst_integration"),
            "dkst_integration.dinki_llm_connections": registry,
            "requests": self.requests, "numpy": MagicMock(), "PIL": MagicMock(),
        }):
            node_type = runpy.run_path(str(ROOT / "dinki_lmstudio.py"),
                                      run_name="dkst_integration.dinki_lmstudio")["DINKI_LMStudio"]
        response = self.requests.post.return_value
        response.status_code = 200
        response.headers = {"Content-Type": "application/json"}
        response.json.return_value = {"choices": [{"message": {"content": "answer"}}]}
        node = node_type()
        self.assertEqual(node.run(connection_alias="Work", generation_mode="Generate on Change")["result"], ("answer",))
        self.assertEqual(self.requests.post.call_args.args[0], "http://192.168.0.9:1234/v1/chat/completions")
        self.assertEqual(self.requests.post.call_args.kwargs["headers"], {"Authorization": "Bearer private-test-key"})
        self.assertIn("Work", node_type.INPUT_TYPES()["optional"]["connection_alias"][0])
        node.run(connection_alias="Work", generation_mode="Generate on Change", seed=123)
        self.assertEqual(self.requests.post.call_count, 1)
        self.call("save", {**self.profile, "overwrite": True, "port": 2345})
        node.run(connection_alias="Work", generation_mode="Generate on Change")
        self.assertEqual(self.requests.post.call_count, 2)
        self.assertEqual(self.requests.post.call_args.args[0], "http://192.168.0.9:2345/v1/chat/completions")

    def test_prompt_boundary_scrubs_old_credentials_and_retains_new_workflow(self):
        old = {"type": "DINKI_LMStudio", "widgets_values": ["private-host", 1234, "private-test-key"],
               "properties": {"api_key": "private-test-key"}}
        new = {"type": "DINKI_LMStudio", "widgets_values": ["Work"], "properties": {"dkstLLMVersion": 2}}
        data = {"prompt": {"1": {"class_type": "DINKI_LMStudio", "inputs": {**self.profile, "connection_alias": "Work"}}},
                "extra_data": {"extra_pnginfo": {"workflow": {"nodes": [old], "definitions": {"subgraphs": [{"nodes": [new]}]}}}}}
        cleaned = self.call("scrub_prompt", data)
        text = json.dumps(cleaned)
        self.assertNotIn("private-test-key", text)
        self.assertNotIn("192.168.0.9", text)
        self.assertNotIn("private-host", text)
        self.assertEqual(new["widgets_values"], ["Work"])
        self.server.PromptServer.instance.add_on_prompt_handler.assert_called_once()

    def test_corrupt_storage_error_does_not_expose_file_content(self):
        path = self.call("profile_path")
        path.parent.mkdir()
        path.write_text("private-test-key")
        with self.assertRaisesRegex(ValueError, "Cannot read DKST LLM"):
            self.call("aliases")
        response = self.route(self.profile)
        self.assertNotIn("private-test-key", json.dumps(response.data))


if __name__ == "__main__":
    unittest.main()
