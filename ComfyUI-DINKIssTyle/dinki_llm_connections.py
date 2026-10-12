"""Server-wide LM Studio profiles. Credentials never enter node inputs."""
import asyncio
import ipaddress
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import uuid

import folder_paths
import requests
from aiohttp import web
from server import PromptServer

NONE = "-- Select connection --"
NODE_TYPES = {"DINKI_LMStudio", "DINKI LM Studio Assistant"}
PRIVATE_INPUTS = {"ip_address", "port", "api_key"}
_LOCK = threading.RLock()


def profile_path():
    return Path(folder_paths.get_user_directory()) / "dkst_llm" / "connections.json"


def _read():
    path = profile_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("connections"), dict):
            raise ValueError()
        for alias, profile in data["connections"].items():
            _alias(alias)
            if not isinstance(profile, dict) or not isinstance(profile.get("revision"), str):
                raise ValueError()
            validated = _connection(profile)
            if any(profile.get(key) != value for key, value in validated.items()):
                raise ValueError()
        return data["connections"]
    except (ValueError, TypeError, AttributeError, OSError):
        raise ValueError("Cannot read DKST LLM connections. Check the server configuration file.") from None


def _write(profiles):
    path = profile_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".connections-", delete=False) as file:
            temporary = file.name
            os.chmod(temporary, 0o600)
            json.dump({"version": 1, "connections": profiles}, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except OSError:
        raise ValueError("Cannot save DKST LLM connections on the server.") from None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _alias(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 80:
        raise ValueError("Alias must contain 1–80 characters.")
    value = value.strip()
    if value == NONE or any(ord(char) < 32 for char in value):
        raise ValueError("Choose a different alias.")
    return value


def _connection(data, previous=None):
    host = data.get("ip_address", "127.0.0.1")
    if not isinstance(host, str):
        raise ValueError("Enter an IP address or hostname without a URL or port.")
    host = host.strip()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if not host or len(host) > 253 or not re.fullmatch(r"[a-zA-Z0-9.-]+", host):
            raise ValueError("Enter an IP address or hostname without a URL or port.") from None
    port = data.get("port", 1234)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("Port must be an integer between 1 and 65535.")
    protocol = data.get("protocol", "http")
    if protocol not in ("http", "https"):
        raise ValueError("Protocol must be HTTP or HTTPS.")
    key = data.get("api_key", (previous or {}).get("api_key", ""))
    if not isinstance(key, str) or len(key) > 8192 or "\r" in key or "\n" in key:
        raise ValueError("Invalid API key.")
    return {"ip_address": host, "port": port, "protocol": protocol, "api_key": key.strip()}


def aliases():
    with _LOCK:
        return sorted(_read())


def resolve(alias):
    with _LOCK:
        profile = _read().get(alias)
        if profile is None:
            raise ValueError("Select a registered connection in Settings > DKST > LLM.")
        return dict(profile)


def revision(alias):
    try:
        return resolve(alias)["revision"]
    except (ValueError, KeyError):
        return "missing"


def connection_url(profile):
    host = profile["ip_address"]
    if ":" in host:
        host = f"[{host}]"
    return f'{profile["protocol"]}://{host}:{profile["port"]}'


def connection_headers(profile):
    key = profile["api_key"]
    return {"Authorization": f"Bearer {key}"} if key else {}


def summaries():
    with _LOCK:
        return [{"alias": alias, **{key: profile[key] for key in ("ip_address", "port", "protocol")},
                 "has_api_key": bool(profile["api_key"])} for alias, profile in sorted(_read().items())]


def save(data):
    alias = _alias(data.get("alias"))
    with _LOCK:
        profiles = _read()
        previous = profiles.get(alias)
        if previous is not None and data.get("overwrite") is not True:
            raise ValueError("Alias already exists. Select it to edit the connection.")
        profile = _connection(data, previous)
        profiles[alias] = {**profile, "revision": uuid.uuid4().hex}
        _write(profiles)
    return alias


def delete(alias):
    alias = _alias(alias)
    with _LOCK:
        profiles = _read()
        if alias not in profiles:
            raise ValueError("Connection no longer exists.")
        del profiles[alias]
        _write(profiles)


def migrate(data):
    """Deduplicate old static connections without putting an address in an alias."""
    profile = _connection(data)
    with _LOCK:
        profiles = _read()
        for alias, existing in profiles.items():
            if all(existing.get(key) == value for key, value in profile.items()):
                return alias
        index = 1
        while f"Migrated LLM {index}" in profiles:
            index += 1
        alias = f"Migrated LLM {index}"
        profiles[alias] = {**profile, "revision": uuid.uuid4().hex}
        _write(profiles)
        return alias


def test_connection(alias):
    profile = resolve(alias)
    try:
        with requests.get(connection_url(profile) + "/v1/models", headers=connection_headers(profile),
                          timeout=(5, 10), allow_redirects=False) as response:
            if not 200 <= response.status_code < 300:
                return {"ok": False, "message": f"Connection test failed (HTTP {response.status_code})."}
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                return {"ok": False, "message": "Server did not return an OpenAI-compatible model list."}
        return {"ok": True, "message": "Connected successfully."}
    except Exception:
        return {"ok": False, "message": "Connection test failed. Check the server address, authentication and availability."}


async def _payload(request):
    if request.content_length is not None and request.content_length > 32 * 1024:
        raise ValueError("Connection request is too large.")
    body = await request.content.read(32 * 1024 + 1)
    if len(body) > 32 * 1024:
        raise ValueError("Connection request is too large.")
    try:
        data = json.loads(body)
    except (ValueError, UnicodeError):
        raise ValueError("Invalid connection request.") from None
    if not isinstance(data, dict):
        raise ValueError("Invalid connection request.")
    return data


@PromptServer.instance.routes.get("/dinki/llm/aliases")
async def list_aliases(request):
    try:
        return web.json_response({"aliases": aliases()})
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.get("/dinki/llm/connections")
async def list_connections(request):
    try:
        return web.json_response({"connections": summaries()})
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.post("/dinki/llm/connections")
async def manage_connections(request):
    try:
        data = await _payload(request)
        action = data.get("action", "save")
        if action == "save":
            return web.json_response({"alias": save(data)})
        if action == "delete":
            delete(data.get("alias"))
            return web.json_response({"ok": True})
        if action == "migrate":
            return web.json_response({"alias": migrate(data)})
        if action == "test":
            return web.json_response(await asyncio.to_thread(test_connection, _alias(data.get("alias"))))
        raise ValueError("Unknown connection action.")
    except ValueError as error:
        # All validation errors above have fixed text; never echo submitted values.
        return web.json_response({"error": str(error)}, status=400)
    except TypeError:
        return web.json_response({"error": "Invalid connection request."}, status=400)


def scrub_prompt(data):
    """Defense for API clients that submit removed inputs: keep them out of history/metadata."""
    for node in (data.get("prompt") or {}).values():
        if isinstance(node, dict) and node.get("class_type") in NODE_TYPES:
            inputs = node.get("inputs", {})
            if isinstance(inputs, dict):
                for name in PRIVATE_INPUTS:
                    inputs.pop(name, None)
    extra = (data.get("extra_data") or {}).get("extra_pnginfo") or {}
    workflow = extra.get("workflow") if isinstance(extra, dict) else None
    def scrub_fields(value):
        if isinstance(value, dict):
            for key in list(value):
                if key in PRIVATE_INPUTS:
                    del value[key]
                else:
                    scrub_fields(value[key])
        elif isinstance(value, list):
            for child in value:
                scrub_fields(child)

    def scrub(value):
        if isinstance(value, dict):
            if value.get("type") in NODE_TYPES:
                scrub_fields(value)
                # Legacy widget arrays contain credentials; unknown old layouts
                # are dropped rather than guessed at the metadata boundary.
                if (value.get("properties") or {}).get("dkstLLMVersion") != 2:
                    value.pop("widgets_values", None)
                value.pop("widgets_values_named", None)
            for child in value.values():
                scrub(child)
        elif isinstance(value, list):
            for child in value:
                scrub(child)
    scrub(workflow)
    return data


PromptServer.instance.add_on_prompt_handler(scrub_prompt)
