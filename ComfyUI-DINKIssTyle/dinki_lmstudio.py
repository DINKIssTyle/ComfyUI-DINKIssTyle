import base64, hashlib, io, json, time, uuid, requests
import numpy as np
from PIL import Image
from .dinki_llm_connections import (
    NONE, aliases, resolve, connection_url, connection_headers,
)

class DINKI_LMStudio:
    def __init__(self):
        self._fallback_scope = uuid.uuid4().hex
        self._cache_scope = None
        self._last_response = None

    @classmethod
    def INPUT_TYPES(cls):
        try:
            registered = aliases()
        except ValueError:
            # A damaged profile file must not prevent all ComfyUI node
            # definitions from loading. Settings/runtime report the problem.
            registered = []
        return {
            "required": {},
            "optional": {
                "assistant_enabled": ("BOOLEAN", {"default": True}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a writer who creates prompts for generative AI images. Respond only with the final English prompt."
                }),
                "image": ("IMAGE",),
                "model_key": ("STRING", {"default": "qwen/qwen3-vl-8b"}),
                "seed": ("INT", {"default": -1}),
                "max_tokens": ("INT", {"default": 1000, "min": 0, "max": 2147483647,
                    "tooltip": "Maximum output tokens. 0 uses the server default. The model context limit still applies."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "timeout_seconds": ("INT", {"default": 300}),
                "auto_unload": ("BOOLEAN", {"default": False}),
                "unload_delay": ("INT", {"default": 0, "min": 0, "max": 600}),
                "connection_alias": ([NONE, *registered], {
                    "default": NONE,
                    "tooltip": "Register connections in Settings > DKST > LLM. Only this alias is saved in the workflow."}),
                "generation_mode": (["Always Generate", "Generate on Change"], {
                    "default": "Always Generate",
                    "tooltip": "Always generate, or reuse the last answer when the prompts and images are unchanged. Seed changes do not trigger regeneration in Generate on Change mode. Model, connection and generation settings changes invalidate the cache."}),
                "cached_response": ("STRING", {"default": "", "dynamicPrompts": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("AI Answer Text",)
    FUNCTION = "run"
    CATEGORY = "DINKIssTyle/LLM"
    DESCRIPTION = (
        "Generate text with LM Studio. Generate on Change reuses the last successful answer "
        "when prompts and image contents are unchanged, regardless of seed randomization. "
        "Model, connection, temperature and max_tokens changes require a new answer. "
        "Save the workflow to retain the last answer."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, connection_alias=NONE, **kwargs):
        # Allow missing aliases to survive workflow loading without selecting a
        # different server. The runtime resolver supplies the actionable error.
        return True

    @classmethod
    def IS_CHANGED(cls, connection_alias=NONE, **kwargs):
        # Always reach run(): ComfyUI's own signature includes seed and upstream
        # nodes, whereas our response cache compares the actual incoming values.
        return float("nan")

    # --- helpers ---
    @staticmethod
    def _saved_response(value):
        try:
            data = json.loads(value)
            if not isinstance(data, dict) or data.get("version") != 1:
                return None
            scope = data.get("scope")
            if not isinstance(scope, str) or uuid.UUID(scope).hex != scope:
                return None
            return data
        except (ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _input_signature(scope, unique_id, user_prompt, system_prompt, image,
                         model_key, connection_alias, profile, temperature, max_tokens):
        digest = hashlib.sha256()
        # Credentials and endpoints are not serialized. Profile revisions also
        # invalidate a saved answer after connection settings change.
        settings = [1, scope, unique_id, user_prompt, system_prompt, model_key,
                    connection_alias, profile.get("revision"), float(temperature), int(max_tokens)]
        digest.update(json.dumps(settings, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if image is not None:
            tensor = image.detach() if hasattr(image, "detach") else image
            array = tensor.cpu().numpy() if hasattr(tensor, "cpu") else tensor
            array = np.ascontiguousarray(array)
            digest.update(json.dumps([array.shape, array.dtype.str]).encode("ascii"))
            if array.size:
                digest.update(memoryview(array).cast("B"))
        return digest.hexdigest()

    @staticmethod
    def _cached_result(data):
        return {"ui": {"dkst_llm_response": [json.dumps(data, ensure_ascii=False)]},
                "result": (data["text"],)}

    def _read_completion(self, response):
        if "text/event-stream" not in response.headers.get("Content-Type", "").lower():
            data = response.json()
            try:
                return data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                raise RuntimeError("Invalid completion response.") from None

        parts = []
        # Small reads allow a completion marker to end the request even if a
        # proxy keeps the HTTP connection open after generation has finished.
        for line in response.iter_lines(chunk_size=1):
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return "".join(parts)
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError("Server returned a stream error.")
            for choice in event.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                content = choice.get("delta", {}).get("content")
                if content:
                    parts.append(content)
                if choice.get("finish_reason") is not None:
                    return "".join(parts)
        raise RuntimeError("LM Studio stream ended without a completion marker.")

    def _convert_single_image_to_base64(self, img_tensor):
        """단일 이미지 텐서(H,W,C)를 base64 문자열로 변환"""
        try:
            tensor = img_tensor.detach() if hasattr(img_tensor, "detach") else img_tensor
            arr = tensor.cpu().numpy() if hasattr(tensor, "cpu") else tensor
            # 값 스케일링 및 타입 변환
            arr = (arr * 255.0).clip(0, 255).astype("uint8")
            pil_img = Image.fromarray(arr)
            
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"[🅳INKIssTyle - Error]Image conversion: {e}")
            return None

    # --- main ---
    def run(
        self,
        assistant_enabled=True,
        user_prompt="",
        system_prompt="",
        image=None,
        model_key="qwen/qwen3-vl-8b",
        seed=-1,
        max_tokens=1000,
        temperature=0.7,
        timeout_seconds=300,
        auto_unload=False,
        unload_delay=0,
        connection_alias=NONE,
        generation_mode="Always Generate",
        cached_response="",
        unique_id=None,
    ):
        # 1) 패스스루 모드
        if not assistant_enabled:
            return (user_prompt or "",)

        try:
            profile = resolve(connection_alias)
        except ValueError as error:
            return (f"Error: {error}",)
        base_url = connection_url(profile)

        saved = self._saved_response(cached_response)
        scope = saved["scope"] if saved else self._fallback_scope
        if scope != self._cache_scope:
            self._cache_scope = scope
            self._last_response = saved
        signature = self._input_signature(scope, unique_id, user_prompt, system_prompt,
                                          image, model_key, connection_alias, profile,
                                          temperature, max_tokens)
        previous = self._last_response
        if (generation_mode == "Generate on Change" and previous
                and previous.get("signature") == signature
                and isinstance(previous.get("text"), str)):
            print("[DINKI LM Studio] Inputs unchanged; reusing the last answer.", flush=True)
            return self._cached_result(previous)

        # 2) 시드 처리
        if seed == -1: # seed가 -1이면 랜덤 (ComfyUI 위젯 특성상 randomize string 대신 -1 int 체크가 일반적이나, string 입력이 있다면 변환)
             seed = int(time.time_ns() % (2**31))

        # 3) 메시지 구성
        url = base_url + "/v1/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 콘텐츠 리스트 생성 (텍스트 + n개의 이미지)
        content_list = [{"type": "text", "text": user_prompt or "Describe the images."}]

        # 이미지 처리 (배치 지원)
        if image is not None:
            # image shape은 보통 [Batch, Height, Width, Channels]
            batch_count = image.shape[0]
            
            for i in range(batch_count):
                # 배치에서 i번째 이미지만 추출 (H,W,C)
                img_slice = image[i]
                image_url = self._convert_single_image_to_base64(img_slice)
                if not image_url:
                    return ("Error: Cannot encode an input image for LM Studio.",)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })

        messages.append({"role": "user", "content": content_list})

        body = {
            "model": model_key,
            "messages": messages,
            "temperature": float(temperature),
            "seed": int(seed),
            "stream": True,
        }
        if int(max_tokens) > 0:
            body["max_tokens"] = int(max_tokens)
        headers = connection_headers(profile)

        # 4) 호출
        resp = None
        try:
            print("[DINKI LM Studio] Sending generation request.", flush=True)
            resp = requests.post(url, json=body, headers=headers,
                                 timeout=(10, timeout_seconds), stream=True, allow_redirects=False)
            print(f"[DINKI LM Studio] HTTP {resp.status_code}; receiving response.", flush=True)
            resp.raise_for_status()
            if not 200 <= resp.status_code < 300:
                return ("Error: LM Studio redirected the request. Check the connection in Settings > DKST > LLM.",)
            text = self._read_completion(resp)
            print(f"[DINKI LM Studio] Response complete ({len(text)} characters).", flush=True)
        except requests.HTTPError:
            return (f"Error: LM Studio request failed (HTTP {resp.status_code}). Check the connection and model settings.",)
        except Exception:
            # requests exceptions can contain the URL; server error bodies may
            # echo credentials. Neither belongs in a text output or metadata.
            return ("Error: LM Studio request failed or returned an incomplete response. Check Settings > DKST > LLM and the model.",)
        finally:
            if resp is not None:
                resp.close()

        # 6) 자동 언로드
        if auto_unload and unload_delay > 0:
            print(f"[DINKI LM Studio] Waiting {unload_delay}s before unloading.", flush=True)
            time.sleep(unload_delay)
            unload_endpoints = [
                base_url + "/v1/models/unload",
                base_url + "/v1/unload"
            ]
            for u in unload_endpoints:
                try:
                    with requests.post(u, json={"model": model_key}, headers=headers,
                                       timeout=2, allow_redirects=False):
                        pass
                except Exception:
                    pass

        print("[DINKI LM Studio] Returning text to ComfyUI.", flush=True)
        # Update only after a complete, successful response. Keep runtime state
        # authoritative over stale workflow snapshots from already queued runs.
        self._last_response = {"version": 1, "scope": scope, "signature": signature, "text": text}
        return self._cached_result(self._last_response)

# 등록
NODE_CLASS_MAPPINGS = {
    "DINKI LM Studio Assistant": DINKI_LMStudio,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI LM Studio Assistant": "DKST LLM (LM Studio)",
}
