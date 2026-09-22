import base64, io, json, time, requests
import numpy as np
from PIL import Image

class DINKI_LMStudio:
    @classmethod
    def INPUT_TYPES(cls):
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
                "ip_address": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 1234, "min": 1, "max": 65535}),
                "api_key": ("STRING", {"default": "", "multiline": False,
                    "dynamicPrompts": False,
                    "tooltip": "API key for servers requiring Bearer authentication. Leave empty if authentication is disabled."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("AI Answer Text",)
    FUNCTION = "run"
    CATEGORY = "DINKIssTyle/LLM"

    # --- helpers ---
    def _read_completion(self, response):
        if "text/event-stream" not in response.headers.get("Content-Type", "").lower():
            data = response.json()
            try:
                return data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                return json.dumps(data, ensure_ascii=False)

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
                raise RuntimeError(json.dumps(event["error"], ensure_ascii=False))
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
            arr = img_tensor.cpu().numpy() if hasattr(img_tensor, "cpu") else img_tensor
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
        ip_address="127.0.0.1",
        port=1234,
        api_key="",
    ):
        # 1) 패스스루 모드
        if not assistant_enabled:
            return (user_prompt or "",)

        # 2) 시드 처리
        if seed == -1: # seed가 -1이면 랜덤 (ComfyUI 위젯 특성상 randomize string 대신 -1 int 체크가 일반적이나, string 입력이 있다면 변환)
             seed = int(time.time_ns() % (2**31))

        # 3) 메시지 구성
        url = f"http://{ip_address}:{port}/v1/chat/completions"
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
                
                if image_url:
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
        headers = {}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"

        # 4) 호출
        resp = None
        try:
            print("[DINKI LM Studio] Sending generation request.", flush=True)
            resp = requests.post(url, json=body, headers=headers,
                                 timeout=(10, timeout_seconds), stream=True)
            print(f"[DINKI LM Studio] HTTP {resp.status_code}; receiving response.", flush=True)
            resp.raise_for_status()
            text = self._read_completion(resp)
            print(f"[DINKI LM Studio] Response complete ({len(text)} characters).", flush=True)
        except requests.HTTPError as e:
            detail = resp.text.strip()
            return (f"Error: {e}" + (f"\nServer response: {detail}" if detail else ""),)
        except Exception as e:
            # 에러 발생 시 상세 내용 반환 (디버깅용)
            return (f"Error: {e}",)
        finally:
            if resp is not None:
                resp.close()

        # 6) 자동 언로드
        if auto_unload and unload_delay > 0:
            print(f"[DINKI LM Studio] Waiting {unload_delay}s before unloading.", flush=True)
            time.sleep(unload_delay)
            unload_endpoints = [
                f"http://{ip_address}:{port}/v1/models/unload",
                f"http://{ip_address}:{port}/v1/unload"
            ]
            for u in unload_endpoints:
                try:
                    requests.post(u, json={"model": model_key}, headers=headers, timeout=2)
                except Exception:
                    pass

        print("[DINKI LM Studio] Returning text to ComfyUI.", flush=True)
        return (text,)

# 등록
NODE_CLASS_MAPPINGS = {
    "DINKI LM Studio Assistant": DINKI_LMStudio,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI LM Studio Assistant": "DKST LM Studio Assistant",
}
