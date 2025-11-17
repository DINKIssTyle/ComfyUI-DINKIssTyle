# ComfyUI/custom_nodes/LMStudio_ImageToText/__init__.py
import base64, io, json, time, requests
from PIL import Image

class DINKI_LMStudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            # ✅ 모든 입력을 위젯으로 처리, image는 선택 연결(옵션)
            "required": {},
            "optional": {
                # 🔘 Assistant 토글 (가장 위에 오도록 첫 항목)
                "assistant_enabled": ("BOOLEAN", {"default": True}),

                # ✍️ 프롬프트
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a writer who creates prompts for generative AI images. Respond only with the final English prompt."
                }),

                # 🖼️ 이미지(옵션 소켓)
                "image": ("IMAGE",),

                # 🧠 모델 & 생성 파라미터
                "model_key": ("STRING", {"default": "qwen/qwen3-vl-8b"}),
                "seed": ("INT", {"default": -1}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "timeout_seconds": ("INT", {"default": 300}),

                # 🧹 생성 후 언로드
                "auto_unload": ("BOOLEAN", {"default": False}),
                "unload_delay": ("INT", {"default": 0, "min": 0, "max": 600}),

                # 🌐 서버
                "ip_address": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 1234, "min": 1, "max": 65535}),
            }
        }

    # 출력: 답변 문자열 하나
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("AI Answer Text",)
    FUNCTION = "run"
    CATEGORY = "DINKIssTyle/LLM"

    # --- helpers ---
    def _tensor_to_data_url(self, image_tensor):
        if image_tensor is None:
            return None
        if isinstance(image_tensor, list) and len(image_tensor) > 0:
            img = image_tensor[0]
        else:
            img = image_tensor
        import numpy as np
        arr = img.cpu().numpy() if hasattr(img, "cpu") else img
        if getattr(arr, "ndim", 0) == 4:
            arr = arr[0]  # (B,H,W,C) → (H,W,C)
        arr = (arr * 255.0).clip(0, 255).astype("uint8")
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

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
    ):
        # 1) 패스스루 모드: 토글이 꺼져 있으면 네트워크 호출 없이 텍스트만 반환
        if not assistant_enabled:
            return (user_prompt or "",)

        # 2) 시드
        if seed == "randomize":
            seed = int(time.time_ns() % (2**31))

        # 3) 메시지 구성 (이미지가 있으면 멀티모달, 없으면 텍스트만)
        url = f"http://{ip_address}:{port}/v1/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content = [{"type": "text", "text": user_prompt or "Describe the image or prompt."}]
        image_url = self._tensor_to_data_url(image)
        if image_url:  # 이미지가 연결된 경우에만 포함
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        messages.append({"role": "user", "content": content})

        body = {
            "model": model_key,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "seed": int(seed),  # 일부 빌드에서 무시될 수 있음
        }

        # 4) 호출
        try:
            resp = requests.post(url, json=body, timeout=timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"LM Studio 요청 실패: {e}")

        # 5) 응답 파싱
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception:
            text = json.dumps(data, ensure_ascii=False)

        # 6) 자동 언로드(선택)
        if auto_unload and unload_delay > 0:
            time.sleep(unload_delay)
            for u in [f"http://{ip_address}:{port}/v1/models/unload",
                      f"http://{ip_address}:{port}/v1/unload"]:
                try:
                    r = requests.post(u, json={"model": model_key}, timeout=5)
                    if r.ok:
                        break
                except Exception:
                    pass

        return (text,)

# 등록
NODE_CLASS_MAPPINGS = {
    "DINKI LM Studio Assistant": DINKI_LMStudio,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI LM Studio Assistant": "DINKI LM Studio Assistant",
}
