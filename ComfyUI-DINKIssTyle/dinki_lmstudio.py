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
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "timeout_seconds": ("INT", {"default": 300}),
                "auto_unload": ("BOOLEAN", {"default": False}),
                "unload_delay": ("INT", {"default": 0, "min": 0, "max": 600}),
                "ip_address": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 1234, "min": 1, "max": 65535}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("AI Answer Text",)
    FUNCTION = "run"
    CATEGORY = "DINKIssTyle/LLM"

    # --- helpers ---
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
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
        }

        # 4) 호출
        try:
            resp = requests.post(url, json=body, timeout=timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            # 에러 발생 시 상세 내용 반환 (디버깅용)
            return (f"Error: {e}",)

        # 5) 응답 파싱
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception:
            text = json.dumps(data, ensure_ascii=False)

        # 6) 자동 언로드
        if auto_unload and unload_delay > 0:
            time.sleep(unload_delay)
            unload_endpoints = [
                f"http://{ip_address}:{port}/v1/models/unload",
                f"http://{ip_address}:{port}/v1/unload"
            ]
            for u in unload_endpoints:
                try:
                    requests.post(u, json={"model": model_key}, timeout=2)
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