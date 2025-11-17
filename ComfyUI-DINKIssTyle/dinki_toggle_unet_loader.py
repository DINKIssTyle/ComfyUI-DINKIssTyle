# ComfyUI/custom_nodes/DINKI_ToggleUNetLoader/dinki_toggle_unet_loader.py

import nodes
import folder_paths
from typing import Type

NONE_LABEL = "-- None --"

class DINKI_ToggleUNetLoader:
    """
    safetensors / GGUF UNet 로더를 하나의 노드로 통합.
    
    - use_gguf == False → safetensors 로드
    - use_gguf == True  → GGUF 로드
    - 선택되지 않은 모델은 무시되며, 실제로 로드되지 않음.
    - 선택된 쪽만 '-- None --' 이면 에러 발생
    """

    @classmethod
    def INPUT_TYPES(cls):
        # safetensors UNet 목록 (ComfyUI 기본 UNETLoader 경로)
        safelist = folder_paths.get_filename_list("diffusion_models")
        safetensor_unets = [NONE_LABEL] + safelist if safelist else [NONE_LABEL]

        # GGUF UNet 목록
        gguf_list = folder_paths.get_filename_list("unet_gguf")
        gguf_unets = [NONE_LABEL] + gguf_list if gguf_list else [NONE_LABEL]

        return {
            "required": {
                "use_gguf": ("BOOLEAN", {
                    "default": False,
                    "label_on": "GGUF",
                    "label_off": "safetensors",
                }),

                "safetensors_unet": (safetensor_unets, {
                    "default": NONE_LABEL,
                    "tooltip": "safetensors UNet 선택"
                }),

                "gguf_unet": (gguf_unets, {
                    "default": NONE_LABEL,
                    "tooltip": "GGUF UNet 선택"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_unet"
    CATEGORY = "DINKIssTyle/Loaders"
    TITLE = "DINKI UNet Loader (safetensors / GGUF)"

    # -------- 내부 헬퍼: GGUF Loader 찾기 --------
    def _get_gguf_loader_class(self) -> Type:
        node_map = getattr(nodes, "NODE_CLASS_MAPPINGS", {})

        if "UnetLoaderGGUFAdvanced" in node_map:
            return node_map["UnetLoaderGGUFAdvanced"]
        if "UnetLoaderGGUF" in node_map:
            return node_map["UnetLoaderGGUF"]

        raise RuntimeError(
            "UnetLoaderGGUF / UnetLoaderGGUFAdvanced 노드를 찾을 수 없습니다.\n"
            "ComfyUI-GGUF 확장이 설치되어 있는지 확인하세요."
        )

    # -------- 실제 로딩 함수 --------
    def load_unet(self, use_gguf: bool, safetensors_unet: str, gguf_unet: str):
        """
        토글에 따라 safetensors 또는 GGUF 모델을 로드.
        선택되지 않은 쪽은 무시됨.
        """

        # ====================================================
        # GGUF 모드
        # ====================================================
        if use_gguf:
            if gguf_unet == NONE_LABEL:
                raise ValueError("GGUF 모드인데 GGUF UNet 모델이 선택되지 않았습니다.")

            loader_class = self._get_gguf_loader_class()
            loader = loader_class()

            # UnetLoaderGGUF.load_unet(unet_name, ...)
            return loader.load_unet(gguf_unet)

        # ====================================================
        # SAFETENSORS 모드
        # ====================================================
        else:
            if safetensors_unet == NONE_LABEL:
                raise ValueError("safetensors 모드인데 safetensors UNet 모델이 선택되지 않았습니다.")

            loader = nodes.UNETLoader()

            # UNETLoader.load_unet(unet_name, weight_dtype="default")
            return loader.load_unet(safetensors_unet, "default")
