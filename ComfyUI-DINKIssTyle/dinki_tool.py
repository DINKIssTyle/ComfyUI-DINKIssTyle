import sys

class DINKI_Note:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # 8방향 이모지 드랍다운
                "direction": (["⬆️", "↗️", "➡️", "↘️", "⬇️", "↙️", "⬅️", "↖️", "⏺️"],),
                # 텍스트 입력창 (멀티라인)
                "text": ("STRING", {"multiline": True, "default": "Text here!"}),
            },
        }

    # 출력은 텍스트를 그대로 내보내서 다른 곳에 연결 가능하게 함 (선택 사항)
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_out",)
    FUNCTION = "do_nothing"
    
    # 노드 카테고리
    CATEGORY = "DINKIssTyle/Utils"

    # 단순 패스스루 함수 (입력받은 텍스트를 그대로 출력)
    def do_nothing(self, direction, text):
        return (text,)


class DINKI_Node_Check:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "selected_node_id": ("STRING", {"default": "None", "multiline": False}),
            },
        }

    RETURN_TYPES = ()      # 출력 없음
    FUNCTION = "get_id"
    CATEGORY = "DINKIssTyle/Utils"
    # OUTPUT_NODE = True   # ← 이 줄을 지우거나 False 로

    def get_id(self, selected_node_id):
        return ()


class DINKI_Anchor:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # 단축키 입력 (예: 1, a, F2 등)
                "shortcut_key": ("STRING", {"default": "1", "multiline": False}),
                # 줌 레벨 목록 (콤마로 구분, % 단위)
                "zoom_levels": ("STRING", {"default": "50, 75, 100", "multiline": False}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "do_nothing"
    CATEGORY = "DINKIssTyle/Utils"
    OUTPUT_NODE = True

    def do_nothing(self, shortcut_key, zoom_levels):
        # 백엔드에서는 아무 작업도 하지 않음
        return ()



class DINKI_Auto_Focus:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # 기능을 켜고 끄는 스위치
                "enable": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Inactive"}),
                # 이동할 때 적용할 줌 비율 (0.5 = 50%, 1.0 = 100%)
                "zoom_level": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1, "display": "slider"}),
                # 이동 속도 (0이면 즉시 이동, 1에 가까울수록 부드럽게...지만 JS 구현 복잡도를 위해 일단 설정값만 둠)
                # 이번 구현에서는 '즉시 이동'을 기본으로 합니다.
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "do_nothing"
    CATEGORY = "DINKIssTyle/Utils"
    OUTPUT_NODE = True

    def do_nothing(self, enable, zoom_level):
        return ()