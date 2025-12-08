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
    CATEGORY = "DINKIssTyle/Util"

    # 단순 패스스루 함수 (입력받은 텍스트를 그대로 출력)
    def do_nothing(self, direction, text):
        return (text,)