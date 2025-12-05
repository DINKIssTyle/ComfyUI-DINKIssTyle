import sys

class DINKI_Node_Switch:
    """
    A logic node that toggles the Bypass status of other nodes based on their IDs.
    The actual bypassing logic is handled by the accompanying JavaScript.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "node_ids": ("STRING", {"multiline": False, "default": "1,2,3"}),
                "active": ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "do_nothing"
    CATEGORY = "DINKIssTyle/Utils"
    OUTPUT_NODE = True

    def do_nothing(self, node_ids, active):
        # The bypassing logic happens in the frontend (JavaScript) before execution.
        # This python method is just a placeholder to satisfy the execution requirement.
        return ()







class DINKI_String_Switch_RT:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # 드랍다운 (JS에서 생성됨)
                "select_string": ("STRING", {"default": "Option 1", "multiline": False}),
                
                # [변경] 슬롯 방식 대신 하나의 멀티라인 텍스트 박스 사용
                "input_text": ("STRING", {"multiline": True, "default": "Option 1\nOption 2\nOption 3", "dynamicPrompts": False}),
            },
            "optional": {
                "text_in": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_text",)
    FUNCTION = "switch_and_combine"
    CATEGORY = "DINKIssTyle/Utils"

    # 함수 인자도 input_text로 변경
    def switch_and_combine(self, select_string, input_text, text_in=None):
        current_text = select_string

        if text_in is None:
            text_in = ""

        # 로직은 동일: 선택된 값(select_string)과 입력값(text_in) 결합
        if text_in and current_text:
            result = f"{text_in}, {current_text}"
        elif text_in:
            result = text_in
        elif current_text:
            result = current_text
        else:
            result = ""

        return (result,)