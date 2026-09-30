import re
import sys
from comfy_execution.graph_utils import ExecutionBlocker, is_link


class _AnyType(str):
    """Accept connections of any ComfyUI socket type."""

    def __ne__(self, other):
        return False


class _DINKI_IfElseBase:
    """Route numbered values through only the selected lazy branch."""

    PAIR_COUNT = 0
    FUNCTION = "select"
    CATEGORY = "DINKIssTyle/Util"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for index in range(1, cls.PAIR_COUNT + 1):
            for branch in ("false", "true"):
                optional[f"on_{branch}_{index}"] = (_AnyType("*"), {"lazy": True})
        optional["empty_as_none"] = (
            "STRING",
            {"default": "", "multiline": False, "advanced": True,
             "tooltip": "Comma-separated output numbers that send None when their selected input is unconnected. Other empty outputs mute downstream nodes."},
        )
        return {
            "required": {"switch": ("BOOLEAN", {"default": False})},
            "optional": optional,
        }

    @classmethod
    def check_lazy_status(cls, switch, **inputs):
        branch = "true" if switch else "false"
        return [
            name
            for index in range(1, cls.PAIR_COUNT + 1)
            if (name := f"on_{branch}_{index}") in inputs and inputs[name] is None
        ]

    def _selected_outputs(self, switch, inputs):
        branch = "true" if switch else "false"
        empty_as_none = {
            int(value) for value in re.split(r"[,\s]+", (inputs.get("empty_as_none") or "").strip())
            if value.isdigit()
        }
        outputs = []
        for index in range(1, self.PAIR_COUNT + 1):
            name = f"on_{branch}_{index}"
            if name in inputs:
                outputs.append(inputs[name])
            elif index in empty_as_none:
                outputs.append(None)
            else:
                outputs.append(ExecutionBlocker(None))
        return tuple(outputs)


class DINKI_IfElseSwitch(_DINKI_IfElseBase):
    """Route ten values and pass the switch state to later branch nodes."""

    PAIR_COUNT = 10
    RETURN_TYPES = (_AnyType("*"),) * 10 + ("BOOLEAN",)
    RETURN_NAMES = tuple(f"output_{index}" for index in range(1, 11)) + ("switch",)

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()["optional"]
        empty_as_none = inputs.pop("empty_as_none")
        inputs["switch"] = ("BOOLEAN", {"forceInput": True})
        inputs["default_switch"] = ("BOOLEAN", {"default": False})
        inputs["empty_as_none"] = empty_as_none
        return {
            "required": {},
            "optional": inputs,
        }

    @classmethod
    def check_lazy_status(cls, default_switch=False, switch=None, **inputs):
        return super().check_lazy_status(default_switch if switch is None else switch, **inputs)

    def select(self, default_switch=False, switch=None, **inputs):
        active = default_switch if switch is None else switch
        return self._selected_outputs(active, inputs) + (bool(active),)


class DINKI_IfElseImageSwitch(_DINKI_IfElseBase):
    """Select the true branch only when a connected IMAGE supplies a value."""

    PAIR_COUNT = 10
    RETURN_TYPES = (_AnyType("*"),) * 10 + ("BOOLEAN",)
    RETURN_NAMES = tuple(f"output_{index}" for index in range(1, 11)) + ("switch",)

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()["optional"]
        empty_as_none = inputs.pop("empty_as_none")
        inputs["image"] = ("IMAGE", {"forceInput": True, "rawLink": True})
        inputs["empty_as_none"] = empty_as_none
        return {
            "required": {},
            "optional": inputs,
            "hidden": {"execution_list": "EXECUTION_LIST", "unique_id": "UNIQUE_ID"},
        }

    @staticmethod
    def _has_image(image, execution_list, unique_id):
        if image is None:
            return False
        if is_link(image):
            if execution_list is None or unique_id is None:
                return False
            cached = execution_list.get_cache(image[0], unique_id)
            if cached is None or cached.outputs is None or image[1] >= len(cached.outputs):
                return False
            values = cached.outputs[image[1]]
        else:
            values = image
        if not isinstance(values, (list, tuple)):
            values = (values,)
        for value in values:
            if value is None or isinstance(value, ExecutionBlocker):
                continue
            numel = getattr(value, "numel", None)
            if numel is None or numel() > 0:
                return True
        return False

    @classmethod
    def check_lazy_status(cls, image=None, execution_list=None, unique_id=None, **inputs):
        active = cls._has_image(image, execution_list, unique_id)
        return super().check_lazy_status(active, **inputs)

    def select(self, image=None, execution_list=None, unique_id=None, **inputs):
        active = self._has_image(image, execution_list, unique_id)
        return self._selected_outputs(active, inputs) + (active,)


class DINKI_IfElseBranch(_DINKI_IfElseBase):
    """Route four values and pass the upstream switch state onward."""

    PAIR_COUNT = 4
    RETURN_TYPES = (_AnyType("*"),) * 4 + ("BOOLEAN",)
    RETURN_NAMES = tuple(f"output_{index}" for index in range(1, 5)) + ("switch",)

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()["optional"]
        empty_as_none = inputs.pop("empty_as_none")
        inputs["switch"] = ("BOOLEAN", {"forceInput": True})
        inputs["empty_as_none"] = empty_as_none
        return {"required": {}, "optional": inputs}

    @classmethod
    def check_lazy_status(cls, switch=False, **inputs):
        return super().check_lazy_status(switch, **inputs)

    def select(self, switch=False, **inputs):
        return self._selected_outputs(switch, inputs) + (bool(switch),)


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
                "node_ids": ("STRING", {"multiline": False, "default": "1,2,3", "advanced": True}),
                "active": ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "do_nothing"
    CATEGORY = "DINKIssTyle/Util"
    OUTPUT_NODE = True

    def do_nothing(self, node_ids, active):
        # The bypassing logic happens in the frontend (JavaScript) before execution.
        # This python method is just a placeholder to satisfy the execution requirement.
        return ()







class DINKI_Node_Change:
    """Activate one group and bypass or mute the other in the frontend."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "node_ids_1": ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "Group 1 node IDs, separated by commas.",
                                      "advanced": True}),
            "node_ids_2": ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "Group 2 node IDs, separated by commas.",
                                      "advanced": True}),
            "active": ("BOOLEAN", {"default": True, "label_on": "Group 1",
                                   "label_off": "Group 2"}),
            "disable_mode": (["Bypass", "Mute"], {"default": "Bypass", "advanced": True}),
            "group_1_label": ("STRING", {"default": "Group 1", "multiline": False,
                                         "dynamicPrompts": False, "advanced": True}),
            "group_2_label": ("STRING", {"default": "Group 2", "multiline": False,
                                         "dynamicPrompts": False, "advanced": True}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "do_nothing"
    CATEGORY = "DINKIssTyle/Util"
    OUTPUT_NODE = True

    def do_nothing(self, node_ids_1, node_ids_2, active, disable_mode="Bypass",
                   group_1_label="Group 1", group_2_label="Group 2"):
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
    CATEGORY = "DINKIssTyle/Util"

    # 함수 인자도 input_text로 변경
    def switch_and_combine(self, select_string, input_text, text_in=None):
        current_text = select_string
        # Some combo menus return only the leaf label for slash-separated values.
        # Recover only an unambiguous match; never guess between duplicate leaves.
        lines = [line.strip() for line in input_text.splitlines() if line.strip()]
        if current_text and current_text not in lines:
            matches = [line for line in lines if line.endswith("/" + current_text)]
            if len(matches) == 1:
                current_text = matches[0]

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
