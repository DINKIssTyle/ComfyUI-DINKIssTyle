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

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "DINKI_Node_Switch": DINKI_Node_Switch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_Node_Switch": "DINKI Node Switch"
}