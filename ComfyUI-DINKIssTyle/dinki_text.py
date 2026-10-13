class DINKI_Text_Note:
    """A workflow note with frontend Markdown preview, Lock and Copy controls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"default": "", "multiline": True})}}

    RETURN_TYPES = ()
    FUNCTION = "note"
    CATEGORY = "DINKIssTyle/Text"

    def note(self, text):
        return ()


class DINKI_Text_Multiline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "output_text"
    CATEGORY = "DINKIssTyle/Text"

    def output_text(self, text):
        return (text,)


class DINKI_Text_Multiline_Cache:
    """Keep the last received string in a saved, editable multiline widget."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "", "multiline": True, "dynamicPrompts": False,
                    "tooltip": "Cached text. New text_in replaces this value; no new text keeps it. Edit here to replace or clear the cache.",
                }),
            },
            "optional": {
                # A normal linked input propagates ExecutionBlocker before this
                # function runs. A raw link lets us fall back for blocked output,
                # while its eager dependency still executes an enabled producer.
                "text_in": ("STRING", {
                    "forceInput": True, "rawLink": True,
                    "tooltip": "Connect incoming text here. Empty, None, muted or blocked input keeps the cached text.",
                }),
            },
            "hidden": {"execution_list": "EXECUTION_LIST", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "output_text"
    CATEGORY = "DINKIssTyle/Text"
    DESCRIPTION = (
        "Cache incoming text in the editable multiline text field. Connect the source to text_in. "
        "New non-empty text replaces the cache; empty strings, None, muted/disconnected inputs "
        "and blocked branches keep the previous text. Whitespace is preserved. "
        "For a list of incoming texts, keep the last non-empty string. "
        "Edit the text field to replace or clear the cache. Save the workflow to retain it."
    )

    @staticmethod
    def _incoming_text(text_in, execution_list, unique_id):
        if (isinstance(text_in, (list, tuple)) and len(text_in) == 2
                and isinstance(text_in[0], str) and isinstance(text_in[1], int)):
            if execution_list is None or unique_id is None:
                return None
            cached = execution_list.get_cache(text_in[0], unique_id)
            index = text_in[1]
            if cached is None or cached.outputs is None or not 0 <= index < len(cached.outputs):
                return None
            values = cached.outputs[index]
        else:
            values = text_in
        if not isinstance(values, (list, tuple)):
            values = (values,)
        # Only strings can update the cache. This also excludes ExecutionBlocker
        # without making the otherwise standalone text module depend on ComfyUI.
        return next((value for value in reversed(values) if isinstance(value, str) and value != ""), None)

    def output_text(self, text="", text_in=None, execution_list=None, unique_id=None):
        incoming = self._incoming_text(text_in, execution_list, unique_id)
        value = text if incoming is None else incoming
        # Do not replay a fallback from an old prompt snapshot onto a newer widget.
        ui = {} if incoming is None else {"dkst_cached_text": [incoming]}
        return {"ui": ui, "result": (value,)}


class DINKI_Text_Split:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "delimiter": ("STRING", {"default": ",", "multiline": True}),
                "clean_whitespace": (
                    "BOOLEAN",
                    {"default": True, "label_on": "true", "label_off": "false"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",) * 10
    RETURN_NAMES = tuple(f"text_{index}" for index in range(1, 11))
    FUNCTION = "split_text"
    CATEGORY = "DINKIssTyle/Text"
    DESCRIPTION = (
        "Split text at an exact delimiter into up to ten outputs. "
        "Enter an actual line break to split lines. Empty delimiters leave text unsplit. "
        "Empty sections keep their positions; output 10 contains any remaining text."
    )

    def split_text(self, text, delimiter=",", clean_whitespace=True):
        parts = text.split(delimiter, 9) if delimiter else [text]
        if clean_whitespace:
            parts = [part.strip() for part in parts]
        return tuple(parts + [""] * (10 - len(parts)))


class DINKI_Text_Concatenate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delimiter": ("STRING", {"default": ", ", "multiline": False}),
                "clean_whitespace": (
                    "BOOLEAN",
                    {"default": True, "label_on": "true", "label_off": "false"},
                ),
            },
            "optional": {
                f"text_{letter}": ("STRING", {"forceInput": True})
                for letter in "abcdefghij"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "concatenate"
    CATEGORY = "DINKIssTyle/Text"

    def concatenate(
        self,
        delimiter=", ",
        clean_whitespace=True,
        text_a=None,
        text_b=None,
        text_c=None,
        text_d=None,
        text_e=None,
        text_f=None,
        text_g=None,
        text_h=None,
        text_i=None,
        text_j=None,
    ):
        texts = (
            text_a,
            text_b,
            text_c,
            text_d,
            text_e,
            text_f,
            text_g,
            text_h,
            text_i,
            text_j,
        )

        if clean_whitespace:
            texts = tuple(text.strip() for text in texts if text is not None)
        else:
            texts = tuple(text for text in texts if text is not None)

        return (delimiter.join(text for text in texts if text != ""),)
