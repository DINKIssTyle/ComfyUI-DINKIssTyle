class DINKI_Text_Note:
    """An editable workflow note with frontend Lock and Copy controls."""

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
