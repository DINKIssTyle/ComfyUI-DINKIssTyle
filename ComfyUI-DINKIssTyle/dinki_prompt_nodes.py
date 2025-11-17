# ComfyUI/custom_nodes/ComfyUI-DINKIssTyle/dinki_prompt_nodes.py

import os
import folder_paths


class PromptLoader:
    def __init__(self):
        self.prompt_data = {}

    def load_prompts(self):
        self.prompt_data = {}
        file_path = os.path.join(folder_paths.get_input_directory(), "prompt_list.csv")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split(',', 1)
                            if len(parts) == 2:
                                self.prompt_data[parts[0].strip()] = parts[1].strip()
                if self.prompt_data:
                    print(f"✅ DINKI_PromptSelector: Loaded {len(self.prompt_data)} prompts.")
                else:
                    print("⚠️ DINKI_PromptSelector: prompt_list.csv is empty or malformed.")
            except Exception as e:
                print(f"❌ DINKI_PromptSelector: Error reading CSV: {e}")
        else:
            print("⚠️ DINKI_PromptSelector: prompt_list.csv not found.")
            self.prompt_data = {"Example Prompt": "Create prompt_list.csv in your input folder."}

    def get_prompt_by_title(self, title):
        return self.prompt_data.get(title, "")


# 전역 인스턴스 (라우터/노드에서 공용 사용)
prompt_loader = PromptLoader()


class DINKI_PromptSelector:
    @classmethod
    def INPUT_TYPES(s):
        prompt_loader.load_prompts()
        titles = ["-- None --"] + sorted(list(prompt_loader.prompt_data.keys()), key=lambda s: s.lower())
        return {"required": {"title": (titles,)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "select_prompt"
    CATEGORY = "DINKIssTyle/Prompt"

    def select_prompt(self, title):
        prompt_loader.load_prompts()
        if title == "-- None --":
            return ("",)
        return (prompt_loader.get_prompt_by_title(title) or "",)


class DINKI_PromptSelectorLive:
    def __init__(self):
        # 실행 간 인스턴스 상태 (중복 방지)
        self._last = {"title": None, "mode": None, "sep": None}

    @classmethod
    def INPUT_TYPES(cls):
        prompt_loader.load_prompts()
        titles = ["-- None --"] + sorted(list(prompt_loader.prompt_data.keys()), key=lambda s: s.lower())
        return {
            "required": {
                "title": (titles,),
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "prompt"}),
                "mode": (["append", "replace", "none"],),
                "separator": ("STRING", {"default": "\n", "placeholder": r"\n, \n\n, ---"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "select_live"
    CATEGORY = "DINKIssTyle/Prompt"

    def _norm_tail(self, s: str) -> str:
        return (s or "").rstrip()

    def select_live(self, title, text, mode, separator):
        prompt_loader.load_prompts()
        # 1) None 선택 시 패스
        if title == "-- None --":
            return (text or "",)

        picked = (prompt_loader.get_prompt_by_title(title) or "").strip()
        sep = "\n" if separator == r"\n" else ("\n\n" if separator == r"\n\n" else (separator or ""))

        # 2) 직전 실행과 동일(제목/모드/구분자) → 재첨부 방지
        if self._last == {"title": title, "mode": mode, "sep": sep} and mode == "append":
            return (text or "",)

        # 3) 모드 처리
        text_curr = text or ""
        if mode == "replace":
            out = picked
        elif mode == "append":
            if not picked:
                out = text_curr
            else:
                tail_plain = self._norm_tail(picked)
                tail_with_sep = self._norm_tail((sep or "") + picked)
                norm_text = self._norm_tail(text_curr)
                if norm_text.endswith(tail_plain) or norm_text.endswith(tail_with_sep):
                    out = text_curr  # 이미 붙어있음
                else:
                    joiner = "" if (not sep or text_curr.endswith(sep)) else sep
                    out = text_curr + (picked if not text_curr else joiner + picked)
        else:  # mode == "none"
            out = text_curr

        # 4) 상태 저장
        self._last = {"title": title, "mode": mode, "sep": sep}
        return (out,)
