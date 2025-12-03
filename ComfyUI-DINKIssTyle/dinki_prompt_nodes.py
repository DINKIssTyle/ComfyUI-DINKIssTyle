import os
import csv
import random
from server import PromptServer
from aiohttp import web

# ============================================================================
# PART 1: Prompt Loader & Selectors (DINKI_Prompt_List.csv)
# ============================================================================

class PromptLoader:
    def __init__(self):
        self.prompt_data = {}

    def load_prompts(self):
        self.prompt_data = {}
        current_node_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(current_node_path, "csv", "DINKI_Prompt_List.csv")
        
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
                    pass 
                else:
                    print("⚠️ DINKI: DINKI_Prompt_List.csv is empty.")
            except Exception as e:
                print(f"❌ DINKI: Error reading CSV: {e}")
        else:
            print(f"⚠️ DINKI: CSV file not found at {file_path}")

    def get_prompt_by_title(self, title):
        return self.prompt_data.get(title, "")

prompt_loader = PromptLoader()

@PromptServer.instance.routes.get("/dinki/prompts")
async def get_prompts_route(request):
    prompt_loader.load_prompts()
    return web.json_response(prompt_loader.prompt_data)

@PromptServer.instance.routes.get("/get-csv-prompts")
async def get_csv_prompts(request):
    prompt_loader.load_prompts()
    titles = ["-- None --"] + sorted(list(prompt_loader.prompt_data.keys()), key=lambda s: s.lower())
    return web.json_response(titles)

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
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        prompt_loader.load_prompts()
        titles = ["-- None --"] + sorted(list(prompt_loader.prompt_data.keys()), key=lambda s: s.lower())
        return {
            "required": {
                "title": (titles,),
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "Text is updated by JS Logic..."}),
                "mode": (["append", "replace", "none"],),
                "separator": ("STRING", {"default": "\\n", "placeholder": r"\n, \n\n, ---"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "select_live"
    CATEGORY = "DINKIssTyle/Prompt"

    def select_live(self, title, text, mode, separator):
        return (text,)


# ============================================================================
# PART 2: Random Prompt Generator (DINKI_Random_Prompt.csv)
# ============================================================================

class DINKI_random_prompt:
    def __init__(self): pass
    
    @classmethod
    def INPUT_TYPES(s):
        current_node_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(current_node_path, "csv", "DINKI_Random_Prompt.csv")
        
        categories = {}
        current_category = None

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row: continue
                        col_a = row[0].strip() if len(row) > 0 else ""
                        col_b = row[1].strip() if len(row) > 1 else ""

                        if col_a:
                            current_category = col_a.replace(":", "")
                            if current_category not in categories:
                                categories[current_category] = []
                        
                        if col_b and current_category:
                            categories[current_category].append(col_b)
            except Exception as e:
                print(f"[DinkiRandomPrompt] Error reading CSV: {e}")
        else:
            print(f"[DinkiRandomPrompt] CSV file not found at: {file_path}")

        inputs = {
            "required": {
                "text_input": ("STRING", {"multiline": True, "default": "", "placeholder": "Optional prefix text..."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "enable_random": ("BOOLEAN", {"default": True, "label_on": "Random", "label_off": "Manual"}),
            },
            "optional": {}
        }

        # [수정] 리스트 생성 방식 안전하게 처리
        for cat_name, values in categories.items():
            if values and len(values) > 0:
                # 1. 원본 리스트 복사하여 사용 (안전)
                safe_values = list(values)
                # 2. 맨 앞에 "-- None --" 추가
                full_list = ["-- None --"] + safe_values
                # 3. 기본값은 실제 데이터의 첫 번째 값 (None이 아닌 값)
                default_val = safe_values[0]
                
                inputs["optional"][cat_name] = (full_list, {"default": default_val})
        
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_string",)
    FUNCTION = "generate_prompt"
    CATEGORY = "DINKIssTyle/Prompt"

    def generate_prompt(self, text_input, seed, enable_random, **kwargs):
        rng = random.Random(seed)
        selected_values = []
        
        current_node_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(current_node_path, "csv", "DINKI_Random_Prompt.csv")

        categories = {}
        current_category = None
        category_order = [] 

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row: continue
                    col_a = row[0].strip() if len(row) > 0 else ""
                    col_b = row[1].strip() if len(row) > 1 else ""
                    
                    if col_a:
                        current_category = col_a.replace(":", "")
                        if current_category not in categories:
                            categories[current_category] = []
                            category_order.append(current_category)
                    
                    if col_b and current_category:
                        categories[current_category].append(col_b)

        # 결과 생성
        for cat in category_order:
            ui_value = kwargs.get(cat, None)
            
            # None 선택 시 (랜덤 모드여도) 건너뛰기
            if ui_value == "-- None --":
                continue

            if enable_random:
                # 랜덤 모드: 카테고리 내에서 무작위 선택
                if cat in categories and categories[cat]:
                    picked = rng.choice(categories[cat])
                    selected_values.append(picked)
            else:
                # 수동 모드: UI 선택값 사용
                if ui_value:
                    selected_values.append(ui_value)
        
        csv_prompt_string = ", ".join(selected_values)
        
        final_string = ""
        has_text = text_input and text_input.strip()
        has_csv_prompt = bool(csv_prompt_string)

        if has_text:
            if has_csv_prompt:
                final_string = f"{text_input}, {csv_prompt_string}"
            else:
                final_string = text_input
        else:
            final_string = csv_prompt_string
        
        return (final_string,)
