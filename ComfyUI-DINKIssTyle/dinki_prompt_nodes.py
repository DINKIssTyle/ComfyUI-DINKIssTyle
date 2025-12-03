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
        
        # 현재 파일(dinki_prompt_nodes.py) 위치 기준
        current_node_path = os.path.dirname(os.path.realpath(__file__))
        # 'csv' 폴더 내의 'DINKI_Prompt_List.csv'
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

# 전역 인스턴스 생성
prompt_loader = PromptLoader()

# --- [API Route] JS와 통신을 위한 라우트 ---
@PromptServer.instance.routes.get("/dinki/prompts")
async def get_prompts_route(request):
    prompt_loader.load_prompts() # 요청 시 갱신
    return web.json_response(prompt_loader.prompt_data)

@PromptServer.instance.routes.get("/get-csv-prompts")
async def get_csv_prompts(request):
    prompt_loader.load_prompts()
    titles = ["-- None --"] + sorted(list(prompt_loader.prompt_data.keys()), key=lambda s: s.lower())
    return web.json_response(titles)
# ---------------------------------------------

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
        pass

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
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        # 현재 파일 위치 기준
        current_node_path = os.path.dirname(os.path.realpath(__file__))
        # 'csv' 폴더 내의 'DINKI_Random_Prompt.csv'
        file_path = os.path.join(current_node_path, "csv", "DINKI_Random_Prompt.csv")
        
        categories = {}
        current_category = None

        # CSV 파일 읽기 및 파싱
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row: continue # 빈 줄 건너뛰기
                        
                        col_a = row[0].strip() if len(row) > 0 else ""
                        col_b = row[1].strip() if len(row) > 1 else ""

                        # 카테고리 인식 (A열)
                        if col_a:
                            current_category = col_a.replace(":", "")
                            if current_category not in categories:
                                categories[current_category] = []
                        
                        # 값 추가 (B열)
                        if col_b and current_category:
                            categories[current_category].append(col_b)
            except Exception as e:
                print(f"[DinkiRandomPrompt] Error reading CSV: {e}")
        else:
            print(f"[DinkiRandomPrompt] CSV file not found at: {file_path}")

        # 입력 위젯 구성
        inputs = {
            "required": {
                # [추가됨] Prefix 입력을 위한 텍스트 박스
                "text_input": ("STRING", {"multiline": True, "default": "", "placeholder": "Optional prefix text..."}),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "enable_random": ("BOOLEAN", {"default": True, "label_on": "Random", "label_off": "Manual"}),
            },
            "optional": {}
        }

        # 카테고리별 드랍다운 메뉴 동적 추가
        for cat_name, values in categories.items():
            if values:
                inputs["optional"][cat_name] = (values, )
        
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_string",)
    FUNCTION = "generate_prompt"
    CATEGORY = "DINKIssTyle/Prompt"

    def generate_prompt(self, text_input, seed, enable_random, **kwargs):
        # 재현성을 위한 시드 설정
        rng = random.Random(seed)
        
        selected_values = []
        
        # 파일 경로 재설정 (실행 시점 경로 확보)
        current_node_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(current_node_path, "csv", "DINKI_Random_Prompt.csv")

        categories = {}
        current_category = None
        category_order = [] # 순서 보장을 위한 리스트

        # 파일을 다시 읽어 카테고리 순서와 전체 데이터 확보
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
            if enable_random:
                # 랜덤 모드: 해당 카테고리 전체 리스트에서 랜덤 선택
                if cat in categories and categories[cat]:
                    picked = rng.choice(categories[cat])
                    selected_values.append(picked)
            else:
                # 수동 모드: UI 드랍다운에서 선택된 값(kwargs) 사용
                if cat in kwargs:
                    selected_values.append(kwargs[cat])
        
        # CSV에서 생성된 랜덤 프롬프트 조합
        csv_prompt_string = ", ".join(selected_values)
        
        # [추가됨] text_input과 CSV 결과 합치기
        final_string = ""
        
        # 입력된 텍스트가 있는지 확인 (공백 제거 후 확인)
        has_text = text_input and text_input.strip()
        has_csv_prompt = bool(csv_prompt_string)

        if has_text:
            if has_csv_prompt:
                # 둘 다 있으면: "텍스트, CSV프롬프트"
                final_string = f"{text_input}, {csv_prompt_string}"
            else:
                # 텍스트만 있으면: "텍스트"
                final_string = text_input
        else:
            # 텍스트가 없으면 CSV 프롬프트만 (비어있어도 빈 문자열)
            final_string = csv_prompt_string
        
        return (final_string,)