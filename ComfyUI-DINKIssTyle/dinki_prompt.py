import os
import csv
import random
from server import PromptServer
from aiohttp import web
import comfy.samplers
import sys

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
                    print("⚠️ 🅳INKIssTyle - DINKI_Prompt_List.csv is empty.")
            except Exception as e:
                print(f"❌ 🅳INKIssTyle - Error reading CSV: {e}")
        else:
            print(f"⚠️ 🅳INKIssTyle - CSV file not found at {file_path}")

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
                print(f"[🅳INKIssTyle - Random Prompt] Error reading CSV: {e}")
        else:
            print(f"[🅳INKIssTyle - Random Prompt] CSV file not found at: {file_path}")

        inputs = {
            "required": {
                "text_input": ("STRING", {"multiline": True, "default": "", "placeholder": "Optional prefix text..."}),
                # [추가] Active 스위치: seed 위에 배치
                "Active": ("BOOLEAN", {"default": True}), 
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {}
        }

        # 리스트 구성: None, Random, 값들...
        for cat_name, values in categories.items():
            if values and len(values) > 0:
                safe_values = list(values)
                
                # 1. 옵션 목록 생성: 맨 앞에 "-- None --"과 "-- Random --" 추가
                full_list = ["-- None --", "-- Random --"] + safe_values
                
                # 2. 기본값은 "-- Random --"
                default_val = "-- None --"
                
                inputs["optional"][cat_name] = (full_list, {"default": default_val})
        
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_string",)
    FUNCTION = "generate_prompt"
    CATEGORY = "DINKIssTyle/Prompt"

    # [수정] Active 인자 추가
    def generate_prompt(self, text_input, Active, seed, **kwargs):
        # [추가] Active가 False(끄기) 상태면 텍스트 입력창의 내용만 바로 반환 (Bypass 기능)
        if not Active:
            return (text_input,)

        rng = random.Random(seed)
        selected_values = []
        
        current_node_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(current_node_path, "csv", "DINKI_Random_Prompt.csv")

        categories = {}
        current_category = None
        category_order = [] 

        # 실행 시 CSV를 다시 읽어 실제 데이터 목록 확보 (Random 선택용)
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

        # 결과 생성 로직
        for cat in category_order:
            ui_value = kwargs.get(cat, None)
            
            # 1. UI에서 선택된 값이 없거나 "-- None --"이면 건너뜀
            if not ui_value or ui_value == "-- None --":
                continue

            # 2. "-- Random --" 선택 시 해당 카테고리 목록에서 무작위 추출
            if ui_value == "-- Random --":
                if cat in categories and categories[cat]:
                    picked = rng.choice(categories[cat])
                    selected_values.append(picked)
            
            # 3. 특정 값을 선택했을 경우 그대로 사용
            else:
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


# ============================================================================
# PART 3: Sampler Preset (JS Version) (DINKI_Sampler_Preset.csv)
# ============================================================================
# [1] API 서버 설정 및 데이터 로드
SAMPLER_PRESET_DATA = {}
ALL_PRESETS_LIST = [] # [핵심 수정] 모든 프리셋 이름을 담을 리스트

def load_sampler_presets():
    global SAMPLER_PRESET_DATA, ALL_PRESETS_LIST
    current_dir = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(current_dir, "csv", "DINKI_Sampler_Preset.csv") 
    
    data = {}
    all_presets = set() # 중복 방지를 위해 set 사용

    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None) # 헤더 스킵
                for row in reader:
                    if len(row) >= 4:
                        model = row[0].strip()
                        preset = row[1].strip()
                        sampler = row[2].strip()
                        scheduler = row[3].strip()
                        
                        display_name = f"{preset} [{sampler} / {scheduler}]"
                        
                        if model not in data:
                            data[model] = []
                        
                        data[model].append({
                            "name": preset,
                            "sampler": sampler,
                            "scheduler": scheduler,
                            "display": display_name
                        })
                        
                        # [핵심] 백엔드 검증 통과를 위해 모든 이름 수집
                        all_presets.add(display_name)
                        
        except Exception as e:
            print(f"[🅳INKIssTyle - Error] Sampler CSV Read: {e}")
            
    # set을 리스트로 변환하고 정렬
    ALL_PRESETS_LIST = sorted(list(all_presets))
    return data

# 서버 시작 시 데이터 로드
SAMPLER_PRESET_DATA = load_sampler_presets()

# JS에서 호출할 API 경로
@PromptServer.instance.routes.get("/dinki/sampler_presets")
async def get_dinki_sampler_presets(request):
    return web.json_response(SAMPLER_PRESET_DATA)


# 연결 호환성을 위한 만능 타입
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class DINKI_Sampler_Preset:
    @classmethod
    def INPUT_TYPES(cls):
        # 모델 목록 확보
        model_list = list(SAMPLER_PRESET_DATA.keys()) if SAMPLER_PRESET_DATA else ["No CSV Data"]
        
        # [핵심 수정] preset 목록에 "Select Model First"만 넣는 게 아니라,
        # CSV에 존재하는 '모든 프리셋'을 넣어줍니다.
        # 이렇게 하면 JS가 어떤 값을 보내도 이 리스트 안에 있으므로 에러가 나지 않습니다.
        # (화면에는 JS가 필터링해서 보여주므로 사용자는 이 긴 목록을 볼 일이 없습니다)
        preset_list = ALL_PRESETS_LIST if ALL_PRESETS_LIST else ["Select Model First"]
        
        return {
            "required": {
                "model": (model_list,), 
                "preset": (preset_list,), 
            }
        }

    # 혹시 모를 검증 에러 방지를 위한 2중 안전장치
    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = (AnyType("*"), AnyType("*"), "STRING")
    RETURN_NAMES = ("sampler_name", "scheduler_name", "info")
    FUNCTION = "process"
    CATEGORY = "DINKIssTyle/Utils"

    def process(self, model, preset):
        target_sampler = "euler"
        target_scheduler = "normal"
        
        # 메모리 데이터에서 매칭되는 값 찾기
        found = False
        if model in SAMPLER_PRESET_DATA:
            for p in SAMPLER_PRESET_DATA[model]:
                if p["display"] == preset:
                    target_sampler = p["sampler"]
                    target_scheduler = p["scheduler"]
                    found = True
                    break
        
        # 만약 정확한 매칭을 못 찾았을 경우 (JS와 Python 싱크 문제 등)
        # 1. 전체 데이터에서라도 이름이 같은 걸 찾아봅니다.
        if not found:
            for m_key, m_val in SAMPLER_PRESET_DATA.items():
                for p in m_val:
                    if p["display"] == preset:
                        target_sampler = p["sampler"]
                        target_scheduler = p["scheduler"]
                        found = True
                        break
                if found: break

        info = f"Model: {model} | Preset: {preset}"
        
        return (target_sampler, target_scheduler, info)






