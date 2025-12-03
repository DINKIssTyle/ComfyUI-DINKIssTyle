import os
import folder_paths # 경로 확인을 위해 필요

class DINKI_Video_Player:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "show_video"
    OUTPUT_NODE = True
    CATEGORY = "DINKIssTyle/Video"

    def show_video(self, filename):
        # 1. 파일명 추출
        video_name = os.path.basename(filename)
        
        # 2. 파일이 temp 폴더에 있는지 output 폴더에 있는지 감지
        # 기본값은 output
        source_type = "output"
        
        # 입력된 절대 경로(filename)가 temp 폴더 경로로 시작하는지 확인
        temp_dir = folder_paths.get_temp_directory()
        if filename.startswith(temp_dir):
            source_type = "temp"
        
        # 3. UI에 파일명뿐만 아니라 type 정보도 함께 전달 (Dictionary 형태)
        # 이렇게 보내야 프론트엔드(JS)가 type="temp" 파라미터를 붙여서 요청할 수 있음
        return {"ui": {"video": [{"filename": video_name, "type": source_type, "subfolder": ""}]}}

# Node Mapping
NODE_CLASS_MAPPINGS = {
    "DINKI_Video_Player": DINKI_Video_Player
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_Video_Player": "DINKI Video Player"
}