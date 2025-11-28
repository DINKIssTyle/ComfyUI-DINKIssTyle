import os

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
        # 전체 경로에서 파일명만 추출합니다.
        # ComfyUI 서버는 output 폴더 내의 파일을 파일명으로 찾습니다.
        video_name = os.path.basename(filename)
        
        # UI 쪽으로 비디오 파일명 정보를 보냅니다.
        # type="output"은 ComfyUI 기본 output 폴더를 의미합니다.
        return {"ui": {"video": [video_name]}}

# Node Mapping
NODE_CLASS_MAPPINGS = {
    "DINKI_Video_Player": DINKI_Video_Player
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINKI_Video_Player": "DINKI Video Player"
}