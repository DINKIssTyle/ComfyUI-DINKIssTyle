# ============================================================
# DINKIssTyle Cross-Platform Helper
# 이 커스텀노드는 매 ComfyUI 부트시 아래 역할을 합니다.
# 1. macOS에서 윈도우에 올린 한글 파일명의 문제를 해결합니다.
# 2. macOS에서 윈도우에 올린 파일로 생긴 macOS 리소스포크를 제거합니다.
# ============================================================

import os
import sys
import unicodedata
import folder_paths  # ComfyUI 기본 경로 모듈

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ============================================================
# Auto Input Cleaner & NFC Normalizer
# ============================================================

def normalize_to_nfc(name: str) -> str:
    """문자열을 NFC로 정규화"""
    return unicodedata.normalize("NFC", name)

def auto_clean_and_normalize_input():
    """
    ComfyUI의 input 폴더를 스캔하여:
    1. macOS 리소스 포크 파일(._*) 삭제 (크기 검증 포함)
    2. 자소 분리된 한글(NFD) 파일명을 NFC(완성형)로 자동 변환
    """
    # 1. ComfyUI input 폴더 경로 가져오기
    try:
        input_dir = folder_paths.get_input_directory()
    except Exception:
        input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../input"))

    if not os.path.exists(input_dir):
        print(f"[DINKIssTyle] 'input' directory not found: {input_dir}")
        return

    print(f"\n[DINKIssTyle] Scanning input folder: {input_dir}")
    
    renamed_count = 0
    deleted_count = 0
    
    # 리소스 포크 파일 최대 허용 크기 (128KB)
    # 보통 ._ 파일은 4KB 내외지만, 아이콘 데이터가 포함될 경우를 대비해 넉넉히 잡음
    MAX_RESOURCE_SIZE = 128 * 1024 

    # 2. 재귀적으로 폴더 탐색
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            full_path = os.path.join(root, fname)
            
            # --------------------------------------------------------
            # [기능 1] macOS 리소스 포크(._) 파일 삭제
            # --------------------------------------------------------
            if fname.startswith("._"):
                try:
                    file_size = os.path.getsize(full_path)
                    # 크기가 작을 때만 삭제 (안전 장치)
                    if file_size <= MAX_RESOURCE_SIZE:
                        os.remove(full_path)
                        print(f"   [DINKIssTyle - DELETE] {fname} ({file_size} bytes) - macOS Resource Fork")
                        deleted_count += 1
                        continue # 삭제했으므로 다음 파일로 넘어감
                except Exception as e:
                    print(f"   [DINKIssTyle - ERROR] Failed to delete {fname}: {e}")
                    continue

            # --------------------------------------------------------
            # [기능 2] 한글 파일명 NFC 정규화
            # --------------------------------------------------------
            new_name = normalize_to_nfc(fname)
            
            if fname != new_name:
                new_full_path = os.path.join(root, new_name)
                
                # 충돌 방지 로직
                if os.path.exists(new_full_path):
                    base, ext = os.path.splitext(new_name)
                    candidate = f"{base}_nfc{ext}"
                    idx = 1
                    while os.path.exists(os.path.join(root, candidate)):
                        candidate = f"{base}_nfc({idx}){ext}"
                        idx += 1
                    new_full_path = os.path.join(root, candidate)
                
                try:
                    os.rename(full_path, new_full_path)
                    print(f"   [DINKIssTyle - RENAME] {fname} -> {os.path.basename(new_full_path)}")
                    renamed_count += 1
                except Exception as e:
                    print(f"   [DINKIssTyle - ERROR] Failed to rename {fname}: {e}")

    # 결과 리포트
    if renamed_count > 0 or deleted_count > 0:
        print(f"[DINKIssTyle] Done. Renamed: {renamed_count}, Deleted: {deleted_count} files.\n")
    else:
        # 변경사항 없으면 로그 생략 (원하면 주석 해제)
        # print("[DINKIssTyle] System Clean. No actions needed.\n")
        pass

# ============================================================
# 실행 (모듈 로드 시 즉시 실행됨)
# ============================================================
auto_clean_and_normalize_input()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]