# DKST Video Combine / Player 하드웨어 인코딩 계획

작성일: 2026-10-10. 아래 계획에 따라 하드웨어 인코딩을 구현했다.
사용 방법은 [Video Tools](DINKI_Video_Tools.md)의 `encoder` 설명을 참고한다.

## VHS에서 확인한 동작

VHS Video Combine은 영상 포맷 JSON의 `main_pass`를 FFmpeg 인자로 사용한다.
일반 `h264-mp4`는 `libx264`, `h265-mp4`는 `libx265`, `av1-webm`은
`libsvtav1`을 사용한다. 이들은 소프트웨어 인코더다.

별도로 `nvenc_h264-mp4`, `nvenc_hevc-mp4`, `nvenc_av1-mp4` 프리셋을 제공하며,
각각 `h264_nvenc`, `hevc_nvenc`, `av1_nvenc`를 선택한다. NVENC 프리셋은
비트레이트와 픽셀 포맷을 설정할 수 있고, 오디오는 AAC로 저장한다.
프리셋 목록을 읽는 과정에는 실제 GPU 초기화 검사가 없으므로 목록에
표시되었다는 사실만으로 실행 가능하다고 판단할 수 없다.

입력 텐서는 CPU NumPy 배열과 raw RGB 바이트로 변환되어 FFmpeg의 stdin으로
전달된다. 따라서 NVENC를 선택해도 프레임 전달과 색 변환 등 CPU 작업은
남는다. 오디오가 있으면 영상 인코딩 후 별도 FFmpeg 호출에서 `-c:v copy`로
영상을 재인코딩하지 않고 오디오를 합친다. GPU 사용 여부는 선택한 인코더에
따르며, ComfyUI의 이미지 생성 장치가 GPU라는 이유로 자동 결정되지 않는다.

근거:

- [VHS Video Combine 구현](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/main/videohelpersuite/nodes.py)
- [일반 H.264 프리셋](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/main/video_formats/h264-mp4.json)
- [일반 H.265 프리셋](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/main/video_formats/h265-mp4.json)
- [일반 AV1 프리셋](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/main/video_formats/av1-webm.json)
- [NVENC H.264](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/main/video_formats/nvenc_h264-mp4.json)
- [NVENC HEVC](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/main/video_formats/nvenc_hevc-mp4.json)
- [NVENC AV1](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/main/video_formats/nvenc_av1-mp4.json)
- [VHS의 FFmpeg 실행 파일 탐색](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/main/videohelpersuite/utils.py)

## DKST 적용 방향

현재 컨테이너/코덱을 묶은 `format` 값은 유지하고, 별도 `encoder` 선택을
추가한다. VHS의 NVENC 사용 원리를 가져오되, 파일 포맷과 실행 장치를 각각
선택할 수 있게 한다.

| 선택 | 동작 |
| --- | --- |
| `Auto` | 선택한 코덱과 픽셀 포맷을 유지하면서 사용 가능한 하드웨어 인코더를 우선 선택. 사용할 수 없으면 CPU 선택 및 이유 표시. |
| `CPU` | 기존 소프트웨어 인코더 사용. |
| `NVIDIA NVENC` | 실제 사용 가능한 NVIDIA 인코더 사용. 사용할 수 없으면 명확한 오류. |
| `Apple VideoToolbox` | 실제 사용 가능한 Mac 하드웨어 인코더 사용. 소프트웨어 내부 대체는 허용하지 않음. |

새 노드는 `Auto`를 기본값으로 제안한다. 기존 저장 워크플로에는 설정
마이그레이션으로 `CPU`를 넣고, 인자를 보내지 않는 기존 API 호출도 `CPU`로
처리해 기존 동작을 보존한다. 새 위젯은 기존 위젯 뒤에 추가하여 저장된
위젯 값의 위치를 바꾸지 않는다.

첫 지원 범위:

| 코덱 | CPU | NVIDIA | Mac |
| --- | --- | --- | --- |
| H.264 | libx264 | h264_nvenc | h264_videotoolbox |
| H.265 / HEVC | libx265 | hevc_nvenc | hevc_videotoolbox |
| AV1 | libsvtav1 | av1_nvenc, 지원 장치에서만 | 이번 범위에서는 제공하지 않음 |
| VP9 / ProRes / FFV1 / GIF / WebP | 현재 경로 | 이번 범위에서는 제공하지 않음 | 이번 범위에서는 제공하지 않음 |

Mac 경로는 GPU 셰이더 연산 여부가 아니라 VideoToolbox 하드웨어 인코딩으로
표시한다. 초기 범위에 AMD AMF와 Intel QSV는 포함하지 않는다.

## 구현 순서

1. **공통 인코더 선택 계층**
   - `dinki_video_combine.py`에서 컨테이너, 비디오 코덱, 오디오 코덱과 실행
     인코더를 분리한다. Combine과 Player에서 동일한 선택 함수를 사용한다.
   - PyAV를 우선 사용한다. PyAV에 해당 인코더가 없고 기존 FFmpeg 실행
     파일에 있다면, 그 실행 파일을 이용하는 선택적 보완 경로를 추가한다.
   - 기존 PATH/설정 경로와 이미 설치된 `imageio_ffmpeg`만 탐색한다.
     패키지 설치, 실행 파일 다운로드, 드라이버 설치는 하지 않는다.
   - FFmpeg 경로와 라이브러리 버전은 진단 정보로 남긴다. 코덱 이름과
     하드웨어 종류가 같아도 PyAV와 실행 파일의 지원 목록은 다를 수 있다.

2. **실제 사용 가능 여부 검사**
   - 코덱 등록 목록을 1차 확인한 뒤, 짧은 테스트 인코딩과 flush로 드라이버,
     장치, 세션 초기화까지 확인한다. 검사는 서버에서 실행한다.
   - 별도 프로세스와 시간 제한으로 검사 중 UI 정지와 장시간 대기를 막는다.
     백엔드·인코더·픽셀 포맷·런타임 버전별 결과를 캐시한다.
   - Mac 하드웨어 경로는 `allow_sw=0`으로 테스트하고 실제 인코딩에도 적용한다.
   - 작은 테스트가 통과해도 실제 해상도/설정에서 초기화가 실패할 수 있다.
     실제 인코딩 시작 시에도 이를 처리한다.

3. **픽셀 포맷과 비트레이트 대응**
   - 선택 인코더가 지원하는 픽셀 포맷만 노출한다. `nv12`, `p010le`처럼
     하드웨어 인코더가 사용하는 포맷도 추가한다.
   - 10-bit YUV를 P010으로 전달해야 하는 경우 샘플링과 정밀도를 보존하는
     내부 변환을 사용하고, 실제 사용한 포맷을 표시한다.
   - GPU 미지원 픽셀 포맷을 조용히 8-bit/4:2:0으로 바꾸지 않는다.
     Auto는 같은 출력을 만들 수 있는 CPU 경로로 돌아간다.
   - 기존 Mbps 설정을 각 인코더의 목표 비트레이트로 전달한다.
     `0`의 자동 품질은 인코더별로 정의한다. CPU CRF 값을 하드웨어 인코더에
     그대로 전달하지 않는다. 자동 비트레이트 산정이 필요하면 적용값을 표시한다.
   - BT.709, 색 범위, HEVC `hvc1`, 홀수 해상도 패딩은 현재 규칙을 유지한다.

4. **실패 처리와 파일 생성**
   - Auto의 CPU 대체는 장치/인코더 초기화 실패에 한정한다. 잘못된 입력,
     디스크 오류, 취소를 GPU 실패로 취급하여 재시도하지 않는다.
   - 직접 선택한 하드웨어 인코더의 실패는 오류로 알린다.
   - 인코딩 중 세션이 실패하면 미완성 파일을 정리하고 작업을 실패 처리한다.
     재시도는 사용자가 다시 실행하며, 완성된 것처럼 파일명을 반환하지 않는다.
   - PyAV의 오디오 처리·트림·무음 패딩·타임스탬프·메타데이터 규칙을 유지한다.
     FFmpeg 보완 경로도 같은 규칙을 적용하고, 영상 재인코딩 없이 오디오를 합친다.
   - 취소/실패 시 subprocess 종료, 파이프 해제, partial 파일 정리를 검증한다.

5. **두 노드와 프리뷰 연결**
   - Combine과 Video Player에 같은 인코더 선택과 지원 상태를 제공한다.
   - 프리뷰 툴바에 실제 사용한 인코더를 표시한다.
     예: `H.264 · NVENC`, `HEVC · VideoToolbox`, `H.264 · CPU`.
   - Auto가 CPU로 돌아간 경우 이유를 짧게 표시한다.
   - Player의 기존 기본 저장 경로는 유지한다. 하드웨어 인코더를 선택한
     재인코딩은 공통 계층을 사용하며, VIDEO passthrough는 그대로 유지한다.
   - H.264 MP4처럼 직접 재생 가능한 출력은 그대로 미리보기한다. 프록시가
     필요한 출력은 같은 계층으로 H.264 프록시를 만들고, 가능하면 하드웨어를
     사용한다. 프록시가 CPU로 처리되었다면 원본 인코더와 구분해 표시한다.
   - Fit, 100%, 해상도, 메뉴 다운로드, `always_save`, filename 출력은 유지한다.

6. **검증과 완료 조건**
   - 단위 테스트: 인코더 선택, Auto의 대체, 직접 선택 실패, 캐시,
     지원 픽셀 포맷, 기존 워크플로/위젯 마이그레이션.
   - 실제 파일 테스트: NVIDIA H.264/HEVC 및 지원 장치의 AV1, Mac H.264/HEVC,
     오디오 동기화, 소수 FPS, 8/10-bit, 메타데이터, 프록시와 다운로드 원본 구분.
   - CPU와 하드웨어가 동일 프레임 수, FPS, 해상도, 허용 오차 내 오디오
     길이를 만드는지 검사한다. 손실 인코더 간 픽셀 완전 일치는 요구하지 않는다.
   - 같은 입력/비트레이트로 전체 처리 시간, 인코딩 시간, 프록시 시간,
     CPU 사용률, 파일 크기를 비교한다. GPU 장치가 없는 테스트는 미검증으로
     기록하고 CPU 또는 mock 통과를 GPU 검증으로 보고하지 않는다.
   - 실제 ComfyUI에서 일반 노드와 Nodes 2.0, 재실행, 워크플로 저장/복구,
     취소, 실행 장치 선택과 프리뷰를 확인한다.

## 현재 환경에서 확인한 범위와 한계

이번 확인은 Mac ARM64의 별도 테스트 런타임(PyAV 19.0.1)에서 수행했다.
`h264_videotoolbox`, `hevc_videotoolbox`로 실제 파일을 인코딩했고,
HEVC 10-bit, 오디오, 30000/1001 FPS, 메타데이터, 프록시와 Auto Player
경로를 검증했다. 기존 FFmpeg를 사용하는 VideoToolbox 보완 경로도 통과했다.
NVENC 코덱은 이 런타임에 등록되지 않았으므로 NVIDIA 실기 테스트는
건너뛰었다. 실제 ComfyUI 화면에서의 검증은 아직 수행하지 않았다.

같은 640×360 합성 노이즈 32프레임, 24 FPS, 목표 8 Mbps, 오디오 입력을
H.264로 처리한 단일 측정 결과는 다음과 같다. 초기 장치 검사는 측정 전에
실행했으며, 처리 시간에는 프레임 변환과 오디오 인코딩이 포함된다.
CPU는 현재 노드의 libx264 설정(2개 인코딩 스레드)을 사용했다.

| 경로 | 전체 시간 | 현재 Python 프로세스 CPU 시간 | 파일 크기 |
| --- | --- | --- | --- |
| CPU (libx264) | 0.475초 | 0.941초 | 1,929,253 bytes |
| VideoToolbox | 0.180초 | 0.126초 | 2,179,106 bytes |

동일 목표 비트레이트도 실제 크기와 화질이 같다는 뜻은 아니다. 이 측정은
하나의 합성 입력 결과이며, 다른 영상·장치의 속도 향상을 보장하지 않는다.

첫 구현에서도 CPU RGB 변환과 하드웨어로의 프레임 전송은 남는다.
GPU 텐서의 직접 전달이나 GPU 색 변환은 별도 최적화 과제로 둔다.
PyAV/FFmpeg 지원 목록과 장치 초기화 결과는 서버 프로세스에서 캐시한다.
지원 환경이 바뀌면 ComfyUI 서버를 재시작하여 다시 검사한다.

### NVENC 사전 검사 해상도 수정 (2026-10-10)

RTX 3080 Ti / Windows 보고서에서 실제 출력은 2272×1280인데도 사전 검사의
128×128 프레임이 NVENC 최소 해상도 제한에 걸렸다. Auto는 CPU로 대체했고,
NVENC 직접 선택은 `Frame Dimension less than the minimum supported value`로
실패했다. PyAV와 FFmpeg 양쪽의 검사 캔버스를 256×256으로 변경했다.
사용자 영상의 출력 해상도는 바꾸지 않는다. 검사 크기도 캐시 키에 포함한다.
업데이트 후 ComfyUI 서버를 재시작하면 기존 실패 캐시가 지워지고 다시 검사한다.

두 검사 경로에서 최소 크기를 거부하는 장치를 재현하여 Auto와 NVENC 직접
선택이 모두 GPU를 선택하는 회귀 테스트를 추가했다. NVIDIA 실기 검증은
현재 Mac 환경에서 수행할 수 없으며, 실제 출력에서의 초기화 실패 처리도 유지한다.

최소 해상도 관련 근거:
[NVIDIA 개발자 포럼 답변](https://forums.developer.nvidia.com/t/minimum-width-in-turing-gpus/155566).

Mac 지원과 옵션 근거:
[FFmpeg VideoToolbox 구현](https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/videotoolboxenc.c).
