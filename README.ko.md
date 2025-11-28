## 소개 (Introduction)

이 저장소는 제가 ComfyUI로 작업하면서 겪었던 다양한 필요를 해결하기 위해 직접 제작한 커스텀 노드들을 모아둔 곳입니다.
주로 **Qwen-Image**, **Flux**, **WAN** 모델을 활용하는 제 개인적인 워크플로우에 맞춰 설계되었으므로, 다른 모델과 함께 사용할 경우 예상치 못한 문제가 발생할 수 있습니다.

## 노드 설명 (Node Descriptions)

### 📐 DINKI Resize and Pad Image / Remove Pad

이 두 노드는 종횡비 변화나 해상도 리사이징에 민감한 **Qwen Image Edit**과 같은 이미지 편집 모델을 활용한 워크플로우에 필수적입니다.

**1. DINKI Resize and Pad Image:** 입력 이미지를 지정된 정사각형 해상도(기본 **1024×1024**) 안에 맞추되, *원본 비율을 유지*하며 리사이징합니다. 남는 공간은 자동으로 패딩(레터박스) 처리하여 채워줍니다.

**2. DINKI Remove Pad from Image:** 처리된 이미지와 첫 번째 노드에서 전달받은 `PAD_INFO`를 사용하여 패딩을 잘라내고, **정확한 원본 비율**로 복원합니다.

#### 💡 왜 필요한가요?
이 워크플로우를 사용하면 Qwen Image Edit과 같은 모델에서 발생하는 **픽셀 밀림(Pixel Shifting) 현상**이나 왜곡을 방지할 수 있습니다. 생성 과정 전반에 걸쳐 피사체의 원래 비율을 유지함으로써 프롬프트 기반의 편집 요청이 최대한 정확하게 처리되도록 돕습니다.

#### 비교 (Comparison)
**Resize and Pad 미사용 (왜곡/밀림 발생):**
![Preview](resource/DINKI_Resize_and_Pad_Image_02.png)

**DINKI Resize and Pad 사용 (정확함):**
![Preview](resource/DINKI_Resize_and_Pad_Image_01.png)

#### 🎛️ 파라미터 가이드 (Parameters Guide)

**DINKI Resize and Pad Image**
| 파라미터 | 설명 |
| :--- | :--- |
| **target_size** | 정사각형 캔버스의 목표 해상도입니다 (예: 1024). 이미지의 긴 변이 이 크기에 맞춰집니다. |
| **resize_and_pad** | **True:** 리사이징 및 패딩을 적용합니다.<br>**False:** 노드를 우회하여 원본 이미지를 그대로 내보냅니다. |
| **upscale_method** | 리사이징 알고리즘입니다 (lanczos, bicubic, area, nearest). |

**DINKI Remove Pad from Image**
| 파라미터 | 설명 |
| :--- | :--- |
| **pad_info** | *Resize and Pad* 노드의 `PAD_INFO` 출력을 이곳에 연결합니다. 크로핑 메타데이터가 포함되어 있습니다. |
| **latent_scale** | (선택 사항) **DINKI Upscale Latent By** 노드의 `latent_scale` 출력을 연결합니다. <br>이미지가 잠재 공간(Latent Space)에서 업스케일링된 경우(예: High-Res Fix), 이를 반영하여 정확하게 크로핑합니다. |
| **remove_pad** | **True:** 패딩을 잘라냅니다.<br>**False:** 입력 이미지를 그대로 반환합니다. |

---

### 🔀 DINKI Cross Output Switch

A/B 테스트나 라우팅 로직을 위한 간단하지만 유용한 유틸리티입니다. 불리언(Boolean) 토글 스위치 하나로 두 입력 이미지의 출력 순서를 맞바꿉니다.

#### 🎛️ 파라미터 가이드

| 파라미터 | 설명 |
| :--- | :--- |
| **image_1 / image_2** | 서로 교체할 두 개의 입력 이미지입니다. |
| **invert** | **False:** 출력 1 = 이미지 1, 출력 2 = 이미지 2.<br>**True:** 출력 1 = 이미지 2, 출력 2 = 이미지 1 (교체됨). |

---

### 👁️ DINKI Image Preview

빈 신호(Empty Signal)를 우아하게 처리하는 강력한 프리뷰 노드입니다. 만약 스위치 등으로 인해 이미지가 전달되지 않는 경우, 에러를 내거나 멈추는 대신 **텍스트가 포함된 대체 이미지(Placeholder)**를 자동으로 생성하여 보여줍니다.

![Preview](resource/DINKI_Image_Preview.png)

#### 🎛️ 파라미터 가이드

| 파라미터 | 설명 |
| :--- | :--- |
| **images** | (선택 사항) 이미지를 연결합니다. 연결이 끊기거나 None일 경우 대체 이미지가 표시됩니다. |
| **placeholder_text** | 대체 이미지에 표시할 텍스트입니다 (예: "Bypassed"). |
| **width / height** | 대체 이미지의 크기입니다. |
| **bg_gray / fg_gray** | 배경 및 텍스트의 밝기(0-255 그레이스케일)를 조절합니다. |
| **font_path** | 커스텀 .ttf 파일 경로입니다. 비워두면 시스템 폰트를 자동으로 찾습니다. |

---

### 📝 DINKI CSV Prompt Selector (Live)

자주 사용하는 프롬프트나 LoRA 트리거를 드롭다운 메뉴에서 선택하여 빠르게 입력할 수 있습니다.

* **설정:** ComfyUI의 **`input`** 폴더 안에 **`prompt_list.csv`** 파일을 생성하세요.
* **CSV 형식:** `제목, 프롬프트 내용`
    ```csv
    LoRA - ToonWorld, ToonWorld
    LoRA - Photo to Anime, transform into anime
    ```
* **실시간 업데이트:** ComfyUI를 재시작할 필요 없이 실행할 때마다 CSV 파일의 변경 사항을 자동으로 불러옵니다.

#### 🎛️ 파라미터 가이드

| 파라미터 | 설명 |
| :--- | :--- |
| **title** | CSV 파일에 정의된 키/제목을 선택합니다. 해당 제목의 프롬프트 텍스트가 출력됩니다. |

---

### ⬆️ DINKI Upscale Latent By

유연성과 파이프라인 통합을 위해 설계된 향상된 잠재 공간(Latent) 업스케일링 노드입니다. "배수 스냅(Snap to Multiple)" 기능을 통해 홀수 해상도 에러를 방지합니다.

#### 🎛️ 파라미터 가이드

| 파라미터 | 설명 |
| :--- | :--- |
| **scale_by** | 업스케일링 배율입니다 (예: 1.5x). |
| **snap_to_multiple** | 결과 해상도가 이 숫자의 배수가 되도록 강제합니다 (기본 8). VAE의 "홀수 차원(Odd Dimension)" 에러를 방지합니다. |
| **enabled** | **True:** 업스케일링을 수행합니다.<br>**False:** 노드를 우회하여 원본 잠재(Latent) 데이터를 반환합니다. |
| **upscale_method** | 잠재 데이터 보간 알고리즘입니다 (nearest-exact, bicubic 등). |

> **출력 참고:** `latent_scale` 출력은 (스냅 적용 후의) *실제* 적용된 스케일 팩터를 제공하며, 이를 **DINKI Remove Pad from Image** 노드로 보낼 수 있습니다.

---

### 🧠 DINKI UNet Loader (safetensors / GGUF)

**safetensors**와 **GGUF** 모델 로딩을 하나의 노드로 통합한 간소화된 로더입니다. 일반 모델과 양자화(Quantized) 모델을 전환할 때마다 별도의 로더 노드를 배치하고 선을 다시 연결할 필요가 없습니다.

![Preview](resource/DINKI_UNet_Loader.png)

#### 🎛️ 파라미터 가이드

| 파라미터 | 설명 |
| :--- | :--- |
| **use_gguf** | **True (GGUF):** `gguf_unet`에서 선택한 모델을 로드합니다.<br>**False (safetensors):** `safetensors_unet`에서 선택한 모델을 로드합니다. |
| **safetensors_unet** | `models/diffusion_models` 폴더의 표준 모델을 선택합니다. |
| **gguf_unet** | `models/unet_gguf` 폴더의 양자화 모델을 선택합니다. |

## 🤖 DINKI LM Studio Assistant

**DINKI LM Studio Assistant**는 ComfyUI를 **LM Studio**와 직접 연결해주는 강력한 브릿지 노드입니다. 이를 통해 로컬 LLM(대형 언어 모델)과 VLM(비전 언어 모델)을 워크플로우 내에서 활용하여 이미지 캡셔닝, 프롬프트 개선, 창의적인 글쓰기 등의 작업을 수행할 수 있습니다.

### ✨ 주요 기능

* **멀티모달 기능:** **Text-to-Text** 및 **Image-to-Text** (비전) 생성을 모두 지원합니다.
* **로컬 & 프라이빗:** LM Studio 서버를 통해 사용자의 로컬 머신에서 전적으로 실행되므로, API 키나 인터넷 연결이 필요 없습니다.
* **배치(Batch) 지원:** 이미지 배치를 자동으로 처리하며, 분석을 위해 각 이미지를 LLM에 개별적으로 전송합니다.
* **메모리 관리:** `auto_unload` 기능을 포함하여 LLM 작업이 끝난 후 VRAM을 확보, Stable Diffusion 생성에 사용할 수 있도록 합니다.
* **유연한 제어:** `temperature`, `max_tokens`, `system_prompt` 등 LLM 파라미터를 완벽하게 제어할 수 있습니다.

---

### 🚀 작동 모드

#### 1. 비전 모드 (이미지 + 텍스트)
**이미지가 연결된 경우**, 노드는 비전 어시스턴트로 작동합니다.
* 이미지는 `user_prompt`와 함께 LLM으로 전송됩니다.
* **사용 예시:** LM Studio에서 비전 모델(Qwen3-VL, Gemma3 등)을 연결하여 이미지 캡션을 달거나, 스타일을 묘사하거나, 구도를 분석합니다.
* *참고: `user_prompt`가 비어 있으면 LLM은 기본적으로 "Describe the images.(이미지를 묘사하세요)"로 작동합니다.*

![Preview](resource/DINKI_LM_Studio_Assistant_01.png)

#### 2. 텍스트 모드 (텍스트 전용)
**이미지가 연결되지 않은 경우**, 노드는 순수 텍스트 생성기로 작동합니다.
* 오직 `user_prompt`와 `system_prompt`만을 기반으로 텍스트를 생성합니다.
* **사용 예시:** 프롬프트 확장, 스타일 생성, 창의적인 글쓰기.

![Preview](resource/DINKI_LM_Studio_Assistant_02.png)

#### 3. 패스스루(Passthrough) 모드
`assistant_enabled`를 **False**로 설정하면 LLM을 완전히 우회하고 입력된 `user_prompt`를 그대로 출력합니다. 노드를 삭제하지 않고 A/B 테스트를 할 때 유용합니다.

![Preview](resource/DINKI_LM_Studio_Assistant_03.png)

---

### 🛠️ 사전 준비 및 설정

1. **LM Studio 설치:** [LM Studio](https://lmstudio.ai/)를 다운로드하여 설치합니다.
2. **모델 로드:**
    * **텍스트 전용:** LLM(Llama 3, Mistral 등)을 로드합니다.
    * **비전용:** 비전 지원 모델(e.g., `Qwen-VL`, `LLaVA`, `BakLLaVA`)을 로드합니다.
3. **로컬 서버 시작:**
    * LM Studio의 **Local Server** 탭(양방향 화살표 아이콘 <->)으로 이동합니다.
    * **Start Server**를 클릭합니다.
    * 포트 번호가 노드 설정(기본값: `1234`)과 일치하는지 확인합니다.

---

### 🎛️ 파라미터 가이드

| 파라미터 | 설명 |
| :--- | :--- |
| **assistant_enabled** | 마스터 토글입니다. `False`일 경우 LLM 호출 없이 입력 텍스트를 그대로 출력합니다. |
| **ip_address** | LM Studio 서버의 IP 주소입니다 (기본값: `127.0.0.1`). |
| **port** | LM Studio 서버의 포트 번호입니다 (기본값: `1234`). |
| **model_key** | 모델 식별 문자열입니다 (예: `qwen/qwen3-vl-8b`). LM Studio 버전에 따라 일반적인 값으로 두어도 되는 경우가 많습니다. |
| **system_prompt** | AI의 페르소나를 정의합니다 (예: "당신은 프롬프트 엔지니어입니다..."). |
| **user_prompt** | 구체적인 지시 사항이나 질문입니다. |
| **max_tokens** | 생성될 응답의 최대 길이입니다. |
| **temperature** | 창의성을 조절합니다 (0.0 = 정확함/결정론적, 1.0+ = 창의적/무작위적). |
| **auto_unload** | `True`일 경우, 생성 후 VRAM에서 모델을 언로드하도록 요청을 보냅니다. VRAM이 부족한 GPU에 필수적입니다. |
| **unload_delay** | 모델을 언로드하기 전 대기할 시간(초)입니다 (`auto_unload`가 True일 때). |

## 📚 DINKI Batch Images

여러 장의 개별 이미지를 **하나의 이미지 배치로 결합**해주는 스마트한 유틸리티 노드입니다.

이미지 크기가 다르면 에러를 뱉는 일반적인 배치 노드와 달리, 이 노드는 들어오는 모든 이미지를 **첫 번째 입력 이미지**의 해상도에 맞춰 자동으로 **리사이징**하여 끊김 없는 배치 처리를 보장합니다.

#### ✨ 주요 기능

* **대량 입력:** 한 번에 최대 **10장의 이미지**를 연결할 수 있습니다.
* **자동 리사이징:** 모든 이미지의 크기(너비/높이)를 **첫 번째 입력 이미지**에 맞춰 자동으로 조절합니다. 더 이상 "Shape Mismatch" 에러를 볼 필요가 없습니다!
* **모드 전환:** 배치를 생성할지, 테스트를 위해 첫 번째 이미지만 통과시킬지 쉽게 전환할 수 있습니다.

#### 💡 워크플로우 팁
이 노드는 **DINKI LM Studio Assistant**와 완벽하게 어울립니다. 여러 장의 참조 이미지를 배치로 묶어서 비전 LLM에 보내면, 한 번에 대량 분석이나 캡셔닝을 수행할 수 있습니다.

---

### 🎛️ 파라미터

| 파라미터 | 설명 |
| :--- | :--- |
| **batch_image** | **True (multiple):** 연결된 모든 이미지를 리사이징하여 하나의 배치로 병합합니다.<br>**False (single):** 나머지는 무시하고 발견된 첫 번째 이미지만 출력합니다 (패스스루 모드). |
| **image1 ~ 10** | 이미지를 이곳에 연결합니다. 입력은 비워둘 수 있으며, 노드가 연결된 입력을 자동으로 감지합니다. |

## DINKI Color Nodes
![Preview](resource/DINKI_Color.png)

#### DINKI Auto Adjustment Node
**DINKI Auto Adjustment** 노드는 다음과 같은 자동 보정 기능을 구현합니다:
- **Auto Tone (자동 톤)**
- **Auto Contrast (자동 대비)**
- **Auto Color (자동 색상)**

#### DINKI Adobe XMP Node
**DINKI Adobe XMP** 노드는 **Adobe Lightroom** 및 **Adobe Camera Raw**의 프리셋을 적용합니다.
현재 지원되는 조정 항목은 다음과 같습니다:

- **Exposure (노출)**
- **Contrast (대비)**
- **Saturation (채도)**
- **Vibrance (활기)**
- **Tone Curve (톤 커브)** (마스터 커브 + RGB 채널)
- **HSL** (색조 / 채도 / 휘도)
- **Vignette (비네팅)**
- **Grain (그레인)**
XMP 프리셋 파일 위치: **~/ComfyUI/input/adobe_xmp**

#### DINKI Color LUT Node
**DINKI Color LUT** 노드는 **.cube** 형식의 컬러 LUT를 적용합니다.
LUT 파일 위치: **~/ComfyUI/input/luts**

#### DINKI AI Oversaturation Fix
AI 생성 이미지에서 자주 발생하는 과도한 채도나 색상 왜곡을 완화합니다.

## 🎚️ DINKI Node Switch
![Preview](resource/DINKI_Node_Switch.gif)
워크플로우를 위한 **리모컨** 역할을 하는 로직 유틸리티 노드입니다. 스위치 하나로 여러 타겟 노드의 **Bypass 상태를 토글**할 수 있습니다.

복잡한 워크플로우에서 "제어 패널(Control Panel)"을 만드는 데 완벽하며, 노드를 일일이 찾아다니지 않고도 전체 섹션(업스케일링, Face Detailer, LoRA 스택 등)을 켜거나 끌 수 있습니다.

#### ✨ 주요 기능

* **원격 제어:** 그래프 내의 어떤 노드 상태든 한 곳에서 관리할 수 있습니다.
* **일괄 토글:** 쉼표로 구분된 노드 ID 목록(예: `10, 15, 23`)을 입력하여 여러 노드를 한 번에 제어합니다.
* **워크플로우 최적화:** 초기 테스트 중에는 무거운 처리 단계(High-res fix 등)를 끄고, 최종 렌더링 시 클릭 한 번으로 다시 켤 수 있어 편리합니다.
* **프론트엔드 통합:** ComfyUI 그래프 인터페이스와 직접 상호작용하여 노드를 시각적으로 끄고 켭니다.

#### 💡 사용 방법
1. **노드 ID 확인:** ComfyUI 설정에서 **"Show Node ID on Node"**를 켜거나 (또는 노드 우클릭 > Properties) ID를 확인합니다.
2. **ID 입력:** 제어하려는 노드의 ID를 `node_ids` 필드에 입력합니다 (예: `5, 12, 44`).
3. **토글:**
    * **On (True):** 타겟 노드가 **활성화(Enabled)**됩니다.
    * **Off (False):** 타겟 노드가 **바이패스(Bypassed/Muted)**됩니다.

---

### 🎛️ 입력

| 파라미터 | 설명 |
| :--- | :--- |
| **node_ids** | 쉼표로 구분된 노드 ID 문자열입니다 (예: `1,2,3`). |
| **active** | 마스터 스위치입니다. 정의된 노드들의 바이패스 상태를 토글합니다. |

## 📸 DINKI Photo Specifications
![Preview](resource/DINKI_photo_specifications.png)

타겟 **메가픽셀(MP)**과 **현실 세계 표준 종횡비**를 선택하여 AI 생성을 위한 **최적의 해상도**를 계산해주는 스마트 유틸리티 노드입니다.

수동으로 픽셀을 입력하는 번거로움을 없애세요. 이 노드는 SDXL, Flux, Z-Image Turbo와 같은 모델에 딱 맞는 크기로 이미지가 생성되도록 보장합니다.

### ✨ 주요 기능

* **현실 세계 표준:** 표준 **사진(Photography)** 비율(3:4, 4:6)부터 전문 **영화 필름(Cinema/Film)** 규격(Academy, IMAX, Super 35)까지 다양한 포맷을 지원합니다.
* **AI 최적화:** 가로/세로 픽셀 값을 자동으로 **8의 배수**로 보정하여 인코딩 에러를 방지하고 잠재 확산(Latent Diffusion) 모델과의 호환성을 보장합니다.
* **메가픽셀 타겟팅:** 모델의 용량에 맞춰 **1MP ~ 4MP** 중에서 선택할 수 있습니다 (기준: 1MP = 1024x1024). 다양한 비율에서도 전체 픽셀 면적을 보존하여 일관된 품질을 유지합니다.
* **즉각적인 방향 전환:** 재계산할 필요 없이 **Portrait(세로)**와 **Landscape(가로)** 모드를 쉽게 토글할 수 있습니다.

#### 💡 워크플로우 팁
제 경험상, 이 노드는 **Z-Image Turbo** 워크플로우와 특히 잘 맞아떨어지며, 가장 효율적인 해상도에서 빠른 생성을 보장합니다.

---

#### 🎛️ 지원 포맷

| 카테고리 | 종횡비 (Aspect Ratios) |
| :--- | :--- |
| **Photo** | 3:4, 3.5:5, 4:6, 5:7, 6:8, 8:10, 10:13, 10:15, 11:14 |
| **Cinema** | 35mm Academy (1.37:1), 35mm Flat (1.85:1), 35mm Scope (2.39:1) |
| **Premium** | 70mm Todd-AO (2.20:1), IMAX 70mm (1.43:1) |
| **Super** | Super 35 (1.85:1 / 2.39:1), Super 16 (1.66:1 / 1.78:1) |

#### 📤 출력
* **width (INT):** 계산된 가로 너비 (8의 배수).
* **height (INT):** 계산된 세로 높이 (8의 배수).
* **info_string (STRING):** 현재 설정 요약 (예: `896x1152 (Photo 3.5:5, 1MP)`).

## 🖼️ DINKI Overlay
![Preview](resource/DINKI_Overlay.png?v=2)

생성된 이미지에 **워터마크, 저작권 텍스트, 자막, 로고 오버레이**를 전문가 수준의 정밀도로 추가할 수 있는 강력하고 다재다능한 ComfyUI 노드입니다.

#### ✨ 주요 기능

* **듀얼 레이어 시스템:** 간단한 토글 스위치를 사용하여 **텍스트**와 **이미지** 오버레이를 동시에 또는 독립적으로 추가할 수 있습니다.
* **고급 텍스트 스타일링:**
    * **커스텀 폰트:** `fonts` 폴더에 있는 `.ttf` 및 `.otf` 파일을 자동으로 감지하여 드롭다운에서 쉽게 선택할 수 있습니다.
    * **스트로크 (테두리):** 복잡한 배경에서도 잘 보이도록 텍스트에 색상 테두리를 추가합니다.
    * **드롭 쉐도우 (그림자):** 조절 가능한 그림자 위치(오프셋), 흐림(퍼짐), 투명도로 깊이감을 더합니다.
    * **멀티라인 지원:** 자동 줄 간격 처리를 지원하여 자막이나 긴 저작권 문구에 완벽합니다.
* **정밀한 위치 선정:** **7가지 프리셋 위치** (예: Top-Left, Bottom-Center, Center) 중 하나를 선택하고 퍼센트 기반 **여백(Margin)**으로 미세 조정할 수 있습니다.
* **적응형 크기 조절:** 텍스트와 로고의 크기를 원본 이미지 크기에 대한 비율(%)로 조절하여, 다양한 해상도(SDXL, Flux 등)에서도 일관된 결과를 얻을 수 있습니다.
* **투명도 제어:** **알파/마스크** (투명 PNG)를 완벽하게 지원하며, 텍스트와 이미지 모두에 대해 투명도(0-100%)를 조절할 수 있습니다.

#### 📂 커스텀 폰트 추가 방법
1. 노드 디렉토리로 이동합니다: `~/ComfyUI/custom_nodes/ComfyUI-DINKIssTyle/fonts/`
2. `.ttf` 또는 `.otf` 폰트 파일을 이 폴더에 붙여넣습니다.
3. ComfyUI를 재시작합니다. 폰트가 **`font_name`** 드롭다운 목록에 자동으로 나타납니다.

#### 💡 투명 PNG (로고) 사용 팁
투명 배경이 있는 로고를 올바르게 오버레이하려면:
1. **Load Image** 노드의 `IMAGE` 출력을 `overlay_image`에 연결합니다.
2. `MASK` 출력을 `overlay_mask`에 연결합니다.
3. *(선택 사항)* `overlay_opacity` 슬라이더를 사용하여 로고를 배경과 자연스럽게 블렌딩합니다.

---

#### 🎛️ 입력 파라미터

| 파라미터 | 설명 |
| :--- | :--- |
| **font_name** | `fonts` 폴더에서 폰트를 선택합니다. |
| **text_content** | 텍스트를 입력합니다. 여러 줄 입력을 지원합니다 (엔터 키). |
| **text_opacity** | 텍스트 투명도를 조절합니다 (0-100). |
| **enable_stroke** | 텍스트 테두리를 토글합니다. 색상과 두께를 설정합니다. |
| **enable_shadow** | 드롭 쉐도우를 토글합니다. 오프셋(X/Y), 퍼짐(블러), 투명도를 조절합니다. |
| **overlay_mask** | (선택 사항) 투명 PNG 로고 지원을 위해 마스크를 이곳에 연결합니다. |
