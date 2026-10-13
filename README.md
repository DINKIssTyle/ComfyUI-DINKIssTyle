# Introduction

This repository stores custom ComfyUI nodes that I created to solve various needs while working with ComfyUI.  
These nodes are primarily designed for my own workflow using **Qwen-Image**, **Z-Image Turbo**, **Flux**, and **WAN**.
Using them with other models may cause unexpected issues.

## Installation and dependencies

Clone this repository into ComfyUI's `custom_nodes` directory and restart ComfyUI.
The repository entry point loads the main node pack. Copying just the inner
`ComfyUI-DINKIssTyle` directory into `custom_nodes` also works. CPH and VoxCPM2
remain separate packs; install their directories separately when needed.

ComfyUI Manager reads the root `requirements.txt` and runs `install.py`.
For a manual install, run this with **the Python that runs ComfyUI** from the
installed pack directory (repository root or inner node directory):

```sh
python install.py
```

Startup also checks and installs missing host packages: NumPy, Pillow, safetensors,
requests, imageio, imageio-ffmpeg, PyAV, psutil and OpenCV with YuNet support.
Compatible OpenCV/headless/contrib builds are reused. A fresh install uses headless
OpenCV so the CPU face detector needs no desktop GUI libraries. Installation
failures print a repair command and individual module errors; other modules keep
loading. After repairing dependencies, restart ComfyUI. Set `DKST_AUTO_INSTALL=0`
to disable startup package installation in managed/offline environments.

**Qwen 2.1 and MiniMax H3** use separate CUDA training dependencies; those pins
are not installed into ComfyUI's environment. Open **Configure Models → Prepare
Runtime** on either trainer (also in Settings → DKST → Training) to install its
`.venv`, view progress/errors and select the prepared Training Python. Save model
paths and use **Check Runtime**. Windows/Linux and an NVIDIA CUDA GPU are required;
Python 3.12 with `venv`/pip is recommended, including for Windows portable ComfyUI.
MiniMax reuses an existing Qwen `.venv` unless it already has its own. Models and
optional accelerated kernels are configured separately. The command-line
`setup_runtime.py` scripts remain available in each trainer directory.

**Face Crop** runs in ComfyUI's Python and uses the bundled YuNet models. It does
not need either training environment.


- [All Nodes (66)](resource/Node_Catalog.md)
- [Comparison Video Tools](resource/DINKI_Video_Tools.md)
- [Image](resource/DINKI_Image.md)
- [Color Nodes](resource/DINKI_Color_Nodes.md)
- [LM Studio Assistant](resource/DINKI_LM_Studio_Assistant.md)
- [Prompts and Strings](resource/DINKI_Prompt_and_String.md)
- [Node Utilities](resource/DINKI_Node_Utils.md)
- [System Monitor (Windows / NVIDIA)](resource/DKST_System_Monitor.md)
- [Internal Processing](resource/DINKI_PS.md)
- [Qwen Image 2.1 LoRA Trainer — setup and usage](ComfyUI-DINKIssTyle/qwen21_trainer/README.md)
- [Face Crop Export — LoRA dataset preparation](ComfyUI-DINKIssTyle/face_crop/README.md)
- [Minimax H3 LoRA Trainer — setup and usage](ComfyUI-DINKIssTyle/minimax_trainer/README.md)


---


**ComfyUI-DINKIssTyle_CPH** (Cross-Platform Helper) scans the ComfyUI `input` folder recursively when it loads:

1. Normalizes decomposed file names, including Korean names uploaded from macOS, to NFC. If the target name exists, it adds an `_nfc` suffix.
2. Removes `._*` macOS resource fork files of 128 KiB or less.
3. Deletes files whose names start with `clipspace-`, including files in subfolders, regardless of size, on every ComfyUI startup.
