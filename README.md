# Introduction

This repository stores custom ComfyUI nodes that I created to solve various needs while working with ComfyUI.  
These nodes are primarily designed for my own workflow using **Qwen-Image**, **Z-Image Turbo**, **Flux**, and **WAN**.
Using them with other models may cause unexpected issues.


- [All Nodes (47)](resource/Node_Catalog.md)
- [Comparison Video Tools](resource/DINKI_Video_Tools.md)
- [Image](resource/DINKI_Image.md)
- [Color Nodes](resource/DINKI_Color_Nodes.md)
- [LM Studio Assistant](resource/DINKI_LM_Studio_Assistant.md)
- [Prompts and Strings](resource/DINKI_Prompt_and_String.md)
- [Node Utilities](resource/DINKI_Node_Utils.md)
- [System Monitor (Windows / NVIDIA)](resource/DKST_System_Monitor.md)
- [Internal Processing](resource/DINKI_PS.md)


---


**ComfyUI-DINKIssTyle_CPH** (Cross-Platform Helper) scans the ComfyUI `input` folder recursively when it loads:

1. Normalizes decomposed file names, including Korean names uploaded from macOS, to NFC. If the target name exists, it adds an `_nfc` suffix.
2. Removes `._*` macOS resource fork files of 128 KiB or less.
