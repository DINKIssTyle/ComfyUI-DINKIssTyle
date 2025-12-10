[Home](./README.md)
- [Comprision Video Tools](resource/DINKI_Video_Tools.md)
- [Image](resource/DINKI_Image.md)
- [Color Nodes](resource/DINKI_Color_Nodes.md)
- [LM Studio Assistant and Batch Images](resource/DINKI_LM_Studio_Assistant.md)
- [Prompts and Strings](resource/DINKI_Prompt_and_String.md)
- [Node Utilities](resource/DINKI_Utils.md)
- [Internal Processing](resource/DINKI_PS.md)

### 📐 DINKI Resize and Pad Image / Remove Pad

This pair of nodes is essential for workflows involving image editing models (like **Qwen Image Edit**) that are sensitive to aspect ratio changes or resolution resizing.

**1. DINKI Resize and Pad Image** Resizes an input image to fit within a target square resolution (default **1024×1024**) while *preserving the original aspect ratio*. It automatically adds padding (letterboxing) to fill the remaining space.

**2. DINKI Remove Pad from Image** Takes the processed image and the `PAD_INFO` from the first node to crop the padding out, restoring the **exact original aspect ratio**.

#### 💡 Why use this?
This workflow prevents **pixel shifting artifacts** and distortion in models like Qwen Image Edit. It ensures that prompt-based editing requests are processed as accurately as possible by maintaining the subject's original proportions throughout the generation process.

#### Comparison
**Without Resize and Pad (Distorted/Shifted):**
![Preview](resource/DINKI_Resize_and_Pad_Image_02.png)

**With DINKI Resize and Pad (Accurate):**
![Preview](resource/DINKI_Resize_and_Pad_Image_01.png)

#### 🎛️ Parameters Guide

**DINKI Resize and Pad Image**
| Parameter | Description |
| :--- | :--- |
| **target_size** | The target resolution for the square canvas (e.g., 1024). The longest side of the image will fit this size. |
| **resize_and_pad** | **True:** Applies resizing and padding.<br>**False:** Bypasses the node (returns original image). |
| **upscale_method** | Algorithm used for resizing (lanczos, bicubic, area, nearest). |

**DINKI Remove Pad from Image**
| Parameter | Description |
| :--- | :--- |
| **pad_info** | Connect the `PAD_INFO` output from the *Resize and Pad* node here. Contains cropping metadata. |
| **latent_scale** | (Optional) Connect the `latent_scale` output from **DINKI Upscale Latent By**. <br>Allows correct cropping even if the image was upscaled in latent space (e.g., during High-Res Fix). |
| **remove_pad** | **True:** Crops the padding.<br>**False:** Returns the input image as-is. |


---



## ⬆️ DINKI Upscale Latent By

An enhanced latent upscaling node designed for flexibility and pipeline integration. It features a "Snap to Multiple" function to prevent odd-resolution errors.

#### 🎛️ Parameters Guide

| Parameter | Description |
| :--- | :--- |
| **scale_by** | The multiplier for upscaling (e.g., 1.5x). |
| **snap_to_multiple** | Ensures the resulting resolution is a multiple of this number (default 8). Prevents "odd dimension" errors in VAEs. |
| **enabled** | **True:** Performs upscaling.<br>**False:** Bypasses the node (returns original latent). |
| **upscale_method** | Algorithm for latent interpolation (nearest-exact, bicubic, etc.). |

> **Output Note:** The `latent_scale` output provides the *actual* scaling factor used (after snapping), which can be sent to **DINKI Remove Pad from Image**.


---


## 🧠 DINKI UNet Loader (safetensors / GGUF)

A streamlined loader that combines **safetensors** and **GGUF** model loading into a single node. This removes the need to place separate loader nodes and rewire connections when switching between standard and quantized models.

![Preview](resource/DINKI_UNet_Loader.png)

#### 🎛️ Parameters Guide

| Parameter | Description |
| :--- | :--- |
| **use_gguf** | **True (GGUF):** Loads the model selected in `gguf_unet`.<br>**False (safetensors):** Loads the model selected in `safetensors_unet`. |
| **safetensors_unet** | Select a standard model from `models/diffusion_models`. |
| **gguf_unet** | Select a quantized model from `models/unet_gguf`. |

