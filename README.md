## Introduction

This repository stores custom ComfyUI nodes that I created to solve various needs while working with ComfyUI.  
These nodes are primarily designed for my own workflow using **Qwen-Image**, **Flux**, and **WAN**.  
Using them with other models may cause unexpected issues.

## Node Descriptions

### DINKI Resize and Pad Image / DINKI Remove Pad from Image

**DINKI Resize and Pad Image** resizes an image to fit within **1024×1024 or 1328*1328** while *preserving the original aspect ratio*.  
It then adds padding so diffusion models can process the image in an optimized square format.

The padding metadata is passed to **DINKI Remove Pad from Image**, which crops the output back using the original aspect ratio.  
The final image may not match the exact pixel size of the input, but its **original proportions are fully preserved**.

This Nodes helps prevent **pixel shifting artifacts** in **Qwen Image Edit**,  
and helps ensure that prompt-based editing requests are processed as accurately as possible.

#### Without DINKI Resize and Pad Image
![Preview](resource/DINKI_Resize_and_Pad_Image_02.png)

#### With DINKI Resize and Pad Image
![Preview](resource/DINKI_Resize_and_Pad_Image_01.png)

### DINKI Cross Output Switch

**DINKI Cross Output Switch** swaps the two input images **A** and **B** based on a Boolean toggle.  
When the toggle is enabled, the node outputs them in reversed order (**B**, **A**) instead of (**A**, **B**).  
This allows easy switching between two image sources in a workflow.

### DINKI Image Preview

**DINKI Image Preview** generates a custom placeholder image containing text when no output image is provided.  
This is useful for debugging or visually confirming that a node executed without producing an image result.

![Preview](resource/DINKI_Image_Preview.png)


### DINKI CSV Prompt Selector (Live)

**DINKI CSV Prompt Selector (Live)** allows you to quickly insert frequently used prompts by selecting them from a dropdown menu.  
The node reads a file named **`prompt_list.csv`** located inside the **`input`** folder.

The CSV format is simple:
LoRA - ToonWorld, ToonWorld  
LoRA - Photo to Anime, transform into anime  
LoRA - 3D Chibi, Convert this image into 3D Chibi Style  
LoRA - InScene, Make a shot in the same scene  


### DINKI Upscale Latent By

**DINKI Upscale Latent By** is a latent-space upscaling node with additional controls not found in the default *Upscale By* node.

Key features:

- **Snap_to_multiple**  
  Ensures the upscaled latent resolution snaps to clean, model-friendly multiples.  
  This helps avoid odd dimensions that may cause issues in certain diffusion pipelines.

- **Boolean Toggle (Bypass Mode)**  
  Allows you to enable or disable the upscaling process instantly.  
  When disabled, the node simply passes the latent through without modification.

This node is useful for flexible latent upscaling workflows where dimension alignment and quick toggling are required.

Additionally, you can pass the upscale metadata to **DINKI Remove Pad from Image**,  
allowing it to crop the final image accurately while preserving the original aspect ratio after upscaling.


### DINKI UNet Loader (safetensors / GGUF)

**DINKI UNet Loader** lets you configure both a **safetensors-based UNet model** and a **GGUF-format UNet model**,  
and loads **whichever model is selected** through a Boolean toggle.

![Preview](resource/DINKI_UNet_Loader.png)

This provides a cleaner workflow compared to placing two separate UNet loader nodes,  
since the node loads the chosen model directly without requiring multiple loader nodes in the graph.


## 🤖 DINKI LM Studio Assistant

**DINKI LM Studio Assistant** is a powerful bridge node that connects ComfyUI directly to **LM Studio**. It enables the use of local Large Language Models (LLMs) and Vision Language Models (VLMs) within your workflows for tasks like image captioning, prompt enhancement, and creative writing.

### ✨ Key Features

* **Multimodal Capabilities:** Supports both **Text-to-Text** and **Image-to-Text** (Vision) generation.
* **Local & Private:** Runs entirely on your local machine via LM Studio server—no API keys or internet required.
* **Batch Support:** Automatically processes image batches, sending them individually to the LLM for analysis.
* **Memory Management:** Includes an `auto_unload` feature to free up VRAM for Stable Diffusion generation after the LLM task is finished.
* **Flexible Control:** Full access to LLM parameters like `temperature`, `max_tokens`, and `system_prompt`.

---

### 🚀 Modes of Operation

#### 1. Vision Mode (Image + Text)
When an **image is connected**, the node operates as a Vision Assistant.
* The image is sent to the LLM along with your `user_prompt`.
* **Use Case:** Connect a Vision model (like Qwen3-VL or Gemma3) in LM Studio to caption images, describe styles, or analyze composition.
* *Note: If `user_prompt` is left empty, the LLM will default to "Describe the images."*

![Preview](resource/DINKI_LM_Studio_Assistant_01.png)

#### 2. Text Mode (Text Only)
When **no image is connected**, the node functions as a pure text generator.
* It generates text based solely on the `user_prompt` and `system_prompt`.
* **Use Case:** Prompt expansion, style generation, or creative writing.

![Preview](resource/DINKI_LM_Studio_Assistant_02.png)

#### 3. Passthrough Mode
By setting `assistant_enabled` to **False**, the node bypasses the LLM entirely and simply outputs your raw `user_prompt`. This is useful for A/B testing without removing the node.

![Preview](resource/DINKI_LM_Studio_Assistant_03.png)

---

### 🛠️ Prerequisites & Setup

1.  **Install LM Studio:** Download and install [LM Studio](https://lmstudio.ai/).
2.  **Load a Model:**
    * For **Text-only**: Load any LLM (Llama 3, Mistral, etc.).
    * For **Vision**: Load a vision-capable model (e.g., `Qwen-VL`, `LLaVA`, `BakLLaVA`).
3.  **Start Local Server:**
    * Go to the **Local Server** tab ( double arrow icon <-> ) in LM Studio.
    * Click **Start Server**.
    * Ensure the port matches the node settings (Default: `1234`).

---

### 🎛️ Parameters Guide

| Parameter | Description |
| :--- | :--- |
| **assistant_enabled** | Master toggle. If `False`, passes input text directly to output without calling the LLM. |
| **ip_address** | The IP of the LM Studio server (Default: `127.0.0.1`). |
| **port** | The port of the LM Studio server (Default: `1234`). |
| **model_key** | The model identifier string (e.g., `qwen/qwen3-vl-8b`). Can often be left generic depending on LM Studio version. |
| **system_prompt** | Defines the AI's persona (e.g., "You are a prompt engineer..."). |
| **user_prompt** | Your specific instruction or query. |
| **max_tokens** | Maximum length of the generated response. |
| **temperature** | Creativity control (0.0 = Precise/Deterministic, 1.0+ = Creative/Random). |
| **auto_unload** | If `True`, sends a request to unload the model from VRAM after generation. Essential for GPUs with limited VRAM. |
| **unload_delay** | Seconds to wait before unloading the model (if `auto_unload` is True). |


## 📚 DINKI Batch Images

A smart utility node designed to **combine multiple individual images into a single image batch**.

Unlike standard batch nodes that error out when image dimensions differ, this node automatically **resizes** all incoming images to match the resolution of the first image, ensuring a seamless batching process.

#### ✨ Key Features

* **Mass Input:** Connect up to **10 different images** at once.
* **Auto-Resizing:** Automatically scales all images to match the dimensions (Width/Height) of the **first input image**. No more "Shape Mismatch" errors!
* **Mode Switching:** Easily toggle between creating a batch or just passing through the first image for testing.

#### 💡 Workflow Tip
This node works perfectly with **DINKI LM Studio Assistant**. Use it to batch multiple reference images together and send them to a Vision LLM for bulk analysis or captioning in a single pass.

---

### 🎛️ Parameters

| Parameter | Description |
| :--- | :--- |
| **batch_image** | **True (multiple):** Resizes and merges all connected images into one batch.<br>**False (single):** Ignores the rest and outputs only the first image found (Pass-through mode). |
| **image1 ~ 10** | Connect your images here. Inputs can be left empty; the node automatically detects active connections. |


## DINKI Color Nodes
![Preview](resource/DINKI_Color.png)

#### DINKI Auto Adjustment Node
The **DINKI Auto Adjustment** node implements the following automatic correction features:
- **Auto Tone**
- **Auto Contrast**
- **Auto Color**

#### DINKI Adobe XMP Node
The **DINKI Adobe XMP** node applies presets from **Adobe Lightroom** and **Adobe Camera Raw**.  
Currently supported adjustments include:

- **Exposure**
- **Contrast**
- **Saturation**
- **Vibrance**
- **Tone Curve** (Master Curve + RGB Channels)
- **HSL** (Hue / Saturation / Luminance)
- **Vignette**
- **Grain**  
XMP preset files should be placed in: **~/ComfyUI/input/adobe_xmp**


#### DINKI Color LUT Node
The **DINKI Color LUT** node applies color LUTs in **.cube** format.  
Place your LUT files in: **~/ComfyUI/input/luts**

#### DINKI AI Oversaturation Fix
Reduces excessive saturation or color distortion often produced by AI-generated images.


## 🎚️ DINKI Node Switch
![Preview](resource/DINKI_Node_Switch.gif)
A logic utility node that acts as a **remote control** for your workflow. It allows you to **toggle the Bypass status** of multiple target nodes simultaneously using a simple switch.

Perfect for creating "Control Panels" in complex workflows, allowing you to turn entire sections (like Upscaling, Face Detailer, or LoRA stacks) on or off without hunting for individual nodes.

#### ✨ Key Features

* **Remote Control:** Manage the state of any node in your graph from a single location.
* **Batch Toggling:** Control multiple nodes at once by entering a comma-separated list of Node IDs (e.g., `10, 15, 23`).
* **Workflow Optimization:** Easily disable heavy processing steps (like high-res fix) during initial testing, then re-enable them for the final render with one click.
* **Frontend Integration:** Directly interacts with the ComfyUI graph interface to visually mute/unmute nodes.

#### 💡 How to Use
1.  **Find Node IDs:** In ComfyUI settings, enable **"Show Node ID on Node"** (or right-click a node > Properties to see its ID).
2.  **Input IDs:** Enter the IDs of the nodes you want to control into the `node_ids` field (e.g., `5, 12, 44`).
3.  **Toggle:**
    * **On (True):** Target nodes are **Enabled** (Active).
    * **Off (False):** Target nodes are **Bypassed** (Muted).

---

### 🎛️ Inputs

| Parameter | Description |
| :--- | :--- |
| **node_ids** | A string of node IDs separated by commas (e.g., `1,2,3`). |
| **active** | The master switch. Toggles the bypass state of the defined nodes. |



## 📸 DINKI Photo Specifications
![Preview](resource/DINKI_photo_specifications.png)

A smart utility node designed to calculate the **optimal resolution** for AI generation by selecting target **megapixels** and **real-world standard aspect ratios**.

Eliminate the guesswork of manual pixel entry. This node ensures your images are generated at the perfect size for models like SDXL, Flux, and Z-Image Turbo.

### ✨ Key Features

* **Real-World Standards:** Supports a wide range of formats, from standard **Photography** ratios (3:4, 4:6) to professional **Cinema/Film** specifications (Academy, IMAX, Super 35).
* **AI Optimization:** Automatically adjusts Width and Height values to **multiples of 8**, preventing encoding errors and ensuring compatibility with latent diffusion models.
* **Megapixel Targeting:** Select from **1MP to 4MP** based on your model's capacity (Base: 1MP = 1024x1024). It maintains consistent quality by preserving the total pixel area across different aspect ratios.
* **Instant Orientation:** Easily toggle between **Portrait** and **Landscape** modes without recalculating.

#### 💡 Workflow Tip
I found this node to work especially well with **Z-Image Turbo** workflows, ensuring fast generation at the most efficient resolutions.

---

#### 🎛️ Supported Formats

| Category | Aspect Ratios |
| :--- | :--- |
| **Photo** | 3:4, 3.5:5, 4:6, 5:7, 6:8, 8:10, 10:13, 10:15, 11:14 |
| **Cinema** | 35mm Academy (1.37:1), 35mm Flat (1.85:1), 35mm Scope (2.39:1) |
| **Premium** | 70mm Todd-AO (2.20:1), IMAX 70mm (1.43:1) |
| **Super** | Super 35 (1.85:1 / 2.39:1), Super 16 (1.66:1 / 1.78:1) |

#### 📤 Outputs
* **width (INT):** Calculated width (multiple of 8).
* **height (INT):** Calculated height (multiple of 8).
* **info_string (STRING):** Summary of current settings (e.g., `896x1152 (Photo 3.5:5, 1MP)`).



## 🖼️ DINKI Overlay
![Preview](resource/DINKI_Overlay.png?v=2)

A powerful and versatile ComfyUI node designed to add **watermarks, copyright text, subtitles, and logo overlays** to your generated images with professional precision.

#### ✨ Key Features

* **Dual Layering System:** Add **Text** and **Image** overlays simultaneously or independently using simple toggle switches.
* **Advanced Text Styling:**
    * **Custom Fonts:** Automatically detects `.ttf` and `.otf` files in the `fonts` folder for easy dropdown selection.
    * **Stroke (Outline):** Add colored outlines to your text for better visibility on complex backgrounds.
    * **Drop Shadow:** Create depth with adjustable shadow position (offset), blur (spread), and opacity.
    * **Multiline Support:** Perfect for subtitles or long copyright notices with automatic line spacing handling.
* **Precise Positioning:** Choose from **7 preset positions** (e.g., Top-Left, Bottom-Center, Center) and fine-tune with percentage-based **margins**.
* **Adaptive Sizing:** Scale text and logos relative to the source image size (%) for consistent results across different resolutions (SDXL, Flux, etc.).
* **Transparency Control:** Full support for **Alpha/Masks** (transparent PNGs) and adjustable opacity (0-100%) for both text and images.

#### 📂 How to Add Custom Fonts
1.  Go to the node's directory: `~/ComfyUI/custom_nodes/ComfyUI-DINKIssTyle/fonts/`
2.  Paste your `.ttf` or `.otf` font files into this folder.
3.  Restart ComfyUI. Your fonts will automatically appear in the **`font_name`** dropdown list.

#### 💡 Usage Tip for Transparent PNGs (Logos)
To properly overlay a logo with a transparent background:
1.  Connect the `IMAGE` output of your **Load Image** node to `overlay_image`.
2.  Connect the `MASK` output to `overlay_mask`.
3.  *(Optional)* Use the `overlay_opacity` slider to blend the logo with the background.

---

#### 🎛️ Input Parameters

| Parameter | Description |
| :--- | :--- |
| **font_name** | Select a font from the `fonts` folder. |
| **text_content** | Enter your text here. Supports multiple lines (enter key). |
| **text_opacity** | Adjust text transparency (0-100). |
| **enable_stroke** | Toggle text outline. Set color and width. |
| **enable_shadow** | Toggle drop shadow. Adjust offset (X/Y), spread (blur), and opacity. |
| **overlay_mask** | (Optional) Connect a mask here to support transparent PNG logos. |



