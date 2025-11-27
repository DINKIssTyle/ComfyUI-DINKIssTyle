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


### DINKI LM Studio Assistant

**DINKI LM Studio Assistant** connects directly to **LM Studio** to handle both image-based and text-based tasks.

- When an **image is provided**, the node sends the image **together with the user’s prompt** to LM Studio.  
  The LLM then produces a response that combines visual analysis with the user’s instructions for richer, context-aware output.  
  **If no user prompt is provided, the LLM generates an image-based prompt derived solely from the visual content.**

![Preview](resource/DINKI_LM_Studio_Assistant_01.png)

- When **no image is connected**, the node instead uses the LLM to generate a text prompt based on the user's input.

![Preview](resource/DINKI_LM_Studio_Assistant_02.png)

- By using the `assistant_enabled` Boolean option, you can switch to a text-only mode without disabling the node entirely.

![Preview](resource/DINKI_LM_Studio_Assistant_03.png)

This makes it versatile for workflows that require automated prompt generation, image captioning, or multimodal understanding.
This allows the LLM to operate normally even when no image is connected.


### DINKI Color Nodes
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


### DINKI Node Switch
![Preview](resource/DINKI_Node_Switch.gif)

You can specify one or multiple **Node IDs**, separated by commas (`,`), and **toggle their bypass state with a mouse click.**  
This is useful when you need quick control in addition to group bypass functionality.


### DINKI Photo Specifications
![Preview](resource/DINKI_photo_specifications.png)

This custom node calculates the **optimal resolution** by defining a target **megapixel count** and selecting **real-world standard aspect ratios** (including Photo and Cinema formats).

It automatically outputs Width and Height values (adjusted to multiples of 8) based on your selected scale and format.

I found this node to work especially well with **Z-Image Turbo**.




