## Introduction

This repository stores custom ComfyUI nodes that I created to solve various needs while working with ComfyUI.  
These nodes are primarily designed for my own workflow using **Qwen-Image**, **Flux**, and **WAN**.  
Using them with other models may cause unexpected issues.

## Node Descriptions

### DINKI Resize and Pad Image / DINKI Remove Pad from Image

**DINKI Resize and Pad Image** resizes an image to fit within **1024×1024** while *preserving the original aspect ratio*.  
It then adds padding so diffusion models can process the image in an optimized square format.

The padding metadata is passed to **DINKI Remove Pad from Image**, which crops the output back using the original aspect ratio.  
The final image may not match the exact pixel size of the input, but its **original proportions are fully preserved**.

### DINKI Cross Output Switch

**DINKI Cross Output Switch** swaps the two input images **A** and **B** based on a Boolean toggle.  
When the toggle is enabled, the node outputs them in reversed order (**B**, **A**) instead of (**A**, **B**).  
This allows easy switching between two image sources in a workflow.

### DINKI Image Preview

**DINKI Image Preview** generates a custom placeholder image containing text when no output image is provided.  
This is useful for debugging or visually confirming that a node executed without producing an image result.


### DINKI CSV Prompt Selector (Live)

**DINKI CSV Prompt Selector (Live)** allows you to quickly insert frequently used prompts by selecting them from a dropdown menu.  
The node reads a file named **`prompt_list.csv`** located inside the **`input`** folder.

The CSV format is simple:
LoRA - ToonWorld, ToonWorld
LoRA - Photo to Anime, transform into anime
LoRA - 3D Chibi, Convert this image into 3D Chibi Style
LoRA - InScene, Make a shot in the same scene


### DINKI Auto Adjustment / DINKI AI Oversaturation Fix

These nodes provide various automatic image adjustments:

- **DINKI Auto Adjustment**  
  Applies automatic enhancements to improve overall image balance and clarity.

- **DINKI AI Oversaturation Fix**  
  Reduces excessive saturation or color distortion often produced by AI-generated images.

Both nodes help refine and stabilize image quality during post-processing in ComfyUI workflows.



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

**DINKI UNet Loader** allows you to preload both a **safetensors-based UNet model** and a **GGUF-format UNet model**,  
and then choose which one to use through a simple Boolean toggle.

This makes it easy to switch between different UNet formats without manually reloading or reconnecting nodes,  
streamlining workflows that require rapid model comparison or format-specific processing.


### DINKI LM Studio Assistant

**DINKI LM Studio Assistant** connects directly to **LM Studio** to handle both image-based and text-based tasks.

- When an **image is provided**, the node sends it to LM Studio and retrieves a detailed description or analysis.
- When **no image is connected**, the node instead uses the LLM to generate a text prompt based on the user's input.

This makes it versatile for workflows that require automated prompt generation, image captioning, or multimodal understanding.

By using the `assistant_enabled` Boolean option, you can switch to a text-only mode  
without disabling the node entirely.  
This allows the LLM to operate normally even when no image is connected.
