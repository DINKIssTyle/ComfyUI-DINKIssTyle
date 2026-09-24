[Home](./README.md)
- [Comparison Video Tools](DINKI_Video_Tools.md)
- [Image](DINKI_Image.md)
- [Color Nodes](DINKI_Color_Nodes.md)
- [LM Studio Assistant](DINKI_LM_Studio_Assistant.md)
- [Prompts and Strings](DINKI_Prompt_and_String.md)
- [Node Utilities](DINKI_Node_Utils.md)
- [Internal Processing](DINKI_PS.md)

# DINKI Color Nodes
![Preview](DINKI_Color.png)

#### DKST Color (Auto Adjust) Node
The **DKST Color (Auto Adjust)** node implements the following automatic correction features:
- **Auto Tone**
- **Auto Contrast**
- **Auto Color**
- **Auto Skin Tone**

#### DKST Color (Adobe XMP) Node
The **DKST Color (Adobe XMP)** node applies presets from **Adobe Lightroom** and **Adobe Camera Raw**.
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


#### DKST Color (LUT) Node
The **DKST Color (LUT)** node applies color LUTs in **.cube** format.
Place your LUT files in: **~/ComfyUI/input/luts**

#### DKST Color (Oversaturation Fix)
Reduces excessive saturation or color distortion often produced by AI-generated images.
