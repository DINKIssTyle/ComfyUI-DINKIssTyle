# Node Catalog

43 nodes across 9 categories. Display names use `DKST Category (Function)`.
Internal node IDs are unchanged for workflow compatibility. `DINKIssTyle/Utils` is now `DINKIssTyle/Util`.

| Category | Display name | Internal node ID |
| :--- | :--- | :--- |
| `DINKIssTyle/Color` | DKST Color (Adobe XMP) | `DINKI_adobe_xmp` |
| `DINKIssTyle/Color` | DKST Color (Auto Adjust) | `DINKI_Auto_Adjustment` |
| `DINKIssTyle/Color` | DKST Color (Deband) | `DINKI_Deband` |
| `DINKIssTyle/Color` | DKST Color (LUT Preview) | `DINKI_Color_Lut_Preview` |
| `DINKIssTyle/Color` | DKST Color (LUT) | `DINKI_Color_Lut` |
| `DINKIssTyle/Color` | DKST Color (Oversaturation Fix) | `DINKI_AIOversaturationFix` |
| `DINKIssTyle/Color` | DKST Color (XMP Preview) | `DINKI_Adobe_XMP_Preview` |
| `DINKIssTyle/Image` | DKST Image (Batch) | `DINKI_BatchImages` |
| `DINKIssTyle/Image` | DKST Image (Grid) | `DINKI_Grid` |
| `DINKIssTyle/Image` | DKST Image (Load) | `DINKI_Image_Load` |
| `DINKIssTyle/Image` | DKST Image (Overlay) | `DINKI_Overlay` |
| `DINKIssTyle/Image` | DKST Image (Photo Specs) | `DINKI_photo_specifications` |
| `DINKIssTyle/Image` | DKST Image (Resize) | `DINKI_Image_Resize` |
| `DINKIssTyle/Image` | DKST Image (Viewer) | `DINKI_Preview_Image` |
| `DINKIssTyle/LLM` | DKST LLM (LM Studio) | `DINKI_LMStudio` |
| `DINKIssTyle/PS` | DKST PS (Latent Source) | `DINKI_Empty_Or_Image_Latent` |
| `DINKIssTyle/PS` | DKST PS (Latent Upscale) | `DINKI_Upscale_Latent_By` |
| `DINKIssTyle/PS` | DKST PS (Mask Mix) | `DINKI_Mask_Weighted_Mix` |
| `DINKIssTyle/PS` | DKST PS (Remove Padding) | `DINKI_Remove_Pad_From_Image` |
| `DINKIssTyle/PS` | DKST PS (Resize & Pad) | `DINKI_Resize_And_Pad` |
| `DINKIssTyle/PS` | DKST PS (UNet Loader) | `DINKI_ToggleUNetLoader` |
| `DINKIssTyle/Prompt` | DKST Prompt (CSV Selector Live) | `DINKI_PromptSelectorLive` |
| `DINKIssTyle/Prompt` | DKST Prompt (CSV Selector) | `DINKI_PromptSelector` |
| `DINKIssTyle/Prompt` | DKST Prompt (Random) | `DINKI_random_prompt` |
| `DINKIssTyle/Text` | DKST Text (Concatenate) | `DINKI_Text_Concatenate` |
| `DINKIssTyle/Text` | DKST Text (Multiline) | `DINKI_Text_Multiline` |
| `DINKIssTyle/Util` | DKST Util (Anchor) | `DINKI_Anchor` |
| `DINKIssTyle/Util` | DKST Util (Auto Focus) | `DINKI_Auto_Focus` |
| `DINKIssTyle/Util` | DKST Util (Base64 Input) | `DINKI_Base64Input` |
| `DINKIssTyle/Util` | DKST Util (Base64 Viewer) | `DINKI_Base64Viewer` |
| `DINKIssTyle/Util` | DKST Util (Cross Switch) | `DINKI_CrossOutputSwitch` |
| `DINKIssTyle/Util` | DKST Util (Image Selector) | `DINKI_ImageSelector` |
| `DINKIssTyle/Util` | DKST Util (Image Signal) | `DINKI_ImagePreview` |
| `DINKIssTyle/Util` | DKST Util (Image to Base64) | `DINKI_Img2Base64` |
| `DINKIssTyle/Util` | DKST Util (Node Change) | `DINKI_Node_Change` |
| `DINKIssTyle/Util` | DKST Util (Node Check) | `DINKI_Node_Check` |
| `DINKIssTyle/Util` | DKST Util (Node Switch) | `DINKI_Node_Switch` |
| `DINKIssTyle/Util` | DKST Util (Note) | `DINKI_Note` |
| `DINKIssTyle/Util` | DKST Util (Sampler Preset) | `DINKI_Sampler_Preset` |
| `DINKIssTyle/Util` | DKST Util (String Switch) | `DINKI_String_Switch_RT` |
| `DINKIssTyle/Video` | DKST Video (Depth Parallax) | `DINKI_DepthParallax_MOV` |
| `DINKIssTyle/Video` | DKST Video (Image Compare) | `DINKI_Image_Comparer_MOV` |
| `DINKIssTyle/Viewer` | DKST Viewer (Video Player) | `DINKI_Video_Player` |

UNet Loader supports safetensors and GGUF. Video Player supports MP4, WEBM, and GIF.
Mask Mix blends masks using weights. Latent Source selects an empty or image-based latent.
