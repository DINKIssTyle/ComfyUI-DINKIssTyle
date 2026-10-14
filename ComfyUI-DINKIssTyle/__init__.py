# ComfyUI/custom_nodes/ComfyUI-DINKIssTyle/__init__.py
from .bootstrap import ensure_dependencies
from .node_loader import load_nodes

# Installation failures are logged; individual modules can still register nodes.
DEPENDENCY_ERRORS = ensure_dependencies()

NODE_MODULES = (
    ('dinki_prompt', ('DINKI_PromptSelector', 'DINKI_PromptSelectorLive', 'DINKI_random_prompt', 'DINKI_Sampler_Preset')),
    ('dinki_prompt_csv_library', ('DINKI_PromptCsvLibrary',)),
    ('dinki_color', ('DINKI_adobe_xmp', 'DINKI_Adobe_XMP_Preview', 'DINKI_AIOversaturationFix', 'DINKI_Auto_Adjustment', 'DINKI_Color_Lut', 'DINKI_Color_Lut_Preview', 'DINKI_Deband')),
    ('dinki_ps', ('DINKI_Upscale_Latent_By', 'DINKI_Mask_Weighted_Mix', 'DINKI_Resize_And_Pad', 'DINKI_Remove_Pad_From_Image', 'DINKI_ToggleUNetLoader', 'DINKI_Empty_Or_Image_Latent')),
    ('dinki_image', ('DINKI_ImageSelector', 'DINKI_CrossOutputSwitch', 'DINKI_ImagePreview', 'DINKI_Image_Resize')),
    ('dinki_viewer', ('DINKI_Video_Player', 'DINKI_Video_Viewer')),
    ('dinki_video_combine', ('DINKI_Video_Combine',)),
    ('dinki_switch', ('DINKI_IfElseBranch', 'DINKI_IfElseImageSwitch', 'DINKI_IfElseSwitch', 'DINKI_Node_Switch', 'DINKI_Node_Change', 'DINKI_Node_Change_3', 'DINKI_Node_Change_4', 'DINKI_Auto_Node_Change', 'DINKI_String_Switch_RT')),
    ('dinki_tool', ('DINKI_Arrange', 'DINKI_Workflow_Lock', 'DINKI_Note', 'DINKI_Node_Check', 'DINKI_Anchor', 'DINKI_Auto_Focus')),
    ('dinki_lmstudio', ('DINKI_LMStudio',)),
    ('dinki_batchImages', ('DINKI_BatchImages',)),
    ('dinki_image_size', ('DINKI_GetImageSize',)),
    ('dinki_photo_specs', ('DINKI_photo_specifications',)),
    ('dinki_photo_studio', ('DINKI_Photo_Studio', 'DINKI_Photo_Studio_Source')),
    ('dinki_multi_lora', ('DINKI_Multi_LoRA_Loader',)),
    ('dinki_ref', ('DINKI_Create_Image_Ref',)),
    ('dinki_qwen_train', ('DINKI_Qwen21_LoRA_Trainer',)),
    ('dinki_minimax_train', ('DINKI_Minimax_H3_LoRA_Trainer',)),
    ('dinki_face_crop', ('DINKI_Face_Crop_Export',)),
    ('dinki_overlay', ('DINKI_Overlay',)),
    ('dinki_comparer', ('DINKI_Image_Comparer_MOV',)),
    ('dinki_grid', ('DINKI_Grid',)),
    ('dinki_base64', ('DINKI_Img2Base64', 'DINKI_Base64Input', 'DINKI_Base64Viewer')),
    ('dinki_depth_parallax', ('DINKI_DepthParallax_MOV',)),
    ('dinki_text', ('DINKI_Text_Note', 'DINKI_Text_Multiline', 'DINKI_Text_Multiline_Cache', 'DINKI_Text_Concatenate', 'DINKI_Text_Split')),
    ('dinki_preview', ('DINKI_Preview_Image',)),
    ('dinki_preview_cache', ('DINKI_Preview_Image_Cache',)),
    ('dinki_image_comparison', ('DINKI_Image_Comparison',)),
    ('dinki_image_crop', ('DINKI_Image_Crop',)),
    ('dinki_load', ('DINKI_Image_Load',)),
    ('dinki_load_crop', ('DINKI_Image_Load_Crop',)),
    ('dinki_video_load', ('DINKI_Video_Load_Crop',)),
    ('dinki_monitor', ()),
    ('dinki_execution_report', ('DINKI_Execution_Report',)),
)

NODE_CLASS_MAPPINGS, _loaded_modules, NODE_IMPORT_ERRORS = load_nodes(__name__, NODE_MODULES)

NODE_DISPLAY_NAME_MAPPINGS = {
    # Prompt
    "DINKI_PromptSelector": "DKST Prompt (CSV Selector)",
    "DINKI_PromptSelectorLive": "DKST Prompt (CSV Selector Live)",
    "DINKI_random_prompt": "DKST Prompt (Random)",
    "DINKI_PromptCsvLibrary": "DKST Prompt (CSV Library)",
    "DINKI_Sampler_Preset": "DKST Util (Sampler Preset)",

    # Color & Correction
    "DINKI_adobe_xmp": "DKST Color (Adobe XMP)",
    "DINKI_Adobe_XMP_Preview": "DKST Color (XMP Preview)",
    "DINKI_AIOversaturationFix": "DKST Color (Oversaturation Fix)",
    "DINKI_Auto_Adjustment": "DKST Color (Auto Adjust)",
    "DINKI_Color_Lut": "DKST Color (LUT)",
    "DINKI_Color_Lut_Preview": "DKST Color (LUT Preview)",
    "DINKI_Deband": "DKST Color (Deband)",
    "DINKI_Photo_Studio": "DKST Color (Photo Studio)",
    "DINKI_Photo_Studio_Source": "DKST Color (Photo Studio Source)",

    # Photoshop/Processing
    "DINKI_Upscale_Latent_By": "DKST PS (Latent Upscale)",
    "DINKI_Mask_Weighted_Mix": "DKST PS (Mask Mix)",
    "DINKI_Resize_And_Pad": "DKST PS (Resize & Pad)",
    "DINKI_Remove_Pad_From_Image": "DKST PS (Remove Padding)",
    "DINKI_ToggleUNetLoader": "DKST PS (UNet Loader)",
    "DINKI_Empty_Or_Image_Latent": "DKST PS (Latent Source)",
    "DINKI_Multi_LoRA_Loader": "DKST PS (Multi LoRA Loader)",
    "DINKI_Create_Image_Ref": "DKST PS (Image Ref Creator)",
    "DINKI_Qwen21_LoRA_Trainer": "DKST PS (Qwen 2.1 LoRA Trainer)",
    "DINKI_Minimax_H3_LoRA_Trainer": "DKST PS (Minimax H3 LoRA Trainer)",
    "DINKI_Face_Crop_Export": "DKST PS (Face Crop Export)",

    # Viewer
    "DINKI_Video_Player": "DKST Video (Sequence Player)",
    "DINKI_Video_Viewer": "DKST Video (Video Player)",
    "DINKI_Video_Combine": "DKST Video (Combine)",

    # Switch
    "DINKI_IfElseBranch": "DKST Util (If/Else Branch)",
    "DINKI_IfElseImageSwitch": "DKST Util (If/Else Image Switch)",
    "DINKI_IfElseSwitch": "DKST Util (If/Else Switch)",
    "DINKI_Node_Switch": "DKST Util (Node Switch)",
    "DINKI_Node_Change": "DKST Util (Node Change)",
    "DINKI_Node_Change_3": "DKST Util (Node Change 3 Groups)",
    "DINKI_Node_Change_4": "DKST Util (Node Change 4 Groups)",
    "DINKI_Auto_Node_Change": "DKST Util (Auto Node Change)",
    "DINKI_String_Switch_RT": "DKST Util (String Switch)",

    # Tool
    "DINKI_Arrange": "DKST Util (Arrange)",
    "DINKI_Workflow_Lock": "DKST Util (Workflow Lock)",
    "DINKI_Note": "DKST Util (Note)",
    "DINKI_Node_Check": "DKST Util (Node Check)",
    "DINKI_Anchor": "DKST Util (Anchor)",
    "DINKI_Auto_Focus": "DKST Util (Auto Focus)",
    "DINKI_Execution_Report": "DKST Util (Execution Report)",


    # Others
    "DINKI_ImageSelector": "DKST Util (Image Selector)",
    "DINKI_CrossOutputSwitch": "DKST Util (Cross Switch)",
    "DINKI_ImagePreview": "DKST Util (Image Signal)",
    "DINKI_LMStudio": "DKST LLM (LM Studio)",
    "DINKI_BatchImages": "DKST Image (Batch)",
    "DINKI_photo_specifications": "DKST Image (Photo Specs)",
    "DINKI_Overlay": "DKST Image (Overlay)",
    "DINKI_Image_Comparer_MOV": "DKST Video (Image Compare)",
    "DINKI_Grid": "DKST Image (Grid)",
    "DINKI_Img2Base64": "DKST Util (Image to Base64)",
    "DINKI_Base64Input": "DKST Util (Base64 Input)",
    "DINKI_Base64Viewer": "DKST Util (Base64 Viewer)",
    "DINKI_DepthParallax_MOV": "DKST Video (Depth Parallax)",
    "DINKI_Image_Resize": "DKST Image (Resize)",
    "DINKI_GetImageSize": "DKST Image (Get Size)",

    # Text
    "DINKI_Text_Note": "DKST Text (Note)",
    "DINKI_Text_Multiline": "DKST Text (Multiline)",
    "DINKI_Text_Multiline_Cache": "DKST Text (Multiline Cache)",
    "DINKI_Text_Concatenate": "DKST Text (Concatenate)",
    "DINKI_Text_Split": "DKST Text (Split)",
    "DINKI_Preview_Image": "DKST Image (Viewer)",
    "DINKI_Preview_Image_Cache": "DKST Image (Preview Cache)",
    "DINKI_Image_Comparison": "DKST Image (Image comparison tool)",
    "DINKI_Image_Crop": "DKST Image (Crop)",
    "DINKI_Image_Load": "DKST Image (Load)",
    "DINKI_Image_Load_Crop": "DKST Image (Load & Crop)",
    "DINKI_Video_Load_Crop": "DKST Video (Load & Crop)",


}

WEB_DIRECTORY = "./js"
NODE_DISPLAY_NAME_MAPPINGS = {
    name: display for name, display in NODE_DISPLAY_NAME_MAPPINGS.items() if name in NODE_CLASS_MAPPINGS
}

if "dinki_execution_report" in _loaded_modules:
    try:
        _loaded_modules["dinki_execution_report"].install_execution_report_hooks()
    except Exception:
        import logging
        logging.getLogger(__name__).exception("DKST: could not install execution report hooks")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
