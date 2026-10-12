import os
import folder_paths # 경로 확인을 위해 필요

class DINKI_Video_Player:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "show_video"
    OUTPUT_NODE = True
    CATEGORY = "DINKIssTyle/Viewer"

    def show_video(self, filename):
        # 1. 파일명 추출
        video_name = os.path.basename(filename)
        
        # 2. 파일이 temp 폴더에 있는지 output 폴더에 있는지 감지
        # 기본값은 output
        source_type = "output"
        
        # 입력된 절대 경로(filename)가 temp 폴더 경로로 시작하는지 확인
        temp_dir = folder_paths.get_temp_directory()
        if filename.startswith(temp_dir):
            source_type = "temp"
        
        # 3. UI에 파일명뿐만 아니라 type 정보도 함께 전달 (Dictionary 형태)
        # 이렇게 보내야 프론트엔드(JS)가 type="temp" 파라미터를 붙여서 요청할 수 있음
        return {"ui": {"video": [{"filename": video_name, "type": source_type, "subfolder": ""}]}}


class DINKI_Video_Viewer:
    """Save and preview a native ComfyUI VIDEO while passing it through."""

    @classmethod
    def INPUT_TYPES(cls):
        from .dinki_video_combine import available_formats, encoding_inputs, format_metadata, encoder_input
        extra_formats = [name for name in available_formats() if name not in ("h264-mp4", "av1-webm")]
        return {
            "required": {
                "video": ("VIDEO",),
                "filename_prefix": ("STRING", {"default": "DKST_Video"}),
                "format": (["auto", "mp4", "mkv", "webm", *extra_formats], {
                    "default": "auto", **format_metadata(available_formats())}),
                "codec": (["auto", "h264", "av1"], {"default": "auto"}),
                "always_save": ("BOOLEAN", {"default": False}),
            },
            # Append controls so positional values in existing workflows keep their meaning.
            "optional": {**encoding_inputs(default_pixel="auto", default_bitrate=0.0), "encoder": encoder_input()},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "preview_video"
    OUTPUT_NODE = True
    CATEGORY = "DINKIssTyle/Video"

    @classmethod
    def VALIDATE_INPUTS(cls, format, codec, pixel_format="auto", bitrate_mbps=0.0, encoder="cpu"):
        if format == "webm" and codec == "h264":
            return "WebM does not support H.264 in ComfyUI Save Video. Choose auto or av1."
        if format not in ("auto", "mp4", "mkv", "webm") or pixel_format != "auto" or bitrate_mbps != 0 or encoder != "cpu":
            from .dinki_video_combine import validate_encoding
            preset = format if format not in ("auto", "mp4", "mkv", "webm") else (
                "av1-webm" if codec == "av1" or format == "webm" else "h264-mp4")
            try:
                validate_encoding(preset, pixel_format, bitrate_mbps, encoder)
            except (ValueError, ImportError) as error:
                return str(error)
        return True

    def preview_video(
        self, video, filename_prefix="DKST_Video", format="auto", codec="auto",
        always_save=False, prompt=None, extra_pnginfo=None,
        pixel_format="auto", bitrate_mbps=0.0, encoder="cpu",
    ):
        from comfy_api.latest import Types
        from comfy.cli_args import args

        validation = self.VALIDATE_INPUTS(format, codec, pixel_format, bitrate_mbps, encoder)
        if validation is not True:
            raise ValueError(validation)

        if format not in ("auto", "mp4", "mkv", "webm") or pixel_format != "auto" or bitrate_mbps != 0 or encoder != "cpu":
            from .dinki_video_combine import save_media
            components = video.get_components()
            native = format in ("auto", "mp4", "mkv", "webm")
            preset = ("av1-webm" if codec == "av1" or format == "webm" else "h264-mp4") if native else format
            container = ("webm" if format == "auto" and codec == "av1" else "mp4" if format == "auto"
                         else "matroska" if format == "mkv" else format) if native else None
            metadata = None
            if not args.disable_metadata:
                metadata = dict(extra_pnginfo or {})
                if prompt is not None: metadata["prompt"] = prompt
            saved = save_media(components.images, components.audio, components.frame_rate, filename_prefix,
                               preset, pixel_format, bitrate_mbps, always_save, metadata, container, encoder)
            saved["result"] = (video,)
            return saved

        resolved_format = "webm" if format == "auto" and codec == "av1" else (
            "mp4" if format == "auto" else format
        )
        width, height = video.get_dimensions()
        target_dir = (folder_paths.get_output_directory() if always_save
                      else folder_paths.get_temp_directory())
        output_folder, base, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, target_dir, width, height
        )
        filename = f"{base}_{counter:05}_.{Types.VideoContainer.get_extension(resolved_format)}"
        descriptor = {
            "filename": filename, "subfolder": subfolder,
            "type": "output" if always_save else "temp",
        }
        metadata = None
        if not args.disable_metadata:
            metadata = dict(extra_pnginfo or {})
            if prompt is not None:
                metadata["prompt"] = prompt
            metadata = metadata or None
        output_path = os.path.join(output_folder, filename)
        video.save_to(
            output_path,
            format=Types.VideoContainer(resolved_format),
            codec=Types.VideoCodec(codec),
            metadata=metadata,
        )

        preview = descriptor
        # HTML video support for MKV and AV1 varies by browser. Keep the saved
        # file intact and use a temporary H.264 MP4 only for on-node playback.
        needs_preview = resolved_format != "mp4" or codec == "av1"
        if not needs_preview and codec == "auto":
            try:
                import av
                with av.open(output_path, mode="r") as container:
                    needs_preview = not container.streams.video or (
                        container.streams.video[0].codec.name != "h264"
                    )
            except Exception:
                needs_preview = True
        if needs_preview:
            preview_folder, preview_base, preview_counter, preview_subfolder, _ = (
                folder_paths.get_save_image_path(
                    f"preview_{filename_prefix}", folder_paths.get_temp_directory(),
                    width, height
                )
            )
            preview_name = f"{preview_base}_{preview_counter:05}_.mp4"
            video.save_to(
                os.path.join(preview_folder, preview_name),
                format=Types.VideoContainer.MP4,
                codec=Types.VideoCodec.H264,
                preset="ultrafast",
            )
            preview = {
                "filename": preview_name, "subfolder": preview_subfolder,
                "type": "temp",
            }

        return {
            "ui": {
                "dkst_video": [descriptor],
                "dkst_video_preview": [preview],
                "resolution": [f"{width} × {height}"],
            },
            "result": (video,),
        }

