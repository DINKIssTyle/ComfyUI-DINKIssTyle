"""VoxCPM2 nodes. Heavy dependencies are imported only when used."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "model"
VOICE_ROOT = ROOT / "voice"
VOX_NAME = "VoxCPM2"
VOX_REPO = "openbmb/VoxCPM2"
WHISPER_NAMES = ("tiny", "base", "small", "medium", "large-v3", "turbo")
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"})

_vox_cache = {"path": None, "model": None}
_whisper_cache = {"path": None, "model": None}


def _vox_names() -> list[str]:
    names = [VOX_NAME]
    if MODEL_ROOT.is_dir():
        names.extend(sorted(p.name for p in MODEL_ROOT.iterdir()
                            if p.is_dir() and p.name != VOX_NAME
                            and (p / "config.json").is_file()
                            and (p / "model.safetensors").is_file()))
    return names


def _vox_path(name: str) -> Path:
    if name not in _vox_names():
        raise ValueError(f"Unknown VoxCPM2 model: {name}")
    return MODEL_ROOT / name


def _whisper_path(name: str) -> Path:
    if name not in WHISPER_NAMES:
        raise ValueError(f"Unknown Whisper model: {name}")
    return MODEL_ROOT / "Whisper" / f"{name}.pt"


def _voice_names() -> list[str]:
    if not VOICE_ROOT.is_dir():
        return []
    return sorted(
        path.name for path in VOICE_ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def _voice_path(name: str) -> Path:
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or "/" in name or "\\" in name or Path(name).suffix.lower() not in AUDIO_EXTENSIONS):
        raise ValueError("Select a valid reference audio file from the voice folder.")
    path = (VOICE_ROOT / name).resolve()
    if path.parent != VOICE_ROOT.resolve():
        raise ValueError("Reference audio must stay inside the voice folder.")
    if not path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {name}")
    return path


def _read_voice_transcript(name: str) -> str:
    path = _voice_path(name).with_suffix(".txt")
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _load_voice_audio(name: str):
    import numpy as np
    import soundfile as sf
    import torch

    path = _voice_path(name)
    try:
        # libsndfile handles WAV, FLAC, OGG, and some MP3 files directly.
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        channels_first = samples.T.copy()
    except (RuntimeError, ValueError):
        # PyAV covers codecs such as M4A, AAC, and Opus without relying on
        # ComfyUI's optional comfy.audio module.
        import av

        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError(f"Reference file has no audio stream: {name}")
            stream = container.streams.audio[0]
            sample_rate = stream.codec_context.sample_rate
            resampler = av.AudioResampler(format="fltp", layout=stream.layout, rate=sample_rate)
            frames = []
            for frame in container.decode(audio=0):
                frames.extend(converted.to_ndarray() for converted in resampler.resample(frame))
            frames.extend(converted.to_ndarray() for converted in resampler.resample(None))
        channels_first = np.concatenate(frames, axis=1) if frames else np.empty((0, 0))

    waveform = torch.from_numpy(channels_first)
    if waveform.ndim != 2 or waveform.numel() == 0 or sample_rate <= 0:
        raise ValueError(f"Reference audio is empty or invalid: {name}")
    return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}


def _check_vox(path: Path) -> None:
    required = ("config.json", "model.safetensors", "audiovae.pth", "tokenizer.json")
    missing = [item for item in required if not (path / item).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing VoxCPM2 model files ({', '.join(missing)}): {path}. "
            "Download the model with DKST VoxCPM2 (Downloader)."
        )


def _check_whisper(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Whisper model not found: {path}. Download it with DKST VoxCPM2 (Downloader)."
        )


def _download_vox(path: Path) -> None:
    from huggingface_hub import snapshot_download

    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=VOX_REPO, local_dir=str(path))
    _check_vox(path)


def _download_whisper(name: str) -> None:
    import whisper

    path = _whisper_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The official loader verifies the checkpoint SHA256 while downloading.
    model = whisper.load_model(name, device="cpu", download_root=str(path.parent))
    _whisper_cache.update(path=str(path), model=model)
    _check_whisper(path)


def _load_vox(path_string: str):
    path = Path(path_string).resolve()
    _check_vox(path)
    from voxcpm import VoxCPM

    if _vox_cache["path"] != str(path):
        # ComfyUI can use cudaMallocAsync. VoxCPM's default torch.compile
        # warm-up uses CUDA graphs, whose pool check is incompatible with that
        # allocator. Eager inference avoids the failing warm-up and generation.
        model = VoxCPM.from_pretrained(str(path), load_denoiser=False, optimize=False)
        _vox_cache.update(path=str(path), model=model)
    return _vox_cache["model"]


def _load_whisper(path_string: str):
    path = Path(path_string).resolve()
    _check_whisper(path)
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    )
    if _whisper_cache["path"] != str(path) or next(_whisper_cache["model"].parameters()).device.type != device:
        _whisper_cache.update(path=str(path), model=whisper.load_model(str(path), device=device))
    return _whisper_cache["model"]


def _validated_audio(audio):
    import torch

    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Reference audio must use the ComfyUI AUDIO format.")
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError("Reference AUDIO must have a batch size of 1.")
    if waveform.shape[1] < 1 or waveform.shape[2] < 1:
        raise ValueError("Reference audio is empty.")
    rate = int(audio["sample_rate"])
    if rate <= 0 or not torch.isfinite(waveform).all():
        raise ValueError("Reference audio has an invalid sample rate or waveform.")
    return waveform.detach().float().cpu(), rate


def _mono_audio(audio):
    waveform, rate = _validated_audio(audio)
    return waveform.mean(dim=1).squeeze(0).contiguous(), rate


def _reference_wav(audio):
    import soundfile as sf

    mono, rate = _mono_audio(audio)
    handle = tempfile.NamedTemporaryFile(prefix="dinki_voxcpm2_", suffix=".wav", delete=False)
    path = handle.name
    handle.close()
    try:
        sf.write(path, mono.numpy(), rate)
    except Exception:
        os.unlink(path)
        raise
    return path


def _transcribe(audio, whisper_path: str) -> str:
    model = _load_whisper(whisper_path)
    mono, rate = _mono_audio(audio)
    if rate != 16000:
        # Whisper accepts a 16 kHz mono array. torchaudio's resampler is used
        # when present; scipy is a fallback for ComfyUI environments without it.
        try:
            import torchaudio.functional as AF

            mono = AF.resample(mono, rate, 16000)
        except (ImportError, OSError):
            from scipy.signal import resample_poly

            divisor = math.gcd(rate, 16000)
            return model.transcribe(
                resample_poly(mono.numpy(), 16000 // divisor, rate // divisor).astype("float32"),
                fp16=False,
            )["text"].strip()
    return model.transcribe(mono.numpy(), fp16=False)["text"].strip()


def _as_comfy_audio(wav, sample_rate: int):
    import torch

    waveform = torch.as_tensor(wav, dtype=torch.float32).detach().cpu()
    if waveform.ndim != 1 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError("VoxCPM2 did not return valid mono audio.")
    return {"waveform": waveform.reshape(1, 1, -1), "sample_rate": int(sample_rate)}


def _validated_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter text to synthesize.")
    return text.strip()


class Downloader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxcpm_model": (_vox_names(),),
                "whisper_model": (list(WHISPER_NAMES), {"default": "base"}),
            },
            "optional": {
                "download_action": (["none", "voxcpm2", "whisper"], {"default": "none"}),
                "request_id": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("VoxCPM2 model path", "Whisper model path")
    FUNCTION = "run"
    CATEGORY = "DINKIssTyle/VoxCPM2"
    OUTPUT_NODE = True  # permits button runs without a downstream output node

    def run(self, voxcpm_model, whisper_model, download_action="none", request_id=""):
        vox_path = _vox_path(voxcpm_model)
        whisper_path = _whisper_path(whisper_model)
        if download_action == "voxcpm2":
            if voxcpm_model != VOX_NAME:
                raise ValueError("Automatic download supports only the official VoxCPM2 model.")
            _download_vox(vox_path)
        elif download_action == "whisper":
            _download_whisper(whisper_model)
        elif download_action != "none":
            raise ValueError("Unsupported download action.")
        status = ("VoxCPM2 download complete" if download_action == "voxcpm2"
                  else "Whisper download complete" if download_action == "whisper"
                  else "Model paths ready")
        return {"ui": {"status": [status], "request_id": [request_id]},
                "result": (str(vox_path), str(whisper_path))}


class ReferenceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "whisper_model_path": ("STRING", {"forceInput": True}),
                "voice_file": (_voice_names() or [""],),
                "transcript_preview": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "transcribe_action": ("BOOLEAN", {"default": False}),
                "request_id": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("sound", "reference_transcript")
    FUNCTION = "run"
    CATEGORY = "DINKIssTyle/VoxCPM2"
    OUTPUT_NODE = True  # permits the transcription button to run this node alone

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, whisper_model_path, voice_file, transcript_preview="", transcribe_action=False, request_id=""):
        audio = _load_voice_audio(voice_file)
        if transcribe_action:
            transcript = _transcribe(audio, whisper_model_path)
            _voice_path(voice_file).with_suffix(".txt").write_text(transcript, encoding="utf-8")
        else:
            transcript = _read_voice_transcript(voice_file)
        return {"ui": {"transcript": [transcript], "request_id": [request_id]},
                "result": (audio, transcript)}


class TTSCloning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxcpm_model_path": ("STRING", {"forceInput": True}),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "cfg": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1, "advanced": True}),
                "inference_steps": ("INT", {"default": 10, "min": 1, "max": 100, "advanced": True}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "reference_transcript": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("sound",)
    FUNCTION = "run"
    CATEGORY = "DINKIssTyle/VoxCPM2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, voxcpm_model_path, text, cfg=2.0, inference_steps=10,
            reference_audio=None, reference_transcript=None):
        final_text = _validated_text(text)
        model = _load_vox(voxcpm_model_path)
        kwargs = {
            "text": final_text,
            "cfg_value": float(cfg),
            "inference_timesteps": int(inference_steps),
        }
        if reference_audio is None:
            wav = model.generate(**kwargs)
        else:
            path = _reference_wav(reference_audio)
            try:
                transcript = (reference_transcript or "").strip()
                if transcript:
                    kwargs["prompt_wav_path"] = path
                    kwargs["prompt_text"] = transcript
                else:
                    kwargs["reference_wav_path"] = path
                wav = model.generate(**kwargs)
            finally:
                os.unlink(path)
        return (_as_comfy_audio(wav, model.tts_model.sample_rate),)
