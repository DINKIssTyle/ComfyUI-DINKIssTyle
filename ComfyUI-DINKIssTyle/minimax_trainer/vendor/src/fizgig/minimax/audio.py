"""Reading and validating audio-only training files for MiniMax H3.

The audio cousin of clip.py, with one deliberate difference in doctrine. A clip's pixels are
refused when off-spec because silently fixing footage changes what trains; audio FORMAT carries
no such risk — resampling a 44.1 kHz mp3 to 32 kHz is lossless of intent — so any container,
rate or channel count decodes. DURATION is the strict axis: the packed sequence quantizes audio
length to the video frame grid, so a file is either one of the valid lengths or it is refused
naming them. Gizmo's audio tab cuts recordings to these lengths; the README carries the same
table for anyone making files by hand.

The spec:

    formats      .wav .mp3 .flac .m4a (any rate/channels — converted on decode)
    duration     one of 0.917 / 1.625 / 2.333 / 3.042 / 3.750 / 4.458 / 5.167 s (±25 ms)
    content      not digital silence
    captions     written in Gizmo, describing only the voice

Duration maps to the clip grid: a file of duration d trains as an audio-only item whose
placeholder video spans the matching 17n+5 pixel-frame count. The 5-frame slot (0.208 s) is
excluded — fifty milliseconds of voice trains nothing worth a training item.

The hop-exact trim matters more than it looks. The audio VAE emits ceil(samples/800) latents but
the model demands round(frames/24*40); at 56 and 107 frames those disagree by one, and the clip
path REFUSES such files at cache time. Audio files are instead trimmed to exactly
`audio_latents_for_frames(frames) * 800` samples, so the latent count always lands on what the
model wants.
"""

import os
import subprocess

import numpy as np

from fizgig.minimax.audio_vae import HOP_LENGTH
from fizgig.minimax.clip import GRID_FRAMES, _ffmpeg
from fizgig.minimax.model import FPS, audio_latents_for_frames

AUDIO_FILE_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")
AUDIO_SAMPLE_RATE = 32000
AUDIO_CHANNELS = 2

# The valid pixel-frame counts for an audio-only item: the clip grid minus the 5-frame slot.
AUDIO_GRID_FRAMES = tuple(f for f in GRID_FRAMES if f > 5)

# ±1 hop (25 ms). Gizmo exports hop-exact so it never needs this; the tolerance exists for
# hand-made files cut to the README's second-precision durations.
DURATION_TOLERANCE_SAMPLES = HOP_LENGTH


class AudioRejected(ValueError):
    """An audio file that cannot become a training item. The message is user-facing — it names
    what was found, the valid lengths, and where to fix it."""


def is_audio(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_FILE_EXTENSIONS


def valid_durations_text() -> str:
    """'0.917, 1.625, ... or 5.167 seconds' — for refusal messages and the README."""
    secs = [f / FPS for f in AUDIO_GRID_FRAMES]
    return ", ".join(f"{s:.3f}" for s in secs[:-1]) + f" or {secs[-1]:.3f} seconds"


def grid_frames_for_samples(n_samples: int, name: str = "audio") -> int:
    """The pixel-frame count whose duration matches, or AudioRejected naming the valid lengths."""
    for frames in AUDIO_GRID_FRAMES:
        target = round(frames / FPS * AUDIO_SAMPLE_RATE)
        if abs(n_samples - target) <= DURATION_TOLERANCE_SAMPLES:
            return frames
    got = n_samples / AUDIO_SAMPLE_RATE
    ceiling = AUDIO_GRID_FRAMES[-1] / FPS
    extra = (" Cut it into segments with Gizmo's audio tab."
             if got > ceiling + 0.025 else
             " Cut it to length with Gizmo's audio tab, or see the audio spec in the README.")
    raise AudioRejected(
        f"{name}: {got:.3f} s of audio. A voice training item has to be exactly "
        f"{valid_durations_text()} long.{extra}")


def read_audio_file(path: str) -> np.ndarray:
    """Decode any supported audio file to float32 (2, L) at 32 kHz in [-1, 1].

    Unlike clips, conversion is welcome here — rate and channel count are decode details, not
    training intent. What IS refused: no decodable audio, and digital silence (a clip treats
    silence as muted and trains its video; an audio file has nothing else to train)."""
    name = os.path.basename(path)
    raw = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", path,
         "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_SAMPLE_RATE), "-"],
        capture_output=True).stdout
    if not raw:
        raise AudioRejected(f"{name}: no decodable audio in this file")
    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1, AUDIO_CHANNELS).T.copy()
    if not np.any(wav):
        raise AudioRejected(f"{name}: decodes to digital silence — nothing to train a voice on")
    return wav


def trim_to_grid(wav: np.ndarray, frames: int) -> np.ndarray:
    """Exactly `audio_latents_for_frames(frames) * 800` samples — the hop-exact length whose
    VAE encoding produces precisely the latent count the model demands for this frame count.
    Trims a long tail; zero-pads a short one (both within the ±25 ms admission tolerance)."""
    want = audio_latents_for_frames(frames) * HOP_LENGTH
    if wav.shape[1] >= want:
        return wav[:, :want]
    return np.pad(wav, ((0, 0), (0, want - wav.shape[1])))


def hop_exact_samples(frames: int) -> int:
    """The exact sample count Gizmo exports for a segment of `frames` — trim_to_grid's target."""
    return audio_latents_for_frames(frames) * HOP_LENGTH
