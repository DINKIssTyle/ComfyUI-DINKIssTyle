"""MiniMax H3's clips: the shared clip checks (families/clips.py) bound to H3's ClipSpec.

The spec (also in the README, so it can be hit by hand):

    container    .mp4
    frame rate   exactly 24 fps - H3's own clock
    frames       on the 17n+5 grid: 5, 22, 39 ... 124
    size         multiples of 32
    audio        32 kHz stereo, or no track at all
    muting       a `_mute` suffix on the stem trains that clip's video only

The names below are the ones H3's own code (caching, audio, RefMod) has always imported.
"""
from fizgig.families import clips as _clips
from fizgig.families.clips import VIDEO_EXTENSIONS, ClipRejected, is_video, probe, read_frames  # noqa: F401
from fizgig.families.minimax import MINIMAX

SPEC = MINIMAX.clip_spec
AUDIO_SAMPLE_RATE = SPEC.audio_rate
AUDIO_CHANNELS = SPEC.audio_channels
SIZE_STEP = SPEC.edge_multiple
MUTE_SUFFIX = SPEC.mute_suffix
# Every length the VAE can encode: 17n+5, which is 5, 22, 39, 56, 73, 90, 107, 124. The long end is not withheld on
# cost grounds: a 124-frame clip is unaffordable on a 16 GB card and reasonable on a 96 GB one, and refusing it here
# would take the choice from the person who knows which they have. Gizmo shows what each length needs.
GRID_FRAMES = _clips.grid_frames(SPEC)


def _ffmpeg() -> str:
    return _clips.ffmpeg()


def is_muted(path: str) -> bool:
    return _clips.is_muted(path, SPEC)


def validate(path: str, info: dict = None, frames: int = None) -> dict:
    return _clips.validate(path, SPEC, MINIMAX.display_name, info=info, frames=frames)


def read_audio(path: str):
    return _clips.read_audio(path, SPEC)


def expected_audio_samples(frames: int) -> int:
    return _clips.expected_audio_samples(frames, SPEC)
