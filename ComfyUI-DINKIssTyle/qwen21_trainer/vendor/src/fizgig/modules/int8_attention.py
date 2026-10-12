"""comfy-kitchen's pure-INT8 SDPA for inference renders, as an opt-in any model family can use.

NVIDIA's kernel from the comfy-kitchen wheel (Apache-2.0): Q / K / V to int8 after a Hadamard rotation, P in uint8,
softmax maths in fp32. Measured on a 5090 (5 Sep 2026, MiniMax H3): 3.2x PyTorch's attention at 1.3k tokens, 6-7x at
9k-18k (a 1024² x 56-frame clip's attention from 6.3 s to 0.84 s per pass), ~1.6% relative error per call against
fp32 (the bf16 path is 0.23%). The longer the sequence, the bigger the win - video models first.

How a family plugs in:
  - its attention calls `attend(q, k, v, mask)` first and uses the result when it is not None ([B, H, S, D] in and
    out); None means "do your own attention" (the switch is off, a gradient is being taken, a mask is in play, or the
    kernel is missing / failed);
  - its description sets int8_attention=True, and the workbench engines turn the switch on around their renders
    (`renders()`); training never turns it on.

Three fallbacks, each announced once: the package missing / not importable (AMD builds skip it), no kernel for this GPU
(compute < 7.5), a call raising at run time.
"""
import contextlib

import torch

_STATE = {"wanted": False, "checked": False, "fn": None}


def set_wanted(on: bool) -> None:
    _STATE["wanted"] = bool(on)


def wanted() -> bool:
    return bool(_STATE["wanted"])


@contextlib.contextmanager
def renders(on: bool = True):
    """The switch on (or as given) for the renders inside, back as it was after."""
    had = _STATE["wanted"]
    _STATE["wanted"] = bool(on)
    try:
        yield
    finally:
        _STATE["wanted"] = had


def _fn():
    st = _STATE
    if not st["checked"]:
        st["checked"] = True
        try:
            import comfy_kitchen as _ck
            if _ck.int8_attention_is_available():
                st["fn"] = _ck.int8_attention
                print("[int8-attention] comfy-kitchen kernel active", flush=True)
            else:
                print("[int8-attention] comfy-kitchen has no kernel for this GPU — PyTorch attention instead",
                      flush=True)
        except Exception as _e:
            print(f"[int8-attention] comfy-kitchen not available ({type(_e).__name__}) — PyTorch attention instead",
                  flush=True)
    return st["fn"]


def kernel_available() -> bool:
    """True when the kernel is really there (import + GPU check + no run-time failure) - not tied to the switch."""
    return _fn() is not None


def attend(q, k, v, mask=None):
    """[B, H, S, D] attention through the int8 kernel when the switch is on, nothing is under grad, and no mask
    excludes a token (an all-True mask is no mask); grouped-query K / V are expanded to the query heads. None = the
    caller's own attention runs."""
    if not _STATE["wanted"] or torch.is_grad_enabled():
        return None
    if mask is not None:
        if mask.dtype != torch.bool or not bool(mask.all()):
            return None
    fn = _fn()
    if fn is None:
        return None
    if k.shape[1] != q.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    try:
        return fn(q, k, v)
    except Exception as _e:
        _STATE["fn"] = None
        print(f"[int8-attention] the kernel failed ({type(_e).__name__}: {_e}) — PyTorch attention for the rest of "
              "the run", flush=True)
        return None
