"""Discover LoRA checkpoints to compare in LoRA Royale."""

import os
import re
from typing import List, Tuple

# Trainer epoch checkpoints: "<name>-000005.safetensors" (get_epoch_ckpt_name).
_EPOCH_RE = re.compile(r"^(?P<name>.+)-(?P<epoch>\d{6})\.safetensors$")


def scan_checkpoints(folder: str):
    """Find LoRA checkpoints in `folder`, sorted ascending.

    Returns a list of (label, path):
      - **Clean training run** (exactly one `<name>-NNNNNN.safetensors` run and no
        other .safetensors) → `label` is the integer epoch number.
      - **Anything else** (a bag of arbitrary LoRAs, multiple runs, or a run mixed
        with stray files) → EVERY .safetensors in the folder, labelled by its
        filename stem (string), sorted by name. This is the 'compare a group of
        random LoRAs' case — not just training sets.

    State directories (`<name>-NNNNNN-state/`) and non-safetensors are ignored.
    """
    if not folder or not os.path.isdir(folder):
        return []

    by_run = {}   # base name -> [(epoch, path)]
    others = []   # non-epoch .safetensors
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    for fn in names:
        if not fn.endswith(".safetensors"):
            continue
        full = os.path.join(folder, fn)
        if not os.path.isfile(full):
            continue
        m = _EPOCH_RE.match(fn)
        if m:
            by_run.setdefault(m.group("name"), []).append((int(m.group("epoch")), full))
        else:
            others.append(full)

    # Clean single training run → epoch labels (the classic LoRA Royale flow).
    if len(by_run) == 1 and not others:
        run = next(iter(by_run.values()))
        return sorted(run, key=lambda t: t[0])

    # Otherwise: compare every .safetensors, labelled by filename stem.
    all_paths = [p for run in by_run.values() for _, p in run] + others
    all_paths = sorted(set(all_paths), key=lambda p: os.path.basename(p).lower())
    return [(os.path.splitext(os.path.basename(p))[0], p) for p in all_paths]


def run_name_for_folder(folder: str) -> str:
    """Best-guess training-run base name in `folder` (the run with the most
    epoch checkpoints), or '' if none. Used for labelling / promote naming."""
    if not folder or not os.path.isdir(folder):
        return ""
    counts = {}
    try:
        names = os.listdir(folder)
    except OSError:
        return ""
    for fn in names:
        m = _EPOCH_RE.match(fn)
        if m:
            counts[m.group("name")] = counts.get(m.group("name"), 0) + 1
    return max(counts, key=counts.get) if counts else ""
