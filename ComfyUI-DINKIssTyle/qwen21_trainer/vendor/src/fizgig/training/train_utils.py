"""Training utilities: loss tracking, checkpoint naming, state management."""

import argparse
import logging
import os
import re
import shutil
from typing import Callable

import accelerate

logger = logging.getLogger(__name__)


# Checkpoint file naming patterns
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"


class LossRecorder:
    """Track per-step losses with a running moving average over the last epoch's worth of steps.

    Slots are indexed by the in-epoch step. A step that records no loss — e.g. an image the
    loss watch excluded from training — must be drop()ed so its slot leaves the average;
    otherwise skipped slots hold stale (or zero-padded) values that bias avr_loss, which the
    adaptive-LR watcher reads as a real signal."""

    def __init__(self):
        self.loss_list: list[float] = []
        self.loss_total: float = 0.0
        self._empty: set[int] = set()  # slots not currently holding a live loss

    def _grow(self, step: int) -> None:
        while len(self.loss_list) <= step:
            self._empty.add(len(self.loss_list))
            self.loss_list.append(0.0)

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        self._grow(step)
        if step not in self._empty:
            self.loss_total -= self.loss_list[step]
        self._empty.discard(step)
        self.loss_list[step] = loss
        self.loss_total += loss

    def drop(self, *, step: int) -> None:
        """Mark an in-epoch step as not-trained (skipped/excluded): its slot leaves the average."""
        self._grow(step)
        if step not in self._empty:
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = 0.0
            self._empty.add(step)

    @property
    def moving_average(self) -> float:
        n = len(self.loss_list) - len(self._empty)
        if n <= 0:
            return 0.0
        return self.loss_total / n


def list_state_dirs(output_dir: str, output_name: str) -> list[tuple[int, str]]:
    """Every `<output_name>-NNNNNN-state/` in output_dir as (epoch_no, full_path), newest first.

    The regex is ANCHORED and scoped to output_name on purpose: several LoRAs commonly share one
    output directory, and a loose `*-state` glob would let one run's pruning delete another's
    states. Shared by both trainers (Klein and Krea 2 agree on this naming) so the pattern lives
    in exactly one place — the GUI's _detect_latest_state_dir mirrors it."""
    pattern = re.compile(rf"^{re.escape(output_name)}-(\d{{6}})-state$")
    found: list[tuple[int, str]] = []
    try:
        entries = os.listdir(output_dir)
    except OSError:
        return found
    for entry in entries:
        m = pattern.match(entry)
        if not m:
            continue
        full = os.path.join(output_dir, entry)
        if os.path.isdir(full):
            found.append((int(m.group(1)), full))
    found.sort(reverse=True)
    return found


def prune_state_dirs(output_dir: str, output_name: str, keep_n) -> None:
    """Keep the keep_n highest-numbered state dirs, delete the rest.

    Two deliberate safety choices:
      * keep_n is clamped to >= 1. A blank/0/negative box must never mean "delete everything" —
        that would take the state just written with it, and the caller then recreates the dir as
        an adaptive-LR sidecar with no weights in it, which resume would happily pick up and choke
        on. Clamping here as well as in the GUI keeps the invariant with the code that relies on it.
      * every rmtree is caught individually. This runs at an epoch boundary of a multi-hour run;
        a Windows AV scanner holding a just-written .safetensors raises PermissionError, and the
        old unguarded call would have taken the whole run down over a housekeeping failure.

    Always call AFTER the state (and any sidecar) is written, never before."""
    try:
        keep_n = max(1, int(keep_n))
    except (TypeError, ValueError):
        keep_n = 1
    for _epoch_no, path in list_state_dirs(output_dir, output_name)[keep_n:]:
        try:
            shutil.rmtree(path)
            logger.info(f"[state] pruned old state: {os.path.basename(path)}")
        except Exception as e:
            logger.warning(f"[state] could not remove {path}: {e}")


_ILLEGAL_NAME_CHARS = '<>:"|?*'


def validate_output_name(output_name: str) -> str:
    """Refuse an output name that cannot become a filename. Returns it unchanged if it can.

    Every function below turns this string into a path, and nothing opens one until the first
    checkpoint save — an epoch in. A name carrying a pasted newline (issue #70) trains for
    sixteen minutes and then dies inside safetensors' Rust writer with OS error 123, which
    names neither the setting nor the character. Worse, `_save_lora` runs before the state
    save, so the run is not even resumable: the epoch is simply gone.

    Checked here rather than in one trainer because all three families take --output_name from
    the same unchecked GUI string, and a direct CLI run skips the GUI's own check entirely.
    """
    name = "" if output_name is None else str(output_name)
    bad = next((c for c in name if c in _ILLEGAL_NAME_CHARS or c < " "), None)
    if bad is not None:
        _shown = repr(bad)[1:-1] if bad < " " else bad          # \n rather than an invisible gap
        raise ValueError(
            f"output name {name!r} cannot contain {_shown!r} — file names can't include that "
            f"character. A name pasted from somewhere else often carries a stray line break.")
    if not name.strip() or name != name.strip() or name.endswith("."):
        raise ValueError(
            f"output name {name!r} cannot be empty, or start or end with a space or a dot — "
            f"the saved file would not come out with the name you gave it.")
    # Both separators, on both platforms: a backslash is legal in a Linux filename, but a LoRA
    # named that way on a pod becomes an unopenable path the moment it reaches Windows.
    if "/" in name or "\\" in name or os.path.basename(name) != name:
        raise ValueError(
            f"output name {name!r} must be just a name, not a path — the folder it saves into "
            f"is set separately.")
    return name


def get_epoch_ckpt_name(model_name: str, epoch_no: int) -> str:
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ".safetensors"


def get_step_ckpt_name(model_name: str, step_no: int) -> str:
    return STEP_FILE_NAME.format(model_name, step_no) + ".safetensors"


def get_last_ckpt_name(model_name: str) -> str:
    return model_name + ".safetensors"


def get_remove_epoch_no(args: argparse.Namespace, epoch_no: int):
    if args.save_last_n_epochs is None:
        return None
    remove_epoch_no = epoch_no - args.save_every_n_epochs * args.save_last_n_epochs
    if remove_epoch_no < 0:
        return None
    return remove_epoch_no


def get_remove_step_no(args: argparse.Namespace, step_no: int):
    if args.save_last_n_steps is None:
        return None
    remove_step_no = step_no - args.save_last_n_steps - 1
    remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
    if remove_step_no < 0:
        return None
    return remove_step_no


def save_state_on_epoch_end(args: argparse.Namespace, accelerator: accelerate.Accelerator, epoch_no: int):
    """Write `<name>-NNNNNN-state/` for this epoch and return its path.

    Pruning is deliberately NOT done here — the caller writes the adaptive-LR sidecar into this
    dir first, so pruning has to happen after that, not inside the save."""
    model_name = args.output_name
    logger.info(f"Saving state at epoch {epoch_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no))
    accelerator.save_state(state_dir)
    return state_dir


def save_and_remove_state_stepwise(args: argparse.Namespace, accelerator: accelerate.Accelerator, step_no: int):
    model_name = args.output_name
    logger.info(f"Saving state at step {step_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, step_no))
    accelerator.save_state(state_dir)

    last_n_steps = args.save_last_n_steps_state if args.save_last_n_steps_state else args.save_last_n_steps
    if last_n_steps is not None:
        remove_step_no = step_no - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
        if remove_step_no > 0:
            state_dir_old = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no))
            if os.path.exists(state_dir_old):
                logger.info(f"Removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def save_state_on_train_end(args: argparse.Namespace, accelerator: accelerate.Accelerator, epoch_no: int):
    """Write the end-of-run state as `<name>-NNNNNN-state/` and return its path.

    Numbered like every other state dir on purpose. This used to be written as a bare
    `<name>-state`, which neither the GUI's state-dir scan nor this trainer's own --resume epoch
    parser could match — so the one state a finished run left behind was the one state nobody
    could resume from. Numbering it is what makes "train 20 more epochs on a finished LoRA" work."""
    model_name = args.output_name
    logger.info("Saving final state.")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no))
    accelerator.save_state(state_dir)
    return state_dir


def get_sanitized_config_or_none(args: argparse.Namespace):
    """Return args dict for logging, with sensitive values filtered out. Returns None if --log_config is not set."""
    if not args.log_config:
        return None

    sensitive_args = {"wandb_api_key", "huggingface_token"}
    sensitive_path_args = {"dit", "vae", "text_encoder", "base_weights", "network_weights", "output_dir", "logging_dir"}

    filtered = {}
    for k, v in vars(args).items():
        if k in sensitive_args or k in sensitive_path_args:
            continue
        if v is None or isinstance(v, (bool, str, float, int)):
            filtered[k] = v
        elif isinstance(v, list):
            filtered[k] = str(v)
        else:
            filtered[k] = str(v)
    return filtered


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    """Return a linear interpolation function from (x1,y1) to (x2,y2)."""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b
