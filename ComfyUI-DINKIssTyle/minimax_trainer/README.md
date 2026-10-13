# DKST PS (Minimax H3 LoRA Trainer)

Photo-based person/style LoRA training for MiniMax H3, using an H3-only Fizgig
engine pinned to `33454e1b0bf06dd2203551abd3bdcdfd8cdaee68`. The ComfyUI node
manages photos, captions, jobs and previews; the separate Python worker performs
real flow-matching LoRA training. This first version accepts still photos and
renders still-image previews. It does not train video/audio, create RefMods,
perform reference distillation or fine-tune the full base model.

## Python environment

Install on the **Windows/Linux machine running ComfyUI**, not the browser machine.
An NVIDIA CUDA GPU is required. Model weights are not downloaded automatically.

If the DKST Qwen trainer already runs successfully, select its existing
`qwen21_trainer/.venv/Scripts/python.exe` in **Configure Models**. The H3 dropdown
also detects this environment; both workers use the same dependency versions.
Click **Check Runtime** to check the H3 engine. No new Python installation is
needed just because this is a second node.

Use **Configure Models → Prepare Runtime** to prepare dependencies from the UI.
It reuses an existing Qwen `.venv` when MiniMax has no environment of its own;
otherwise it creates/updates `minimax_trainer/.venv`. Installation progress and
errors are shown in the dialog. Select the resulting Training Python and save.
**Check Runtime** validates the pinned dependency versions and CUDA availability.
Python with `venv`/pip is required on the server; Python 3.12 is recommended.

To make an independent environment from the command line, run from `minimax_trainer`:

```powershell
py -3.12 setup_runtime.py
```

If `py -3.12` says no suitable runtime was found, install Python 3.12 on the
ComfyUI server first, or invoke the installer with an existing compatible
Python executable (Python 3.10+). The script creates this directory's `.venv`;
select its **python.exe file**, not the `Scripts` directory. It never installs
packages at custom-node import time or changes ComfyUI's Python packages.
Optional accelerated kernels can be installed explicitly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

## Configure models

Restart ComfyUI and refresh the browser. Add **DKST PS (Minimax H3 LoRA Trainer)**,
or open [the starter workflow](../../sample_workflows/Minimax_H3_LoRA_Trainer.json).
Click **Configure Models**, also available in Settings → DKST → Training → MiniMax H3.
Model dropdowns include ComfyUI's registered paths and `extra_model_paths.yaml`.
External files can be entered manually. Paths are stored on the server in
`input/dkst_minimax_h3/settings.json`, outside the workflow.

| Setting | Model |
| --- | --- |
| Training Python | Existing Qwen trainer venv, or `minimax_trainer/.venv/Scripts/python.exe` |
| fl2va DiT | `minimax_h3_fl2va_pruned_int8_convrot.safetensors`; the full BF16 H3 file can also be loaded with runtime NF4 |
| ref2va DiT | Optional `minimax_h3_ref2va_pruned_int8_convrot.safetensors`; required when Training Base is ref2va |
| Video VAE | MiniMax H3 Video VAE, e.g. `minimax_h3_video_vae.safetensors` |
| Text encoder | H3 `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (recommended), or the corresponding BF16 encoder |
| Training adapter | Circlestone H3 adapter recommended for photos; optional, blank disables it. An Ostris adapter must match the selected base |
| Turbo LoRA | Optional H3 Turbo v4, used only for fl2va previews |

H3 uses **Qwen3-VL-32B**, not Qwen Image's 8B encoder or its VAE. H3 INT8 ConvRot
text encoders are not supported by this engine: use the H3 NVFP4 AWQ or BF16 file.
Dropdown entries show available files, not proof of compatibility. Before GPU
allocation the worker checks the selected DiT/text-encoder layout and every
required VAE parameter's name and shape, so an unrelated VAE cannot silently
leave the encoder randomly initialized. Use the corresponding H3 base checkpoint
for fl2va/ref2va; the filenames are not used to infer or convert a model's task.

H3's tokenizer and special-token assets are bundled with attribution, avoiding a
first-run tokenizer download. Original model weights remain external.

## Train

1. Add original photos, and write pose, clothing, view and background descriptions.
   The trigger is added automatically. The selection checkbox selects photos for
   removal; every photo remaining in the grid trains.
2. Optional: choose an LLM alias from Settings → DKST → LLM and manually enter an
   image-capable model name. **Auto Caption (Empty)** fills only blank captions.
   Manual captioning works without any LLM. Review generated captions before training.
3. Start with **Fast**, 50 epochs, 512 resolution (about 0.25 MP), precision Auto,
   compile Off, Training Base fl2va, Training Mode Default, Likeness and Style.
   Bucketing preserves the original aspect ratios. Standard raises rank to 16;
   epochs remain the explicit number entered in the node.
4. Clear/finish the generation queue and click **Start Training**. ComfyUI releases
   its cached models. Separate subprocesses cache video-VAE latents, cache 32B text
   embeddings, then train; each stage releases its GPU allocations when it exits.
5. Inspect sample images, epoch checkpoints and the full log. Use the arrows,
   position sliders or mouse wheel to scroll photos and samples. **Pause** requests
   a save at the trainer's safe boundary; **Resume** uses the latest complete state.
   **Stop** terminates the worker stage and keeps existing saved checkpoints.
6. Load a selected saved checkpoint with a normal ComfyUI LoRA loader or DKST Multi
   LoRA Loader, on the appropriate H3 base, and include the trigger in the prompt.

Recipes use rank/alpha 8 (Fast) or 16 (Standard), Automagic v3 starting at `1e-6`,
EMA 0.98 and caption dropout 0.05. Default trains blocks 20–49; More Blocks trains
6–49. Neither trains blocks 0–5 or the text token refiner. Likeness and Style gives
60% of steps to low noise; Model default gives 8%. The frozen training adapter is
active for training, disabled for previews, and excluded from the saved LoRA.
Reference distillation remains disabled even when training on ref2va.

**Sample Every = 0** disables previews. Otherwise the worker renders 768×768 stills
on its ordinary 20-step, CFG-free H3 schedule. A configured fl2va Turbo uses 6
steps at strength 0.75 for previews only. ref2va uses ordinary previews. The node
shows inline samples; there is no IMAGE output port. **Run/Queue only reports the
latest job's LoRA path and status; it never starts training.**

Qwen and H3 share a GPU training lock: a second training start/resume is rejected
until the first worker has actually paused/stopped/finished. Generation submitted
through the ComfyUI UI is blocked during either trainer's active work, including
photo preparation. Both nodes retain separate settings and job IDs.

## Output, recovery and logs

Output: `<first ComfyUI loras folder>/DKST_MinimaxH3/<project>/<job-id>/`.
Each epoch saves a checkpoint; the final file is `<project>.safetensors` and uses
H3 Kohya `lora_unet_blocks_*.lora_down/up.weight` pairs. Refresh ComfyUI's model
list to select a new LoRA. Only a validated final artifact marks a job completed;
no automatic "best checkpoint" is chosen.

Projects live in `input/dkst_minimax_h3/projects`. Job snapshots are immutable;
changing photos/options starts a new job. Latent cache identity includes photos,
VAE identity, engine version and resolution. Caption/encoder changes invalidate
text caches while keeping reusable latents. Model-file changes block resume.

**View Full Log** shows the entire Training or Worker log, starting from its first
line, with Refresh and Download. The compact log inside the node shows only a tail.
Pause during caching resumes by safely repeating caching. Pause during training
resumes a complete optimizer/RNG/LoRA state; an interrupted stage without such a
state starts training again rather than treating an epoch checkpoint as resumable.

## Engine and validation

This H3 subset retains upstream Apache-2.0, GPL-3.0 and third-party notices in
`vendor/`. Reproduce it from the pinned Fizgig source checkout with:

```sh
python scripts/vendor_minimax_trainer.py /path/to/Fizgig
```

`vendor/UPSTREAM.json` records source revision, modified modules and file hashes,
including the bundled tokenizer assets. Other training families and the full
workbench/fine-tune engines are excluded. Media helpers needed by H3's imports
remain, but the DKST photo gallery and job preparation do not accept video/audio.

CPU checks exercise the real tiny H3 model's flow loss, gradients, frozen base,
selected-block LoRA updates, frozen adapter exclusion and save/load parity. Managed
job, command, caption/cache, API, GPU-lock and shared UI regressions are covered.
Full-size CUDA training and preview quality must be verified on the training GPU;
CPU tests cannot establish likeness quality or GPU memory requirements.

References: [Fizgig H3 guide](https://github.com/shootthesound/Fizgig/blob/33454e1b0bf06dd2203551abd3bdcdfd8cdaee68/docs/MINIMAX_H3.md),
[H3 family recipe](https://github.com/shootthesound/Fizgig/blob/33454e1b0bf06dd2203551abd3bdcdfd8cdaee68/src/fizgig/families/minimax.py).
