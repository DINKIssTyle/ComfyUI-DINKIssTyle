# DKST Qwen Image 2.1 LoRA Trainer

Qwen-only Fizgig engine + a ComfyUI gallery node. This trains model LoRA weights
from original photos and captions. Image Ref Creator remains independent; its
pooled `.safetensors` reference files are not training datasets.

## Install on the training machine

An NVIDIA CUDA GPU is required for the full trainer. Use a separate Python 3.12
environment, keeping ComfyUI's packages unchanged. From this directory on Windows:

```powershell
py -3.12 setup_runtime.py
```

This explicitly installs dependencies into this directory's `.venv`. It does not
download model weights. Optional accelerated kernels can be installed afterward:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

Restart ComfyUI, refresh the browser and add **DKST PS (Qwen 2.1 LoRA Trainer)**.
Click **Configure Models**, also available in Settings → DKST → Training.
Select files from the dropdowns. DiT, VAE and text encoders use ComfyUI's registered
model lists, including `extra_model_paths.yaml`; both adapters use the `loras`
list. **Refresh Models** reloads the lists. The local training `.venv` Python is
detected automatically after installation. For files outside those folders, choose
**Enter a server path manually…**. Previously saved external paths remain selectable.
The list shows available files; selecting one does not verify model compatibility.

Use these initial model files:

| Setting | Initial supported combination |
| --- | --- |
| Python | `qwen21_trainer/.venv/Scripts/python.exe` |
| DiT | `qwen_image_2.1_bf16.safetensors` |
| VAE | Qwen 2.1 `qwen_image_2_1_vae_bf16.safetensors` (ComfyUI layout), or official `vae/diffusion_pytorch_model.safetensors` (Diffusers layout) |
| Text encoder | `qwen3vl_8b_bf16.safetensors` from Comfy-Org Qwen Image 2.1 |
| Training adapter | `fizgig_qwen_image_2.1_training_adapter.safetensors` |
| Turbo LoRA | Optional `qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors` |

The loader recognizes the ComfyUI Qwen 2.1 VAE key layout, converts keys in
memory and removes singleton temporal kernel axes for the image specialization.
It checks every weight name and shape before strict loading. Original Qwen Image
and Wan video VAEs are not compatible; mismatches produce a short diagnostic.

Click **Check Runtime**. Qwen 2.1's VAE and tokenizer are different from the
original Qwen Image/Krea VAE and bundled Qwen3-VL-4B tokenizer. The worker obtains
Qwen 2.1 processor/tokenizer assets through Hugging Face's cache on first use;
prepopulate that cache for offline use. Model paths are saved in
`input/dkst_qwen21/settings.json`, not in the workflow.

The text encoder loader supports ComfyUI `int8_tensorwise` checkpoints, including
INT8 ConvRot: it restores the stored scales and inverse rotation on the CPU
before loading HF weights. Low-VRAM runtime quantization can then operate on the
decoded weights. This briefly requires memory for dense CPU weights. Other packed
quantization formats are rejected with a diagnostic; use a BF16 encoder instead.
Training compatibility with a specific file, including a Qwen3.5 encoder, must be
checked on the target machine; the UI does not treat a filename as proof of compatibility.
Start with the combination above and use `precision=auto` for runtime quantization.

## Train a person

1. Add original photos in the gallery. Caption each photo's pose, viewpoint,
   clothes and background. The trigger is prefixed automatically if absent.
   An empty caption uses only the trigger; varied captions separate identity
   from the scene. Include sharp portraits and wider shots; face crops may be
   added as ordinary photos. Cropping is manual; optional LLM captions are described below.
2. Enter a project name and trigger (for example `person_01`, `dkst_person01`).
3. Start with **Fast**, 704 resolution, 30 epochs, compile off. 704×704 is an
   area target around 0.5 MP; bucketing preserves aspect ratios. Standard uses
   rank 16 and lower Adaptive LR. Both use AdamW8bit and EMA 0.98.
4. Finish or clear the generation queue, then click **Start Training**.
   ComfyUI releases cached models, and fresh subprocesses cache latents, cache
   text, and train. **Run/Queue only reports status; it never starts training.**
5. Inspect logs, checkpoints and sample images. Sample Every = 0 disables previews.
   The optional Turbo LoRA uses 8 steps/CFG 1; without it previews use 25 steps/CFG 3.
6. Load a selected checkpoint with the standard LoRA loader or DKST Multi LoRA
   Loader. Include the trigger in the prompt. The frozen training adapter is
   disabled for previews and excluded from the saved person LoRA.

Output is inside the first ComfyUI `loras` directory:
`DKST_Qwen21/<project>/<job-id>/<project>.safetensors`. Each epoch is also saved.
Refresh ComfyUI's model list to select new LoRAs. There is no automatic "best
checkpoint" decision: compare held-out photos and new poses/clothes/backgrounds.

## Jobs, caching and controls

The gallery, individual caption fields and training log support wheel scrolling
inside the node. **View Full Log** opens a scrollable, selectable complete log
from its first line, with Training/Worker log selection, Refresh and Download Log.
This includes error headers that may be absent from the node's short log tail.

Projects live in `input/dkst_qwen21/projects`. Each job snapshots its photos,
captions and configuration. The thumbnail selection checkbox selects photos
for removal; every photo remaining in the gallery is used for training.

Latent caches are keyed by image content, VAE identity, engine version and bucket
resolution. Text caches are invalidated when captions or encoder files change.
Repeated jobs with the same photos/latent settings reuse VAE caches. Captions
are immutable during a running job; edit them in the node and start a new job.

**Pause** waits for the current cache stage, or the current training epoch, then
saves state. **Resume** restores the latest complete LoRA/optimizer/RNG/EMA state.
If stopped before a training state exists it starts training from the beginning
after reusing valid caches. **Stop** terminates the current stage process tree;
the last completed epoch state remains available. Changing models invalidates
resume. Browsers can reconnect through the stored job ID or Refresh Jobs.

One DKST training job runs at a time. Starts are rejected while the ComfyUI
generation queue is occupied. Generation submitted from this browser UI is
blocked during an active training job. External API clients must also avoid
submitting generation concurrently. Process separation does not create extra VRAM.

The coordinator keeps running and writes a heartbeat if ComfyUI restarts. A
stopped coordinator is reported as interrupted; inspect its log before resuming.
Do not manually delete active project directories or run the raw vendor CLI
against a managed job's caches while it is active.

## Source and scope

Vendored from [shootthesound/Fizgig](https://github.com/shootthesound/Fizgig/tree/33454e1b0bf06dd2203551abd3bdcdfd8cdaee68).
`vendor/UPSTREAM.json` records the revision, files and changes. Retained Krea and
MiniMax modules are checkpoint/quantization utilities, not registered trainers.
Only `qwen_image21` is registered. GUI, workbench, other model drivers, video,
auto-recaption and full fine-tune dependencies are excluded. Some generic LoRA
helpers retain optional network-format readers; the managed runner always uses LoRA.
See `vendor/LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md` for source attributions.

To reproduce the subset from that upstream checkout:

```sh
python scripts/vendor_qwen_trainer.py /path/to/Fizgig
```

Local CPU checks cover real Qwen block forward/backward, LoRA-only updates and
save/reload, dataset preparation, cache invalidation and subprocess controls.
Full-resolution CUDA training and identity quality require testing with actual
models and photos on the target GPU. No training-time or likeness guarantee is
inferred from these CPU checks.

## Optional automatic captions

LLM captioning is optional. Leave **Caption LLM alias (optional)** unselected and
**Caption model (optional)** empty to train using manually entered descriptions.
Starting training or queueing the node never calls an LLM.

To generate descriptions, select an existing alias from Settings → DKST → LLM,
enter the exact name of an image-capable model, and click **Auto Caption (Empty)**.
**Refresh LLM Aliases** updates the alias list after editing connections.
The server uses that alias's existing address, port and authentication; the node
stores only the alias and model name. The selected LLM receives each photo as a
JPEG resized to at most 1024 pixels on its longest side, through the existing
OpenAI-compatible chat API. Captioning uses ComfyUI's environment and does not
need the separate training Python installation.

Photos with existing descriptions are skipped. Generated English captions describe
pose, clothing, framing and background and appear in the gallery for editing.
The trigger is added separately when the training dataset is prepared.
Review captions before **Start Training**. Clear a photo's description to regenerate it.
**Stop Captioning** stops after the current photo. Earlier completed captions remain
if a request fails; manually edited or removed photos never receive stale results.
Captioning is unavailable while a training job is active.

Qwen and MiniMax H3 trainer jobs share a GPU reservation. Finish or actually pause
one before starting/resuming the other. Their settings and job IDs remain separate;
the same training Python environment can be selected by both nodes.
