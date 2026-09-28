# DKST VoxCPM2 ComfyUI Nodes

Three nodes: **DKST VoxCPM2 (Downloader)**,
**DKST VoxCPM2 (Reference Audio)**, and **DKST VoxCPM2 (TTS & Cloning)**.

## Installation

Place this folder directly under `ComfyUI/custom_nodes/` and start ComfyUI.
The node checks for missing Python dependencies at startup and installs them
into the Python environment running ComfyUI. The first startup may take several
minutes. If automatic installation fails, use that same Python interpreter to
install the requirements manually:

```bash
python -m pip install -r ComfyUI/custom_nodes/ComfyUI-DINKIssTyle_VoxCPM2/requirements.txt
```

Restart ComfyUI after installation. The nodes appear under `DINKIssTyle/VoxCPM2`.
Python dependencies are installed at startup; model checkpoints are downloaded
only when you click a download button.

## Workflow

1. Add **DKST VoxCPM2 (Downloader)**. Select `VoxCPM2` and a Whisper checkpoint,
   then use the two download buttons. Checkpoints are stored in `model/`.
2. Connect its Whisper model path to **DKST VoxCPM2 (Reference Audio)**. Select a
   file from the `voice/` dropdown, use **Upload Voice**, or copy an audio file
   into `voice/` and select **Refresh Voices**. Supported extensions are WAV,
   MP3, FLAC, OGG, M4A, AAC, and Opus. Uploaded files with duplicate names get
   a numbered suffix.
   Audio decoding uses SoundFile and PyAV, so it also works on ComfyUI versions
   without `comfy.audio`.
3. **Transcribe Reference (Whisper)** transcribes the selected file and saves
   `voice/<audio stem>.txt`. Pressing the button again replaces that transcript.
   The multiline preview shows the saved transcript. The node outputs both the
   reference `AUDIO` and transcript `STRING`.
4. Connect the VoxCPM2 model path to **DKST VoxCPM2 (TTS & Cloning)**. Enter the
   target text in the multiline `text` field. Without reference audio, it performs TTS. With
   reference audio, it clones the voice. A connected reference transcript adds
   transcript-guided cloning. Its `sound` output connects to **Save Audio**.

`cfg` defaults to 2.0 and `inference_steps` to 10. Both controls are behind
ComfyUI's Advanced toggle. The synthesis node does not require a Whisper path.

VoxCPM's `torch.compile` optimization is disabled for these nodes. Its CUDA
graph warm-up is incompatible with ComfyUI installations using the
`cudaMallocAsync` allocator; inference uses the eager path instead.

Model downloads are explicit. The downloader's path outputs are constructed from
the current selection; missing checkpoint files are reported when a model is
used. Checkpoints in `model/` and audio/transcripts in `voice/` are ignored by Git.
