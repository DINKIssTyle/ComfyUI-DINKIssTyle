# DKST VoxCPM2 ComfyUI Nodes

Three nodes: **DKST VoxCPM2 (Downloader)**, **DKST VoxCPM2 (TTS)**, and
**DKST VoxCPM2 (Cloning)**.

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

1. Add **DKST VoxCPM2 (Downloader)**. Select `VoxCPM2` and a Whisper checkpoint, then use the two
   download buttons. Checkpoints are stored in this package's `model/` folder.
2. Connect both model path outputs to **DKST VoxCPM2 (TTS)** or **DKST VoxCPM2 (Cloning)**. Whisper is
   only used for reference transcription; ordinary synthesis does not load it.
3. For cloning, connect a standard ComfyUI `AUDIO` source such as **Load Audio**.
   Enter the target text and optionally the exact transcript of the reference
   clip. If the transcript is blank, cloning uses reference-only mode. If it is
   filled, cloning uses VoxCPM2 prompt-audio and prompt-text mode.
4. **Transcribe Reference (Whisper)** queues a transcription-only execution of the
   connected audio and fills the transcript widget. Run the workflow normally
   afterward to create audio. Both synthesis nodes output standard `AUDIO`,
   suitable for **Save Audio**.

The speech synthesis node accepts text either in its multiline widget or from
the optional `text_input` STRING connection. A connected STRING takes priority.
Synthesis defaults to CFG 2.0 and 10 inference steps. Cloning uses the same
settings internally, matching the previous GUI.

VoxCPM's `torch.compile` optimization is disabled for these nodes. Its CUDA
graph warm-up is incompatible with ComfyUI installations using the
`cudaMallocAsync` allocator; inference uses the eager path instead.

Model downloads are explicit. The downloader's path outputs are constructed from
the current selection; missing checkpoint files are reported when a model is
used. Large checkpoint files under `model/` are ignored by Git.
