# Tests

Preview Cache snapshots, producer skipping and viewer restoration:

```sh
python3 -m unittest discover -s tests -p test_preview_cache.py
node --test tests/dinki_preview_cache.test.cjs tests/dinki_preview_viewer.test.cjs tests/dinki_preview_menu.test.cjs
```

Python tests require NumPy and Pillow and use a small CPU tensor adapter instead
of installing PyTorch. Real floating-point snapshot and PNG files verify lossless
batch retention, immutable refreshes, workflow recreation, missing/blocked input,
bypassed original images, missing/corrupt cache recovery and partial-write cleanup.
Lazy dependency requests, hidden-widget serialization and existing viewer controls
are covered. On ComfyUI, replace the depth branch's Preview Image with Preview Cache,
run Capture once, then change the source image and verify that the depth producer
does not run and the original depth map remains. Refresh with the producer enabled,
then return to Capture once. Save/reload the workflow and restart the server to
verify retained pixels and previews; other direct consumers can still run the depth
producer independently.

Multiline Cache text retention and incoming-result persistence:

```sh
python3 -m unittest discover -s tests -p test_text_nodes.py
node --test tests/dinki_text_cache.test.cjs
```

Checks cover raw-link output lookup, empty/None/blocked input, separate node
state, manual replacement/clearing, exact whitespace, saved widget values and
frontend notifications. The frontend test also runs the real Python node and
reloads its received text before further empty/blocked and replacement runs.
On ComfyUI, connect a text source to `text_in` and keep the cache enabled when
muting its source; verify the received text is shown and saved in the workflow.

Auto Node Change completion switching, group routing and workflow lifecycle:

```sh
node --test tests/dinki_auto_node_change.test.cjs tests/dinki_node_change.test.cjs tests/dinki_node_switch.test.cjs
python3 -m unittest discover -s tests -p test_if_else_switch.py
```

Tests simulate prompt export, queue responses and completion events, including
cached completion before HTTP responds, errors/cancellation, manual changes,
workflow replacement, nested promoted controls, duplicate events and unsupported
overlapping submissions. Confirm sequential runs with a live ComfyUI frontend;
this node is not supported for queued/batch or API-only execution.

Host dependency installation, isolated node loading and trainer runtime setup:

```sh
python3 -m unittest discover -s tests -p test_dependencies.py
python3 -m unittest discover -s tests -p test_qwen_training_routes.py
node --test tests/dinki_qwen_train.test.cjs
```

Dependency tests require `packaging`; subprocess/pip operations are mocked and
do not install packages or require CUDA. Checks cover fresh-install entry points,
partial module import failure, offline startup, OpenCV variant reuse, trainer pin
validation, setup progress/errors and training/setup exclusion. Full CUDA package
installation and both worker checks must be verified on Windows/Linux with an
NVIDIA GPU. Face Crop's real YuNet test needs OpenCV.

Face crop geometry, disk exports, background control, ComfyUI outputs and UI:

```sh
python3 -m unittest discover -s tests -p test_face_crop.py
node --test tests/dinki_face_crop.test.cjs tests/dinki_ref_grid.test.cjs
```

Python tests require NumPy and Pillow. OpenCV enables the real bundled YuNet
inference check against the local 4×4 photo fixture (otherwise that check skips).
Tests cover padding and square boundaries, no upscaling, EXIF orientation, clean
PNG metadata, independent cells, variable image sizes, missing/corrupt sources,
path containment, unique folders, manifests, cancellation and restart behavior.
Routes use a stubbed ComfyUI server; fragmented/oversized JSON and image-list
outputs are verified. UI tests cover result restoration, option submission,
progress/cancellation, stale responses, polling cleanup and canceled-canvas-wheel
scrolling, arrows and sliders. Confirm loading on a running ComfyUI server.
ZIP checks cover archive contents/CRC, portable manifests, original-image
exclusion, cache reuse/invalidation, failed-write cleanup, concurrent preparation,
missing crops, completed-export gating, fragmented requests and Unicode attachment
headers. UI checks cover duplicate-click prevention, native base-aware download
links, error retry and suppression of stale downloads after removal/reconfiguration.

Qwen 2.1 LoRA training engine, project/worker control and UI tests:

```sh
python3 -m unittest discover -s tests -p 'test_qwen_training*.py'
python3 -m unittest discover -s tests -p 'test_qwen_caption.py'
python3 -m unittest discover -s tests -p 'test_qwen_vae.py'
python3 -m unittest discover -s tests -p 'test_qwen_text_weights.py'
node --test tests/dinki_qwen_train.test.cjs tests/dinki_ref_grid.test.cjs
```

Python tests need `torch`, `numpy`, `Pillow`, `einops` and `safetensors`. They run
the real Qwen driver/DiT on a tiny CPU model, check finite flow-matching gradients,
LoRA-only updates, frozen-adapter exclusion and save/reload. Project tests cover
aspect-preserving snapshots, caption/encoder cache invalidation, path validation,
single-job controls, complete-state selection and real subprocess failure,
pause/cancellation and final-artifact validation. Worker lifecycle tests substitute
small subprocesses for GPU training; they do not generate full-model person LoRAs.
UI tests cover caption persistence without focus loss, job ID serialization,
button actions, model dropdown saving/refresh and manual paths, generation blocking
and polling cleanup. Route tests cover registered model aliases, missing files and
training Python discovery. Caption tests use real photo resizing and saved connection
profiles with a mocked HTTP transport, covering optional use, manual model names,
server-side authentication and error redaction. UI tests also cover sequential captioning,
partial results, cancellation and preservation of captions edited during requests.
VAE checks cover ComfyUI key conversion, singleton temporal kernels, strict
checkpoint validation and real CPU safetensors loading/encode/decode. Log tests
cover complete-log retrieval and download, restricted job paths, native scrolling
and fragmented JSON requests.
Text-encoder checks load real INT8 safetensors through the encoder constructor
with a tiny substituted HF model, compare restored embedding/linear numerics and
reject malformed or unsupported quantization metadata without dropping scales.
Validate full CUDA
training, ComfyUI loading and identity quality on the target machine.

Image Ref Creator import, latent refinement and safetensors storage tests:

```sh
python3 -m unittest discover -s tests -p 'test_ref_nodes.py'
node --test tests/dinki_ref_grid.test.cjs
```

Run the Python tests in an environment with `torch`, `numpy`, `Pillow` and
`safetensors` (for example, ComfyUI's Python environment). They load Creator and
its storage helpers without importing the entire plugin, stub ComfyUI paths,
and cover refinement inside ComfyUI's inference mode, temporal channel/frame
ordering, independent photo encoding with video VAEs, token estimates and saved
tensor/metadata roundtrips. The JavaScript tests cover the retained gallery widget.
On a running ComfyUI instance, check Add Images, selection/removal, thumbnail
sizing and workflow save/reload. These checks do not measure generation quality
or demonstrate that stored references replace an identity LoRA.

LLM connection profiles, alias dropdown, legacy migration and credential exclusion:

```sh
python3 -m unittest discover -s tests -p 'test_l*m*.py'
node --experimental-vm-modules --test tests/dinki_llm_connections.test.cjs
```

These tests stub ComfyUI/HTTP services while exercising actual profile-file persistence,
permissions, auth, cache revisions, settings DOM and workflow/API serialization.
They also cover cyclic widget references, runtime LinkMap proxies, loading without
waiting for migration APIs, migration/refresh failures, linked legacy inputs and
the bundled legacy sample workflow with promoted generation inputs. On a
running ComfyUI instance, also check the DKST → LLM settings panel, alias selection
with regular nodes and Nodes 2.0, generation against LM Studio, and the `workflow`
and `prompt` metadata written by image/video save nodes. Previously generated files
are outside the migration scope.

Note UI and Markdown rendering tests:

```sh
npm ci --prefix tests
npm run test:note --prefix tests
```

The Markdown tests use jsdom and the actual bundled Marked and DOMPurify modules.
The startup tests link the actual extension module graph in a VM, including a
renderer dependency failure, rather than stripping its imports. The test script
enables Node's experimental VM modules for those integration tests.
jsdom is a test-only dependency; ComfyUI does not need Node.js or npm to render notes.

Python text node tests:

```sh
python3 -m unittest discover -s tests -p 'test_text_nodes.py'
```

In ComfyUI, check both the regular node renderer and Nodes 2.0: Edit/Preview,
Lock/Unlock, source Copy, workflow save/reload, resizing, text selection, scroll
gestures, and canvas zoom. Include a Korean note, wide table, long code line and
an image when checking layout.

Photo Studio editing, layout and full-resolution image exports:

```sh
python3 -m unittest discover -s tests -p test_photo_studio.py
node --experimental-vm-modules --test tests/dinki_photo_studio.test.cjs tests/dinki_photo_studio_editor.test.cjs
```

Python checks require Torch, NumPy, Pillow and aiohttp. They cover adjustments,
depth, session ownership, RGBA preservation and exports beyond the preview size
limit. Auto checks include balanced and warm scenes, backlit highlights, saturated
channels, existing clipping, transparent pixels and flat images. Auto preserves
the current white balance and creative controls; explicit White Balance Auto
remains available in the Color panel. UI checks exercise the actual modules with
jsdom: compact node creation
after native size expansion, promoted setting rows, image context menus, draft
exports without workflow execution, centered zoom, minimap navigation and aligned
Slide/Difference comparison. Apply stores settings and connections; the node's
full-resolution context-menu image is produced by workflow execution. Before
execution its image menu uses the displayed draft preview.
Geometry includes conservative automatic straightening, manual vertical/horizontal
perspective, rotation, aspect, scale and offsets. Empty transformed areas are black
for RGB and transparent for RGBA. Crop supports free/original/fixed aspect ratios,
eight drag handles, frame movement and keyboard adjustments. Its percentage bounds
are evaluated at native resolution, without upscaling. Depth previews and focus
sampling follow geometry and crop; the Before image uses matching crop dimensions.
Tests check batch/RGBA transforms, transparent-edge colors, native crop pixels,
preview/export/node agreement, connected controls, gesture undo and migration of
presets and workflows created before these spatial controls existed.
`tests/photo_studio_browser.py` provides an isolated browser harness with the
real Python preview/export routes. Confirm node layout in regular ComfyUI and
Nodes 2.0 on the target machine as well.

Image Viewer background updates and preview restoration:

```sh
node --test tests/dinki_preview_viewer.test.cjs tests/dinki_preview_menu.test.cjs
```

These tests cover results received while hidden, browser focus/visibility return,
offscreen canvas refresh, workflow output restoration, subgraph output keys,
failed image retries, removed nodes, merged batches, repeated filenames and
preservation of the selected batch image. Also verify returning from another
browser tab after generation in ComfyUI, with regular nodes and Nodes 2.0.

Video Load & Crop tests (real PyAV codecs and Torch tensors):

```sh
python3 -m unittest discover -s tests -p 'test_video_load.py'
node --experimental-vm-modules --test tests/dinki_video_load.test.cjs tests/dinki_image_crop.test.cjs
```

The Python test environment needs `av`, `torch`, `numpy`, `Pillow`, and `aiohttp`.
ComfyUI services are stubbed, while decoding, encoding, tensor processing, VFR
sampling, fractional FPS, and trimmed audio synchronization use real libraries.
The JavaScript tests link both actual extensions and exercise trim handles, length
input, FPS reset, workflow restoration, stale requests, and preview fallback. They
also check upload destination and duplicate-name handling, the file picker, drag
and drop, upload failure recovery, and a file-selection change during upload.
On a running ComfyUI instance, also verify normal nodes and Nodes 2.0 at different
canvas zoom levels, resizing, actual playback, and connection to native Save Video.

Video Combine and extended Video Player tests:

```sh
python3 -m unittest discover -s tests -p 'test_video_combine.py'
python3 -m unittest discover -s tests -p 'test_video_viewer.py'
python3 -m unittest discover -s tests -p 'test_video_encoding.py'
node --test tests/dinki_video_viewer.test.cjs
```

Combine integration tests use real PyAV/Pillow encoders and Torch tensors. They
cover available video formats, audio, fractional FPS, chroma padding, 10-bit
precision, GIF/WebP timing, cancellation cleanup, filename collisions, metadata,
and Player passthrough. The UI tests exercise both nodes' Fit/100% previews,
animation switching, format-dependent choices, downloads, and workflow restore.
Verify playback and layout in a running ComfyUI instance with regular nodes and
Nodes 2.0 as well; these automated tests do not launch the ComfyUI application.

Enable hardware integration tests on a machine with an available encoder:

```sh
DKST_RUN_HARDWARE_TESTS=1 python3 -m unittest discover -s tests -p 'test_video_combine.py'
```

These exercise real VideoToolbox 8/10-bit output, the existing FFmpeg adapter,
Auto in Video Player, audio and fractional FPS, and a CPU/hardware timing
comparison. NVENC tests run when the runtime registers it; they are skipped on
this Mac host. Device initialization must be allowed by the execution sandbox.
Selection tests separately verify bounded subprocess probes, cache behavior,
precision-preserving fallback and explicit device failures. Pipe tests cover
cancellation when a process stops reading and input errors without deadlock.
