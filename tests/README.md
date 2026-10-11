# Tests

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
