# Tests

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
