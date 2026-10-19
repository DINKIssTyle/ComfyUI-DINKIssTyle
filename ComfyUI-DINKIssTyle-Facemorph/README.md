# DKST Face Morph Studio

A standalone ComfyUI node pack for **correspondence-based shape morphing and appearance blending** across 2–10 photos.
Faces are analyzed automatically, and contour points can be corrected manually, as in traditional morphing software.
Sliders update a live preview without using the execution queue. Run renders IMAGE and MASK outputs from original photos.

## Installation

Copy or link `ComfyUI-DINKIssTyle-Facemorph` into ComfyUI's `custom_nodes` directory.
The repository's main node-pack loader does not automatically load this separate pack.

```text
ComfyUI/
└── custom_nodes/
    └── ComfyUI-DINKIssTyle-Facemorph/
        ├── __init__.py
        ├── nodes.py
        ├── core/
        └── web/
```

Restart ComfyUI after installing the folder. On startup, the pack checks its required packages
and automatically installs missing or unusable dependencies, including MediaPipe, using **the Python interpreter running ComfyUI**.
Installer output and failures are captured explicitly in the ComfyUI console, including when the launcher hides child-process streams.
Existing working packages are reused. If the environment has no pip, startup prepares it with `ensurepip` and retries.
If `ensurepip` is unavailable and `uv` is on PATH, startup uses `uv pip install --python` with the exact ComfyUI interpreter.
Installer subprocesses use UTF-8 mode on Windows. Package-resolution or network errors are reported with their original details;
those errors do not trigger installer fallback or automatic version downgrades.
Python 3.10 or newer is recommended. PyTorch and aiohttp are supplied by the ComfyUI environment.

ComfyUI Manager can also run the pack's `install.py`. For a manual repair, run this from the pack folder using ComfyUI Python:

```sh
python install.py
```

Add `DKST Face Morph Studio` from `DINKIssTyle/Facemorph` and click **Open Face Morph Studio**.
Alternatively, open `sample_workflows/FaceMorph_Studio.json` for a basic setup connected to Save Image.
When the editor opens, it checks for the official MediaPipe face landmark and multiclass selfie segmentation models. If missing, a setup modal downloads them automatically.
A failed download offers Retry or Cancel; installed models skip this setup.
Required Python packages are installed automatically at startup; the face and hair models are downloaded automatically when opening the editor if missing.
Set `DKST_AUTO_INSTALL=0` to disable automatic package installation in managed or offline environments.
If a package remains unavailable after installation, restart ComfyUI; the startup error includes a repair command
with the exact Python interpreter and requirements path.

Avoid installing multiple OpenCV distributions that provide the same `cv2` module.
A working OpenCV build is reused, including standard, headless, and contrib distributions.
MediaPipe is installed with `--no-deps`, then its wheel metadata is used to install missing or incompatible dependencies
while skipping replacement OpenCV packages when `cv2` already works. This prevents Windows `WinError 5` failures caused
by overwriting a loaded `cv2.pyd`. If OpenCV is absent, the installer supplies it first.
MediaPipe and OpenCV are managed by `install.py` and startup rather than listed unconditionally in `requirements.txt`.
After updating this pack, fully stop and restart ComfyUI so the updated installer can run.

## Usage

Click the **?** button in the upper-right corner for usage and storage help. Explanatory text is hidden during editing.

1. Click **Add photos**, drag and drop, or paste clipboard images with **Ctrl+V / Cmd+V** while the editor is open. The total limit is 10 photos. Clipboard images use the same upload and automatic face analysis as files; text paste in inputs is unchanged.
2. The largest detected face is selected by default. When a photo contains multiple faces, select its face number to change the selection.
3. Select a base photo. Its background, hair, neck, and aspect ratio determine the output composition.
4. Select global weights or a facial region, then adjust each photo's **Shape / Appearance and texture** weights.
5. Drag correspondence points in the source view. Releasing a point rebuilds the constrained mesh.
6. Leave **Hair protection** enabled to exclude source hair and preserve the base hairstyle. Use the selected photo’s **Mask** action to review or correct segmentation.
7. Click **Apply**, then click ComfyUI **Run**. Connect IMAGE to Save Image and use MASK for further processing.

Weights are **relative values** between 0 and 100. A=50 and B=50 produces a 50:50 mix;
A=100 and B=50 produces approximately 67:33. Region weights follow global weights until overridden.
Use the region's link button to restore global control. If all weights are zero, the base photo is used.
Left and right labels refer to the viewer's screen coordinates.

Supported regions include face outline and skin, eyes, nose, inner mouth, lips, and eyebrows.
Paired regions are adjusted together by default, with optional individual overrides:

```text
Eyes
  ├─ Left eye (screen)
  └─ Right eye (screen)
Eyebrows
  ├─ Left eyebrow (screen)
  └─ Right eyebrow (screen)
```

Select **Eyes** or **Eyebrows** to adjust both sides with one Shape or Appearance slider per photo.
Select a child entry only when you want an independent adjustment. Each property inherits in this order:
individual region → paired region → global photo weight.
Moving a paired slider clears individual overrides for that property and applies the new value to both sides.
The other property is preserved. Use the child's **Use paired region weights** button to restore inheritance,
or the parent's **Link both sides to global weights** button to reset the pair and both child overrides.
Previously saved workflows are upgraded automatically while preserving photos and facial point edits.

## Face Liquify: final face adjustments

The **Face Liquify** panel sits to the right of Live morph preview. It applies a separate local deformation
**after** all photos have been morphed into a separate face layer, **before** that layer is composited over the base photo. It does not change source photos,
source correspondence points, or mixing weights.

- Eyes: size, height, width, tilt, and distance. Both eyes follow the paired controls by default.
  Expand **Individual eyes** for left/right size, height, width, and tilt. Moving a paired control relinks
  that property; **Link both eyes** clears all individual eye overrides.
- Nose: height and width.
- Mouth: smile, upper lip, lower lip, width, and height.
- Face shape: forehead, chin height, jawline, and face width.

Every slider ranges from -100 to 100; **0 is unchanged**. Numeric fields accept exact values.
**Enabled** bypasses or enables this stage in both preview and Run while retaining its values.
**Reset all** restores zero adjustments. Undo, Apply, and workflow saving include these settings.
The adjustment centers follow the current morphed face as mixing weights change.

The browser applies the final warp in a separate WebGL pass. Run uses the same face-relative coordinate system
and inverse deformation fields on the premultiplied face layer and its alpha, processed in strips. The base photograph stays fixed; protected base hair is composited in front.
At zero settings, this stage is skipped entirely, preserving the previous result exactly.
Image borders are pinned. This is an independent landmark-guided implementation, not Photoshop's proprietary algorithm;
extreme combinations can stretch features. It does not include brush-based Liquify tools.

## Output and storage

- Outputs: ComfyUI `IMAGE [1,H,W,3]` and `MASK [1,H,W]`, float32, values from 0 to 1.
- Resolution: base photo size or a longest edge of 1024/2048/4096px. Base photo size is also limited to 4096px.
- Preview: starts at a 640px longest edge and adapts down to 384px on slower devices; browser WebGL, linear interpolation, and an 8-bit accumulation buffer.
- Run: original RGB photos, OpenCV high-quality interpolation, and 192-row strips to limit intermediate memory usage.
- `color_match`: optionally matches face-region color statistics to the base photo. Applies only to Run.
- Upload limits: 32MiB per photo and 40 megapixels after decoding. EXIF orientation is applied; originals are stored as RGB PNGs.
- Photos: `ComfyUI/input/DKST_Facemorph/<sha256>.png`.
- Models: `ComfyUI/models/facemorph/face_landmarker.task` and `selfie_multiclass.tflite`.
- Mask cache: `ComfyUI/input/DKST_Facemorph/masks/<sha256>.png` (red: usable face, green: protected hair).
- Photo references, points, region weights, hair options, normalized mask strokes, boundary controls, and Liquify settings are saved in the workflow through the node's `project` widget.
- When moving a workflow to another computer, also copy the referenced originals from `input/DKST_Facemorph`.
- Removing a photo from the editor does not delete its file. Saved analysis and cached masks are reused. Masks can be regenerated from originals and workflow corrections with the installed model.

## Morphing method and current scope

Photos are aligned in scale and rotation using eye positions, with consistent semantic correspondence indices across photos.
The mesh uses selected contour points from the detected face. Edge flips recover boundaries around eyes, lips, eyebrows, and the face outline.
This reduces incorrect mouth-region connections caused by blindly triangulating all 468 facial points.

Shape weights determine target coordinates. Each original photo is warped into that geometry, then mixed using appearance weights.
Outer support points create a soft transition into the base photo. Region weights are interpolated between vertices.
The browser and Python share the same points, triangles, per-vertex weights, and mask definitions.
Different interpolation filters and preview resolution mean their results are not pixel-identical.

Front-facing photos with similar angles and expressions work best. Large pose differences, glasses, hair occlusion,
and open-mouth versus closed-mouth combinations can cause overlap or stretching and require point correction.
If a collapsed eye or inner-mouth contour crosses itself, its edge constraints are relaxed and a warning is shown.
A folded target mesh prevents rendering and displays correction guidance. The previous preview is cleared to avoid showing stale results.

The editor supports moving existing correspondence points. It does not currently support adding or deleting arbitrary points,
freehand Bezier curves, 3D pose correction, hidden-surface generation, generative face restoration, multiband blending, or animation export.
It does not restore details absent from the originals or automatically resolve substantially different shooting conditions.
Photos are registered directly in the editor rather than through an upstream ComfyUI IMAGE input.

## Hair protection

**Hair protection** is enabled by default for new and existing workflows. It uses the official
[MediaPipe multiclass selfie segmentation model](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter)
through the existing MediaPipe runtime; no additional Python package is required. The model download is approximately 16 MB,
pinned to a version and verified with SHA-256 before installation.

Each selected face is analyzed within a padded head crop. Soft visible-face and hair masks are cached by photo,
model version, and crop. Manual corrections produce separate immutable cache entries. Ordinary morph and Liquify sliders
reuse the masks entirely in WebGL; they do not run segmentation or send render requests.
Changing correspondence bounds can rebuild the crop and mask once when a point is released.

Source appearance weights are multiplied by the warped usable-face mask and normalized again over the remaining sources.
If all sources are invalid, the base photograph remains visible. Covered landmarks have reduced shape confidence;
this confidence adjustment is softened if necessary to prevent an otherwise valid mesh from folding.
Liquify deforms only the blended face layer. The original base hair stays in front, and the returned MASK excludes it.
Run uses the same cached masks and composition rules, with higher-quality image interpolation.
Color match uses usable-face pixels when protection is enabled.

Select a photo and click **Mask**:

- **Include face** makes painted pixels usable and removes their hair protection.
- **Exclude hair** removes painted pixels from blending and protects them when this photo is the base.
- **Restore** returns painted pixels to the automatic mask.
- **Forehead inset / Temple inset** tighten the visible-face boundary; **Edge feather** softens the mask edge.
- **Reset corrections** discards manual strokes and boundary changes. **Apply** saves corrections; **Cancel** leaves the workflow unchanged.

Green marks usable face; pink marks protected hair. Strokes are normalized to photo coordinates and saved in the workflow.
When moving a project, copying the originals is sufficient to regenerate caches, or copy the `masks` directory too.
The model is a 256px semantic segmenter, not a strand-level matting model. Fine wisps, unusual lighting, accessories,
profile poses, and heavily occluded faces can require manual correction. Protection does not invent skin hidden in every source
or synthesize hairstyles. Preview precision and image interpolation differ from Run.

Implementation details and validation scope: [Hair protection design](HAIR_PROTECTION_PLAN.md).

## Validation

```sh
python -m unittest discover -s tests -v
node --test tests/test_math.cjs tests/test_liquify.cjs
```

Tests cover identity warps, normalized weights, paired and individual region overrides, legacy workflow migration, zero weights, alignment,
mesh errors, image and mask outputs, cancellation, API validation, unsupported model rejection, final face adjustment identity/bypass, paired eyes, image and mask warping, JavaScript/Python numerical parity, source exclusion, soft validity normalization, zero-coverage fallback, fixed base hair after Liquify, mask correction replay, and immutable caches.
`tests/landmarks.json` contains coordinates detected by MediaPipe from Matplotlib's Grace Hopper sample, without the photo itself.

References: [Morpheus help](https://www.morpheussoftware.net/morpheusmorph/help/help.php),
[FantaMorph Face Mixer](https://www.fantamorph.com/how_to_use_face_mixer.html),
[RE:Flex](https://revisionfx.com/products/reflex/resolve/),
[MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker).
