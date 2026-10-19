# Hair protection design

Status: implemented. The remaining limitations are listed under Validation scope below.
Goal: preserve the base hairstyle and composition while using visible facial anatomy from other photos. Local neck/background resampling around a changed face silhouette prevents double contours.

## Original leakage paths

The original face mask was derived from a landmark mesh. All facial vertices are opaque, including forehead points;
a zero-alpha support ring is placed 12% outside the facial oval. Interpolation creates a soft boundary, but it has no knowledge of hair.
Consequently, hair inside the face outline (bangs, loose strands) and hair within the outer transition ring can enter the blend.
Landmark-based alignment does not remove an occluding strand from the source texture.

Face Liquify originally deformed the completed composite. Its local fields can also move adjacent hair or background near the forehead and jaw.
Protecting the mixing mask alone would not address that final deformation.

## Recommended pipeline

1. Analyze photos once when registered. Cache separate soft masks for usable face pixels and occluding hair.
   Preserve soft strand boundaries where the detector supports them; allow manual corrections when it does not.
2. Derive an occlusion score from mask overlap around each feature. Reduce a source's shape contribution to covered regions,
   rather than letting inferred points under bangs strongly determine the target forehead or eye shape.
3. Warp each source's color and validity mask using the same correspondence mapping.
   Multiply the source's per-pixel appearance weight by its warped valid-face mask, then normalize over valid sources.
   Hair from a non-base source must have zero contribution even when it lies inside the landmark oval.
4. Build the mixed face as a premultiplied color + alpha layer, with a separate base context behind it.
   Invalid pixels fall back to that context instead of dividing by zero or using excluded source hair.
   If the entire face has no usable contributions, retain the original base.
5. Warp the base context through the base-to-target mesh. Apply the same Face Liquify field to that context
   and the completed face layer/alpha. Otherwise face contraction reveals the original jaw under the shrinking alpha.
   Use a distance field to smoothly pin displacement near protected hair along the sampled path. The transition width grows with travel distance, preventing abrupt edge folds. Limit the morph context to a feathered neighborhood of the old/new face bounds.
6. Composite the adjusted face over that context. Use original base hair as a protected foreground layer,
   so the base's bangs and loose strands stay in front of the new face.
   Attenuate adjusted face alpha at that protected boundary to avoid double edges.
7. Return a face MASK that follows Liquify and excludes protected hair.

Merely putting original hair over an unconstrained full-image warp can leave displaced hair copies.
The face/context split plus swept-path hair pinning avoids pulling protected hair into exposed face/background areas.
Keeping the underlying base completely unwarped is also insufficient: its original facial outline becomes visible
when the morphed face or Liquify alpha contracts.

## Implementation stages

### 1. Boundary controls and manual protection

Add an advanced mask editor for each photo, with Include face and Exclude hair strokes.
Keep mask editing out of the primary photo cards. Expose it through a compact Mask action and the existing help dialog.
Provide forehead/temple boundary tightening and edge feather controls as a quick first measure.
These geometric controls reduce outer-boundary leakage but cannot identify bangs inside the face by themselves.

### 2. Automatic hair and visible-face masks

Select and validate a local segmentation model for the supported ComfyUI environments before adding a runtime dependency.
The model must distinguish hair over the forehead/cheeks from usable face pixels; a simple head silhouette is insufficient.
Generate masks during photo analysis and cache by image hash, model version, and manual-edit revision.
Sliders reuse cached masks; they do not trigger repeated inference or run the ComfyUI workflow.
Manual corrections remain available for wispy strands and detection failures.

### 3. Foreground composition and validation

Use the same soft masks, alpha rules, and normalization in WebGL and Python.
Test dissimilar hairstyles, bangs covering one eye, hair across cheeks, profile poses, dark hair on dark skin/background,
base changes, missing usable pixels, and Liquify adjustments near the forehead and jaw.
Verify that excluded source hair never contributes, base hair remains stable, zero-coverage pixels stay finite,
and workflow reload reproduces mask corrections and final output.

## Suggested defaults

- Hair source: Base photo.
- Protect base hair: enabled.
- Exclude source hair from face blending: enabled when masks are available.
- Occluded pixels: retain the base photo, with a clear mask-review indication if analysis needs correction.
- No hairstyle synthesis or invented skin behind fully occluded regions.

## Implemented model and storage

Automatic masks use the official MediaPipe multiclass selfie segmenter, float32 version 1.
Hair is class 1 and face skin is class 3. A padded crop around the selected face limits interference from other heads.
The existing MediaPipe package supplies ImageSegmenter; startup dependencies do not change.
The editor downloads missing models in its setup modal. The TFLite model is pinned by SHA-256:
`c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0`.

Masks are RGB PNGs at up to 768px on the longest full-photo edge: red is usable face, green is protected hair, and blue stores distance to hair normalized by one quarter of the mask’s longest edge.
The model itself uses 256px input, so increasing the cache size does not invent segmentation detail.
Automatic and corrected masks have separate immutable content-derived names under `input/DKST_Facemorph/masks`.
The workflow stores normalized brush strokes and forehead/temple/feather values, so caches can be rebuilt.
Cached head crops are tied to selected-face bounds, and each correction revision receives its own mask identifier.
Only source files and model files are needed to reconstruct the result; mask cache copying is optional.

Both rasterizers accumulate masked appearance weights, normalize RGB, and use the union of active source validity
for face coverage. A separate base context follows the same mesh and Liquify field as face color/alpha,
with hair distance sampled at five locations along the destination-to-source path. Its smooth transition spans three times the proposed travel, with a minimum of three mask pixels. The context’s morph displacement fades out within 25% of the old/new face bounding box.
The original base hair is then overlaid in original base coordinates. Covered shape weights use smoothed hair
confidence with a minimum confidence of 0.1; an adaptive relaxation preserves an otherwise valid triangulation.
User-created folded geometry is still rejected.

## Validation scope

Automated synthetic tests verify excluded source color, soft weight normalization, all-invalid fallback,
fixed protected base pixels and image borders after Liquify, old-outline removal after contraction, reduced shape confidence, correction replay,
restore strokes, boundary tightening, cache immutability, and input validation.
Browser checks exercise real WebGL pixel values, Mask Apply/Cancel and workflow reopen,
and ensure weight changes do not issue prepare or render requests.
Real-photo smoke checks use the installed model on Grace Hopper and Lena portraits.

The broad hairstyle/lighting matrix listed above remains a manual quality-review target, not a claim of exhaustive testing.
This semantic model cannot guarantee strand-level accuracy. Mask review is required for difficult hair, accessories,
and strong pose differences. Linear preview sampling and cubic Run sampling can differ at mask boundaries.

Reference: [MediaPipe Image Segmenter](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter).
