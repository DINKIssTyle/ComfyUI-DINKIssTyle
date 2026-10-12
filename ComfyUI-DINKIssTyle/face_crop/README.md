# DKST PS (Face Crop Export)

Prepare face images for a LoRA dataset using a multi-photo gallery and a small
CPU-only YuNet detector. No training environment, GPU model, LLM or account is
required. Original photos remain unchanged.

## Use

1. Restart ComfyUI after installing/updating the node files and refresh the browser.
2. Add **DKST PS (Face Crop Export)** from `DINKIssTyle/PS`, or open the
   [starter workflow](../../sample_workflows/Face_Crop_Export.json).
3. Use **Add Images** to upload photos. All photos left in the gallery are processed;
   checkboxes select photos for removal. The gallery has wheel scrolling, arrows
   and a position slider.
4. Set the output folder, filename prefix, face selection and padding. For a 4×4
   contact sheet, select **grid_split: 4x4** to detect faces separately within each
   cell. Use **Off** for ordinary photographs. All faces or the largest face are
   selected per image/cell; there is no person identification or identity matching.
5. Click **Execute** to detect, crop and save. This button starts a background CPU
   task; no ComfyUI generation queue is needed. **Cancel** finishes the current
   detection and stops before further crops; files already saved are kept.
6. Review crops in the scrollable result grid. Click a crop to open its PNG.
   **Download Manifest** provides all detection boxes, crop boxes, source names,
   confidence scores, options and skipped-image reasons. **Refresh Results**
   reloads a saved result. The result job is retained with the workflow.
7. If downstream nodes need the crops, connect `images` and use normal **Queue/Run**
   after export finishes. It outputs the saved crops without exporting again.
   `images` is a ComfyUI image list: every item is a one-image RGB tensor with its
   own dimensions. `folder` is the absolute export folder; `info` summarizes the
   result. List processing can run a downstream node once for each crop.

Each Execute creates a new folder, for example:

```text
ComfyUI/output/DKST_FaceCrops/my_person/20261012-113000_<job-id>/
    face_00001_01.png
    face_00001_02.png
    face_00002_01.png
    manifest.json
```

The first number is the source image index; the second is the face index within
that source. In 4×4 mode cells are processed top to bottom, left to right.

## Options

| Setting | Default | Meaning |
| --- | --- | --- |
| output_folder | DKST_FaceCrops/my_person | Relative to ComfyUI's `output`; `/` separates folders. |
| filename_prefix | face | Prefix for exported PNGs. |
| face_selection | All faces | Export every detection or only the largest per image/cell. |
| crop_shape | BBox ratio | Padded box proportions, or a square. |
| padding_percent | 30 | Add 30% of face-box width on each side and 30% of height above/below. |
| limit_size | on | Downscale oversized crops only; never upscale. |
| max_size | 1024 | Longest edge limit, active when limit_size is on. |
| grid_split | Off | Optional 4×4 splitting with padding confined to each cell. |
| confidence | 0.8 | Lower to include uncertain detections; review false positives. |

Padded windows shift inward at image edges. Square crops are limited to the
shorter image/cell edge and can truncate a very tall/wide face at that boundary.
EXIF orientation is applied before detection. Alpha images are composited on
white. PNG exports contain no source EXIF or workflow metadata. Detection uses a
preview of at most 1024 pixels on the longest edge, but crops come from the
original-resolution image. Very small faces, extreme profiles and occlusions may
be missed; these are recorded as skipped. A 4×4 sheet with gutters or uneven panel
sizes may need adjustment before using equal-cell splitting.

Folder components and prefixes accept 1–64 letters, numbers, underscores or
hyphens; reserved Windows names and paths outside ComfyUI's output are rejected.
Gallery sources must be inside the server's ComfyUI input, output or temp folders.
Job records are stored in `input/dkst_face_crop/jobs`. Keep these records if saved
workflows should restore their result. After a server restart, unfinished jobs are
marked interrupted; start a new export to process the remaining photos. Review
partial results before sending them downstream.

## OpenCV dependency

Most image-processing installations already have OpenCV. If Execute reports
OpenCV missing, install [requirements-face-crop.txt](../requirements-face-crop.txt)
using **ComfyUI's actual Python** and restart ComfyUI. The separate Qwen/H3 training
Python is not used by this node.

For a Stability Matrix installation with its environment at the following path,
run this in PowerShell from the installed custom node folder (replace the Python
path if your installation uses a different environment):

```powershell
& "S:\StabilityMatrix\Packages\ComfyUI\venv\Scripts\python.exe" -m pip install -r .\requirements-face-crop.txt
```

For ComfyUI Portable, use its `python_embeded\python.exe` instead. Install in only
the environment that starts ComfyUI. An existing `opencv-contrib-python` build
with `FaceDetectorYN` also works and does not require installing a second OpenCV
package. The dependency is imported only when Execute is used, so its absence
does not prevent other DKST nodes from loading.

## Bundled model and license

The two tiny YuNet ONNX models are bundled; no model download occurs at runtime.
OpenCV 4 uses `2023mar`; OpenCV 5 uses the dynamic-shape `2026may` model. Model
hashes and the upstream revision are recorded in [UPSTREAM.json](models/UPSTREAM.json).
The model files are covered by their bundled [MIT license](models/LICENSE).
Upstream: [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
