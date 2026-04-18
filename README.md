# DSTMap Generator

Toolkit for generating **Difference-based Spatio-Temporal Maps (DSTMaps)** from raw facial videos, aimed at **remote photoplethysmography (rPPG)** and heart-rate estimation.

The pipeline first builds a **frame-difference video** to isolate temporal signal components, then aligns the lower facial region using 2D landmarks, and finally aggregates per-ROI color statistics into compact **RGB / YUV STMap images** suitable for 2D-CNN or Vision-Transformer input.

---

## What is a DSTMap?

A conventional STMap stacks per-ROI pixel means across time — each column is a frame, each row is a facial sub-region. A **DSTMap** applies the same construction on a **difference video** (normalized inter-frame differences) instead of raw pixel values.

Because the photoplethysmographic signal lives in the *temporal derivative* of skin color, differentiating before spatially aggregating:

- suppresses identity-level appearance (skin tone, lighting bias, pose),
- emphasizes pulse-band variations at the frame scale,
- lets the downstream model skip learning the derivative operator itself.

Both **RGB** and **YUV (BT.601)** STMaps are generated so that models can exploit complementary information across color spaces.

---

## Repository structure

```
├── Diffvidmaker/           # Raw video  ->  per-channel diff video (.avi, MJPG)
├── STmap_Generator/        # Face alignment  +  RGB / YUV STMap construction
│                           #   outputs STmap_RGB.png / STmap_YUV.png
├── DATALOADER/             # Dataset parsers for UBFC-rPPG and PURE
└── DataTransmit_UBFC_PURE.py  # Utility: pairs videos with their ground-truth
                               #   physiological .txt files
```

---

## Pipeline

```
       raw video                   diff video                 aligned diff frames              DSTMap (.png)
   ┌──────────────┐            ┌──────────────┐            ┌──────────────┐            ┌──────────────┐
   │ facial video │    ───►    │  Diffvidmaker│    ───►    │  face align  │    ───►    │  STMap build │
   └──────────────┘            └──────────────┘            └──────────────┘            └──────────────┘
         │                    B/G/R channel-wise            landmarks from raw          4×8 = 32 ROIs
         │                    Δ = f_t − f_{t−1}             applied to diff             on lower face
         │                                                  (jaw + chin anchors)        YUV → min-max
         └──────────────────────── landmarks ──────────────────►                         → PNG image
```

### 1 · Diff-video generation (`Diffvidmaker/`)

For each pair of consecutive frames the pipeline computes the **per-BGR-channel difference** in `int16` precision, then writes the result as an **MJPG-encoded `.avi`**. This emphasises the temporal variations tied to the blood-volume pulse while keeping a file format that is cheap to decode downstream.

### 2 · Face alignment (`STmap_Generator/`)

- 2D facial landmarks (68 points) are detected on the **raw** video using [`face_alignment`](https://github.com/1adrianb/face-alignment). Landmark detection on diff frames would be unreliable, so the raw stream is used as the landmark source and the same trajectory is then applied to the diff stream.
- Missing frames are recovered by **cubic B-spline interpolation** (`scipy.interpolate.splrep` / `splev`) over each of the 136 coordinate channels.
- Each diff frame is then **affine-warped** using three landmark anchors:

  | Source landmark | Index | Destination (128 × 128) |
  |---|---|---|
  | Left jaw corner | `lmk[1]` | `(0, 48)` |
  | Right jaw corner | `lmk[15]` | `(128, 48)` |
  | Chin tip | `lmk[8]` | `(64, 128)` |

This warp forces the jaw line and chin tip to identical pixel positions across every frame and every subject, so the resulting STMap has a **consistent anatomical layout along its vertical axis**.

### 3 · STMap construction (`STmap_Generator/`)

- Only the **lower half of the aligned face** is used (`y ≥ 64`), concentrating the STMap on the densely perfused chin / jaw / mouth region.
- The cropped region is divided into a **4 × 8 grid** (width × height) = **32 ROIs**. Each ROI is the mean pixel value over its block.
- Every frame contributes one 32-element vector, producing a matrix of shape `(T, 32, 3)`. Each ROI is then **min-max normalized over the time axis** and rescaled to `[0, 255]`, so every row of the final image has full dynamic range regardless of its baseline intensity.
- After transposing to `(32, T, 3)` and casting to `uint8`, the result is saved as a **PNG image** in both RGB and YUV form:

```
STmap_RGB.png     # (32, T, 3) uint8 — rows = 32 ROIs, cols = frames
STmap_YUV.png     # (32, T, 3) uint8 — YUV (BT.601), channels independently
```

---

## Color-space conversion (BT.601)

The pipeline uses the **ITU-R BT.601** matrix for RGB ↔ YUV conversion:

```
Y =  0.299·R + 0.587·G + 0.114·B
U = −0.168736·R − 0.331264·G + 0.5·B        (+128 offset)
V =  0.5·R − 0.418688·G − 0.081312·B        (+128 offset)
```

U and V are shifted by +128 to keep values in `[0, 255]`.

---

## Supported datasets

- [**UBFC-rPPG**](https://sites.google.com/view/ybenezeth/ubfcrppg)
- [**PURE**](https://www.tu-ilmenau.de/neurob/data-sets-code/pulse/)

Dataset-specific parsers live in `DATALOADER/`. Ground-truth physiological traces (`.txt`) are paired with their source videos via the helper script `DataTransmit_UBFC_PURE.py`.

---

## Prerequisites

```bash
pip install torch torchvision opencv-python face-alignment scipy tqdm matplotlib numpy
```

- Python 3.8+
- CUDA-capable GPU strongly recommended for face alignment
- Tested with PyTorch ≥ 1.13

---

## Getting started

### 1 · Generate a diff video

```python
from Diffvidmaker.diff import compute_frame_difference, save_frame_differences
import cv2

raw = 'path/to/raw.avi'
out = 'path/to/raw_DIFF.avi'

cap = cv2.VideoCapture(raw)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(3)), int(cap.get(4)))
cap.release()

diffs = compute_frame_difference(raw)
save_frame_differences(diffs, out, fps, size)
```

### 2 · Build the DSTMap

```python
# STmap_Generator/main.py  (edit the paths at the bottom, then run)
python STmap_Generator/main.py
```

Produces `STmap_RGB.png` and `STmap_YUV.png` in the configured output directory.

### 3 · Load into your model

Use the parsers under `DATALOADER/` to pair DSTMap images with the corresponding ground-truth pulse waveforms for training or evaluation.

---

## Implementation notes

- **Raw → diff alignment transfer.** Landmark detection is run on the raw video, *not* on the diff video — detection on diff frames would fail because they carry almost no appearance cue. The detected landmark trajectory is then reused to align the diff stream.
- **Lower-face crop.** Keeping only `y ≥ 64` of the aligned face focuses the STMap on tissue with dense superficial vasculature (chin, perioral area) and avoids including the eyes, which contribute motion artifacts rather than pulse signal.
- **Per-ROI temporal normalization.** Each ROI is min-max scaled over time independently. This equalizes rows that sit on very different baselines (e.g. shaded vs. well-lit regions) before they land in the same image.
- **Adding a new dataset.** Implement a parser under `DATALOADER/` that yields `(stmap_path, gt_signal_path)` pairs — the rest of the pipeline is dataset-agnostic.
