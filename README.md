# DSTMap Generator

This repository contains tools and scripts for generating Spatial-Temporal Maps (STMaps) from raw facial videos, primarily for remote photoplethysmography (rPPG) applications such as physiological signal (heart rate) estimation. 

The pipeline extracts facial regions, computes frame differences, and generates rich RGB and YUV STMaps suitable for deep learning models.

## Repository Structure

* **`STmap_Generator/`**: The core component that handles face alignment (using `face_alignment`) and creating the STMaps. It outputs both RGB (`_stmap_rgb.png`) and YUV (`_stmap_yuv.png`) STMaps by processing the aligned facial frames and their temporal differences.
* **`Diffvidmaker/`**: Contains scripts to process raw videos frame-by-frame, computing the normalized difference between consecutive frames. This generates "diff videos" that emphasize subtle temporal changes like blood volume pulses.
* **`DATALOADER/`**: Scripts and utilities for parsing and loading popular rPPG datasets like **UBFC-rPPG** and **PURE**.
* **Utility Scripts** (e.g., `DataTransmit_UBFC_PURE.py`): Helper scripts for organizing and copying ground truth data files (e.g., `.txt` files containing physiological signals) corresponding to the dataset videos.

## Pipeline Overview

1. **Diff Video Generation (`Diffvidmaker`)**: Raw facial videos are processed to compute spatial-temporal differences between consecutive frames, saved as `.avi` uncompressed diff videos.
2. **Face Alignment (`STmap_Generator`)**: 2D facial landmarks are extracted to crop and stably align the facial Regions of Interest (ROI) over time.
3. **STMap Extraction (`STmap_Generator`)**: The aligned facial frames from both the raw and diff videos are concatenated or reshaped into STMaps. These condense the temporal progression of skin color changes into a single 2D image for 2D-CNN or Vision Transformer based models.

## Prerequisites

* Python 3.x
* `torch`, `torchvision` (CUDA recommended for faster face alignment)
* `opencv-python` (`cv2`)
* `face_alignment`
* `tqdm`
