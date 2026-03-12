# 3D Instance Segmentation (InstaScene Clustering)

This tool performs offline 3D instance clustering for trained FriendlySplat
scenes. It projects 2D SAM-style instance masks across multiple views onto the
3D Gaussians to build consistent 3D instance labels and point clouds.

At a high level it:

1. loads Gaussian splats from a checkpoint or PLY export;
2. loads COLMAP cameras from the scene;
3. loads per-image instance masks from `<data_dir>/sam/...`;
4. projects mask evidence onto Gaussians with `gsplat` rasterization;
5. clusters multi-view mask observations into 3D instances;
6. exports instance-colored point clouds and per-Gaussian labels.

## Installation

```bash
# Basic installation
pip install -e ".[segment]" --no-build-isolation

# Optional: GPU DBSCAN for faster post-processing on large scenes
pip install cupy-cuda12x
pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
```

Note: if `cupy` / `cuml` are missing, the tool automatically falls back to CPU
DBSCAN. When installing them, make sure the package build matches your local
CUDA version.

## Quick Start

The easiest workflow is to point the tool at a FriendlySplat result directory.
It will load the latest checkpoint by default, infer the dataset path when
possible, and write outputs under `cluster_result/`.

```bash
fs-segment \
  --result-dir /path/to/results \
  --split train
```

Alternative workflows:

- Specific step: `--result-dir /path/to/results --step 30000`
- Explicit checkpoint: `fs-segment --ckpt-paths /path/to/ckpt_step030000.pt --data-dir /path/to/scene`
- Explicit PLY: `fs-segment --ply-paths /path/to/splats_step030000.ply --data-dir /path/to/scene`
- Faster post-processing on large scenes: add `--use-gpu-dbscan`

## Data Preparation (Masks)

The tool expects per-image 2D instance label maps, not binary masks. Pixels
with the same positive integer inside one frame belong to the same 2D instance.
`0` is background.

Place `.png`, `.jpg`, `.jpeg`, or `.npy` masks matching your image stems in one
of these folders:

- `<data_dir>/sam/mask_sorted`
- `<data_dir>/sam/mask`
- `<data_dir>/sam/mask_filtered`

Override this with `--mask-dir-name` if needed.
Mask resolution must exactly match the corresponding image resolution. If a mask
image has multiple channels, only the first channel is used. For `.npy`, store
the integer label map directly.
Labels are interpreted per frame rather than globally across the whole scene.

This repository does not bundle mask inference itself. If you need a practical
pipeline for generating these per-image label maps, the modified EntitySeg setup
from InstaScene is a reasonable reference:

- https://github.com/zju3dv/instascene

In practice, EntitySeg remains a stable and useful choice for panorama-style
full-image segmentation before running this clustering stage.

## Key Parameters and Quality Tuning

- `--data-factor` (default: `1.0`): must exactly match the resolution of your
  masks. For example, if masks were generated at half resolution, use
  `--data-factor 2`. Mismatched resolutions break Gaussian-to-pixel
  correspondences.
- `--point-filter-threshold` (default: `0.5`): multi-view consistency
  threshold. Higher values are stricter and remove more noise; lower values keep
  more points but may introduce contamination.
- `--dbscan-eps` (default: `0.1`): spatial distance threshold for 3D
  DBSCAN clustering. Increase it if one object is split into many pieces;
  decrease it if nearby objects are being fused together.
- `--dbscan-min-points` (default: `4`): minimum number of 3D points
  required to form a valid cluster.
- `--split` (default: `train`): controls which camera set is used for mask
  projection. `all` usually gives better 3D coverage but costs more and may
  include less reliable views.
- `--use-gpu-dbscan`: use cuML DBSCAN when available. This is often worthwhile
  on large scenes.

There are also a few important thresholds currently kept internal rather than
exposed as CLI flags:

- `DEFAULT_PIXEL_GAUSSIAN_THRESHOLD = 0.25`
- `DEFAULT_OVERLAP_RATIO = 0.8`
- `DEFAULT_UNDERSEGMENT_THRESHOLD = 0.8`

If clustering quality consistently fails in edge cases, these are worth
inspecting in [instascene_gauscluster.py](instascene_gauscluster.py).

## Outputs

Outputs are written to:

```text
<result_dir>/cluster_result/
```

Main files:

- `color_cluster.ply`: point cloud colored by instance id
- `instance_labels.npy`: per-Gaussian integer instance labels
- `color_cluster_knn.ply`: KNN-filled colored point cloud
- `instance_labels_knn.npy`: KNN-filled labels for previously unlabeled points
