# SfM Preprocessing (HLOC + pycolmap)

A fast, lightweight wrapper to build a FriendlySplat-ready COLMAP scene from an
input image folder using [HLOC](https://github.com/cvg/Hierarchical-Localization).

> Note: This tool works best for temporally ordered images such as video frames.
> For highly unordered, sparse, or geometrically difficult scenes, consider more
> robust tools like Metashape or RealityScan.

## Installation

Install the FriendlySplat `sfm` dependencies and the HLOC submodule:

```bash
# 1. Install FriendlySplat sfm extra
pip install -e ".[sfm]" --no-build-isolation

# 2. Install HLOC (clone recursively to preserve third_party submodules)
git clone --recursive https://github.com/AshadowZ/Hierarchical-Localization.git
pip install -e ./Hierarchical-Localization

# Optional: install pixsfm for feature refinement
# pip install pixsfm
```

## Data Preparation

Training on extremely high-resolution images rarely yields noticeable
improvements and usually just slows the pipeline down. A good default is to keep
the longest edge around 1K-2K (for example, 1600px).

Scenario A: from a video. Extract frames at about 1 FPS and downscale them with
`ffmpeg`:

```bash
mkdir -p input_frames
ffmpeg -i input.mp4 -q:v 2 -vf "fps=1,scale='min(1600,iw)':'min(1600,ih)':force_original_aspect_ratio=decrease" input_frames/frame_%05d.jpg
```

Scenario B: from existing high-resolution images. Batch resize them with
ImageMagick:

```bash
mkdir -p input_frames_1600
# The '>' keeps smaller images unchanged and only downsizes larger ones.
magick mogrify -path input_frames_1600 -resize '1600x1600>' input_frames/*.jpg
```

## Quick Start

The tool reads images, runs feature matching, and exports a self-contained COLMAP
scene without modifying your original data.

Recommended for video-like captures:

```bash
fs-sfm \
  --input-image-dir /path/to/input-frames \
  --output-dir /path/to/data-dir \
  --camera-model OPENCV \
  --matching-method sequential \
  --feature-type superpoint_aachen \
  --matcher-type superpoint+lightglue \
  --retrieval-type megaloc \
  --use-single-camera-mode True
```

We recommend `OPENCV` as it automatically undistorts images for FriendlySplat.
Use `PINHOLE` only if your images are already perfectly undistorted.

Use `--use-single-camera-mode True` when all images come from the same physical
camera. If the images are not from a single camera, turn it off.

## Alternative Workflows

Add or modify flags based on your dataset characteristics:

- Unordered images: `--matching-method retrieval --retrieval-type megaloc`
- Small datasets: `--matching-method exhaustive`
- Panoramas: `--is-panorama --pano-downscale 2`
- High quality refinement: `--refine-pixsfm`

## Output and Training

The exported output directory is designed to work directly with FriendlySplat.
Intermediate files under `_sfm_work/` are deleted automatically unless you pass
`--keep-work-dir`.

```text
out_scene/
├── images/      # standardized and undistorted images
└── sparse/0/    # final COLMAP model
```

You can pass this directory directly to training:

```bash
fs-train --io.data-dir /path/to/out_scene ...
```

## Tips and Troubleshooting

- Camera model: use `--use-single-camera-mode True` by default unless you need
  per-image camera handling. Supported models include `PINHOLE`,
  `SIMPLE_PINHOLE`, `RADIAL`, `OPENCV`, and others.
- Missing dependencies:
  `hloc`: ensure you cloned HLOC with `--recursive`.
  `pycolmap`, `huggingface_hub`, `safetensors`: rerun
  `pip install -e ".[sfm]" --no-build-isolation`.
- Existing output directory: use `--overwrite` if `images/` or `sparse/` already
  exist in the destination directory.
