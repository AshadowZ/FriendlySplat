# SfM preprocessing (HLOC + pycolmap)

This tool builds a COLMAP scene from an input image folder using the
Hierarchical-Localization (HLOC) toolbox and `pycolmap`.

Its goal is to produce a FriendlySplat-ready scene without modifying your
original dataset directory.

At a high level it:

1. reads images from `--input-image-dir`;
2. optionally splits panoramas into a fixed 5-view rig;
3. runs HLOC feature extraction, pairing, matching, and reconstruction;
4. exports a self-contained COLMAP scene under `--output-dir`;
5. writes the result in a layout directly consumable by FriendlySplat.

This is intentionally a lightweight, toy-style SfM wrapper built on top of HLOC.

Practical positioning:

- it is fast
- it is convenient
- it can run from Python packages alone
- but it is not the most robust SfM pipeline for difficult scenes

In practice, it works best on temporally ordered image sets such as frames
sampled from a video. If your image collection is more unordered, sparse, or
geometrically difficult, this pipeline can fail or drift badly.

If your scene is challenging and you care about reconstruction reliability more
than setup simplicity, it is usually better to obtain camera poses from a more
robust SfM tool such as Metashape or RealityScan, then feed the resulting scene
into FriendlySplat.

## Output layout

Given:

```bash
--output-dir /path/to/out_scene
```

the script writes:

- `/path/to/out_scene/images/`
- `/path/to/out_scene/sparse/0/`
- `/path/to/out_scene/_sfm_work/`

Meaning:

- `images/`: exported 3DGS-ready image set
- `sparse/0/`: final COLMAP model
- `_sfm_work/`: intermediate HLOC files and raw sparse reconstruction

By default `_sfm_work/` is deleted at the end. Use `--keep-work-dir` if you
want to keep it for debugging.

## Install

Install FriendlySplat with the sfm extra:

```bash
pip install -e ".[sfm]" --no-build-isolation
```

This installs:

- `pycolmap==3.13.0`
- Python-side helper dependencies used by the SfM wrapper

`hloc` itself must be installed separately from a local clone so that its
repository layout and `third_party/` submodules are preserved.

Recommended HLOC setup:

```bash
git clone --recursive https://github.com/AshadowZ/Hierarchical-Localization.git
pip install -e /path/to/Hierarchical-Localization
```

If you already cloned the repository without submodules, run:

```bash
cd /path/to/Hierarchical-Localization
git submodule update --init --recursive
pip install -e .
```

Optional dependency:

- `pixsfm` (only needed when `--refine-pixsfm` is enabled)

Optional install:

```bash
pip install pixsfm
```

## Inputs

Required inputs:

- `--input-image-dir`: directory containing the original images
- `--output-dir`: destination directory for the exported COLMAP scene
- `--camera-model`: camera model used by COLMAP / pycolmap

Accepted image extensions:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.tiff`
- `.tif`

The script copies images into the output scene and standardizes filenames, so
it does not modify your original image folder.

## Basic usage

Recommended package entrypoint:

```bash
fs-sfm \
  --input-image-dir input_image_dir \
  --output-dir data_dir \
  --camera-model PINHOLE \
  --matching-method sequential \
  --feature-type superpoint_inloc \
  --retrieval-type megaloc \
  --matcher-type superpoint+lightglue \
  --use-single-camera-mode True
```

Legacy script path also works:

```bash
python3 tools/sfm/run_hloc_sfm.py \
  --input-image-dir /path/to/images \
  --output-dir /path/to/out_scene \
  --camera-model PINHOLE \
  --matching-method sequential
```

## Common workflows

### 1. Standard sequential SfM

```bash
fs-sfm \
  --input-image-dir /path/to/images \
  --output-dir /path/to/out_scene \
  --camera-model PINHOLE \
  --matching-method sequential
```

This is the default recommended mode for video-like captures.

Notes:

- internally, sequential mode also uses retrieval to improve loop closure
- this is usually the best first choice for temporally ordered image sets

### 2. Retrieval-based pairing

```bash
fs-sfm \
  --input-image-dir /path/to/images \
  --output-dir /path/to/out_scene \
  --camera-model PINHOLE \
  --matching-method retrieval \
  --retrieval-type netvlad \
  --num-matched 50
```

This is useful when image order is weak or when the dataset is more unordered.

### 3. Exhaustive pairing

```bash
fs-sfm \
  --input-image-dir /path/to/images \
  --output-dir /path/to/out_scene \
  --camera-model PINHOLE \
  --matching-method exhaustive
```

This can work well on small image sets, but may become expensive as the dataset grows.

### 4. Panorama mode

```bash
fs-sfm \
  --input-image-dir /path/to/panoramas \
  --output-dir /path/to/out_scene \
  --camera-model PINHOLE \
  --is-panorama \
  --pano-downscale 2
```

Panorama mode:

- splits each panorama into a fixed 5-view rig
- writes split images into `_sfm_work/images/pano_camera*/`
- builds a rig-aware reconstruction

Current expectation:

- panorama mode works with `PINHOLE` or `SIMPLE_PINHOLE`
- non-supported models are forced back to `PINHOLE`

### 5. Enable PixSfM refinement

```bash
fs-sfm \
  --input-image-dir /path/to/images \
  --output-dir /path/to/out_scene \
  --camera-model PINHOLE \
  --matching-method sequential \
  --refine-pixsfm
```

This requires:

```bash
pip install pixsfm
```

## Important arguments

Core I/O:

- `--input-image-dir`: source image directory
- `--output-dir`: destination COLMAP scene root
- `--overwrite`: delete existing exported content under `--output-dir`
- `--keep-work-dir`: keep `_sfm_work/` for debugging

Camera setup:

- `--camera-model`: one of `PINHOLE`, `SIMPLE_PINHOLE`, `RADIAL`, `SIMPLE_RADIAL`, `OPENCV`, `FULL_OPENCV`
- `--use-single-camera-mode`: use a single shared camera model for all frames

Matching / retrieval:

- `--matching-method {exhaustive,sequential,retrieval}`
- `--feature-type`: HLOC local feature extractor config
- `--matcher-type`: HLOC matcher config
- `--retrieval-type`: global retrieval descriptor for retrieval-based pairing
- `--num-matched`: number of retrieval candidates when `--matching-method retrieval`

Optional refinement / acceleration:

- `--refine-pixsfm`: enable PixSfM refinement

Panorama mode:

- `--is-panorama`: treat inputs as panoramas
- `--pano-downscale`: downscale factor applied before panorama splitting

## Practical tips

- For ordinary video or phone captures, start with `--matching-method sequential`.
- For unordered image folders, try `--matching-method retrieval`.
- For smaller datasets, `exhaustive` can be a useful sanity check.
- Keep `--use-single-camera-mode True` unless you know you need per-image camera handling.
- If reconstruction quality is poor, first revisit image quality, overlap, and camera model choice before changing advanced HLOC settings.
- Panorama mode is useful when your source data is equirectangular and you want a conventional SfM-friendly rig decomposition.

## Troubleshooting

- `missing dependency hloc`:
  install HLOC from a local clone with submodules, then run `pip install -e /path/to/Hierarchical-Localization`
- `missing dependency pycolmap`:
  reinstall the sfm extra with `pip install -e ".[sfm]" --no-build-isolation`
- `refine_pixsfm=True requires pixsfm`:
  install `pixsfm` separately
- `No images found`:
  check the input folder and supported image extensions
- existing `images/` or `sparse/` under `--output-dir`:
  rerun with `--overwrite`
- panorama mode behaving unexpectedly:
  use `PINHOLE` or `SIMPLE_PINHOLE`

## Output compatibility

The exported output directory is designed to work directly with FriendlySplat:

```text
<output-dir>/
  images/
  sparse/0/
```

That means you can pass it directly to training:

```bash
fs-train --io.data-dir /path/to/out_scene ...
```
