# SfM preprocessing (HLOC + pycolmap)

This folder contains a standalone Structure-from-Motion (SfM) preprocessing tool
based on the **Hierarchical-Localization (HLOC)** toolbox and **pycolmap**.

It is designed to fit FriendlySplat's tooling style:

- it **does not** modify your original dataset folder;
- it writes a self-contained COLMAP scene under an explicit `--output-dir`;
- the result is directly compatible with FriendlySplat's COLMAP dataparser
  (expects `images/` + `sparse/0/`).

## Output layout

Given `--output-dir /path/to/out_scene`, the script writes:

- `/path/to/out_scene/images/` (3DGS-ready images)
- `/path/to/out_scene/sparse/0/` (COLMAP model)
- `/path/to/out_scene/_sfm_work/` (intermediate HLOC artifacts + raw recon)

## Install notes

This tool requires external dependencies:

- `hloc` toolbox (from the Hierarchical-Localization repo)
- `pycolmap`
- optional: `pixsfm` (only when `--refine_pixsfm` is enabled)

Python package dependencies used by the wrapper script are listed in:

- `tools/requirements.txt`

## Usage

Basic (sequential matching):

```bash
python3 tools/sfm/run_hloc_sfm.py \
  --input-image-dir /path/to/images \
  --output-dir /path/to/out_scene \
  --camera-model PINHOLE \
  --matching-method sequential
```

Retrieval-based pairing:

```bash
python3 tools/sfm/run_hloc_sfm.py \
  --input-image-dir /path/to/images \
  --output-dir /path/to/out_scene \
  --camera-model PINHOLE \
  --matching-method retrieval \
  --retrieval-type netvlad
```

Panorama mode (split to a 5-view rig before SfM):

```bash
python3 tools/sfm/run_hloc_sfm.py \
  --input-image-dir /path/to/panoramas \
  --output-dir /path/to/out_scene \
  --camera-model PINHOLE \
  --is-panorama \
  --pano-downscale 2
```

Notes:

- Use `--overwrite` to delete existing `images/` and `sparse/` under `--output-dir`.
- Use `--keep-work-dir` to keep intermediate files under `_sfm_work/` for debugging.
