# Geometry priors (MoGe)

This folder contains scripts for generating dense depth/normal priors that can be consumed by FriendlySplat via `DataConfig.depth_dir_name` / `DataConfig.normal_dir_name` / `DataConfig.sky_mask_dir_name`.

## Install

`moge` is a pip-installable package.

Suggested environment:

- `pip install moge`
- `pip install opencv-python tqdm`
- Optional (better depth alignment): `pip install scikit-learn`

## Generate priors

Run from repo root:

```bash
python3 tools/geometry_prior/moge_infer.py --data-dir /path/to/scene \
  --save-depth \
  --save-sky-mask
```

Outputs are written under the scene root:

- `moge_normal/<image_stem>.png` (always generated)
- `moge_depth/<image_stem>.npy` (when `--save-depth`)
- `sky_mask/<image_stem>.png` (when `--save-sky-mask`, 255=invalid, 0=valid)

`--align-depth-with-colmap` is enabled by default and will align each depth map to COLMAP sparse depths when a COLMAP model is found under `sparse/0`, `sparse`, or `colmap/sparse/0`.

If you only need normals, run with `--no-align-depth-with-colmap` and without `--save-depth` to skip depth processing.

RGBA images: if the input image has an alpha channel, pixels with `alpha==0` are treated as non-existent; exported depth is forced to `0.0` there and exported normal is forced to `127,127,127`.

## Use in training

Set these in your training config:

- `data.normal_dir_name="moge_normal"`
- `data.depth_dir_name="moge_depth"` (if you exported depth)
- `data.sky_mask_dir_name="sky_mask"` (if you exported masks)

Note: When training with auxiliary priors/masks, the priors/masks must match the training image resolution.
For example, if you train with `data_factor=2` (using `images_2/`), generate priors/masks at `images_2/` resolution
or the dataset loader will raise a shape mismatch error.
