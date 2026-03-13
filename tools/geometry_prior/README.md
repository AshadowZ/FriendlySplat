# Geometry Priors (MoGe)

Generate dense depth and normal priors using
[MoGe](https://github.com/Ruicheng/moge) to improve FriendlySplat's geometry
reconstruction.

## Installation

```bash
pip install -e ".[priors]" --no-build-isolation
```

## Quick Start

Run the inference script from the repository root. This is the recommended
standard command:

```bash
python tools/geometry_prior/moge_infer.py \
  --data-dir /path/to/scene \
  --factor 1 \
  --model-id Ruicheng/moge-2-vitl-normal \
  --out-normal-dir moge_normal \
  --out-depth-dir moge_depth \
  --align-depth-with-colmap \
  --save-depth \
  --verbose
```

## Key Parameters Explained

- `--align-depth-with-colmap`: enabled by default and crucial for usable depth.
  MoGe predicts relative depth. This flag fits it to your COLMAP sparse scene
  under `sparse/0/`, turning it into scene-consistent depth.
- `--save-depth`: by default the script always exports normals, but depth `.npy`
  files are only written when you pass this flag.
- `--save-sky-mask`: optionally export invalid-depth / sky-like masks under
  `sky_mask/`, which can be consumed by FriendlySplat via
  `data.sky_mask_dir_name`.
- `--factor`: controls which image folder is used, such as `images_2/` or
  `images_2p5/`. It must match your training `--data.data-factor`, otherwise
  FriendlySplat will hit image-prior shape mismatches.
- `--model-id`: HuggingFace model weights. `Ruicheng/moge-2-vitl-normal` is the
  default and recommended high-quality model.
- `--remove-depth-edge`: enabled by default. It masks sharp depth
  discontinuities to reduce floating artifacts during splat training.
- `--verbose`: prints per-image alignment details and is useful when depth
  fitting behaves unexpectedly.

## Use in Training

Once generated, pass the output directory names to training:

```bash
fs-train --io.data-dir /path/to/scene \
  --data.data-factor 1 \
  --data.normal-dir-name moge_normal \
  --data.depth-dir-name moge_depth \
  --data.sky-mask-dir-name sky_mask
```

Tip: if you only need normals, skip `--save-depth` and add
`--no-align-depth-with-colmap` to avoid the heavier COLMAP depth fitting step.
