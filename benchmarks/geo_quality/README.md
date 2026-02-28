# geo_quality (DTU + TnT)

This folder contains batch scripts for geometry-focused benchmarking on:

- **TnT (Tanks & Temples)**: mesh + F1 eval
- **DTU**: mesh + official DTU geometry eval

## Common assumptions

### Data root layout

All scripts take a `--data-root` that contains dataset folders in fixed locations:

```
<data-root>/
  tnt_dataset/
    tnt/
      <Scene>/
        images/            # input images (full res)
        sparse/            # COLMAP sparse model
        <Scene>_COLMAP_SfM.log
        <Scene>_trans.txt
        transforms.json    # (optional)
        ...

  dtu_dataset/
    dtu/
      scanXX/
        images/            # input images (full res; often RGBA)
        sparse/            # COLMAP sparse model
        mask/              # DTU foreground masks (for DTU eval mesh culling)
        cameras.npz        # DTU camera pack used by DTU eval/culling
        ...
    dtu_eval/              # official DTU eval assets
      ObsMask/
      Points/
        stl/
```

Notes:

- `preprocess_*.py` modifies the dataset folders in-place (writes `images_2/`, priors, masks).
- DTU **evaluation** requires `scanXX/mask/` and `scanXX/cameras.npz` (mesh culling step before calling `eval.py`).

### Output directory layout

Training/eval outputs are written under `--data-root / --out-dir-name` (default paths shown):

- TnT default: `<data-root>/benchmark/geo_benchmark/tnt_benchmark/<Scene>/<exp-name>/...`
- DTU default: `<data-root>/benchmark/geo_benchmark/dtu_benchmark/scanXX/<exp-name>/...`

Both runners append `extra_args` to the end of the `friendly_splat/trainer.py` command line (after `--`), so any
trainer flag you pass there overrides the script defaults.

## Resolution policy (matches prior baseline)

This benchmark follows the same practice as the reference baselines:

- **Training is half-resolution only**: `--data.data-factor 2` (i.e. use `images_2/`)
- **TSDF meshing is half-resolution by default**: `--tsdf-render-factor 2` (renders depth at half-res before fusion)

This is intentionally aligned with the 2DGS benchmark setup.

## DTU

### Preprocess (images_2 + MoGe normals + optional invalid_mask)

```bash
python3 benchmarks/geo_quality/preprocess_dtu_batch.py \
  --data-root /path/to/data \
  --scans default
```

What it writes into each `dtu_dataset/dtu/scanXX/`:

- `images_2/` (half-res, RGB only; generated from `images/`)
- `moge_normal/` (half-res normal priors from MoGe)

If DTU images are RGBA and you want an alpha-derived invalid/background mask:

```bash
python3 benchmarks/geo_quality/preprocess_dtu_batch.py \
  --data-root /path/to/data \
  --scans default \
  --export-alpha-mask
```

This writes `invalid_mask/*.png` (half-res) where:

- `255` = invalid/background (`alpha < 0.5`)
- `0` = valid/foreground (`alpha >= 0.5`)

### Train

```bash
python3 benchmarks/geo_quality/run_train_dtu_batch.py \
  --data-root /path/to/data \
  --scans default
```

Defaults are hardcoded in `benchmarks/geo_quality/run_train_dtu_batch.py`.
To override ad-hoc, append trainer args after `--`, e.g.:

```bash
python3 benchmarks/geo_quality/run_train_dtu_batch.py \
  --data-root /path/to/data \
  --scans scan24 \
  -- --strategy.grow-grad2d 0.0004
```

Notes:

- Training uses `images_2/` (`--data.data-factor 2`) and MoGe normals (`--data.normal-dir-name moge_normal`).
- By default this DTU runner does **not** pass `invalid_mask/` into the trainer; if you want to use it, override with:
  `-- --data.sky-mask-dir-name invalid_mask`

### Mesh + geometry evaluation

```bash
python3 benchmarks/geo_quality/run_eval_dtu_batch.py \
  --data-root /path/to/data \
  --scans default
```

This does:

1) TSDF mesh reconstruction from the exported splat PLY via `tools/mesh/tsdf_mesh_from_ply.py`
2) mesh culling using DTU `mask/` + `cameras.npz`
3) official DTU eval via vendored `benchmarks/geo_quality/dtu_eval/eval.py` (uses `dtu_dataset/dtu_eval/`)

If your official DTU eval assets are not under `<data-root>/dtu_dataset/dtu_eval/`, pass `--dtu-official-dir` explicitly.

DTU TSDF defaults (borrowed from **PGSR**-style DTU TSDF fusion):

- `--voxel-length 0.002`
- `--sdf-trunc 0.008` (`4 * voxel_length`)
- `--depth-trunc 5.0`

DTU TSDF mask policy:

- By default (`--tsdf-use-mask`), TSDF fusion uses an **object mask derived from `invalid_mask/`** (inverted) and
  zeroes depth outside the mask.

Summary:

- A Markdown summary is written to `<data-root>/benchmark/geo_benchmark/dtu_benchmark/summary_<exp-name>.md`.

## TnT (Tanks & Temples)

### Preprocess (images_2 + MoGe priors)

```bash
python3 benchmarks/geo_quality/preprocess_tnt_batch.py \
  --data-root /path/to/data \
  --scenes default
```

Writes into each `tnt_dataset/tnt/<Scene>/`:

- `images_2/` (half-res)
- `moge_normal/` (normal priors)
- `moge_depth/` (depth priors)
- `invalid_mask/` (sky/invalid mask, 255=invalid, 0=valid)

### Train

```bash
python3 benchmarks/geo_quality/run_train_tnt_batch.py \
  --data-root /path/to/data \
  --scenes default
```

This runner follows the 2DGS baseline practice: half-res only (`--data.data-factor 2`) and enables MoGe priors by default.

### Mesh + F1 eval

```bash
python3 benchmarks/geo_quality/run_eval_tnt_batch.py \
  --data-root /path/to/data \
  --scenes default
```

TnT evaluation is aligned with the **2DGS** setup:

- TSDF meshing uses `tools/mesh/tsdf_mesh_from_ply.py` at half resolution by default (`--tsdf-render-factor 2`).
- TSDF hyperparameters use the 2DGS-style per-scene presets:
  - Large-scale scenes (Meetingroom/Courthouse/Church): `voxel_length=0.006`, `sdf_trunc=0.024`, `depth_trunc=4.5`
  - 360 scenes (Barn/Caterpillar/Ignatius/Truck): `voxel_length=0.004`, `sdf_trunc=0.016`, `depth_trunc=3.0`
- TSDF fusion uses an object mask derived from `invalid_mask/` (inverted) when available.

Summary:

- A Markdown summary is written to `<data-root>/benchmark/geo_benchmark/tnt_benchmark/summary_<exp-name>.md`.
