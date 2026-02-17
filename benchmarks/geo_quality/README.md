# geo_quality (DTU)

This folder contains scripts for geometry-focused benchmarking workflows.

## DTU MoGe priors

Generate dense priors for each `DTU/scanXXX` scene by running `tools/depth_prior/moge_infer.py`:

```bash
python3 benchmarks/geo_quality/run_moge_priors_dtu_batch.py \
  --data-root /path/to/data \
  --scans default
```

If DTU images are RGBA and you want to mask out the alpha background during training, enable alpha-mask export:

```bash
python3 benchmarks/geo_quality/run_moge_priors_dtu_batch.py \
  --data-root /path/to/data \
  --scans default \
  --export-alpha-mask
```

This writes `DTU/scanXXX/invalid_mask/*.png` where white (255) indicates background (`alpha==0`) and black (0) indicates foreground (`alpha>0`). Use it as `data.sky_mask_dir_name="invalid_mask"` to ignore background pixels (and encourage transparency) during training.

## DTU training (with priors + invalid_mask as sky_mask)

Train each scan with `moge_depth/` + `moge_normal/` enabled, and use `invalid_mask/` as `sky_mask_dir_name`:

```bash
python3 benchmarks/geo_quality/run_train_dtu_batch.py \
  --data-root /path/to/data \
  --scans default
```

This writes outputs under `<data-root>/geo_benchmark/DTU/scanXXX/<exp-name>/`.

## DTU mesh + geometry evaluation

Given trained splat PLYs, reconstruct a TSDF mesh and run the DTU geometry eval script:

```bash
python3 benchmarks/geo_quality/run_eval_dtu_batch.py \
  --data-root /path/to/data \
  --scans default \
  --dtu-official-dir /path/to/DTU_Official
```

By default, the script uses the eval code at `DTU/eval_dtu/evaluate_single_scene.py` under `--data-root`.

Expected dataset layout:

```
<data-root>/
  DTU/
    scan24/
      images/
      sparse/
    scan105/
      images/
      sparse/
```

Outputs are written into each scan folder:

- `DTU/scanXXX/moge_normal/` (PNG)
- `DTU/scanXXX/moge_depth/` (NPY)

Notes:

- The script is sequential (one scan at a time).
- Use `--dry-run` to print the commands first.
- Use `--no-skip-existing` to regenerate outputs.
