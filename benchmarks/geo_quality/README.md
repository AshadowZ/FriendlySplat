# geo_quality (DTU)

This folder contains scripts for geometry-focused benchmarking workflows.

## DTU MoGe priors

Generate dense normal priors for each `DTU/scanXXX` scene by running `tools/depth_prior/moge_infer.py`:

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

This writes `DTU/scanXXX/invalid_mask/*.png` where white (255) indicates background (`alpha<0.5`) and black (0) indicates foreground (`alpha>=0.5`). Use it as `data.sky_mask_dir_name="invalid_mask"` to ignore background pixels (and encourage transparency) during training.

## DTU training (with priors + invalid_mask as sky_mask)

Train each scan with `moge_normal/` enabled, and use `invalid_mask/` as `sky_mask_dir_name`:

```bash
python3 benchmarks/geo_quality/run_train_dtu_batch.py \
  --data-root /path/to/data \
  --scans default
```

Defaults (can be overridden via CLI flags):

- `--data-preload cuda`
- `--postprocess.use-bilateral-grid` (requires `fused_bilagrid`)
- `--strategy-impl improved`
- `--densification-budget 1000000`
- `--prune-opa 0.05`
- `--prune-scale3d 0.1`
- `--flat-reg-weight 1.0`
- `--scale-ratio-reg-weight 1.0`

By default, the script enables invalid-mask sky masking.
Use `--no-use-invalid-mask` to disable sky masking.

This writes outputs under `<data-root>/geo_benchmark/DTU/scanXXX/<exp-name>/`.

## DTU mesh + geometry evaluation

Given trained splat PLYs, reconstruct a TSDF mesh and run the DTU geometry eval script:

```bash
python3 benchmarks/geo_quality/run_eval_dtu_batch.py \
  --data-root /path/to/data \
  --scans default \
  --dtu-official-dir /path/to/DTU_Official
```

If you have the DTU official eval assets bundled under `<data-root>/DTU/eval_dtu/` (i.e. it contains `ObsMask/` and `Points/stl/`),
you can omit `--dtu-official-dir` and the script will auto-detect it.

By default, the script uses FriendlySplat's internal DTU mesh culling implementation and runs `eval.py` directly.

The internal culling treats pixels with `alpha > 0.5` in `DTU/scanXXX/mask/*.png` as foreground.

By default, TSDF fusion applies the per-frame DTU object mask at `DTU/scanXXX/mask/` (PGSR-style: depth is set to 0 outside the mask).
Use `--no-tsdf-use-mask` to disable or `--tsdf-mask-dir-name` to change the folder name.

TSDF defaults are aligned with PGSR's DTU settings:

- `--voxel-length 0.002`
- `--sdf-trunc 0.008` (i.e. `4 * voxel_length`)
- `--depth-trunc 5.0`

Expected dataset layout:

```
<data-root>/
  DTU/
    scan24/
      images/
      mask/
      sparse/
    scan105/
      images/
      mask/
      sparse/
```

Outputs are written into each scan folder:

- `DTU/scanXXX/moge_normal/` (PNG)

Evaluation summary:

- A Markdown summary table is written to `<data-root>/geo_benchmark/summary_<exp-name>.md`.
- By default, per-scan TSDF cache files under `<result_dir>/mesh/cache/` are deleted after evaluation. Use `--no-delete-mesh-cache` to keep them.

Notes:

- The script is sequential (one scan at a time).
- Use `--dry-run` to print the commands first.
- Use `--no-skip-existing` to regenerate outputs.
