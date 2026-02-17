# Visual Quality Benchmarks

This folder contains scripts/utilities for benchmarking FriendlySplat visual quality
across common 3DGS datasets (Mip-NeRF360 / Tanks&Temples-Vis / DeepBlending).

## Batch training

Single-GPU, sequential batch training (Improved-GS-style per-scene budgets):

```bash
python3 benchmarks/strategies_visual_quality/run_train_batch.py \
  --data-root /media/joker/HV/3DGS/PublicDataset \
  --datasets mipnerf360,deepblending \
  --scenes bicycle,bonsai,drjohnson \
  --budget-profile normal \
  --device cuda:0 \
  --max-steps 30000
```

By default, outputs are written under:

- `<data-root>/strategy_benchmark/<dataset>/<scene>/<strategy>/`

For example:

- `<data-root>/strategy_benchmark/mipnerf360/bicycle/improved/`
- `<data-root>/strategy_benchmark/tanksandtemples_vis/truck/mcmc/`
- `<data-root>/strategy_benchmark/deepblending/drjohnson/default/`

This benchmark runner is designed to benchmark different densification strategies.
It always exports the final PLY and enables TensorBoard logging.

By default, it runs all three strategies (`--strategy-impl all`): improved, default, and mcmc.

For `--strategy-impl improved`, it also applies an Improved-GS-style alignment:

- Enables `data.benchmark_train_split=True` so training uses a disjoint split from evaluation (Improved-GS-style)
- Sets key strategy switches and optimizer hyperparameters to Improved-GS defaults
- Enables the MU-style splat optimizer step schedule by default (`optim.mu_enable=True`) (disable with `--no-mu`)

For `--strategy-impl default`, it applies a gsplat DefaultStrategy-style baseline:

- Disables `strategy.absgrad` and enables `strategy.verbose`
- Sets `strategy.refine_scale2d_stop_iter=0` and `strategy.prune_scale3d=0.1`
- Aligns the main optimizer LRs (means/scales/quats/opacities/SH) and sets means `lr_final=0.01*lr_init`

For `--strategy-impl mcmc`, it aligns to gsplat's MCMC preset:

- Sets `init.init_opacity=0.5` and `init.init_scale=0.1`
- Enables MCMC regularizers (`reg.opacity_reg_weight=0.01`, `reg.scale_l1_reg_weight=0.01`)
- Sets `strategy.refine_stop_iter=25_000` and aligns key MCMC knobs (noise/min_opacity)
- Aligns the main optimizer LRs (means/scales/quats/opacities/SH) and sets means `lr_final=0.01*lr_init`

When `--budget-profile` is enabled, the per-scene budgets are applied as:

- ImprovedStrategy: `strategy.densification_budget`
- MCMCStrategy: `strategy.mcmc_cap_max`

By default it also uses `preload='cuda'` for faster throughput (at the cost of GPU memory).
To disable preloading:

```bash
python3 benchmarks/strategies_visual_quality/run_train_batch.py \
  --data-root /media/joker/HV/3DGS/PublicDataset \
  --datasets mipnerf360 \
  --scenes bicycle \
  --data-preload none
```

If a scene's `images/` are downsampled compared to the COLMAP intrinsics, set `data_factor` to match
your `images_<factor>/` folder (e.g. `images_4/`):

```bash
python3 benchmarks/strategies_visual_quality/run_train_batch.py \
  --data-root /media/joker/HV/3DGS/PublicDataset \
  --datasets mipnerf360 \
  --scenes garden \
  --scene-data-factor garden=4
```

If you have removed `images/` and only keep `images_<factor>/` on disk, the script will also auto-infer
`data_factor` from existing image folders unless you override it explicitly.

You can also specify per-scene factors with a repeatable flag:

```bash
python3 benchmarks/strategies_visual_quality/run_train_batch.py \
  --data-root /media/joker/HV/3DGS/PublicDataset \
  --datasets mipnerf360 \
  --scenes garden,bicycle \
  --scene-data-factor garden=4 \
  --scene-data-factor bicycle=2
```

List available dataset keys/scenes:

```bash
python3 benchmarks/strategies_visual_quality/run_train_batch.py --list --data-root .
```

## Batch evaluation

Evaluate visual quality metrics (PSNR/SSIM/LPIPS) from exported PLYs and write a table
that also includes final Gaussian count and training wall-clock time (from TensorBoard when available):

```bash
python3 benchmarks/strategies_visual_quality/run_eval_batch.py \
  --data-root /media/joker/HV/3DGS/PublicDataset \
  --datasets mipnerf360 \
  --scenes garden \
  --device cuda:0
```

Outputs are written under:

- `<data-root>/strategy_benchmark/summary.md` (single strategy)
- `<data-root>/strategy_benchmark/summary_all.md` (all strategies)

This evaluator automatically picks the latest `ply/splats_step*.ply` under each scene output directory.
Use `--step` to evaluate a specific training step.

When training used `data.normalize_world_space=True`, it evaluates in training space and maps PLY geometry
back to that space to avoid a camera/splat/SH coordinate mismatch.
