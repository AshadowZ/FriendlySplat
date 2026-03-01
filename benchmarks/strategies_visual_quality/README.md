# Visual Quality Benchmarks

This folder contains scripts/utilities for benchmarking FriendlySplat visual quality
for the Mip-NeRF360 benchmark scenes (360v2).

Dataset layout assumption:

- `<data-root>/360v2/<scene>/{images/, sparse/, ...}`

## Batch training

Single-GPU, sequential batch training (Improved-GS-style per-scene budgets):

```bash
python3 benchmarks/strategies_visual_quality/run_train_batch.py \
  --data-root /media/joker/HV/3DGS/PublicDataset \
  --scenes bicycle,bonsai
```

By default, outputs are written under:

- `<data-root>/benchmark/strategies_benchmark/<scene>/<strategy>/`

For example:

- `<data-root>/benchmark/strategies_benchmark/bicycle/improved/`

Dataset keys used by these scripts:

- `360v2`

This benchmark runner is designed to benchmark different densification strategies.
It always exports the final PLY and enables TensorBoard logging.

By default, it runs the gsplat `default` strategy. Use `--strategy-impl all` to run
`improved`, `default`, and `mcmc` sequentially for each scene.

For `--strategy-impl improved`, it also applies an Improved-GS-style alignment:

- Enables `data.benchmark_train_split=True` so training uses a disjoint split from evaluation (Improved-GS-style)
- Sets key strategy switches and optimizer hyperparameters to Improved-GS defaults
- Enables the MU-style splat optimizer step schedule by default (`optim.mu_enable=True`)

For `--strategy-impl default`, it applies a gsplat DefaultStrategy-style baseline:

- Disables `strategy.absgrad` and enables `strategy.verbose`
- Sets `strategy.refine_scale2d_stop_iter=0` and `strategy.prune_scale3d=0.1`
- Aligns the main optimizer LRs (means/scales/quats/opacities/SH) and sets means `lr_final=0.01*lr_init`

For `--strategy-impl mcmc`, it aligns to gsplat's MCMC preset:

- Sets `init.init_opacity=0.5` and `init.init_scale=0.1`
- Enables MCMC regularizers (`reg.opacity_reg_weight=0.01`, `reg.scale_l1_reg_weight=0.01`)
- Sets `strategy.refine_stop_iter=25_000` and aligns key MCMC knobs (noise/min_opacity)
- Aligns the main optimizer LRs (means/scales/quats/opacities/SH) and sets means `lr_final=0.01*lr_init`

Per-scene budgets are applied as:

- ImprovedStrategy: `strategy.densification_budget`
- MCMCStrategy: `strategy.mcmc_cap_max`

This runner uses the Improved-GS "normal" per-scene budget table when training with `--strategy-impl improved` or `--strategy-impl mcmc`.

This runner hardcodes baseline trainer hyperparameters (device, preload, max_steps, etc).
To override temporarily, pass trainer flags after `--` (these are appended last and override defaults).
Examples:

```bash
python3 benchmarks/strategies_visual_quality/run_train_batch.py \
  --data-root /media/joker/HV/3DGS/PublicDataset \
  --scenes bicycle -- \
  --data.preload none
```

Scene resolution is controlled by per-scene `data.data_factor` presets in `run_train_batch.py`.
Edit the mapping in the script if you need different per-scene factors.

## Batch evaluation

Evaluate visual quality metrics (PSNR/SSIM/LPIPS) from exported PLYs and write a table
that also includes final Gaussian count and training wall-clock time (from TensorBoard when available):

```bash
python3 benchmarks/strategies_visual_quality/run_eval_batch.py \
  --data-root /media/joker/HV/3DGS/PublicDataset \
  --scenes garden \
  --device cuda:0
```

Outputs are written under:

- `<data-root>/benchmark/strategies_benchmark/summary.md` (single strategy)
- `<data-root>/benchmark/strategies_benchmark/summary_all.md` (all strategies)

This evaluator automatically picks the latest `ply/splats_step*.ply` under each scene output directory.
Use `--step` to evaluate a specific training step.

When training used `data.normalize_world_space=True`, it evaluates in training space and maps PLY geometry
back to that space to avoid a camera/splat/SH coordinate mismatch.
