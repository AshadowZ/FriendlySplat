# Pruning Visual Quality Benchmarks

This folder contains scripts/utilities for benchmarking FriendlySplat visual quality
under different **post-densification pruning** methods.

The benchmark runs the same densification preset and then prunes down to a target
`final_budget` using either:

- Pure densification baseline (`pure_densify`): cap densification to `final_budget` (no pruning).
- GNS pruning (`gns.gns_enable=True`)
- Speedy-style hard pruning (`hard_prune.enable=True`)

All outputs (logs/TensorBoard/PLYs/summaries) are written under:

- `<data-root>/pruning_benchmark/`

## Batch training

Single-GPU, sequential batch training:

```bash
python3 benchmarks/pruning_visual_quality/run_train_batch.py \
  --data-root /media/joker/HV/3DGS/PublicDataset \
  --datasets mipnerf360,deepblending \
  --scenes bicycle,bonsai,drjohnson \
  --device cuda:0 \
  --max-steps 30000
```

By default, the script:

- Uses `strategy.impl=improved`.
- Sets `strategy.densification_budget = 3 * final_budget` for pruning runs (and `final_budget` for `pure_densify` baseline).
- Forces pruning to start strictly after densification (`strategy.refine_stop_iter`).
- Exports the final PLY (`ply/splats_stepXXXXXX.ply`) and enables TensorBoard logging.

By default, it uses per-scene budgets aligned with the upstream GNS benchmark defaults
(600k/300k per scene). Use `--final-budget` or `--scene-final-budget` to override.

Notes:

- `strategy.refine_stop_iter` is 0-based (internal `step`).
- GNS `gns.reg_start/reg_end` and hard-prune `hard_prune.start_step/stop_step` are 1-based (train steps).
- `--densify-stop-step` is 1-based and matches the GNS repo's common schedule (default: 15000).

List available dataset keys/scenes:

```bash
python3 benchmarks/pruning_visual_quality/run_train_batch.py --list --data-root .
```

## Batch evaluation

Evaluate PSNR/SSIM/LPIPS from exported PLYs and write a summary table under
`<data-root>/pruning_benchmark/summary.md`:

```bash
python3 benchmarks/pruning_visual_quality/run_eval_batch.py \
  --data-root /media/joker/HV/3DGS/PublicDataset \
  --datasets mipnerf360 \
  --scenes garden \
  --device cuda:0
```

The evaluator automatically picks the latest `ply/splats_step*.ply` under each
scene output directory. Use `--step` to evaluate a specific training step.
