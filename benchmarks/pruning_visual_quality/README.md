# Pruning Visual Quality Benchmarks

This folder contains scripts for benchmarking FriendlySplat visual quality under
post-densification pruning on **360v2 only**.

Dataset layout assumption:

- `<data-root>/360v2/<scene>/{images/, sparse/, ...}`

Output layout:

- `<data-root>/benchmark/pruning_benchmark/<scene>/<pruner>/...`

Pruners:

- `pure_densify`
- `gns`
- `speedy`

## Batch training

Run training sequentially on one GPU:

```bash
python3 benchmarks/pruning_visual_quality/run_train_batch.py \
  --data-root /media/joker/HV/3DGS/PublicDataset \
  --scenes bicycle,bonsai \
  --pruners all
```

`--scenes` supports `all|csv`.
`--pruners` supports `all|csv` (`pure_densify,gns,speedy`).

The script hardcodes baseline trainer/pruning hyperparameters (device, preload,
max steps, pruning schedule, TB cadence, etc.). To override temporarily, pass
trainer flags after `--`.

Example:

```bash
python3 benchmarks/pruning_visual_quality/run_train_batch.py \
  --data-root /media/joker/HV/3DGS/PublicDataset \
  --scenes kitchen \
  --pruners gns -- \
  --optim.max-steps 50000
```

## Batch evaluation

Evaluate PSNR/SSIM/LPIPS from exported PLYs:

```bash
python3 benchmarks/pruning_visual_quality/run_eval_batch.py \
  --data-root /media/joker/HV/3DGS/PublicDataset \
  --scenes kitchen \
  --pruners all \
  --device cuda:0
```

Summary output:

- `<data-root>/benchmark/pruning_benchmark/summary_all.md` (when `--pruners all`)
- `<data-root>/benchmark/pruning_benchmark/summary.md` (single or custom pruner set)

The evaluator automatically picks the latest `ply/splats_step*.ply` under each
scene output directory. Use `--step` to evaluate a specific step.
