<p align="center">
  <img src="assets/training.gif" alt="FriendlySplat" width="100%">
</p>

<a href="https://ashadowz.github.io/FriendlySplat/">
  <img src="https://img.shields.io/badge/Project_Page-FriendlySplat-green" alt="Project Page">
</a>

FriendlySplat is a user-friendly, open-source Gaussian Splatting toolkit, integrating SOTA
features into a unified platform for training, pruning, meshing and segmentation.

## Installation

This repo builds the local `gsplat` CUDA extension and installs the FriendlySplat package.

```bash
git clone https://github.com/AshadowZ/FriendlySplat.git
cd FriendlySplat

pip install -e ".[train,viewer]" --no-build-isolation
```

Or install with the full optional toolchain:

```bash
pip install -e ".[train,viewer,mesh,segment,sfm,priors]" --no-build-isolation
```

Notes:

- A CUDA-enabled PyTorch environment is expected for normal training.
- `--no-build-isolation` is recommended so the local `gsplat` extension reuses the current environment's
  PyTorch/CUDA toolchain instead of creating a separate build env.
- `setup.py` builds the local `gsplat` extension in this repo.
- Some optional tools under `tools/` and `benchmarks/` have extra dependencies; check the README in
  each subfolder if you use them.

## Quick Start

Train on a COLMAP scene:

```bash
fs-train \
  --io.data-dir /path/to/scene \
  --io.result-dir results/scene \
  --io.device cuda \
  --io.export-ply True \
  --io.save-ckpt True
```

Open the viewer on the latest checkpoint or PLY in a result directory:

```bash
fs-view \
  --result-dir results/scene \
  --device cuda \
  --port 8080
```

Common knobs live in [`friendly_splat/trainer/configs.py`](friendly_splat/trainer/configs.py), including:

- dataset normalization and train / test split
- densification strategy selection
- pruning schedules
- evaluation cadence
- viewer settings
- TensorBoard logging

## Useful Entry Points

- Training: `fs-train ...`
- Viewer: `fs-view ...`
- SfM preprocessing: `fs-sfm ...`
- TSDF meshing: `fs-mesh ...`
- Instance clustering: `fs-segment ...`
- MoGe prior generation: `fs-prior-moge ...`
- Strategy benchmark training: `python3 benchmarks/strategies_visual_quality/run_train_batch.py ...`
- Strategy benchmark evaluation: `python3 benchmarks/strategies_visual_quality/run_eval_batch.py ...`
- Pruning benchmark training: `python3 benchmarks/pruning_visual_quality/run_train_batch.py ...`
- Pruning benchmark evaluation: `python3 benchmarks/pruning_visual_quality/run_eval_batch.py ...`

## Current Scope

FriendlySplat is aimed at single-machine experimentation and benchmarking.
There are config fields reserved for distributed training, but the current repo is centered on
single-GPU workflows and benchmark scripts.

## License

See [LICENSE](LICENSE).
