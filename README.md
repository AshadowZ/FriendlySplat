<p align="center">
  <img src="assets/training.gif" alt="FriendlySplat" width="100%">
</p>

<a href="https://ashadowz.github.io/FriendlySplat/">
  <img src="https://img.shields.io/badge/Project_Page-FriendlySplat-green" alt="Project Page">
</a>

FriendlySplat is a user-friendly, open-source Gaussian Splatting toolkit, integrating SOTA
features into a unified platform for training, pruning, meshing and segmentation.

## Installation

This repo builds the local `gsplat` CUDA extension and installs the extra Python dependencies used
by FriendlySplat.

```bash
git clone <your-repo-url>
cd FriendlySplat

pip install -r friendly_splat/requirements.txt
pip install -e .
```

Notes:

- A CUDA-enabled PyTorch environment is expected for normal training.
- `setup.py` builds the local `gsplat` extension in this repo.
- Some optional tools under `tools/` and `benchmarks/` have extra dependencies; check the README in
  each subfolder if you use them.

## Quick Start

Train on a COLMAP scene:

```bash
python3 friendly_splat/trainer.py \
  --io.data-dir /path/to/scene \
  --io.result-dir results/scene \
  --io.device cuda \
  --io.export-ply True \
  --io.save-ckpt True
```

Open the viewer on the latest checkpoint or PLY in a result directory:

```bash
python3 friendly_splat/viewer.py \
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

- Training: `python3 friendly_splat/trainer.py ...`
- Viewer: `python3 friendly_splat/viewer.py ...`
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
