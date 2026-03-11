<p align="center">
  <img src="assets/training.gif" alt="FriendlySplat" width="100%">
</p>

<a href="https://ashadowz.github.io/FriendlySplat/">
  <img src="https://img.shields.io/badge/Project_Page-FriendlySplat-green" alt="Project Page">
</a>

FriendlySplat is a user-friendly, open-source Gaussian Splatting toolkit, integrating SOTA
features into a unified platform for training, pruning, meshing and segmentation.

## To-Do List

☐ Improve the `Examples` section.<br>
☐ Better organize the already integrated features and the features still planned.<br>
☐ Build proper docs to replace the current collection of README files.

## Installation

Dependence: Please install [Pytorch](https://pytorch.org/get-started/locally/) first. Example environment setup:

```bash
conda create -n friendly-splat python=3.10
conda activate friendly-splat
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

For faster environment setup and dependency resolution, we recommend installing
[`uv`](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install FriendlySplat with the full optional toolchain:

```bash
git clone https://github.com/AshadowZ/FriendlySplat.git
cd FriendlySplat

uv pip install -e ".[train,viewer,mesh,segment,sfm,priors]" --no-build-isolation
```

If you only want the training and viewer extras:

```bash
uv pip install -e ".[train,viewer]" --no-build-isolation
```

The equivalent `pip` commands also work:

```bash
pip install -e ".[train,viewer,mesh,segment,sfm,priors]" --no-build-isolation
```

Notes:

- A CUDA-enabled PyTorch environment is expected for normal training.
- `uv pip` is usually faster than plain `pip`.
- `--no-build-isolation` lets the local `gsplat` CUDA kernel reuse your current PyTorch/CUDA toolchain.
- Some tools still have extra dependencies; check the README in each subfolder if needed.

## Expected Dataset Layout

FriendlySplat expects a COLMAP-style dataset directory under `--io.data-dir`:

```text
data_dir/
  images/
  sparse/0/
  depth_prior/        # optional
  normal_prior/       # optional
  dynamic_mask/       # optional
  sky_mask/           # optional
```

`images/` stores the training images. `sparse/0/` stores the COLMAP reconstruction.
The prior and mask folders are optional, and are only needed if you enable the
corresponding inputs in the config.

## Quick Start

Train on a COLMAP scene:

```bash
fs-train \
  --io.data-dir /path/to/data-dir \
  --io.result-dir /path/to/result-dir \
  --io.device cuda:0 \
  --io.export-ply \
  --io.save-ckpt \
  --data.preload none \
  --postprocess.use-bilateral-grid \
  --optim.visible-adam \
  --strategy.impl improved \
  --strategy.densification-budget 1000000
```

Open the viewer on the latest checkpoint or PLY in a result directory:

```bash
fs-view \
  -result-dir /path/to/result-dir \
  --device cuda \
  --port 8080
```

## Examples

This repo provides some examples to help you decide which extra tricks are worth
enabling, and how to tune the many magic-number-like hyperparameters in
`friendly_splat/trainer/configs.py`. This part is still under construction. For now,
you can also use Codex / Claude Code to read the repo and help generate a training
command for your scene.

## Development and Contribution

Issues and pull requests are welcome. The codebase is still evolving, and many
features may not have been widely tested yet, so issue reports are especially welcome.

FriendlySplat is built with substantial help from the broader Gaussian Splatting
community. We first thank
[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and
[gsplat](https://github.com/nerfstudio-project/gsplat) for efficient CUDA kernels and
strong feature integration.

We also thank [Improved-GS](https://github.com/XiaoBin2001/Improved-GS),
[AbsGS](https://github.com/TY424/AbsGS),
[taming-3dgs](https://github.com/humansensinglab/taming-3dgs),
[3dgs-mcmc](https://github.com/ubc-vision/3dgs-mcmc), and
[mini-splatting](https://github.com/fatPeter/mini-splatting) for high-quality
densification implementations and references.

For pruning-related ideas and code references, we thank
[GNS](https://github.com/XiaoBin2001/GNS),
[speedy-splat](https://github.com/j-alex-hanson/speedy-splat),
[GaussianSpa](https://github.com/noodle-lab/GaussianSpa), and
[LightGaussian](https://github.com/VITA-Group/LightGaussian).

We further thank [CityGaussian](https://github.com/Linketic/CityGaussian) for valuable
code references on urban-scale scene reconstruction, and
[InstaScene](https://zju3dv.github.io/instascene/) together with
[MaskClustering](https://github.com/PKU-EPIC/MaskClustering) for 2D-to-3D lifting
references.

Finally, special thanks to [XiaoBin2001](https://github.com/XiaoBin2001) for helpful
suggestions throughout development.
