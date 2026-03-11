<div align="center">

<p align="center">
  <img src="assets/training.gif" alt="FriendlySplat" width="100%">
</p>

<a href="https://ashadowz.github.io/FriendlySplat/">
  <img src="https://img.shields.io/badge/Project_Page-FriendlySplat-green" alt="Project Page">
</a>
<img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
<img src="https://img.shields.io/badge/Runtime-PyTorch%20%2B%20CUDA-orange" alt="PyTorch CUDA">

<p align="center">
  FriendlySplat is a user-friendly, open-source Gaussian Splatting toolkit,
  integrating SOTA features into a unified platform for training, pruning,
  meshing and segmentation.
</p>

</div>

## 🤔 Why FriendlySplat

If you mostly care about getting a reconstruction quickly and hate setting up
environments, more mature commercial tools such as
[KIRI Engine](https://www.kiriengine.app/) and [Polycam](https://poly.cam/) are
probably a better fit.

If you are trying to scale up training on very large synthetic scenes, you should
also take a look at [Marble World](https://marble.worldlabs.ai/).

If you are looking for a usable codebase for Gaussian Splatting development, or are
simply curious about GS techniques and want something relatively easy to read and
modify, FriendlySplat is a good place to start.

## ✨ Highlights

- **Feature-rich workflow toolbox**: this repo integrates SfM preprocessing, more
  densification and pruning strategies, more regularization terms, downstream mesh
  extraction, GS scene segmentation, and other practical extensions around the core
  training loop.
- **Fast experimentation without hiding the code**: the repo stays easy to read and
  modify, with benchmark scripts and tool folders kept close to the actual implementation.

## 📝 To-Do List

☐ Improve the `Examples` section.<br>
☐ Clearly list the features that are already integrated and the features planned for future integration.<br>
☐ Build proper docs to replace the current collection of README files.

## 📦 Installation

Dependence: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.
Example environment setup:

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

The equivalent `pip` command also works:

```bash
pip install -e ".[train,viewer,mesh,segment,sfm,priors]" --no-build-isolation
```

Notes:

- `uv pip` is usually faster than plain `pip`.
- `--no-build-isolation` lets the local `gsplat` CUDA kernel reuse your current
  PyTorch/CUDA toolchain.
- Some tools still have extra dependencies; in particular, the `sfm` extra does not
  fully install HLOC by itself, so check the README in each subfolder when needed,
  especially [tools/sfm/README.md](tools/sfm/README.md).

## 🗂️ Expected Dataset Layout

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

- `images/` stores the training images.
- `sparse/0/` stores the COLMAP reconstruction.
- The prior and mask folders are optional and only needed if you enable the
  corresponding inputs in the config.

## 🚀 Quick Start

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
  --result-dir /path/to/result-dir \
  --device cuda \
  --port 8080
```

## 🧪 Examples

This repo provides some examples to help you decide which extra tricks are worth
enabling, and how to tune the many magic-number-like hyperparameters in
`friendly_splat/trainer/configs.py`. This part is still under construction. For now,
you can also use Codex / Claude Code to read the repo and help generate a training
command for your scene.

## 🛠️ Development and Contribution

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

We also thank [PGSR](https://github.com/zju3dv/PGSR),
[2DGS](https://github.com/hbb1/2d-gaussian-splatting),
[GGGS](https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting),
[dn-splatter](https://github.com/maturk/dn-splatter),
[mvsanywhere](https://github.com/nianticlabs/mvsanywhere), and
[2DGS++](https://github.com/hugoycj/2d-gaussian-splatting-great-again) for their
explorations of geometry regularization and high-quality code releases.

We further thank [CityGaussian](https://github.com/Linketic/CityGaussian) for valuable
code references on urban-scale scene reconstruction, and
[InstaScene](https://zju3dv.github.io/instascene/) together with
[MaskClustering](https://github.com/PKU-EPIC/MaskClustering) for 2D-to-3D lifting
references.

Finally, special thanks to [XiaoBin2001](https://github.com/XiaoBin2001) for helpful
suggestions throughout development.
