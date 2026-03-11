# FriendlySplat

FriendlySplat is a user-friendly, open-source Gaussian Splatting toolkit.
It integrates SOTA-style features into a unified platform for training, pruning,
meshing, and segmentation experiments.

Project page:
- https://ashadowz.github.io/friendlysplat/

Core capabilities:
- Fast 3DGS training workflows with configurable optimization and export steps
- Geometry-aware supervision with optional depth, normal, sky, and mask priors
- Live viewer support for metrics, rendering inspection, and camera frustums
- Pruning-oriented experimentation in the same training stack
- Mesh / splat export and checkpoint-based scene inspection

Repository layout:
- `main`: FriendlySplat source code, training pipeline, viewer, and tools
- `gh-pages`: static project website assets

This branch contains the website for FriendlySplat rather than the training code.
If you are looking for the implementation, switch to the `main` branch.
