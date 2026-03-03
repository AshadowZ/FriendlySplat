# gsplat demos

Small, self-contained scripts to sanity-check `gsplat` rendering/backprop in this repo.

## Image fitting

`image_fitting.py` optimizes a set of random Gaussians (3DGS) to fit a target image using plain Adam + MSE.

Run from repo root:

```bash
# Fit a synthetic image (white background, red top-left, blue bottom-right).
python tools/gsplat_demo/image_fitting.py --height 256 --width 256 \
  --save-gif

# Fit an existing RGB image.
python tools/gsplat_demo/image_fitting.py --img-path /path/to/image.jpg \
  --save-gif

# Run without densify/prune (densify is enabled by default).
python tools/gsplat_demo/image_fitting.py --height 256 --width 256 \
  --no-densify --save-gif
```

Outputs (default):

- `results/gsplat_demo/training.gif`

Notes:

- Requires a Python environment with `torch`, `gsplat`, and `Pillow`.
- If `--device cuda:0` is requested but CUDA is unavailable, the script falls back to CPU with a warning.
- Densify/prune is enabled by default (demo heuristic). Densify clones each selected Gaussian once. Key defaults: `num_points=5000`, `iterations=6000`, `lr=0.005`, `densify_to_iter=4000`, `densify_interval=200`, `densify_topk=1500`, `max_points=60000`, `prune_opacity_threshold=0.05`.
