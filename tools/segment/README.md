# Instance Clustering

This folder contains an offline instance clustering tool for FriendlySplat scenes.

The tool:

- loads FriendlySplat Gaussian checkpoints or PLY exports
- loads COLMAP cameras from the scene
- loads per-image SAM masks from `<data_dir>/sam/...`
- builds mask-to-Gaussian correspondences with `gsplat` rasterization
- clusters masks into 3D instances across views
- exports instance-colored point clouds

The script entry point is [instascene_gauscluster.py](/home/joker/learning/FriendlySplat/tools/segment/instascene_gauscluster.py).

## Install

Install the normal FriendlySplat dependencies first.

Additional dependencies used by this tool are listed in [tools/requirements.txt](/home/joker/learning/FriendlySplat/tools/requirements.txt).

Optional GPU DBSCAN:

- `cupy`
- `cuml`

## Expected Mask Layout

The tool looks for masks under one of:

- `<data_dir>/sam/mask_sorted`
- `<data_dir>/sam/mask`
- `<data_dir>/sam/mask_filtered`

You can override this with `--mask-dir-name`.

Each mask file should match the training image stem, for example:

- `images/frame_0001.png` -> `sam/mask_sorted/frame_0001.png`
- `images/frame_0001.jpg` -> `sam/mask_sorted/frame_0001.npy`

Label `0` is treated as background. Positive integer labels are treated as instance masks inside that frame.

## Usage

From a FriendlySplat result directory:

```bash
python3 tools/segment/instascene_gauscluster.py \
  --result-dir /path/to/results \
  --data-dir /path/to/scene \
  --split train
```

By default, `--result-dir` prefers checkpoints under `ckpts/` and falls back to
PLY files under `ply/` if no checkpoint is available. You can change this with
`--prefer-source ply`.

To load a specific training step from `result_dir`, pass `--step`:

```bash
python3 tools/segment/instascene_gauscluster.py \
  --result-dir /path/to/results \
  --step 60000
```

From explicit checkpoints:

```bash
python3 tools/segment/instascene_gauscluster.py \
  --ckpt-paths /path/to/ckpt_step030000.pt \
  --data-dir /path/to/scene
```

From explicit PLY exports:

```bash
python3 tools/segment/instascene_gauscluster.py \
  --ply-paths /path/to/splats_step030000.ply \
  --data-dir /path/to/scene
```

With GPU DBSCAN:

```bash
python3 tools/segment/instascene_gauscluster.py \
  --result-dir /path/to/results \
  --data-dir /path/to/scene \
  --use-gpu-dbscan
```

## Outputs

By default outputs are written to `cluster_result/` under the FriendlySplat
results directory. When you launch from `--result-dir`, this is
`<result_dir>/cluster_result/`. When you launch from explicit `--ckpt-paths`
or `--ply-paths`, the script infers the same results root from the input file
location:

- `color_cluster.ply`
- `instance_labels.npy`
- `color_cluster_knn.ply`
- `instance_labels_knn.npy`

## Notes

- The tool uses FriendlySplat's `ColmapDataParser`, so `--data-factor`, `--normalize-world-space`, `--align-world-axes`, and split settings should match how the scene was trained.
- When `--result-dir` resolves to a checkpoint, omitted `--normalize-world-space` and `--align-world-axes` values are inherited from `result_dir/cfg.yml`.
- When the selected source is a PLY, `--normalize-world-space` and `--align-world-axes` both default to `False` unless explicitly provided. This matches the assumption that exported PLY splats are already aligned with the COLMAP world coordinates.
- There is no `--output-dir` parameter. Outputs always go under the inferred FriendlySplat results directory as `cluster_result/`.
- Pixel-to-Gaussian tracking depends on the installed `gsplat` build supporting `track_pixel_gaussians=True`.
