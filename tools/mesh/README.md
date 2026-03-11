# Mesh extraction (TSDF) from FriendlySplat PLY exports

This tool reconstructs a triangle mesh from a FriendlySplat / gsplat PLY export by:

1. loading an exported splat PLY (`splats_stepXXXXXX.ply`);
2. rendering RGB + depth for the COLMAP cameras in a scene;
3. integrating the rendered frames with Open3D TSDF fusion;
4. optionally post-processing the mesh by keeping the largest connected components.

It is designed as a standalone post-processing step and does **not** depend on
`friendly_splat/trainer/*`.

## Install

Install FriendlySplat with the mesh extra:

```bash
pip install -e ".[mesh]" --no-build-isolation
```

## Inputs

You need two inputs:

- `--ply_path`: an **uncompressed** FriendlySplat / gsplat splat PLY
- `--data_dir`: a COLMAP scene root compatible with `ColmapDataParser`

Expected scene layout:

```text
<data_dir>/
  images/
  sparse/0/
```

Notes:

- This tool assumes the input PLY is already in the same coordinate system as the COLMAP scene.
- `ply_compressed` exports are not supported here.
- The script renders from the COLMAP cameras found under `--data_dir`.

## Outputs

By default, outputs are written to:

```text
<ply_dir>/../mesh/
```

Main outputs:

- `tsdf_mesh.ply`: raw TSDF mesh before post-processing
- `tsdf_mesh_post.ply`: post-processed mesh after connected-component filtering

If caching is enabled, RGB/depth frame caches are also written under:

- `<output_dir>/cache/` by default
- or `--cache_dir` if you provide one

By default, the cache is deleted after meshing finishes. Use `--no-delete-cache`
if you want to keep the rendered RGB/depth `.npy` files for debugging.

## Basic usage

Recommended package entrypoint:

```bash
fs-mesh \
  --ply_path results/my_scene/ply/splats_step030000.ply \
  --data_dir /path/to/colmap_scene \
  --render_factor 2 \
  --interval 2 \
  --output_dir results/my_scene/mesh
```

Legacy script path also works:

```bash
python tools/mesh/tsdf_mesh_from_ply.py \
  --ply_path results/my_scene/ply/splats_step030000.ply \
  --data_dir /path/to/colmap_scene \
  --render_factor 2 \
  --interval 2 \
  --output_dir results/my_scene/mesh
```

## Common workflows

### 1. Basic reconstruction

```bash
fs-mesh \
  --ply_path results/my_scene/ply/splats_step030000.ply \
  --data_dir /path/to/scene
```

This uses:

- full render resolution (`--render_factor 1`)
- every frame (`--interval 1`)
- automatic TSDF parameter estimation when `--voxel_length` / `--sdf_trunc` are omitted

### 2. Faster reconstruction at lower render resolution

```bash
fs-mesh \
  --ply_path results/my_scene/ply/splats_step030000.ply \
  --data_dir /path/to/scene \
  --render_factor 2 \
  --interval 2
```

This is usually the first thing to try when the scene is large or meshing is slow.

### 3. TSDF with per-frame masks

If you have per-frame object masks aligned with COLMAP images, you can zero out
depth outside the mask during TSDF integration:

```bash
fs-mesh \
  --ply_path results/my_scene/ply/splats_step030000.ply \
  --data_dir /path/to/scene \
  --mask_dir mask \
  --mask_dilate 0
```

Mask behavior:

- if `--mask_dir` is relative, it is resolved relative to `--data_dir`
- masks are matched to images by integer filename id or exact filename stem
- pixels with mask value `< 0.5` are removed from TSDF integration

### 4. TSDF with 3D AABB culling

To reduce TSDF volume growth and memory usage, you can keep only depth samples
that back-project into an axis-aligned box in **world coordinates**:

```bash
fs-mesh \
  --ply_path results/my_scene/ply/splats_step030000.ply \
  --data_dir /path/to/scene \
  --aabb_min -1.0 -1.0 -1.0 \
  --aabb_max  1.0  1.0  1.0
```

This is especially useful for scenes with large amounts of empty space or strong floaters.

### 5. Keep cached render frames after meshing

```bash
fs-mesh \
  --ply_path results/my_scene/ply/splats_step030000.ply \
  --data_dir /path/to/scene \
  --no-delete-cache
```

## Important arguments

- `--render_factor`: render downscale factor before TSDF integration. `2` means half resolution.
- `--interval`: render every N-th frame.
- `--device`: rendering device, e.g. `cuda:0` or `cpu`.
- `--sh_degree`: SH degree to render. `-1` uses the max degree stored in the splats.
- `--output_dir`: output directory. Defaults to `<ply_dir>/../mesh`.
- `--cache_dir`: directory for cached RGB/depth `.npy` frames.
- `--delete_cache` / `--no-delete-cache`: whether cached frames are deleted after meshing. Default is delete.
- `--write_workers`: number of background workers used to write cached frames.
- `--queue_size`: number of pending rendered frames buffered before disk write.
- `--post_process_clusters`: keep the largest K connected mesh components. `0` disables filtering.

TSDF parameters:

- `--voxel_length`: TSDF voxel size in scene units
- `--sdf_trunc`: TSDF truncation distance in scene units
- `--depth_trunc`: max depth used during TSDF integration

In practice, TSDF mesh quality is dominated mostly by `voxel_length` and `sdf_trunc`.

- `voxel_length` mainly controls mesh detail level:
  smaller values usually produce finer geometry, but also increase memory usage and runtime.
- `sdf_trunc` is usually best treated as a multiple of `voxel_length`.
  A good empirical range is roughly `4x` to `10x` of `voxel_length`.

If `--voxel_length` and `--sdf_trunc` are omitted, the script estimates them automatically.
Current behavior:

- `voxel_length = median_camera_distance / 192`
- `sdf_trunc = 5 * voxel_length`

## Practical tips

- Start with `--render_factor 2 --interval 2` for quick iteration.
- If you are tuning mesh quality, focus on `--voxel_length` and `--sdf_trunc` first.
- If the mesh is too noisy, reduce floaters first with masks or AABB culling.
- If thin structures disappear, try a smaller `--voxel_length`.
- If the mesh has many holes, try increasing both `--voxel_length` and `--sdf_trunc`.
- If meshing is slow due to disk I/O, increase `--write_workers` and `--queue_size`.
- If you only care about the cleaned mesh, use `tsdf_mesh_post.ply`.
- In practice, `--voxel_length` and `--sdf_trunc` usually need to be tuned per model; there is no single setting that works best for every scene.

## Troubleshooting

- `PLY not found` / unsupported format:
  use an uncompressed `splats_stepXXXXXX.ply` export.
- `mask_dir not found`:
  check whether `--mask_dir` should be absolute or relative to `--data_dir`.
- Very large cache size:
  increase `--interval`, increase `--render_factor`, or keep the default cache deletion behavior.
- Mesh contains too much background:
  use `--mask_dir` and/or `--aabb_min` + `--aabb_max`.

## Relationship to benchmarks

This tool is used by geometry-oriented benchmark scripts in this repo. Those
benchmark runners may override TSDF hyperparameters for specific datasets such
as DTU or Tanks & Temples.
