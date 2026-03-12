# Mesh Extraction (TSDF) from FriendlySplat PLY Exports

This standalone tool reconstructs a triangle mesh from an uncompressed
FriendlySplat or gsplat PLY export (`splats_stepXXXXXX.ply`).

It follows a simple pipeline:

1. Load the exported splat PLY.
2. Render RGB and depth for the COLMAP cameras in the scene.
3. Integrate the rendered frames with Open3D TSDF fusion.
4. Optionally keep only the largest connected mesh components.

## Installation

Install FriendlySplat with the optional mesh toolchain:

```bash
pip install -e ".[mesh]" --no-build-isolation
```

## Quick Start

You need two inputs:

- `--ply_path`: path to an uncompressed FriendlySplat / gsplat `.ply` file
- `--data_dir`: root of a COLMAP scene containing `images/` and `sparse/0/`

The input PLY must already be in the same coordinate system as the COLMAP scene.
`ply_compressed` exports are not supported here.

Basic reconstruction:

```bash
fs-mesh \
  --ply_path results/my_scene/ply/splats_step030000.ply \
  --data_dir /path/to/scene
```

Note: by default, this runs at full render resolution and estimates TSDF
parameters automatically. Outputs are written to `<ply_dir>/../mesh/`.
That means `--render_factor 1` and `--interval 1` unless you override them.

## Common Workflows

### Per-Frame Masks

If you have aligned object masks, you can zero out depth outside the mask during
TSDF integration:

```bash
fs-mesh \
  --ply_path results/my_scene/ply/splats_step030000.ply \
  --data_dir /path/to/scene \
  --mask_dir mask \
  --mask_dilate 0
```

Pixels with mask value `< 0.5` are ignored. Relative `--mask_dir` paths are
resolved under `--data_dir`.

### AABB Cropping

To reduce memory use and remove distant floaters, restrict integration to a
world-space axis-aligned bounding box:

```bash
fs-mesh \
  --ply_path results/my_scene/ply/splats_step030000.ply \
  --data_dir /path/to/scene \
  --aabb_min -1.0 -1.0 -1.0 \
  --aabb_max 1.0 1.0 1.0
```

## Outputs and Caching

Main outputs:

- `tsdf_mesh.ply`: raw TSDF mesh
- `tsdf_mesh_post.ply`: post-processed mesh after connected-component filtering

The tool also caches rendered RGB and depth `.npy` files on disk to save RAM.
These caches are deleted automatically unless you pass `--no-delete-cache`.

## Tuning TSDF Quality

Mesh quality depends heavily on TSDF parameters and filtering settings. The tool
estimates reasonable defaults, but tuning them often matters.

Core TSDF parameters:

- `--voxel_length`: controls mesh detail. Smaller values preserve finer geometry
  but increase memory use and runtime. Auto-estimate:
  `median_camera_distance / 192`.
- `--sdf_trunc`: TSDF truncation distance. A good starting point is roughly
  `4x-10x` of `voxel_length`. Auto-estimate: `5 * voxel_length`.
- `--depth_trunc`: Max depth for TSDF integration. Crucial for 360 degree
  object scans. Set this slightly larger than the camera-to-object distance to
  discard distant backgrounds (e.g., walls). This prevents memory explosion
  when using a small `--voxel_length` on object-centric captures.

Filtering and post-processing:

- `--post_process_clusters` (default: `50`): keep only the largest connected
  triangle clusters. Set `0` to disable post-processing.
- `--mask_dilate` (default: `0`): dilate foreground masks when using
  `--mask_dir`, which can help avoid clipping object boundaries too aggressively.

Troubleshooting tips:

- Too much background or giant floaters: use `--depth_trunc`, `--mask_dir`, or
  `--aabb_min` / `--aabb_max`.
- Thin structures disappearing: decrease `--voxel_length`.
- Mesh has many holes: increase both `--voxel_length` and `--sdf_trunc`.
- Meshing is too slow or memory-heavy: increase `--render_factor` and/or
  `--interval` for quicker iterations.
