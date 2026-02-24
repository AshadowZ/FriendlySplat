# Mesh extraction (TSDF) from FriendlySplat PLY exports

This directory contains a standalone mesh-extraction script that:

1) loads a FriendlySplat/gsplat-style splat PLY (`splats_stepXXXXXX.ply`);
2) renders RGB + depth for a COLMAP scene using the splats;
3) runs Open3D TSDF fusion and extracts a triangle mesh.

It intentionally does **not** depend on `friendly_splat/trainer/*`.

## Install

This tool requires Open3D:

```bash
pip install -r tools/requirements.txt
```

## Usage

```bash
python tools/mesh/tsdf_mesh_from_ply.py \
  --ply_path results/my_scene/ply/splats_step030000.ply \
  --data_dir /path/to/colmap_scene \
  --render_factor 2 \
  --interval 2 \
  --output_dir results/my_scene/mesh
```

### Optional: 2D mask culling (per-frame)

If you have per-frame object masks aligned with COLMAP images (e.g. `mask/*.png`),
you can zero out depth outside the mask during TSDF fusion:

```bash
python tools/mesh/tsdf_mesh_from_ply.py \
  --ply_path results/my_scene/ply/splats_step030000.ply \
  --data_dir /path/to/colmap_scene \
  --mask_dir mask \
  --mask_dilate 0
```

### Optional: 3D AABB culling (PGSR-style)

To reduce TSDF volume growth (and memory usage), you can provide a 3D axis-aligned
bounding box in **world coordinates**. Depth pixels that back-project outside the
box will be set to 0 before TSDF integration:

```bash
python tools/mesh/tsdf_mesh_from_ply.py \
  --ply_path results/my_scene/ply/splats_step030000.ply \
  --data_dir /path/to/colmap_scene \
  --aabb_min -1.0 -1.0 -1.0 \
  --aabb_max  1.0  1.0  1.0
```

Notes:
- Use `--interval` to skip frames (faster; also reduces cached frames on disk).
- This tool assumes the input PLY is already in the same coordinate system as the COLMAP scene.
- Use `--render_factor > 1` (or `-r 2`) to render TSDF inputs at lower resolution (e.g. 2 = half-res).
- If disk I/O becomes a bottleneck, tune `--write_workers` / `--queue_size`.
