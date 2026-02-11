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
  --data_factor 1 \
  --interval 2 \
  --output_dir results/my_scene/mesh
```

Notes:
- Use `--interval` to skip frames (faster, less memory).
- If you used `--data_factor > 1` during training, set the same value here.
This tool assumes the input PLY is already in the same coordinate system as the COLMAP scene.
- If disk I/O becomes a bottleneck, tune `--write_workers` / `--queue_size`.
