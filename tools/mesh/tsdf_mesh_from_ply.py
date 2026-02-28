from __future__ import annotations

import argparse
import math
import shutil
import threading
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import open3d as o3d
import torch
import tqdm

# This script consumes a gsplat-style splat PLY and runs TSDF fusion to extract a mesh.
# Allow running as a standalone script from the repo root without installation.
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from friendly_splat.data.colmap_dataparser import ColmapDataParser
from friendly_splat.modules.gaussian import GaussianModel


@dataclass(frozen=True)
class CameraFrame:
    camtoworld: np.ndarray  # [4,4]
    K: np.ndarray  # [3,3]
    height: int
    width: int
    image_path: str


class DiskNpyArray:
    """Expose a list of `.npy` files as an array, loading on-demand."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = [str(p) for p in paths]

    def __len__(self) -> int:
        return int(len(self.paths))

    def __getitem__(self, idx: int) -> np.ndarray:
        return np.load(self.paths[int(idx)])


def _mask_to_alpha01(mask: np.ndarray) -> np.ndarray:
    """Convert an arbitrary mask image array into float32 alpha in [0, 1]."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    if np.issubdtype(m.dtype, np.integer):
        denom = float(np.iinfo(m.dtype).max)
        if denom <= 0:
            return np.zeros(m.shape[:2], dtype=np.float32)
        return (m.astype(np.float32) / denom).clip(0.0, 1.0)

    mf = m.astype(np.float32, copy=False)
    maxv = float(np.nanmax(mf)) if mf.size > 0 else 0.0
    if maxv <= 1.0:
        return mf.clip(0.0, 1.0)
    if maxv <= 255.0:
        return (mf / 255.0).clip(0.0, 1.0)
    return (mf / maxv).clip(0.0, 1.0)


@dataclass(frozen=True)
class _WriteItem:
    done_event: Optional["torch.cuda.Event"]
    color_cpu: torch.Tensor
    depth_cpu: torch.Tensor
    color_path: str
    depth_path: str


def _load_ply_splats(ply_path: str) -> dict[str, torch.Tensor]:
    """Load splat parameters from an uncompressed gsplat-style PLY.

    Supported format:
      - header: `format binary_little_endian 1.0`
      - vertex properties: x/y/z, f_dc_*, f_rest_*, opacity, scale_*, rot_*

    Notes:
      - This loader is intended for PLY files produced by `gsplat.export_splats(..., format="ply")`
        (and by FriendlySplat PLY exports).
      - `ply_compressed` is not supported.
    """
    ply_path = str(ply_path)
    with open(ply_path, "rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if line == b"":
                raise ValueError("Unexpected EOF while reading PLY header.")
            s = line.decode("utf-8", errors="strict").rstrip("\n")
            header_lines.append(s)
            if s.strip() == "end_header":
                break

        if not header_lines or header_lines[0].strip() != "ply":
            raise ValueError("Not a PLY file (missing leading 'ply').")

        fmt = None
        vertex_count = None
        in_vertex = False
        vertex_props: list[str] = []

        for s in header_lines:
            parts = s.strip().split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = " ".join(parts[1:])
            if parts[0] == "element":
                # Only parse vertex block; ignore faces/other elements.
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                continue
            if parts[0] == "property" and in_vertex:
                # `property float x` => name at parts[-1]
                vertex_props.append(parts[-1])

        if fmt is None or fmt.strip() != "binary_little_endian 1.0":
            raise ValueError(
                f"Unsupported PLY format: {fmt!r}. Only 'binary_little_endian 1.0' is supported."
            )
        if vertex_count is None or vertex_count <= 0:
            raise ValueError(f"Invalid vertex count: {vertex_count!r}.")
        if not vertex_props:
            raise ValueError("PLY has no vertex properties.")

        prop_to_idx = {name: i for i, name in enumerate(vertex_props)}

        def _req(name: str) -> int:
            if name not in prop_to_idx:
                raise KeyError(f"PLY is missing required vertex property {name!r}.")
            return int(prop_to_idx[name])

        # Required canonical properties.
        ix = _req("x")
        iy = _req("y")
        iz = _req("z")
        iop = _req("opacity")
        is0 = _req("scale_0")
        is1 = _req("scale_1")
        is2 = _req("scale_2")
        ir0 = _req("rot_0")
        ir1 = _req("rot_1")
        ir2 = _req("rot_2")
        ir3 = _req("rot_3")
        idc0 = _req("f_dc_0")
        idc1 = _req("f_dc_1")
        idc2 = _req("f_dc_2")

        # Optional higher-order SH.
        rest: list[tuple[int, int]] = []
        for name, idx in prop_to_idx.items():
            if name.startswith("f_rest_"):
                suffix = name[len("f_rest_") :]
                if suffix.isdigit():
                    rest.append((int(suffix), int(idx)))
        rest.sort(key=lambda t: t[0])
        rest_cols = [idx for _suffix, idx in rest]

        num_props = int(len(vertex_props))
        data_bytes = f.read(int(vertex_count) * num_props * 4)
        expected = int(vertex_count) * num_props
        # Note: np.frombuffer returns a non-writable view; we keep it as-is to avoid
        # an extra copy. Downstream `torch.from_numpy` will warn once and then
        # silence the warning for this program.
        arr = np.frombuffer(data_bytes, dtype=np.dtype("<f4"), count=expected)
        if int(arr.size) != expected:
            raise ValueError(
                f"PLY vertex data truncated: expected {expected} float32 values, got {arr.size}."
            )
        arr = arr.reshape(int(vertex_count), num_props)

        means = arr[:, [ix, iy, iz]]
        opacities = arr[:, iop]
        scales = arr[:, [is0, is1, is2]]
        quats = arr[:, [ir0, ir1, ir2, ir3]]

        sh0 = arr[:, [idc0, idc1, idc2]].reshape(int(vertex_count), 1, 3)

        if rest_cols:
            rest_flat = arr[:, rest_cols]
            if int(rest_flat.shape[1]) % 3 != 0:
                raise ValueError(
                    f"Invalid f_rest_* property count: {rest_flat.shape[1]} (must be divisible by 3)."
                )
            k = int(rest_flat.shape[1] // 3)
            shN = rest_flat.reshape(int(vertex_count), 3, k).transpose(0, 2, 1)
        else:
            shN = np.zeros((int(vertex_count), 0, 3), dtype=np.float32)

    return {
        "means": torch.from_numpy(means).float(),
        "scales": torch.from_numpy(scales).float(),
        "quats": torch.from_numpy(quats).float(),
        "opacities": torch.from_numpy(opacities).float(),
        "sh0": torch.from_numpy(sh0).float(),
        "shN": torch.from_numpy(shN).float(),
    }


def _build_gaussian_model(*, splats: dict[str, torch.Tensor], device: torch.device) -> GaussianModel:
    params = {k: torch.nn.Parameter(v.to(device=device)) for k, v in splats.items()}
    model = GaussianModel(params)
    model.eval()
    return model


def _infer_hw_from_image(*, image_path: str) -> tuple[int, int]:
    # Avoid importing PIL; `imageio` is already a dependency of FriendlySplat data loading.
    import imageio.v3 as iio  # noqa: WPS433

    if str(image_path).strip() == "":
        raise ValueError("image_path is empty while inferring (H, W).")
    img = iio.imread(str(image_path))
    if img.ndim < 2:
        raise ValueError(f"Unexpected image shape: {img.shape}")
    height, width = int(img.shape[0]), int(img.shape[1])
    return height, width


def _iter_frames(
    *,
    parsed_scene,
    split: str,
    default_hw: tuple[int, int],
    K_scale_xy: tuple[float, float],
    interval: int,
) -> Iterable[CameraFrame]:
    idxs = parsed_scene.indices
    h_default, w_default = int(default_hw[0]), int(default_hw[1])
    kx, ky = float(K_scale_xy[0]), float(K_scale_xy[1])
    if int(interval) <= 0:
        raise ValueError(f"interval must be > 0, got {interval}")

    for j, image_index in enumerate(idxs.tolist()):
        if j % int(interval) != 0:
            continue
        camtoworld = parsed_scene.camtoworlds[int(image_index)].astype(np.float32)
        K = parsed_scene.Ks[int(image_index)].astype(np.float32)
        if not np.isfinite(kx) or not np.isfinite(ky) or kx <= 0.0 or ky <= 0.0:
            raise ValueError(f"Invalid K_scale_xy={K_scale_xy}. Expected positive finite floats.")
        if kx != 1.0 or ky != 1.0:
            K = K.copy()
            K[0, :] *= float(kx)
            K[1, :] *= float(ky)
        yield CameraFrame(
            camtoworld=camtoworld,
            K=K,
            height=int(h_default),
            width=int(w_default),
            image_path=str(parsed_scene.image_paths[int(image_index)]),
        )


def _estimate_med_camera_dist(camtoworlds: np.ndarray, max_samples: int = 200) -> float:
    centers = camtoworlds[:, :3, 3].astype(np.float64)
    n = int(centers.shape[0])
    if n <= 1:
        return 1.0
    m = min(int(max_samples), n)
    sample_idx = np.linspace(0, n - 1, num=m, dtype=np.int64)
    c = centers[sample_idx]  # [m,3]
    diffs = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    tri = dists[np.triu_indices(m, k=1)]
    if tri.size == 0:
        return 1.0
    return float(np.median(tri))


def _create_tsdf_mesh(
    *,
    depths_np: DiskNpyArray,
    colors_np: DiskNpyArray,
    Ks: np.ndarray,  # [N,3,3]
    camtoworlds: np.ndarray,  # [N,4,4]
    hws: np.ndarray,  # [N,2]
    voxel_length: Optional[float],
    sdf_trunc: Optional[float],
    depth_trunc: Optional[float],
    mask_paths: Optional[list[str]],
    mask_dilate: int,
    aabb_bounds: Optional[np.ndarray],
):
    n = int(len(depths_np))
    if n <= 0:
        raise ValueError("No frames provided for TSDF integration.")
    if mask_paths is not None and int(len(mask_paths)) != n:
        raise ValueError(f"mask_paths length mismatch: got {len(mask_paths)}, expected {n}")

    med_dist = _estimate_med_camera_dist(camtoworlds)
    if voxel_length is None:
        voxel_length = float(med_dist) / 192.0
    if sdf_trunc is None:
        sdf_trunc = float(voxel_length) * 5.0
    print(f"[tsdf] voxel_length={float(voxel_length):.6g}, sdf_trunc={float(sdf_trunc):.6g}, frames={n}")

    # Resolution diagnostics (useful when running TSDF at reduced render_factor).
    # Most datasets use a constant resolution for all frames; if not, print all unique sizes.
    unique_hws = np.unique(np.asarray(hws, dtype=np.int32), axis=0)
    if int(unique_hws.shape[0]) == 1:
        h0, w0 = int(unique_hws[0, 0]), int(unique_hws[0, 1])
        print(f"[tsdf] integration resolution: H={h0}, W={w0}", flush=True)
    else:
        sizes = ", ".join(f"(H={int(h)},W={int(w)})" for h, w in unique_hws.tolist())
        print(f"[tsdf] integration resolutions (unique): {sizes}", flush=True)

    tsdf_vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    integration_depth_trunc = float(depth_trunc) if depth_trunc is not None else 10000.0
    use_mask = mask_paths is not None
    if use_mask:
        import cv2  # noqa: WPS433

        mask_threshold = 0.5
        if int(mask_dilate) < 0:
            raise ValueError(f"mask_dilate must be >= 0, got {mask_dilate}")
        kernel = None
        if int(mask_dilate) > 0:
            k = int(mask_dilate) * 2 + 1
            kernel = np.ones((k, k), dtype=np.uint8)

    bounds = None
    if aabb_bounds is not None:
        bounds = np.asarray(aabb_bounds, dtype=np.float32)
        if bounds.shape == (2, 3):
            bounds = bounds.T
        if bounds.shape != (3, 2):
            raise ValueError(f"aabb_bounds must have shape (3,2), got {bounds.shape}")
        if not np.all(np.isfinite(bounds)):
            raise ValueError("aabb_bounds contains non-finite values.")
        mins = bounds[:, 0]
        maxs = bounds[:, 1]
        if not np.all(maxs >= mins):
            raise ValueError(f"Invalid aabb_bounds (max < min): {bounds}")
        print(
            "[aabb] enabled: "
            f"x=[{mins[0]:.6g},{maxs[0]:.6g}], "
            f"y=[{mins[1]:.6g},{maxs[1]:.6g}], "
            f"z=[{mins[2]:.6g},{maxs[2]:.6g}]",
            flush=True,
        )

    aabb_cache: dict[tuple[int, int, float, float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    aabb_valid_before = 0
    aabb_valid_after = 0

    for i in tqdm.tqdm(range(n), desc="TSDF Integration", unit="frame"):
        depth = depths_np[i].astype(np.float32)  # [H,W]
        color = colors_np[i]
        if color.dtype != np.uint8:
            color = (np.clip(color, 0.0, 1.0) * 255.0).astype(np.uint8)

        h, w = int(hws[i, 0]), int(hws[i, 1])
        if depth.shape[0] != h or depth.shape[1] != w:
            raise ValueError(f"Cached depth shape mismatch: got {depth.shape}, expected {(h, w)}")
        if color.shape[0] != h or color.shape[1] != w:
            raise ValueError(f"Cached color shape mismatch: got {color.shape}, expected {(h, w, 3)}")

        if use_mask:
            mp = Path(str(mask_paths[i])).expanduser().resolve()
            m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if m is None:
                raise FileNotFoundError(f"Failed to read mask: {mp}")
            if m.shape[0] != h or m.shape[1] != w:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            alpha01 = _mask_to_alpha01(m)
            if kernel is not None:
                m_u8 = (alpha01 * 255.0).round().clip(0.0, 255.0).astype(np.uint8)
                m_u8 = cv2.dilate(m_u8, kernel, iterations=1)
                alpha01 = (m_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)
            depth[alpha01 < float(mask_threshold)] = 0.0

        if bounds is not None:
            K = Ks[i].astype(np.float32, copy=False)
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            if not (fx > 0.0 and fy > 0.0 and np.isfinite(fx) and np.isfinite(fy)):
                raise ValueError(f"Invalid intrinsics for aabb crop: fx={fx}, fy={fy}")

            key = (int(h), int(w), round(fx, 6), round(fy, 6), round(cx, 6), round(cy, 6))
            cached = aabb_cache.get(key)
            if cached is None:
                u = np.arange(int(w), dtype=np.float32)
                v = np.arange(int(h), dtype=np.float32)
                xdir = (u - float(cx)) / float(fx)  # [W]
                ydir = (v - float(cy)) / float(fy)  # [H]
                aabb_cache[key] = (xdir, ydir)
                xdir, ydir = xdir, ydir
            else:
                xdir, ydir = cached

            # World point: p_w = t + d * (R @ [xdir, ydir, 1]).
            c2w = camtoworlds[i].astype(np.float32, copy=False)
            R = c2w[:3, :3]
            t = c2w[:3, 3]

            valid_before = int(np.count_nonzero(depth > 0.0))
            keep = depth > 0.0

            coeff = (R[0, 0] * xdir[None, :]) + (R[0, 1] * ydir[:, None]) + float(R[0, 2])
            xw = float(t[0]) + depth * coeff
            keep &= (xw >= float(bounds[0, 0])) & (xw <= float(bounds[0, 1]))

            coeff = (R[1, 0] * xdir[None, :]) + (R[1, 1] * ydir[:, None]) + float(R[1, 2])
            yw = float(t[1]) + depth * coeff
            keep &= (yw >= float(bounds[1, 0])) & (yw <= float(bounds[1, 1]))

            coeff = (R[2, 0] * xdir[None, :]) + (R[2, 1] * ydir[:, None]) + float(R[2, 2])
            zw = float(t[2]) + depth * coeff
            keep &= (zw >= float(bounds[2, 0])) & (zw <= float(bounds[2, 1]))

            depth[~keep] = 0.0
            valid_after = int(np.count_nonzero(keep))
            aabb_valid_before += valid_before
            aabb_valid_after += valid_after

        o3d_depth = o3d.geometry.Image(depth)
        o3d_color = o3d.geometry.Image(color)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=1.0,
            depth_trunc=integration_depth_trunc,
            convert_rgb_to_intensity=False,
        )

        K = Ks[i]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w,
            height=h,
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
        )
        extrinsic_world_to_cam = np.linalg.inv(camtoworlds[i])
        tsdf_vol.integrate(rgbd, intrinsic, extrinsic_world_to_cam)

    if bounds is not None and aabb_valid_before > 0:
        frac = float(aabb_valid_after) / float(max(aabb_valid_before, 1))
        print(
            f"[aabb] kept depth pixels: {aabb_valid_after}/{aabb_valid_before} ({frac*100.0:.2f}%)",
            flush=True,
        )

    mesh = tsdf_vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def _post_process_mesh(*, mesh, cluster_to_keep: int):
    if int(cluster_to_keep) <= 0:
        return mesh
    mesh_0 = mesh
    (
        triangle_clusters,
        cluster_n_triangles,
        _cluster_area,
    ) = mesh_0.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if int(cluster_n_triangles.size) == 0:
        return mesh_0

    total_clusters = int(len(cluster_n_triangles))
    actual_keep = min(int(cluster_to_keep), total_clusters)
    n_cluster_threshold = np.sort(cluster_n_triangles.copy())[-actual_keep]
    n_cluster_threshold = max(int(n_cluster_threshold), 50)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster_threshold
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    return mesh_0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a TSDF mesh from a gsplat-style 3DGS PLY export."
    )
    parser.add_argument("--ply_path", type=str, required=True, help="Path to an uncompressed gsplat-style PLY.")
    parser.add_argument("--data_dir", type=str, required=True, help="COLMAP scene directory.")
    parser.add_argument(
        "--render_factor",
        "--resolution",
        "-r",
        dest="render_factor",
        type=int,
        default=1,
        help=(
            "Additional render downscale factor applied during TSDF meshing. "
            "For example, --render_factor 2 renders at half resolution. "
            "This affects both rasterization resolution and TSDF integration."
        ),
    )
    parser.add_argument("--interval", type=int, default=1, help="Render every N-th frame.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: <ply_dir>/../mesh).")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for rendering (e.g. cuda:0, cpu).")
    parser.add_argument("--sh_degree", type=int, default=-1, help="SH degree to render (-1 uses max from ckpt).")
    parser.add_argument("--voxel_length", type=float, default=None, help="TSDF voxel size in scene units.")
    parser.add_argument("--sdf_trunc", type=float, default=None, help="TSDF truncation distance in scene units.")
    parser.add_argument("--depth_trunc", type=float, default=None, help="Max depth for TSDF integration.")
    parser.add_argument(
        "--aabb_min",
        type=float,
        nargs=3,
        default=None,
        help="Optional AABB min corner in world coords: x y z (enables 3D bbox depth culling).",
    )
    parser.add_argument(
        "--aabb_max",
        type=float,
        nargs=3,
        default=None,
        help="Optional AABB max corner in world coords: x y z (enables 3D bbox depth culling).",
    )
    parser.add_argument(
        "--write_workers",
        type=int,
        default=4,
        help="Number of background workers for writing cached .npy files.",
    )
    parser.add_argument(
        "--queue_size",
        type=int,
        default=8,
        help="Maximum number of pending frames buffered for async write.",
    )
    parser.add_argument(
        "--post_process_clusters",
        type=int,
        default=50,
        help="Keep K largest connected components (0 disables).",
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for RGB/depth .npy files.")
    parser.add_argument(
        "--delete_cache",
        action="store_true",
        help="Delete the cached RGB/depth .npy directory after mesh extraction.",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help=(
            "Optional directory of per-frame object masks. When set, TSDF integration zeroes depth where mask<0.5. "
            "If the path is relative, it is interpreted relative to --data_dir. "
            "Masks are matched to images by integer filename id (e.g. 0000.png -> 000.png) or by exact stem."
        ),
    )
    parser.add_argument(
        "--mask_dilate",
        type=int,
        default=0,
        help="Optional mask dilation radius in pixels (0 disables).",
    )
    args = parser.parse_args()

    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    # Open3D is a compiled extension; in some environments it can crash (SIGSEGV)
    # during TSDF integration when used with NumPy 2.x. Prefer failing fast with a
    # clear message over crashing inside native code.
    numpy_major = int(str(np.__version__).split(".", maxsplit=1)[0])
    if numpy_major >= 2:
        raise RuntimeError(
            "TSDF meshing requires NumPy 1.x in this setup (Open3D may segfault with NumPy 2.x). "
            f"Detected numpy=={np.__version__}. "
            "Fix: reinstall with `pip install 'numpy<2' --force-reinstall`, then reinstall Open3D "
            "(e.g. `pip install open3d==0.18.0 --force-reinstall`)."
        )

    # Import gsplat lazily so we can print early diagnostics before the first
    # run potentially triggers CUDA extension compilation.
    print("[gsplat] importing rasterization (first run may compile CUDA extensions)...", flush=True)
    from gsplat.rendering import rasterization  # noqa: WPS433

    ply_path = Path(args.ply_path).expanduser().resolve()
    if not ply_path.is_file():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (ply_path.parent.parent if ply_path.parent.name == "ply" else ply_path.parent) / "mesh"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir is not None
        else output_dir / "cache"
    )
    if cache_dir.exists():
        print(f"[cache] clearing existing cache dir: {cache_dir}")
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
        else:
            cache_dir.unlink()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cache] dir={cache_dir} (write_workers={int(args.write_workers)}, queue_size={int(args.queue_size)})")

    data_dir = str(args.data_dir)
    mask_dir: Optional[Path] = None
    if args.mask_dir is not None:
        raw = Path(str(args.mask_dir)).expanduser()
        mask_dir = (Path(data_dir).expanduser().resolve() / raw).resolve() if not raw.is_absolute() else raw.resolve()

    dataparser = ColmapDataParser(
        data_dir=data_dir,
        factor=1,
        test_every=8,
        benchmark_train_split=False,
        depth_dir_name=None,
        normal_dir_name=None,
        dynamic_mask_dir_name=None,
        sky_mask_dir_name=None,
    )
    parsed_scene = dataparser.get_dataparser_outputs(split="train")
    if int(parsed_scene.indices.size) <= 0:
        raise ValueError(f"Train split is empty for data_dir={data_dir!r}.")
    print(f"[data] loaded train cameras: {int(parsed_scene.indices.size)} (interval={int(args.interval)})")
    first_image_index = int(parsed_scene.indices[0])
    h0, w0 = _infer_hw_from_image(image_path=parsed_scene.image_paths[first_image_index])
    render_factor = int(args.render_factor)
    if render_factor <= 0:
        raise ValueError(f"--render_factor must be > 0, got {render_factor}")
    if render_factor != 1:
        h_render = max(1, int(round(float(h0) / float(render_factor))))
        w_render = max(1, int(round(float(w0) / float(render_factor))))
    else:
        h_render, w_render = int(h0), int(w0)
    kx = float(w_render) / max(float(w0), 1.0)
    ky = float(h_render) / max(float(h0), 1.0)
    print(
        f"[render] render_factor={render_factor} (H,W)=({h_render},{w_render}) from ({h0},{w0})",
        flush=True,
    )

    mask_by_int: dict[int, Path] = {}
    mask_by_stem: dict[str, Path] = {}
    if mask_dir is not None:
        if not mask_dir.exists():
            raise FileNotFoundError(f"mask_dir not found: {mask_dir}")
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in sorted(mask_dir.glob(ext)):
                if p.name.startswith("."):
                    continue
                stem = p.stem
                if stem.isdigit():
                    mask_by_int[int(stem)] = p
                mask_by_stem[stem] = p
        if not mask_by_int and not mask_by_stem:
            raise FileNotFoundError(f"No mask images found under: {mask_dir}")

        print(
            f"[mask] enabled (dir={mask_dir}, threshold=0.5, dilate={int(args.mask_dilate)})",
            flush=True,
        )

    mask_paths: Optional[list[str]] = [] if mask_dir is not None else None

    aabb_bounds = None
    if args.aabb_min is not None or args.aabb_max is not None:
        if args.aabb_min is None or args.aabb_max is None:
            raise ValueError("Provide both --aabb_min and --aabb_max.")
        mins = np.asarray(args.aabb_min, dtype=np.float32).reshape(3)
        maxs = np.asarray(args.aabb_max, dtype=np.float32).reshape(3)
        aabb_bounds = np.stack([mins, maxs], axis=1)  # [3,2]

    splats = _load_ply_splats(str(ply_path))
    print(f"[ply] loaded {int(splats['means'].shape[0])} splats from: {ply_path}")
    gaussian_model = _build_gaussian_model(splats=splats, device=device)

    sh_degree = int(args.sh_degree)
    if sh_degree < 0:
        sh_degree = int(gaussian_model.max_sh_degree)

    # Materialize render-time tensors once. This is the main performance win
    # compared to calling `render_splats(...)` which re-derives these each frame.
    render_tensors = gaussian_model.to_render_tensors(sh_degree=int(sh_degree))
    means = render_tensors["means"]
    quats = render_tensors["quats"]
    scales = render_tensors["scales"]
    opacities = render_tensors["opacities"]
    sh_coeffs = render_tensors["colors"]

    rendered = 0
    color_paths: list[str] = []
    depth_paths: list[str] = []
    all_Ks: list[np.ndarray] = []
    all_c2w: list[np.ndarray] = []
    all_hws: list[tuple[int, int]] = []

    total_frames = int(parsed_scene.indices.size)
    total_to_render = int(math.ceil(float(total_frames) / float(max(int(args.interval), 1))))
    pbar = tqdm.tqdm(total=total_to_render, desc="Rendering RGB+Depth", unit="frame")

    write_workers = max(1, int(args.write_workers))
    write_queue_size = max(1, int(args.queue_size))
    write_q: "queue.Queue[Optional[_WriteItem]]" = queue.Queue(maxsize=write_queue_size)
    write_err: list[BaseException] = []
    write_err_lock = threading.Lock()
    write_stats = {"time": 0.0, "count": 0}
    write_stats_lock = threading.Lock()

    def _writer_loop(worker_id: int) -> None:
        while True:
            item = write_q.get()
            if item is None:
                write_q.task_done()
                return
            try:
                if item.done_event is not None:
                    item.done_event.synchronize()
                t0 = time.perf_counter()
                np.save(item.color_path, item.color_cpu.numpy())
                np.save(item.depth_path, item.depth_cpu.numpy())
                dt = time.perf_counter() - t0
                with write_stats_lock:
                    write_stats["time"] += float(dt)
                    write_stats["count"] += 1
            except BaseException as e:  # noqa: BLE001
                with write_err_lock:
                    write_err.append(e)
            finally:
                write_q.task_done()

    writer_threads = [
        threading.Thread(target=_writer_loop, args=(i,), daemon=True)
        for i in range(write_workers)
    ]
    for t in writer_threads:
        t.start()

    copy_stream: Optional["torch.cuda.Stream"] = None
    if device.type == "cuda":
        copy_stream = torch.cuda.Stream(device=device)

    render_t0 = time.perf_counter()
    with torch.no_grad():
        for frame in _iter_frames(
            parsed_scene=parsed_scene,
            split="train",
            default_hw=(h_render, w_render),
            K_scale_xy=(kx, ky),
            interval=int(args.interval),
        ):
            if mask_paths is not None:
                stem = Path(str(frame.image_path)).stem
                mp: Optional[Path] = None
                if stem.isdigit():
                    mp = mask_by_int.get(int(stem))
                if mp is None:
                    mp = mask_by_stem.get(stem)
                if mp is None:
                    raise FileNotFoundError(
                        f"Could not find mask for image={frame.image_path} (stem={stem!r}) under mask_dir={mask_dir}"
                    )
                mask_paths.append(str(mp))

            c2w = torch.from_numpy(frame.camtoworld).to(device=device, dtype=torch.float32)[None]
            K = torch.from_numpy(frame.K).to(device=device, dtype=torch.float32)[None]

            renders, _alphas, meta = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=sh_coeffs,
                viewmats=torch.linalg.inv(c2w),  # world-to-camera
                Ks=K,
                width=int(frame.width),
                height=int(frame.height),
                sh_degree=int(sh_degree),
                packed=False,
                rasterize_mode="classic",
                backgrounds=None,
                render_mode="RGB+ED",
                sparse_grad=False,
                absgrad=False,
                near_plane=0.01,
                far_plane=1e10,
                radius_clip=0.0,
                eps2d=0.3,
                camera_model="pinhole",
            )

            color = renders[0, ..., 0:3].clamp(0.0, 1.0)
            expected_depth = renders[0, ..., -1:]
            meta = meta if isinstance(meta, dict) else {}
            median_depth = meta.get("render_median")
            if isinstance(median_depth, torch.Tensor):
                depth = median_depth[0]
            elif isinstance(expected_depth, torch.Tensor):
                depth = expected_depth
            else:
                raise RuntimeError("Renderer did not return depth (expected_depth/meta['render_median']).")

            color_path = str(cache_dir / f"color_{rendered:06d}.npy")
            depth_path = str(cache_dir / f"depth_{rendered:06d}.npy")

            # Producer/consumer pipeline:
            # - render (GPU)
            # - quantize + async D2H copy into pinned CPU tensors (GPU->CPU)
            # - background thread writes .npy to disk (CPU I/O)
            if device.type == "cuda":
                assert copy_stream is not None
                color_u8 = (color * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
                depth_f16 = depth.squeeze(-1).to(torch.float16)
                color_cpu = torch.empty_like(color_u8, device="cpu", pin_memory=True)
                depth_cpu = torch.empty_like(depth_f16, device="cpu", pin_memory=True)

                copy_stream.wait_stream(torch.cuda.current_stream(device=device))
                with torch.cuda.stream(copy_stream):
                    # Make allocator stream-aware: these source tensors are used on copy_stream.
                    color_u8.record_stream(copy_stream)
                    depth_f16.record_stream(copy_stream)
                    color_cpu.copy_(color_u8, non_blocking=True)
                    depth_cpu.copy_(depth_f16, non_blocking=True)
                    done = torch.cuda.Event()
                    done.record(copy_stream)
                write_q.put(
                    _WriteItem(
                        done_event=done,
                        color_cpu=color_cpu,
                        depth_cpu=depth_cpu,
                        color_path=color_path,
                        depth_path=depth_path,
                    )
                )
            else:
                color_np = (color.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
                depth_np = depth.detach().cpu().numpy().squeeze(-1).astype(np.float16)
                color_cpu = torch.from_numpy(color_np)
                depth_cpu = torch.from_numpy(depth_np)
                write_q.put(
                    _WriteItem(
                        done_event=None,
                        color_cpu=color_cpu,
                        depth_cpu=depth_cpu,
                        color_path=color_path,
                        depth_path=depth_path,
                    )
                )

            color_paths.append(color_path)
            depth_paths.append(depth_path)
            all_Ks.append(frame.K.astype(np.float32))
            all_c2w.append(frame.camtoworld.astype(np.float32))
            all_hws.append((int(frame.height), int(frame.width)))
            rendered += 1
            pbar.update(1)

    pbar.close()
    render_dt = time.perf_counter() - render_t0
    print(f"[render] rendered={rendered} frames in {render_dt:.2f}s ({(render_dt/max(rendered,1)):.3f}s/frame)")

    # Flush pending disk writes before TSDF integration.
    for _ in range(write_workers):
        write_q.put(None)
    write_q.join()
    for t in writer_threads:
        t.join()
    if write_err:
        raise RuntimeError(f"Failed to write cached frames (first error): {write_err[0]!r}")
    if write_stats["count"] > 0:
        avg = float(write_stats["time"]) / float(write_stats["count"])
        print(f"[cache] wrote {int(write_stats['count'])} frames to disk (avg {avg:.4f}s/frame)")

    if rendered <= 0:
        raise RuntimeError("No frames rendered. Check --interval and --data_dir.")

    # Release GPU memory before TSDF (Open3D TSDF is CPU-side).
    del gaussian_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    depths_np = DiskNpyArray(depth_paths)
    colors_np = DiskNpyArray(color_paths)
    Ks_np = np.stack(all_Ks, axis=0)
    c2w_np = np.stack(all_c2w, axis=0)
    hws_np = np.asarray(all_hws, dtype=np.int32)

    mesh_raw = _create_tsdf_mesh(
        depths_np=depths_np,
        colors_np=colors_np,
        Ks=Ks_np,
        camtoworlds=c2w_np,
        hws=hws_np,
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        depth_trunc=args.depth_trunc,
        mask_paths=mask_paths,
        mask_dilate=int(args.mask_dilate),
        aabb_bounds=aabb_bounds,
    )
    out_raw = output_dir / "tsdf_mesh.ply"
    o3d.io.write_triangle_mesh(str(out_raw), mesh_raw)
    print(f"Saved mesh: {out_raw}")

    mesh_post = _post_process_mesh(mesh=mesh_raw, cluster_to_keep=int(args.post_process_clusters))
    mesh_post.compute_vertex_normals()

    out_post = output_dir / "tsdf_mesh_post.ply"
    o3d.io.write_triangle_mesh(str(out_post), mesh_post)
    print(f"Saved mesh: {out_post}")

    if bool(args.delete_cache):
        try:
            inside_output = cache_dir.resolve().is_relative_to(output_dir.resolve())
        except Exception:  # noqa: BLE001
            inside_output = False
        if not inside_output:
            raise RuntimeError(
                f"Refusing to delete cache_dir outside output_dir (cache_dir={cache_dir}, output_dir={output_dir})."
            )
        if cache_dir.exists() and cache_dir.is_dir():
            shutil.rmtree(cache_dir)
            print(f"[cache] deleted cache dir: {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
