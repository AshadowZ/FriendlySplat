from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from friendly_splat.data.colmap_dataparser import ColmapDataParser
from friendly_splat.modules.gaussian import GaussianModel
from gsplat.rendering import rasterization
from tools.segment.gauscluster_core import (
    HAS_CUML,
    export_color_cluster,
    iterative_cluster_masks,
    post_process_clusters,
    read_mask,
    remedy_undersegment,
)


@dataclass(frozen=True)
class ResolvedContext:
    source_kind: str
    source_paths: list[str]
    data_dir: str
    results_dir: Path
    normalize_world_space: bool
    align_world_axes: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instance clustering for FriendlySplat Gaussian scenes.",
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="FriendlySplat result directory containing ckpts/ or ply/.",
    )
    source.add_argument(
        "--ckpt-paths",
        type=str,
        nargs="+",
        default=None,
        help="One or more checkpoint paths containing FriendlySplat splats.",
    )
    source.add_argument(
        "--ply-paths",
        type=str,
        nargs="+",
        default=None,
        help="One or more uncompressed splat PLY paths.",
    )
    parser.add_argument(
        "--prefer-source",
        choices=("ckpt", "ply"),
        default="ckpt",
        help="When --result-dir is used, prefer this source type first.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Optional 1-based step to load from result-dir.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="COLMAP scene root. If omitted with --result-dir, try to infer from cfg.yml.",
    )
    parser.add_argument(
        "--data-factor",
        type=float,
        default=1.0,
        help="Image downscale factor. Must match the mask resolution used for segmentation.",
    )
    parser.add_argument(
        "--normalize-world-space",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Match training-time world normalization. When omitted with --result-dir, "
            "inherit from result_dir/cfg.yml if available."
        ),
    )
    parser.add_argument(
        "--align-world-axes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Match training-time axis alignment. When omitted with --result-dir, "
            "inherit from result_dir/cfg.yml if available."
        ),
    )
    parser.add_argument(
        "--test-every",
        type=int,
        default=8,
        help="Same train/test split period as FriendlySplat training.",
    )
    parser.add_argument(
        "--benchmark-train-split",
        action="store_true",
        help="Use benchmark train split behavior from FriendlySplat.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="train",
        help="Which camera split to cluster over.",
    )
    parser.add_argument(
        "--mask-dir-name",
        type=str,
        default=None,
        help="Optional subfolder name under <data_dir>/sam/ containing masks.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for rasterization, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--radius-clip",
        type=float,
        default=3.0,
        help="Radius clip passed to gsplat rasterization while building correspondences.",
    )
    parser.add_argument(
        "--pixel-gaussian-threshold",
        type=float,
        default=0.25,
        help="Per-pixel Gaussian contribution threshold for correspondence tracking.",
    )
    parser.add_argument(
        "--max-gaussians-per-pixel",
        type=int,
        default=100,
        help="Maximum stored Gaussian ids per pixel while tracking correspondences.",
    )
    parser.add_argument(
        "--point-filter-threshold",
        type=float,
        default=0.5,
        help="Post-filter threshold on per-point detection consistency.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.1,
        help="DBSCAN epsilon for 3D point clustering.",
    )
    parser.add_argument(
        "--dbscan-min-points",
        type=int,
        default=4,
        help="DBSCAN min_points for 3D point clustering.",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.8,
        help="Overlap threshold used when removing duplicate objects.",
    )
    parser.add_argument(
        "--undersegment-threshold",
        type=float,
        default=0.8,
        help="Threshold used when assigning under-segmented masks back to instances.",
    )
    parser.add_argument(
        "--use-gpu-dbscan",
        action="store_true",
        help="Use cuML DBSCAN during post-processing if available.",
    )
    return parser.parse_args()


def _load_result_cfg(result_dir: Path) -> dict | None:
    cfg_path = result_dir / "cfg.yml"
    if not cfg_path.is_file():
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else None


def _resolve_bool_option(
    explicit_value: bool | None,
    cfg_section: dict | None,
    key: str,
) -> bool:
    if explicit_value is not None:
        return bool(explicit_value)
    if isinstance(cfg_section, dict):
        cfg_value = cfg_section.get(key)
        if isinstance(cfg_value, bool):
            return cfg_value
    return False


def _find_result_source(
    result_dir: Path,
    prefer_source: str,
    step: int | None,
) -> tuple[str, list[str]]:
    def _find_ckpts() -> list[str]:
        ckpt_dir = result_dir / "ckpts"
        if not ckpt_dir.is_dir():
            return []
        if step is not None:
            path = ckpt_dir / f"ckpt_step{int(step):06d}.pt"
            return [str(path)] if path.is_file() else []
        return [str(p) for p in sorted(ckpt_dir.glob("ckpt_step*.pt"))[-1:]]

    def _find_plys() -> list[str]:
        ply_dir = result_dir / "ply"
        if not ply_dir.is_dir():
            return []
        if step is not None:
            path = ply_dir / f"splats_step{int(step):06d}.ply"
            return [str(path)] if path.is_file() else []
        return [str(p) for p in sorted(ply_dir.glob("splats_step*.ply"))[-1:]]

    ordered = ["ckpt", "ply"] if prefer_source == "ckpt" else ["ply", "ckpt"]
    for source_kind in ordered:
        paths = _find_ckpts() if source_kind == "ckpt" else _find_plys()
        if paths:
            return source_kind, paths
    raise FileNotFoundError(
        f"No usable {'/'.join(ordered)} source found under result_dir={result_dir}"
    )


def _infer_data_dir(cfg: dict | None) -> str | None:
    if not isinstance(cfg, dict):
        return None
    io_cfg = cfg.get("io")
    if not isinstance(io_cfg, dict):
        return None
    data_dir = io_cfg.get("data_dir")
    if not isinstance(data_dir, str) or len(data_dir.strip()) == 0:
        return None
    return data_dir


def _infer_results_dir(source_kind: str, source_paths: Sequence[str]) -> Path:
    if len(source_paths) == 0:
        raise ValueError("No source paths were resolved.")
    first_source = Path(source_paths[0]).expanduser().resolve()
    parent = first_source.parent
    if source_kind == "ckpt" and parent.name == "ckpts":
        return parent.parent
    if source_kind == "ply" and parent.name == "ply":
        return parent.parent
    return parent


def resolve_context(args: argparse.Namespace) -> ResolvedContext:
    data_dir = args.data_dir
    result_cfg = None
    if args.result_dir is not None:
        result_dir = Path(args.result_dir).expanduser().resolve()
        if not result_dir.is_dir():
            raise FileNotFoundError(f"result_dir not found: {result_dir}")
        result_cfg = _load_result_cfg(result_dir)
        source_kind, source_paths = _find_result_source(
            result_dir=result_dir,
            prefer_source=str(args.prefer_source),
            step=args.step,
        )
        if data_dir is None:
            data_dir = _infer_data_dir(result_cfg)
        if data_dir is None:
            raise ValueError(
                "--data-dir is required when it cannot be inferred from result_dir/cfg.yml."
            )
        results_dir = result_dir
    elif args.ckpt_paths is not None:
        if data_dir is None:
            raise ValueError("--data-dir is required with --ckpt-paths.")
        source_kind = "ckpt"
        source_paths = [str(Path(p).expanduser().resolve()) for p in args.ckpt_paths]
        results_dir = _infer_results_dir(source_kind, source_paths)
    elif args.ply_paths is not None:
        if data_dir is None:
            raise ValueError("--data-dir is required with --ply-paths.")
        source_kind = "ply"
        source_paths = [str(Path(p).expanduser().resolve()) for p in args.ply_paths]
        results_dir = _infer_results_dir(source_kind, source_paths)
    else:
        raise ValueError(
            "Specify one of --result-dir, --ckpt-paths, or --ply-paths."
        )

    data_cfg = result_cfg.get("data") if isinstance(result_cfg, dict) else None
    normalize_world_space = False
    align_world_axes = False
    if source_kind == "ckpt":
        normalize_world_space = _resolve_bool_option(
            args.normalize_world_space,
            data_cfg,
            "normalize_world_space",
        )
        align_world_axes = _resolve_bool_option(
            args.align_world_axes,
            data_cfg,
            "align_world_axes",
        )
    else:
        normalize_world_space = bool(args.normalize_world_space)
        align_world_axes = bool(args.align_world_axes)

    return ResolvedContext(
        source_kind=source_kind,
        source_paths=list(source_paths),
        data_dir=str(Path(data_dir).expanduser().resolve()),
        results_dir=results_dir,
        normalize_world_space=normalize_world_space,
        align_world_axes=align_world_axes,
    )


def discover_mask_dir(data_dir: str, override: str | None = None) -> str:
    sam_root = Path(data_dir) / "sam"
    if not sam_root.is_dir():
        raise FileNotFoundError(f"sam directory not found under {data_dir}")

    candidates: list[Path] = []
    if override is not None:
        candidates.append(sam_root / override)
    candidates.extend(
        [
            sam_root / "mask_sorted",
            sam_root / "mask",
            sam_root / "mask_filtered",
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            print(f"[Masks] Using SAM masks from: {candidate}")
            return str(candidate)
    raise FileNotFoundError(f"No mask directory found inside {sam_root}")


def collect_view_data(
    dataparser: ColmapDataParser,
    split: str,
    mask_dir: str,
) -> list[Dict[str, object]]:
    if split == "all":
        indices = np.arange(len(dataparser.image_names), dtype=np.int64)
    else:
        parsed_scene = dataparser.get_dataparser_outputs(split=split)
        indices = parsed_scene.indices

    view_data: list[Dict[str, object]] = []
    pbar = tqdm(indices, total=len(indices), desc="[Cameras] Preparing views")
    for local_frame_idx, global_idx in enumerate(pbar):
        image_path = dataparser.image_paths[int(global_idx)]
        image_name = Path(image_path).stem
        width, height = dataparser.image_sizes[int(global_idx)]
        K = dataparser.Ks[int(global_idx)]
        camtoworld = dataparser.camtoworlds[int(global_idx)]

        mask_path = None
        for ext in (".png", ".jpg", ".jpeg", ".npy"):
            candidate = Path(mask_dir) / f"{image_name}{ext}"
            if candidate.exists():
                mask_path = str(candidate)
                break
        if mask_path is None:
            raise FileNotFoundError(f"No mask found for {image_name} in {mask_dir}")

        mask_np = read_mask(mask_path)
        mask_h, mask_w = mask_np.shape
        if (mask_w, mask_h) != (int(width), int(height)):
            raise ValueError(
                f"Mask size {mask_h}x{mask_w} does not match camera resolution "
                f"{int(height)}x{int(width)} for {mask_path}."
            )

        view_data.append(
            {
                "frame_idx": int(local_frame_idx),
                "global_idx": int(global_idx),
                "image_name": image_name,
                "image_path": image_path,
                "width": int(width),
                "height": int(height),
                "K": torch.from_numpy(K).float(),
                "camtoworld": torch.from_numpy(camtoworld).float(),
                "mask_path": mask_path,
            }
        )
        pbar.set_postfix_str(f"{local_frame_idx + 1}/{len(indices)} current: {image_name}")

    return view_data


def load_render_gaussians(
    source_kind: str,
    source_paths: Sequence[str],
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
    render_tensors_list: list[Dict[str, torch.Tensor]] = []
    max_degrees: list[int] = []

    print(f"[Gaussians] Loading {len(source_paths)} source file(s) on {device} ...")
    for source_path in source_paths:
        if source_kind == "ckpt":
            model = GaussianModel.from_ckpt(
                ckpt_path=str(source_path),
                device=device,
                requires_grad=False,
            )
        elif source_kind == "ply":
            model = GaussianModel.from_splat_ply(
                ply_path=str(source_path),
                device=device,
                requires_grad=False,
            )
        else:
            raise ValueError(f"Unsupported source_kind={source_kind!r}")
        max_degree = int(model.max_sh_degree)
        max_degrees.append(max_degree)
        render_tensors_list.append(model.to_render_tensors(sh_degree=max_degree))
        print(f"  - Loaded {source_path}")

    if len(set(max_degrees)) != 1:
        raise ValueError(
            f"All input sources must share the same max SH degree, got {max_degrees}."
        )
    sh_degree = max_degrees[0]
    keys = ("means", "quats", "scales", "opacities", "colors")
    merged = {
        key: torch.cat([item[key] for item in render_tensors_list], dim=0).contiguous()
        for key in keys
    }
    point_positions = merged["means"].detach().cpu()
    print(
        f"[Gaussians] Total={merged['means'].shape[0]}, "
        f"feature_dim={(sh_degree + 1) ** 2}, sh_degree={sh_degree}"
    )
    return merged, point_positions, sh_degree


def build_mask_gaussian_tracker(
    render_tensors: Dict[str, torch.Tensor],
    sh_degree: int,
    view_data: list[Dict[str, object]],
    device: torch.device,
    radius_clip: float,
    pixel_gaussian_threshold: float,
    max_gaussians_per_pixel: int,
) -> Dict:
    means = render_tensors["means"]
    quats = render_tensors["quats"]
    scales = render_tensors["scales"]
    opacities = render_tensors["opacities"]
    colors = render_tensors["colors"]

    num_points = int(means.shape[0])
    num_frames = len(view_data)
    gaussian_in_frame_matrix = np.zeros((num_points, num_frames), dtype=bool)
    mask_gaussian_pclds: Dict[str, np.ndarray] = {}
    global_frame_mask_list: list[tuple[int, int]] = []

    pbar = tqdm(view_data, desc="Build mask->Gaussian tracking", total=num_frames)
    for frame_idx, view in enumerate(pbar):
        width = int(view["width"])
        height = int(view["height"])
        viewmats = torch.linalg.inv(
            torch.as_tensor(view["camtoworld"], device=device)
        ).unsqueeze(0)
        Ks = torch.as_tensor(view["K"], device=device).unsqueeze(0)

        with torch.no_grad():
            _renders, _alphas, meta = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmats,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=int(sh_degree),
                render_mode="RGB",
                packed=False,
                radius_clip=float(radius_clip),
                distributed=False,
                with_ut=False,
                with_eval3d=False,
                track_pixel_gaussians=True,
                max_gaussians_per_pixel=int(max_gaussians_per_pixel),
                pixel_gaussian_threshold=float(pixel_gaussian_threshold),
            )

        pixel_gaussians = meta.get("pixel_gaussians")
        if pixel_gaussians is None or pixel_gaussians.numel() == 0:
            raise RuntimeError(
                "Pixel-to-Gaussian correspondences are unavailable for "
                f"view={view['image_name']}. Ensure your gsplat build supports "
                "track_pixel_gaussians."
            )

        gaus_ids = pixel_gaussians[:, 0].long()
        pixel_ids = pixel_gaussians[:, 1].long()

        segmap_np = read_mask(str(view["mask_path"]))
        segmap = torch.from_numpy(segmap_np).long().to(device)
        labels = segmap.view(-1)[pixel_ids]
        valid_mask = labels > 0
        if int(valid_mask.sum().item()) == 0:
            continue

        gaus_ids = gaus_ids[valid_mask]
        labels = labels[valid_mask]

        unique_labels = labels.unique()
        frame_gauss_set: set[int] = set()
        for lbl in unique_labels:
            lbl_int = int(lbl.item())
            lbl_mask = labels == lbl
            lbl_gauss = torch.unique(gaus_ids[lbl_mask]).cpu().numpy()
            mask_gaussian_pclds[f"{frame_idx}_{lbl_int}"] = lbl_gauss
            global_frame_mask_list.append((frame_idx, lbl_int))
            frame_gauss_set.update(int(x) for x in lbl_gauss.tolist())

        if len(frame_gauss_set) > 0:
            gaussian_in_frame_matrix[list(frame_gauss_set), frame_idx] = True

    return {
        "gaussian_in_frame_matrix": gaussian_in_frame_matrix,
        "mask_gaussian_pclds": mask_gaussian_pclds,
        "global_frame_mask_list": global_frame_mask_list,
    }


def main() -> None:
    args = parse_args()
    context = resolve_context(args)
    output_dir = context.results_dir / "cluster_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_gpu_dbscan and not HAS_CUML:
        raise RuntimeError(
            "Detected --use-gpu-dbscan but cuML is missing; install cuML or remove the flag."
        )

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    render_tensors, point_positions, sh_degree = load_render_gaussians(
        source_kind=context.source_kind,
        source_paths=context.source_paths,
        device=device,
    )

    print(f"[Dataset] Loading COLMAP data from {context.data_dir}")
    dataparser = ColmapDataParser(
        data_dir=context.data_dir,
        factor=float(args.data_factor),
        normalize_world_space=context.normalize_world_space,
        align_world_axes=context.align_world_axes,
        test_every=int(args.test_every),
        benchmark_train_split=bool(args.benchmark_train_split),
    )
    mask_dir = discover_mask_dir(context.data_dir, args.mask_dir_name)
    view_data = collect_view_data(dataparser, args.split, mask_dir)
    print(f"[Summary] Prepared {len(view_data)} camera views with SAM masks.")

    print("\n================ Data validation ================")
    print(f"Data directory: {context.data_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Split: {args.split}")
    print(f"Source kind: {context.source_kind}")
    print(f"Results directory: {context.results_dir}")
    print(f"Number of cameras: {len(view_data)}")
    print(f"Number of Gaussians: {render_tensors['means'].shape[0]}")
    print(f"SH degree: {sh_degree}")
    print("----------------------------------------")
    print("Sample views:")
    for sample in view_data[:3]:
        print(
            f"  - {sample['image_name']} | Resolution {sample['width']}x{sample['height']} | "
            f"mask: {sample['mask_path']}"
        )
    print("========================================\n")

    tracker = build_mask_gaussian_tracker(
        render_tensors=render_tensors,
        sh_degree=sh_degree,
        view_data=view_data,
        device=device,
        radius_clip=float(args.radius_clip),
        pixel_gaussian_threshold=float(args.pixel_gaussian_threshold),
        max_gaussians_per_pixel=int(args.max_gaussians_per_pixel),
    )
    total_masks = len(tracker["global_frame_mask_list"])
    total_points_hit = int(tracker["gaussian_in_frame_matrix"].sum())
    print("============== Tracking statistics (Stage 1) ==============")
    print(f"Valid mask count: {total_masks}")
    print(f"Total Gaussians with pixel contributions: {total_points_hit}")
    for frame_id, mask_id in tracker["global_frame_mask_list"][:5]:
        gs_ids = tracker["mask_gaussian_pclds"][f"{frame_id}_{mask_id}"]
        print(f"  - frame {frame_id:03d}, mask {mask_id}: Gaussian count {len(gs_ids)}")
    print("=============================================")

    print("\nStarting iterative clustering (Stage 2)...")
    clustering_result = iterative_cluster_masks(tracker)
    print("Clustering completed.")
    print(f"Instance count (post-clustering nodes): {len(clustering_result['nodes'])}")

    print("\nStarting post-processing (Stage 3: DBSCAN + point filtering)...")
    clustering_result = post_process_clusters(
        clustering_result,
        point_positions=point_positions,
        point_filter_threshold=float(args.point_filter_threshold),
        dbscan_eps=float(args.dbscan_eps),
        dbscan_min_points=int(args.dbscan_min_points),
        overlap_ratio=float(args.overlap_ratio),
        use_gpu_dbscan=bool(args.use_gpu_dbscan),
    )
    print("Post-processing completed.")
    print(
        f"Instance count (post-processing): {len(clustering_result['total_point_ids_list'])}"
    )

    print("\nStarting under-segmented mask repair (Stage 4)...")
    clustering_result = remedy_undersegment(
        clustering_result,
        threshold=float(args.undersegment_threshold),
    )
    print("Repair completed.")
    print(f"Final instance count: {len(clustering_result['total_point_ids_list'])}")

    export_color_cluster(
        clustering_result,
        point_positions=point_positions,
        save_dir=output_dir,
    )
    print(f"[Done] Outputs written to {output_dir}")


def entrypoint() -> None:
    main()


if __name__ == "__main__":
    entrypoint()
