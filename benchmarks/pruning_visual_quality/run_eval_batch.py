#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

# Allow running as a standalone script from the repo root without installation.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class DatasetDef:
    key: str
    dir_name: str
    scenes: tuple[str, ...]


_DATASETS: dict[str, DatasetDef] = {
    "mipnerf360": DatasetDef(
        key="mipnerf360",
        dir_name="Mip-NeRF360",
        scenes=(
            "bicycle",
            "bonsai",
            "counter",
            "flowers",
            "garden",
            "kitchen",
            "room",
            "stump",
            "treehill",
        ),
    ),
    "tanksandtemples_vis": DatasetDef(
        key="tanksandtemples_vis",
        dir_name="Tanks&Temples-Vis",
        scenes=("train", "truck"),
    ),
    "deepblending": DatasetDef(
        key="deepblending",
        dir_name="DeepBlending",
        scenes=("drjohnson", "playroom"),
    ),
}


def _split_csv(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    items: list[str] = []
    for part in value.split(","):
        s = part.strip()
        if not s:
            continue
        items.append(s)
    return items


def _iter_selected_scenes(
    *,
    dataset: DatasetDef,
    include_scenes: set[str],
    exclude_scenes: set[str],
) -> Iterable[str]:
    for scene in dataset.scenes:
        if scene in exclude_scenes:
            continue
        if include_scenes and scene not in include_scenes:
            continue
        yield scene


def _infer_data_factor_from_scene_dir(scene_data_dir: Path) -> Optional[int]:
    if (scene_data_dir / "images").exists():
        return 1
    for factor in (2, 4, 8):
        if (scene_data_dir / f"images_{factor}").exists():
            return factor
    return None


def _read_cfg_snapshot(scene_out_dir: Path) -> Optional[dict]:
    cfg_path = scene_out_dir / "cfg.yml"
    if not cfg_path.exists():
        return None
    if yaml is None:
        raise ModuleNotFoundError(
            "Missing dependency 'pyyaml'. Install it to enable cfg.yml parsing."
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return None
    return data


_PLY_STEP_RE = re.compile(r"^splats_step(?P<step>\\d+)\\.ply$")


def _find_latest_ply_path(scene_out_dir: Path) -> tuple[Path, int]:
    ply_dir = scene_out_dir / "ply"
    if not ply_dir.exists():
        raise FileNotFoundError(f"Missing PLY directory: {ply_dir}")
    best: Optional[tuple[int, Path]] = None
    for p in ply_dir.iterdir():
        if not p.is_file():
            continue
        m = _PLY_STEP_RE.match(p.name)
        if not m:
            continue
        step = int(m.group("step"))
        if best is None or step > best[0]:
            best = (step, p)
    if best is None:
        raise FileNotFoundError(f"No PLY files found under: {ply_dir}")
    return best[1], best[0]


def _find_ply_path(*, scene_out_dir: Path, step: int) -> Path:
    ply_path = scene_out_dir / "ply" / f"splats_step{int(step):06d}.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing PLY: {ply_path}")
    return ply_path


def _load_ply_splats(ply_path: Path) -> dict[str, torch.Tensor]:
    """Load splat tensors from an uncompressed gsplat-style PLY.

    This matches the format produced by `gsplat.export_splats(..., format="ply")`
    (and by FriendlySplat PLY exports).
    """
    with open(str(ply_path), "rb") as f:
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
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                continue
            if parts[0] == "property" and in_vertex:
                vertex_props.append(parts[-1])

        if fmt is None or fmt.strip() != "binary_little_endian 1.0":
            raise ValueError(
                f"Unsupported PLY format: {fmt!r}. Only 'binary_little_endian 1.0' is supported."
            )
        if vertex_count is None or int(vertex_count) <= 0:
            raise ValueError(f"Invalid vertex count: {vertex_count!r}.")
        if not vertex_props:
            raise ValueError("PLY has no vertex properties.")

        prop_to_idx = {name: i for i, name in enumerate(vertex_props)}

        def _req(name: str) -> int:
            if name not in prop_to_idx:
                raise KeyError(f"PLY is missing required vertex property {name!r}.")
            return int(prop_to_idx[name])

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
        import numpy as np

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
    from friendly_splat.modules.gaussian import GaussianModel

    params = {k: torch.nn.Parameter(v.to(device=device)) for k, v in splats.items()}
    model = GaussianModel(params)
    model.eval()
    return model


def _read_train_time_s(scene_out_dir: Path) -> Optional[float]:
    tb_dir = scene_out_dir / "tb"
    if tb_dir.exists():
        try:
            from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore
                EventAccumulator,
            )

            ea = EventAccumulator(str(tb_dir))
            ea.Reload()
            events = ea.Scalars("train/total_time_s")
            if events:
                return float(events[-1].value)
        except Exception:
            pass

    log_path = scene_out_dir / "logs" / "train.log"
    if log_path.exists():
        pat = re.compile(r"Training time: wall=(?P<sec>[0-9.]+)s")
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = pat.search(line)
                    if m:
                        return float(m.group("sec"))
        except Exception:
            pass
    return None


def _rotmat_to_quat_wxyz(rot: torch.Tensor) -> torch.Tensor:
    """Convert a single 3x3 rotation matrix to a wxyz quaternion."""
    r = rot.detach().cpu().double()
    t = float(r[0, 0] + r[1, 1] + r[2, 2])
    if t > 0.0:
        s = torch.sqrt(torch.tensor(t + 1.0)).item() * 2.0
        w = 0.25 * s
        x = float(r[2, 1] - r[1, 2]) / s
        y = float(r[0, 2] - r[2, 0]) / s
        z = float(r[1, 0] - r[0, 1]) / s
    elif float(r[0, 0]) > float(r[1, 1]) and float(r[0, 0]) > float(r[2, 2]):
        s = torch.sqrt(torch.tensor(1.0 + float(r[0, 0]) - float(r[1, 1]) - float(r[2, 2]))).item() * 2.0
        w = float(r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = float(r[0, 1] + r[1, 0]) / s
        z = float(r[0, 2] + r[2, 0]) / s
    elif float(r[1, 1]) > float(r[2, 2]):
        s = torch.sqrt(torch.tensor(1.0 + float(r[1, 1]) - float(r[0, 0]) - float(r[2, 2]))).item() * 2.0
        w = float(r[0, 2] - r[2, 0]) / s
        x = float(r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = float(r[1, 2] + r[2, 1]) / s
    else:
        s = torch.sqrt(torch.tensor(1.0 + float(r[2, 2]) - float(r[0, 0]) - float(r[1, 1]))).item() * 2.0
        w = float(r[1, 0] - r[0, 1]) / s
        x = float(r[0, 2] + r[2, 0]) / s
        y = float(r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = torch.tensor([w, x, y, z], dtype=rot.dtype, device=rot.device)
    return q / torch.linalg.norm(q).clamp(min=1e-12)


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication in wxyz convention: a ⊗ b."""
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([w, x, y, z], dim=-1)


def _apply_colmap_to_train_transform_inplace(
    *,
    splats: dict[str, torch.Tensor],
    transform_colmap_to_train: torch.Tensor,
) -> None:
    """Map PLY-exported splats (COLMAP coords) back into training coords.

    FriendlySplat currently exports PLY geometry (means/quats/scales) in COLMAP coordinates
    but keeps SH coefficients as-is. Since SH evaluation uses world-space view directions,
    evaluating with COLMAP cameras would require rotating SH coefficients as well.

    For benchmarking, we instead evaluate in the training coordinate system by:
      - mapping means/quats/scales back using the same similarity transform used by the dataparser
      - leaving SH coefficients unchanged
    """
    T = transform_colmap_to_train.to(device=splats["means"].device, dtype=splats["means"].dtype)
    A = T[:3, :3]
    t = T[:3, 3]

    # means: [N,3] row-vector convention in dataparser.
    splats["means"] = splats["means"] @ A.T + t

    # uniform similarity scale (columns have norm == scale).
    col_norms = torch.linalg.norm(A, dim=0)
    length_scale = float(col_norms.mean().item())
    splats["scales"] = splats["scales"] + float(torch.log(torch.tensor(max(length_scale, 1e-12))).item())

    rot = A / float(max(length_scale, 1e-12))
    q_rot = _rotmat_to_quat_wxyz(rot)
    splats["quats"] = _quat_mul_wxyz(q_rot, splats["quats"])


@dataclass(frozen=True)
class _EvalCfg:
    eval: EvalConfig
    optim: OptimConfig
    strategy: StrategyConfig


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run_eval_batch.py",
        description=(
            "Batch evaluator for FriendlySplat visual quality metrics (PSNR/SSIM/LPIPS) "
            "with extra columns for #Gaussians and training time (pruning benchmark outputs)."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Dataset root directory (contains Mip-NeRF360/, Tanks&Temples-Vis/, DeepBlending/...).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset keys to run (or 'all').",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="Optional comma-separated scene names to include.",
    )
    parser.add_argument(
        "--exclude-scenes",
        type=str,
        default=None,
        help="Optional comma-separated scene names to exclude.",
    )
    parser.add_argument(
        "--pruners",
        type=str,
        default="all",
        help="Comma-separated methods to evaluate: pure_densify,gns,speedy (or 'all').",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default=None,
        help=(
            "Output markdown filename under <data-root>/pruning_benchmark/. "
            "Defaults to 'summary.md'."
        ),
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help=(
            "Optional training step to evaluate (selects ply/splats_stepXXXXXX.ply). "
            "When omitted, uses the latest exported PLY under the scene output directory."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for rendering/metrics (e.g. cuda:0).",
    )
    parser.add_argument(
        "--eval-preload",
        type=str,
        choices=("none", "cuda"),
        default="cuda",
        help="DataLoader preload mode for evaluation.",
    )
    parser.add_argument(
        "--metrics-backend",
        type=str,
        choices=("inria", "gsplat"),
        default="inria",
        help="Metric backend to use during benchmarking (default: inria).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on evaluated images per scene.",
    )
    args = parser.parse_args(argv)
    if torch is None:
        raise ModuleNotFoundError(
            "Missing dependency 'torch'. Please run this script inside the FriendlySplat environment."
        )

    from friendly_splat.data.colmap_dataparser import ColmapDataParser
    from friendly_splat.data.dataloader import DataLoader
    from friendly_splat.data.dataset import InputDataset
    from friendly_splat.trainer.configs import EvalConfig, OptimConfig, StrategyConfig
    from friendly_splat.trainer.eval_runtime import build_eval_summary, run_evaluation

    data_root = Path(str(args.data_root)).expanduser().resolve()
    include_scenes = set(_split_csv(args.scenes))
    exclude_scenes = set(_split_csv(args.exclude_scenes))

    datasets_raw = str(args.datasets).strip().lower()
    if datasets_raw == "all":
        dataset_keys = list(_DATASETS.keys())
    else:
        dataset_keys = [x.strip() for x in _split_csv(datasets_raw)]

    pruners_raw = str(args.pruners).strip().lower()
    if pruners_raw == "all":
        pruners = ("pure_densify", "gns", "speedy")
    else:
        aliases = {
            "pure_densify": "pure_densify",
            "pure-densify": "pure_densify",
            "dense": "pure_densify",
            "densify": "pure_densify",
            "no_prune": "pure_densify",
            "baseline": "pure_densify",
            "gns": "gns",
            "speedy": "speedy",
        }
        pruners_norm = []
        for x in _split_csv(pruners_raw):
            k = aliases.get(x.strip().lower().replace("-", "_"))
            if k is None:
                pruners_norm.append(x.strip())
            else:
                if k not in pruners_norm:
                    pruners_norm.append(k)
        pruners = tuple(pruners_norm)
        unknown = sorted(set(pruners) - {"pure_densify", "gns", "speedy"})
        if unknown:
            raise KeyError(
                f"Unknown pruners: {unknown}. Expected: pure_densify,gns,speedy,all."
            )

    out_dir = data_root / "pruning_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / (str(args.out_name) if args.out_name is not None else "summary.md")

    device = torch.device(str(args.device))

    rows: list[dict[str, object]] = []
    any_failed = False

    for dataset_key in dataset_keys:
        dataset = _DATASETS.get(dataset_key)
        if dataset is None:
            raise KeyError(f"Unknown dataset key {dataset_key!r}")

        dataset_dir = data_root / dataset.dir_name
        if not dataset_dir.exists():
            print(f"[skip] missing dataset dir: {dataset_dir}", flush=True)
            continue

        for scene in _iter_selected_scenes(
            dataset=dataset, include_scenes=include_scenes, exclude_scenes=exclude_scenes
        ):
            scene_data_dir = dataset_dir / scene
            if not scene_data_dir.exists():
                print(f"[skip] missing scene data: {scene_data_dir}", flush=True)
                continue

            for pruner in pruners:
                scene_out_dir = out_dir / dataset.key / scene / str(pruner)
                if not scene_out_dir.exists() and str(pruner) == "pure_densify":
                    # Backward-compat: older runs used 'dense' as the baseline folder name.
                    legacy = out_dir / dataset.key / scene / "dense"
                    if legacy.exists():
                        scene_out_dir = legacy
                if not scene_out_dir.exists():
                    print(
                        f"[skip] missing outputs: {dataset.key}/{scene} (pruner={pruner})",
                        flush=True,
                    )
                    continue

                try:
                    cfg_dict = _read_cfg_snapshot(scene_out_dir)
                    data_factor = None
                    test_every = 8
                    benchmark_train_split = True
                    normalize_world_space = True
                    align_world_axes = True
                    default_optim = OptimConfig()
                    default_strategy = StrategyConfig()
                    sh_degree = int(default_optim.sh_degree)
                    sh_degree_interval = int(default_optim.sh_degree_interval)
                    packed = bool(default_optim.packed)
                    sparse_grad = bool(default_optim.sparse_grad)
                    antialiased = bool(default_optim.antialiased)
                    absgrad = bool(default_strategy.absgrad)
                    if cfg_dict is not None:
                        data_cfg = cfg_dict.get("data") or {}
                        if isinstance(data_cfg, dict):
                            data_factor = data_cfg.get("data_factor")
                            test_every = int(data_cfg.get("test_every", test_every))
                            benchmark_train_split = bool(
                                data_cfg.get("benchmark_train_split", benchmark_train_split)
                            )
                            normalize_world_space = bool(
                                data_cfg.get("normalize_world_space", normalize_world_space)
                            )
                            # Backward compatibility: older configs rotated by default.
                            align_world_axes = bool(
                                data_cfg.get(
                                    "align_world_axes",
                                    data_cfg.get("normalize_world_space_rotate", True),
                                )
                            )
                        optim_cfg = cfg_dict.get("optim") or {}
                        if isinstance(optim_cfg, dict):
                            sh_degree = int(optim_cfg.get("sh_degree", sh_degree))
                            sh_degree_interval = int(
                                optim_cfg.get("sh_degree_interval", sh_degree_interval)
                            )
                            packed = bool(optim_cfg.get("packed", packed))
                            sparse_grad = bool(optim_cfg.get("sparse_grad", sparse_grad))
                            antialiased = bool(optim_cfg.get("antialiased", antialiased))
                        strategy_cfg = cfg_dict.get("strategy") or {}
                        if isinstance(strategy_cfg, dict):
                            absgrad = bool(strategy_cfg.get("absgrad", absgrad))

                    if data_factor is None:
                        inferred = _infer_data_factor_from_scene_dir(scene_data_dir)
                        data_factor = int(inferred) if inferred is not None else 1
                    data_factor = int(data_factor)

                    if args.step is None:
                        ply_path, ply_step = _find_latest_ply_path(scene_out_dir)
                    else:
                        ply_step = int(args.step)
                        ply_path = _find_ply_path(scene_out_dir=scene_out_dir, step=int(ply_step))
                    step_index = int(ply_step) - 1
                    if step_index < 0:
                        raise ValueError(f"--step must be > 0, got {ply_step}")
                    train_time_s = _read_train_time_s(scene_out_dir)

                    cam_space = "train" if bool(normalize_world_space) else "colmap"
                    normalize_world_space_for_eval = (
                        bool(normalize_world_space) and cam_space == "train"
                    )

                    dataparser = ColmapDataParser(
                        data_dir=str(scene_data_dir),
                        factor=int(data_factor),
                        normalize_world_space=bool(normalize_world_space_for_eval),
                        align_world_axes=bool(align_world_axes),
                        test_every=int(test_every),
                        benchmark_train_split=bool(benchmark_train_split),
                        depth_dir_name=None,
                        normal_dir_name=None,
                        dynamic_mask_dir_name=None,
                        sky_mask_dir_name=None,
                    )
                    parsed = dataparser.get_dataparser_outputs(split="test")
                    dataset_test = InputDataset(parsed)

                    preload = str(args.eval_preload)
                    num_workers = 0 if preload == "cuda" else None
                    eval_loader = DataLoader(
                        dataset_test,
                        batch_size=1,
                        num_workers=num_workers,
                        device=device,
                        infinite_sampler=False,
                        prefetch_to_gpu=False,
                        preload=preload,  # type: ignore[arg-type]
                        seed=42,
                    )

                    splats = _load_ply_splats(ply_path)
                    num_gaussians = int(splats["means"].shape[0])
                    if cam_space == "train" and bool(normalize_world_space):
                        # Map geometry back into training coords to match the camera space (and SH basis).
                        transform = (
                            torch.from_numpy(dataparser.transform).float().to(device=device)
                        )
                        _apply_colmap_to_train_transform_inplace(
                            splats=splats,
                            transform_colmap_to_train=transform,
                        )
                    gaussian_model = _build_gaussian_model(splats=splats, device=device)
                    if int(gaussian_model.num_gaussians) != int(num_gaussians):
                        num_gaussians = int(gaussian_model.num_gaussians)

                    eval_cfg = replace(
                        EvalConfig(),
                        lpips_net="vgg",
                        metrics_backend=str(args.metrics_backend),
                        max_images=int(args.max_images) if args.max_images is not None else None,
                        compute_cc_metrics=False,
                    )
                    optim_cfg = replace(
                        OptimConfig(),
                        sh_degree=int(sh_degree),
                        sh_degree_interval=int(sh_degree_interval),
                        packed=bool(packed),
                        sparse_grad=bool(sparse_grad),
                        antialiased=bool(antialiased),
                    )
                    strategy_cfg = replace(StrategyConfig(), absgrad=bool(absgrad))
                    eval_out = run_evaluation(
                        cfg=_EvalCfg(
                            eval=eval_cfg, optim=optim_cfg, strategy=strategy_cfg
                        ),  # type: ignore[arg-type]
                        step=int(step_index),
                        eval_loader=eval_loader,
                        gaussian_model=gaussian_model,
                        bilateral_grid=None,
                    )
                    stats = dict(eval_out.stats)
                    summary = build_eval_summary(eval_step=int(step_index), stats=stats)
                    print(
                        f"[ok] {dataset.key}/{scene} (pruner={pruner}) {summary}",
                        flush=True,
                    )

                    row: dict[str, object] = {
                        "dataset": dataset.key,
                        "scene": scene,
                        "pruner": str(pruner),
                        "step": int(ply_step),
                        "data_factor": int(data_factor),
                        "psnr": float(stats["psnr"]),
                        "ssim": float(stats["ssim"]),
                        "lpips": float(stats["lpips"]),
                        "num_eval_images": int(stats["num_eval_images"]),
                        "num_gaussians": int(num_gaussians),
                        "train_time_s": float(train_time_s)
                        if train_time_s is not None
                        else None,
                    }
                    rows.append(row)
                except Exception as e:
                    any_failed = True
                    print(
                        f"[fail] {dataset.key}/{scene} (pruner={pruner}): {e}",
                        flush=True,
                    )

    if not rows:
        print("[warn] no evaluated scenes; nothing to write.", flush=True)
        return 1 if any_failed else 0

    fieldnames = [
        "dataset",
        "scene",
        "pruner",
        "step",
        "data_factor",
        "psnr",
        "ssim",
        "lpips",
        "num_eval_images",
        "num_gaussians",
        "train_time_s",
    ]

    def _fmt(x: object) -> str:
        if x is None:
            return ""
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    rows.sort(
        key=lambda r: (
            str(r.get("dataset")),
            str(r.get("scene")),
            str(r.get("pruner")),
        )
    )

    from collections import defaultdict

    def _mean(values: list[float]) -> Optional[float]:
        if not values:
            return None
        return float(sum(values) / len(values))

    agg: dict[tuple[str, str], dict[str, object]] = {}
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["dataset"]), str(row["pruner"]))
        grouped[key].append(row)
    for (dataset_name, pruner_name), items in grouped.items():
        agg[(dataset_name, pruner_name)] = {
            "dataset": dataset_name,
            "pruner": pruner_name,
            "scenes": len({str(x["scene"]) for x in items}),
            "psnr": _mean([float(x["psnr"]) for x in items]),
            "ssim": _mean([float(x["ssim"]) for x in items]),
            "lpips": _mean([float(x["lpips"]) for x in items]),
            "num_gaussians": _mean([float(x["num_gaussians"]) for x in items]),
            "train_time_s": _mean(
                [float(x["train_time_s"]) for x in items if x.get("train_time_s") is not None]
            ),
        }

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(_fmt(row.get(k)) for k in fieldnames) + " |\n")
        f.write("\n")
        f.write(
            "| dataset | pruner | scenes | psnr_mean | ssim_mean | lpips_mean | num_gaussians_mean | train_time_s_mean |\n"
        )
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for dataset_name in sorted({k[0] for k in agg.keys()}):
            for pruner_name in ("pure_densify", "gns", "speedy"):
                key = (dataset_name, pruner_name)
                if key not in agg:
                    continue
                a = agg[key]
                f.write(
                    "| "
                    + " | ".join(
                        [
                            str(a["dataset"]),
                            str(a["pruner"]),
                            str(a["scenes"]),
                            _fmt(a["psnr"]),
                            _fmt(a["ssim"]),
                            _fmt(a["lpips"]),
                            _fmt(a["num_gaussians"]),
                            _fmt(a["train_time_s"]),
                        ]
                    )
                    + " |\n"
                )
        f.write("\n")

    print(f"[write] {md_path}", flush=True)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
