#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import yaml

# Allow running as a standalone script from the repo root without installation.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from friendly_splat.data.colmap_dataparser import ColmapDataParser  # noqa: E402
from friendly_splat.modules.gaussian import GaussianModel  # noqa: E402
from friendly_splat.trainer.io_utils import export_ply  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"YAML must be a dict, got {type(obj)!r}: {path}")
    return obj


def _load_manifest(partition_dir: Path) -> dict[str, Any]:
    manifest_path = partition_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.json: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"manifest.json must be a dict, got {type(obj)!r}: {manifest_path}")
    return obj


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_block_id(block_id: str) -> Tuple[int, int]:
    # Expected: block_00_00
    parts = str(block_id).split("_")
    if len(parts) != 3 or parts[0] != "block":
        raise ValueError(f"Invalid block_id (expected 'block_XX_YY'): {block_id!r}")
    return int(parts[1]), int(parts[2])


def _half_open_bounds(
    *,
    x_edges: list[float],
    y_edges: list[float],
    bx: int,
    by: int,
) -> tuple[float, float, float, float, bool, bool]:
    # Use [min, max) for interior blocks to avoid duplicates on boundaries.
    # Last block uses <= max.
    xmin = float(x_edges[bx])
    xmax = float(x_edges[bx + 1])
    ymin = float(y_edges[by])
    ymax = float(y_edges[by + 1])
    x_inclusive_max = (bx + 1) == (len(x_edges) - 1)
    y_inclusive_max = (by + 1) == (len(y_edges) - 1)
    return xmin, xmax, ymin, ymax, x_inclusive_max, y_inclusive_max


def _mask_in_block_xy(
    *,
    xy: torch.Tensor,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    x_inclusive_max: bool,
    y_inclusive_max: bool,
) -> torch.Tensor:
    x = xy[:, 0]
    y = xy[:, 1]
    m = (x >= float(xmin)) & (y >= float(ymin))
    if x_inclusive_max:
        m = m & (x <= float(xmax))
    else:
        m = m & (x < float(xmax))
    if y_inclusive_max:
        m = m & (y <= float(ymax))
    else:
        m = m & (y < float(ymax))
    return m


def _torch_load_dict(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(obj)!r}: {path}")
    return obj


def _available_steps(ckpt_dir: Path) -> set[int]:
    steps: set[int] = set()
    for p in ckpt_dir.glob("ckpt_step*.pt"):
        stem = p.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            steps.add(int(digits))
    return steps


def _pick_merge_step(
    *,
    trained_blocks_dir: Path,
    block_ids: Iterable[str],
    step: int | None,
) -> int:
    if step is not None:
        if int(step) <= 0:
            raise ValueError(f"--step must be > 0, got {step}")
        return int(step)

    common: set[int] | None = None
    for block_id in block_ids:
        ckpt_dir = trained_blocks_dir / block_id / "ckpts"
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"Missing ckpt dir for {block_id}: {ckpt_dir}")
        s = _available_steps(ckpt_dir)
        if not s:
            raise FileNotFoundError(f"No ckpt_step*.pt under: {ckpt_dir}")
        common = s if common is None else (common & s)
        if common is not None and len(common) == 0:
            raise ValueError(
                "No common checkpoint step across all blocks. "
                "Pass --step explicitly or ensure all blocks were trained to the same final step."
            )
    assert common is not None
    return max(common)


def _build_merge_cfg(*, coarse_dir: Path, manifest: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    cfg_path = coarse_dir / "cfg.yml"
    cfg: dict[str, Any] = _load_yaml(cfg_path) if cfg_path.is_file() else {}
    io_cfg = cfg.get("io") if isinstance(cfg.get("io"), dict) else {}
    data_cfg = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}

    io_cfg["data_dir"] = str(manifest.get("data_dir", io_cfg.get("data_dir", "")))
    io_cfg["result_dir"] = str(out_dir)

    data_cfg["data_factor"] = float(manifest.get("data_factor", data_cfg.get("data_factor", 1.0)))
    data_cfg["normalize_world_space"] = bool(
        manifest.get("normalize_world_space", data_cfg.get("normalize_world_space", True))
    )
    data_cfg["align_world_axes"] = bool(
        manifest.get("align_world_axes", data_cfg.get("align_world_axes", False))
    )
    data_cfg["test_every"] = int(manifest.get("test_every", data_cfg.get("test_every", 8)))
    data_cfg["benchmark_train_split"] = bool(
        manifest.get("benchmark_train_split", data_cfg.get("benchmark_train_split", False))
    )

    cfg["io"] = io_cfg
    cfg["data"] = data_cfg
    return cfg


def _resolve_scene_transform_from_manifest(manifest: dict[str, Any]) -> torch.Tensor:
    data_dir = manifest.get("data_dir")
    if not isinstance(data_dir, str) or not data_dir:
        raise ValueError("manifest.json is missing 'data_dir'.")

    dataparser = ColmapDataParser(
        data_dir=str(data_dir),
        factor=float(manifest.get("data_factor", 1.0)),
        normalize_world_space=bool(manifest.get("normalize_world_space", True)),
        align_world_axes=bool(manifest.get("align_world_axes", False)),
        test_every=int(manifest.get("test_every", 8)),
        benchmark_train_split=bool(manifest.get("benchmark_train_split", False)),
        depth_dir_name=None,
        normal_dir_name=None,
        dynamic_mask_dir_name=None,
        sky_mask_dir_name=None,
        train_image_list_file=None,
    )
    parsed = dataparser.get_dataparser_outputs(split="train")
    return torch.from_numpy(parsed.transform).float()


def _merge_partition_ckpts(
    *,
    manifest: dict[str, Any],
    trained_blocks_dir: Path,
    step: int,
) -> tuple[GaussianModel, dict[str, Any]]:
    blocks = manifest.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("manifest.json missing non-empty 'blocks' list.")

    x_edges = manifest.get("x_edges")
    y_edges = manifest.get("y_edges")
    if not isinstance(x_edges, list) or not isinstance(y_edges, list):
        raise ValueError("manifest.json missing x_edges/y_edges.")

    merged: dict[str, list[torch.Tensor]] = {k: [] for k in ("means", "scales", "quats", "opacities", "sh0", "shN")}
    per_block_counts: list[dict[str, Any]] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_id = block.get("block_id")
        if not isinstance(block_id, str) or not block_id:
            continue

        bx, by = _parse_block_id(block_id)
        xmin, xmax, ymin, ymax, x_inc, y_inc = _half_open_bounds(
            x_edges=[float(v) for v in x_edges],
            y_edges=[float(v) for v in y_edges],
            bx=int(bx),
            by=int(by),
        )

        ckpt_path = trained_blocks_dir / block_id / "ckpts" / f"ckpt_step{int(step):06d}.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {block_id}: {ckpt_path}")

        ckpt = _torch_load_dict(ckpt_path)
        splats = ckpt.get("splats")
        if not isinstance(splats, dict):
            raise KeyError(f"Checkpoint missing 'splats' dict: {ckpt_path}")

        means = splats.get("means")
        if isinstance(means, torch.nn.Parameter):
            means = means.detach()
        if not torch.is_tensor(means):
            raise TypeError(f"splats['means'] must be a tensor: {ckpt_path}")
        means = means.to(dtype=torch.float32, device="cpu").contiguous()

        mask = _mask_in_block_xy(
            xy=means[:, :2],
            xmin=float(xmin),
            xmax=float(xmax),
            ymin=float(ymin),
            ymax=float(ymax),
            x_inclusive_max=bool(x_inc),
            y_inclusive_max=bool(y_inc),
        )
        keep = int(mask.sum().item())
        per_block_counts.append({"block_id": block_id, "keep": keep, "total": int(means.shape[0])})

        for key in merged.keys():
            value = splats.get(key)
            if isinstance(value, torch.nn.Parameter):
                value = value.detach()
            if not torch.is_tensor(value):
                raise TypeError(f"splats[{key!r}] must be a tensor: {ckpt_path}")
            value = value.to(dtype=torch.float32, device="cpu").contiguous()
            merged[key].append(value[mask])

        print(
            f"[merge] {block_id}: keep={keep} / total={int(means.shape[0])}",
            flush=True,
        )

    merged_tensors = {k: torch.cat(v, dim=0) for k, v in merged.items()}
    gaussian_model = GaussianModel(
        params={k: torch.nn.Parameter(t, requires_grad=False) for k, t in merged_tensors.items()}
    )
    gaussian_model.eval()

    report = {
        "num_blocks": len(per_block_counts),
        "num_gaussians": int(gaussian_model.num_gaussians),
        "per_block": per_block_counts,
        "step": int(step),
    }
    return gaussian_model, report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge partition-trained blocks into a single ckpt + PLY.")
    p.add_argument("--partition-dir", type=str, required=True)
    p.add_argument("--trained-blocks-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--step", type=int, default=None, help="1-based step to merge (default: max common step).")
    p.add_argument("--ply-format", type=str, default="ply", choices=("ply", "ply_compressed"))
    return p


def main() -> int:
    args = _build_parser().parse_args()
    partition_dir = Path(args.partition_dir).expanduser().resolve()
    trained_blocks_dir = Path(args.trained_blocks_dir).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else (partition_dir.parent / "merged")
    )

    manifest = _load_manifest(partition_dir)
    blocks = manifest.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("manifest.json missing non-empty 'blocks' list.")
    block_ids = [b["block_id"] for b in blocks if isinstance(b, dict) and isinstance(b.get("block_id"), str)]
    if not block_ids:
        raise ValueError("manifest.json blocks list has no valid block_id entries.")

    step = _pick_merge_step(trained_blocks_dir=trained_blocks_dir, block_ids=block_ids, step=args.step)
    print(f"[step] {step}", flush=True)

    print("[1/4] Merge block checkpoints", flush=True)
    gaussian_model, report = _merge_partition_ckpts(
        manifest=manifest,
        trained_blocks_dir=trained_blocks_dir,
        step=int(step),
    )
    print(f"[merged] gaussians={int(gaussian_model.num_gaussians)}", flush=True)

    print("[2/4] Write merged ckpt + cfg.yml", flush=True)
    _ensure_dir(out_dir / "ckpts")
    _ensure_dir(out_dir / "ply")
    coarse_dir = Path(str(manifest.get("coarse_dir", ""))).expanduser().resolve()
    merge_cfg = _build_merge_cfg(coarse_dir=coarse_dir, manifest=manifest, out_dir=out_dir)
    with open(out_dir / "cfg.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(merge_cfg, f, sort_keys=False, allow_unicode=True)

    ckpt_out = out_dir / "ckpts" / f"ckpt_step{int(step):06d}.pt"
    ckpt_obj: Dict[str, object] = {
        "step": int(step) - 1,
        "train_step": int(step),
        "cfg": merge_cfg,
        "splats": dict(gaussian_model.splats.state_dict().items()),
    }
    torch.save(ckpt_obj, ckpt_out)
    print(f"[ok] wrote ckpt: {ckpt_out}", flush=True)

    report_path = out_dir / "merge_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[ok] wrote report: {report_path}", flush=True)

    print("[3/4] Export merged PLY", flush=True)
    scene_transform = _resolve_scene_transform_from_manifest(manifest)
    export_ply(
        step=int(step) - 1,
        ply_dir=str(out_dir / "ply"),
        ply_format=str(args.ply_format),
        gaussian_model=gaussian_model,
        active_sh_degree=int(gaussian_model.max_sh_degree),
        scene_transform=scene_transform,
    )

    print("[4/4] Done", flush=True)
    print(f"[out_dir] {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
