#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"JSON must be a dict, got {type(obj)!r}: {path}")
    return obj


def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) < 0:
        return "-"
    s = float(seconds)
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{(s / 60.0):.1f}m"
    return f"{(s / 3600.0):.2f}h"


def _try_read_tb_scalar(tb_dir: Path, tag: str) -> Optional[float]:
    if not tb_dir.is_dir():
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore
            EventAccumulator,
        )
    except Exception:
        return None

    event_files = sorted(tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None

    # Read the newest file first (most likely contains the end-of-train scalar).
    for event_path in reversed(event_files):
        try:
            ea = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
            ea.Reload()
            if tag not in ea.Tags().get("scalars", []):
                continue
            items = ea.Scalars(tag)
            if not items:
                continue
            return float(items[-1].value)
        except Exception:
            continue
    return None


def _estimate_wall_time_seconds(result_dir: Path) -> Optional[float]:
    ckpt_dir = result_dir / "ckpts"
    if not ckpt_dir.is_dir():
        return None
    ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"))
    if len(ckpts) < 2:
        return None
    mtimes = [p.stat().st_mtime for p in ckpts]
    return float(max(mtimes) - min(mtimes))


def _train_time_seconds(result_dir: Path) -> Optional[float]:
    # Prefer TensorBoard scalar if available; else fallback to a coarse mtime estimate.
    tb_s = _try_read_tb_scalar(result_dir / "tb", tag="train/total_time_s")
    if tb_s is not None:
        return float(tb_s)
    return _estimate_wall_time_seconds(result_dir)


_SPLATS_STEP_RE = re.compile(r"^splats_step(\d+)\.ply$")


def _pick_latest_splats_ply(result_dir: Path) -> Optional[Path]:
    ply_dir = result_dir / "ply"
    if not ply_dir.is_dir():
        return None
    best_step = None
    best_path = None
    for p in ply_dir.glob("splats_step*.ply"):
        m = _SPLATS_STEP_RE.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        if best_step is None or step > best_step:
            best_step = step
            best_path = p
    if best_path is not None:
        return best_path
    cands = sorted(ply_dir.glob("*.ply"))
    return cands[-1] if cands else None


def _read_ply_vertex_count(ply_path: Path) -> Optional[int]:
    # Header is ASCII for both ASCII and binary PLY; only need `element vertex N`.
    if not ply_path.is_file():
        return None
    try:
        with open(ply_path, "rb") as f:
            for _ in range(512):
                line = f.readline()
                if line == b"":
                    break
                s = line.decode("utf-8", errors="ignore").strip()
                if s == "end_header":
                    break
                parts = s.split()
                if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
                    try:
                        n = int(parts[2])
                        return n if n >= 0 else None
                    except Exception:
                        return None
    except Exception:
        return None
    return None


def _pick_latest_metrics(eval_dir: Path) -> Optional[Path]:
    if not eval_dir.is_dir():
        return None
    cands = sorted(eval_dir.glob("metrics_step*.json"))
    return cands[-1] if cands else None


def _format_metric(x: Optional[object], fmt: str) -> str:
    if x is None:
        return "-"
    try:
        v = float(x)
    except Exception:
        return "-"
    if not math.isfinite(v):
        return "-"
    return format(v, fmt)


@dataclass(frozen=True)
class BlockRow:
    block_id: str
    keep: Optional[int]
    total: Optional[int]
    train_time_s: Optional[float]
    ply_gaussians: Optional[int]


def _collect_blocks(
    *,
    partition_dir: Path,
    trained_blocks_dir: Path,
    merge_report: Optional[dict[str, Any]],
) -> tuple[int, list[BlockRow]]:
    manifest_path = partition_dir / "manifest.json"
    num_blocks = 0
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        blocks = manifest.get("blocks")
        if isinstance(blocks, list):
            num_blocks = len(blocks)
    if num_blocks <= 0:
        blocks_dir = partition_dir / "blocks"
        if blocks_dir.is_dir():
            block_ids: set[str] = set()
            for p in blocks_dir.glob("block_*_train_images.txt"):
                name = p.name
                # block_00_00_train_images.txt -> block_00_00
                if name.endswith("_train_images.txt"):
                    block_ids.add(name[: -len("_train_images.txt")])
            if block_ids:
                num_blocks = len(block_ids)
    if num_blocks <= 0 and trained_blocks_dir.is_dir():
        num_blocks = len([p for p in trained_blocks_dir.glob("block_*") if p.is_dir()])

    per_block_keep: dict[str, tuple[Optional[int], Optional[int]]] = {}
    if isinstance(merge_report, dict):
        per_block = merge_report.get("per_block")
        if isinstance(per_block, list):
            for item in per_block:
                if not isinstance(item, dict):
                    continue
                bid = item.get("block_id")
                if not isinstance(bid, str):
                    continue
                keep = item.get("keep")
                total = item.get("total")
                per_block_keep[bid] = (
                    int(keep) if isinstance(keep, int) else None,
                    int(total) if isinstance(total, int) else None,
                )

    rows: list[BlockRow] = []
    if trained_blocks_dir.is_dir():
        for block_dir in sorted(trained_blocks_dir.glob("block_*")):
            if not block_dir.is_dir():
                continue
            bid = block_dir.name
            keep, total = per_block_keep.get(bid, (None, None))
            splats_ply = _pick_latest_splats_ply(block_dir)
            rows.append(
                BlockRow(
                    block_id=bid,
                    keep=keep,
                    total=total,
                    train_time_s=_train_time_seconds(block_dir),
                    ply_gaussians=_read_ply_vertex_count(splats_ply) if splats_ply is not None else None,
                )
            )
    return int(num_blocks), rows


def _build_scene_section(*, scene_root: Path, scene_name: str) -> str:
    coarse_dir = scene_root / "coarse"
    partition_dir = scene_root / "partition"
    trained_blocks_dir = partition_dir / "trained_blocks"
    merged_dir = scene_root / "merged"

    merge_report = None
    merge_report_path = merged_dir / "merge_report.json"
    if merge_report_path.is_file():
        merge_report = _load_json(merge_report_path)

    num_blocks, block_rows = _collect_blocks(
        partition_dir=partition_dir,
        trained_blocks_dir=trained_blocks_dir,
        merge_report=merge_report,
    )

    coarse_time_s = _train_time_seconds(coarse_dir)
    merged_time_s = _train_time_seconds(merged_dir)

    coarse_ply = _pick_latest_splats_ply(coarse_dir)
    coarse_gaussians = _read_ply_vertex_count(coarse_ply) if coarse_ply is not None else None
    merged_ply = _pick_latest_splats_ply(merged_dir)
    merged_ply_gaussians = _read_ply_vertex_count(merged_ply) if merged_ply is not None else None

    merged_gaussians = None
    merged_step = None
    if isinstance(merge_report, dict):
        merged_gaussians = merge_report.get("num_gaussians")
        merged_step = merge_report.get("step")

    metrics_path = _pick_latest_metrics(merged_dir / "eval")
    metrics = _load_json(metrics_path) if metrics_path is not None else None

    blocks_time_total_s = 0.0
    blocks_time_count = 0
    for r in block_rows:
        if r.train_time_s is not None and math.isfinite(float(r.train_time_s)) and float(r.train_time_s) >= 0:
            blocks_time_total_s += float(r.train_time_s)
            blocks_time_count += 1

    lines: list[str] = []
    lines.append(f"## {scene_name}")
    lines.append("")
    lines.append(
        f"- Coarse: `{coarse_dir}` (train_time={_format_seconds(coarse_time_s)}, gaussians={coarse_gaussians if coarse_gaussians is not None else '-'})"
    )
    lines.append(f"- Partition: `{partition_dir}` (blocks={num_blocks if num_blocks else '-'})")
    merged_gaussians_s = merged_gaussians if merged_gaussians is not None else merged_ply_gaussians
    lines.append(
        f"- Merged: `{merged_dir}` (train_time={_format_seconds(merged_time_s)}, step={merged_step if merged_step else '-'}, gaussians={merged_gaussians_s if merged_gaussians_s is not None else '-'})"
    )
    if block_rows:
        blocks_time_s = _format_seconds(blocks_time_total_s) if blocks_time_count else "-"
        lines.append(f"- Blocks: trained={len(block_rows)} (sum_train_time={blocks_time_s})")
    if metrics_path is not None and metrics is not None:
        lines.append(f"- Eval: `{metrics_path}`")
        psnr = _format_metric(metrics.get("psnr"), ".3f")
        ssim = _format_metric(metrics.get("ssim"), ".4f")
        lpips = _format_metric(metrics.get("lpips"), ".4f")
        cc_psnr = _format_metric(metrics.get("cc_psnr"), ".3f")
        cc_ssim = _format_metric(metrics.get("cc_ssim"), ".4f")
        cc_lpips = _format_metric(metrics.get("cc_lpips"), ".4f")
        cc_lpips_inria = _format_metric(metrics.get("cc_lpips_inria"), ".4f")
        lines.append(f"  - metrics: psnr={psnr} ssim={ssim} lpips={lpips}")
        cc_lpips_suffix = f" cc_lpips_inria={cc_lpips_inria}" if cc_lpips_inria != "-" else ""
        lines.append(
            f"  - cc_metrics: cc_psnr={cc_psnr} cc_ssim={cc_ssim} cc_lpips={cc_lpips}{cc_lpips_suffix}"
        )
        lines.append(
            f"  - num_eval_images={metrics.get('num_eval_images','-')} sec/img={_format_metric(metrics.get('seconds_per_image'), '.4f')}"
        )
    else:
        lines.append("- Eval: -")

    if block_rows:
        lines.append("")
        lines.append("### Blocks")
        lines.append("")
        lines.append("| block | gaussians | keep_gaussians | total_gaussians | train_time |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in block_rows:
            lines.append(
                f"| `{r.block_id}` | {r.ply_gaussians if r.ply_gaussians is not None else (r.total if r.total is not None else '-')} | {r.keep if r.keep is not None else '-'} | {r.total if r.total is not None else '-'} | {_format_seconds(r.train_time_s)} |"
            )

    lines.append("")
    return "\n".join(lines)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate MatrixCity benchmark summary.md (coarse/partition/merge/eval)."
    )
    p.add_argument(
        "--benchmark-root",
        type=str,
        required=True,
        help="e.g. /media/joker/p3500/3DGS_Dataset/benchmark/urban_benchmark/matrix_benchmark",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output markdown path (default: <benchmark-root>/summary.md).",
    )
    p.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Optional scene filter (repeatable). If omitted, summarize all scenes under benchmark root.",
    )
    return p


def main() -> int:
    args = _build_argparser().parse_args()
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out is not None
        else (benchmark_root / "summary.md")
    )
    scenes_filter = set(str(s) for s in (args.scene or []))

    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"benchmark root not found: {benchmark_root}")

    scene_dirs = [
        p for p in sorted(benchmark_root.iterdir()) if p.is_dir() and not p.name.startswith(".")
    ]
    if scenes_filter:
        scene_dirs = [p for p in scene_dirs if p.name in scenes_filter]

    sections: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections.append("# MatrixCity benchmark summary")
    sections.append("")
    sections.append(f"- Generated: {now}")
    sections.append(f"- Benchmark root: `{benchmark_root}`")
    sections.append("")

    if not scene_dirs:
        sections.append("_No scenes found._")
    else:
        for scene_dir in scene_dirs:
            sections.append(_build_scene_section(scene_root=scene_dir, scene_name=scene_dir.name))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
