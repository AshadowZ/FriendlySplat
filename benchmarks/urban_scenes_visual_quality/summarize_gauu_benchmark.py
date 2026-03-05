#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Optional


_METRICS_STEP_RE = re.compile(r"^metrics_step(\d+)\.json$")


def _load_json_dict(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"JSON root must be dict: {path}")
    return obj


def _format_float(x: Optional[float], digits: int) -> str:
    if x is None or not math.isfinite(float(x)):
        return "-"
    return f"{float(x):.{digits}f}"


def _format_seconds(x: Optional[float]) -> str:
    if x is None or not math.isfinite(float(x)):
        return "-"
    return f"{float(x):.1f}"


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _try_read_tb_total_time_s(result_dir: Path) -> Optional[float]:
    tb_dir = result_dir / "tb"
    if not tb_dir.is_dir():
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore
            EventAccumulator,
        )
    except Exception:
        return None

    event_files = sorted(tb_dir.glob("events.out.tfevents.*"))
    for event_path in reversed(event_files):
        try:
            ea = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
            ea.Reload()
            if "train/total_time_s" not in ea.Tags().get("scalars", []):
                continue
            items = ea.Scalars("train/total_time_s")
            if items:
                return float(items[-1].value)
        except Exception:
            continue
    return None


def _fallback_ckpt_wall_time_s(result_dir: Path) -> Optional[float]:
    ckpt_dir = result_dir / "ckpts"
    if not ckpt_dir.is_dir():
        return None
    ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"))
    if len(ckpts) < 2:
        return None
    mtimes = [float(p.stat().st_mtime) for p in ckpts]
    return max(mtimes) - min(mtimes)


def _train_time_s(result_dir: Path) -> Optional[float]:
    tb_s = _try_read_tb_total_time_s(result_dir)
    if tb_s is not None:
        return tb_s
    return _fallback_ckpt_wall_time_s(result_dir)


def _pick_latest_metrics(eval_dir: Path) -> Optional[Path]:
    if not eval_dir.is_dir():
        return None
    best_step = None
    best_path = None
    for path in eval_dir.glob("metrics_step*.json"):
        m = _METRICS_STEP_RE.match(path.name)
        if not m:
            continue
        step = int(m.group(1))
        if best_step is None or step > best_step:
            best_step = step
            best_path = path
    return best_path


def _extract_float(metrics: dict[str, Any], key: str) -> Optional[float]:
    val = metrics.get(key)
    if val is None:
        return None
    try:
        f = float(val)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def _extract_int(metrics: dict[str, Any], key: str) -> Optional[int]:
    val = metrics.get(key)
    if val is None:
        return None
    try:
        return int(val)
    except Exception:
        return None


def _resolve_scene_dirs(root: Path, scenes_filter: list[str]) -> list[Path]:
    if scenes_filter:
        out: list[Path] = []
        for name in scenes_filter:
            path = root / name
            if path.is_dir():
                out.append(path)
        return out

    preferred = ["Modern_Building", "Residence", "Russian_Building"]
    present = {p.name: p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")}
    ordered: list[Path] = []
    for name in preferred:
        if name in present:
            ordered.append(present[name])
    for name in sorted(present.keys()):
        if name not in preferred:
            ordered.append(present[name])
    return ordered


def _write_summary_md(
    *,
    out_path: Path,
    benchmark_root: Path,
    rows: list[dict[str, object]],
) -> None:
    cols = [
        "scene",
        "step",
        "num_gaussians",
        "train_time_s",
        "train_time_h",
        "psnr",
        "ssim",
        "lpips",
        "cc_psnr",
        "cc_ssim",
        "cc_lpips",
        "num_eval_images",
        "sec_per_img",
    ]

    def _cell(row: dict[str, object], key: str) -> str:
        v = row.get(key)
        if v is None:
            return "-"
        if isinstance(v, float):
            if not math.isfinite(v):
                return "-"
            if key in ("psnr", "cc_psnr"):
                return f"{v:.3f}"
            if key in ("ssim", "lpips", "cc_ssim", "cc_lpips", "sec_per_img"):
                return f"{v:.4f}"
            if key in ("train_time_h",):
                return f"{v:.3f}"
            return f"{v:.1f}"
        return str(v)

    psnr_vals = [float(r["psnr"]) for r in rows if isinstance(r.get("psnr"), (float, int))]
    ssim_vals = [float(r["ssim"]) for r in rows if isinstance(r.get("ssim"), (float, int))]
    lpips_vals = [float(r["lpips"]) for r in rows if isinstance(r.get("lpips"), (float, int))]
    cc_psnr_vals = [float(r["cc_psnr"]) for r in rows if isinstance(r.get("cc_psnr"), (float, int))]
    cc_ssim_vals = [float(r["cc_ssim"]) for r in rows if isinstance(r.get("cc_ssim"), (float, int))]
    cc_lpips_vals = [float(r["cc_lpips"]) for r in rows if isinstance(r.get("cc_lpips"), (float, int))]
    train_time_vals = [float(r["train_time_s"]) for r in rows if isinstance(r.get("train_time_s"), (float, int))]
    gauss_vals = [float(r["num_gaussians"]) for r in rows if isinstance(r.get("num_gaussians"), (float, int))]

    mean_row: dict[str, object] = {
        "scene": "MEAN",
        "step": "-",
        "num_gaussians": _mean(gauss_vals),
        "train_time_s": _mean(train_time_vals),
        "train_time_h": (_mean(train_time_vals) / 3600.0) if train_time_vals else None,
        "psnr": _mean(psnr_vals),
        "ssim": _mean(ssim_vals),
        "lpips": _mean(lpips_vals),
        "cc_psnr": _mean(cc_psnr_vals),
        "cc_ssim": _mean(cc_ssim_vals),
        "cc_lpips": _mean(cc_lpips_vals),
        "num_eval_images": "-",
        "sec_per_img": "-",
    }

    lines: list[str] = []
    lines.append("# GauU benchmark summary")
    lines.append("")
    lines.append(f"- benchmark_root: `{benchmark_root}`")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_cell(row, col) for col in cols) + " |")
    if rows:
        lines.append("| " + " | ".join(_cell(mean_row, col) for col in cols) + " |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize GauU benchmark metrics into summary.md.")
    p.add_argument(
        "--benchmark-root",
        type=str,
        default="/media/joker/p3500/3DGS_Dataset/benchmark/urban_benchmark/gauu_benchmark",
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
        help="Optional scene filter (repeatable).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out is not None
        else (benchmark_root / "summary.md")
    )
    scenes_filter = [str(x) for x in (args.scene or [])]

    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"benchmark root not found: {benchmark_root}")

    scene_dirs = _resolve_scene_dirs(benchmark_root, scenes_filter)
    rows: list[dict[str, object]] = []

    for scene_dir in scene_dirs:
        metrics_path = _pick_latest_metrics(scene_dir / "eval")
        metrics = _load_json_dict(metrics_path) if metrics_path is not None else {}

        train_time_s = _train_time_s(scene_dir)
        row: dict[str, object] = {
            "scene": scene_dir.name,
            "step": _extract_int(metrics, "train_step"),
            "num_gaussians": _extract_int(metrics, "num_gaussians"),
            "train_time_s": train_time_s,
            "train_time_h": (train_time_s / 3600.0) if train_time_s is not None else None,
            "psnr": _extract_float(metrics, "psnr"),
            "ssim": _extract_float(metrics, "ssim"),
            "lpips": _extract_float(metrics, "lpips"),
            "cc_psnr": _extract_float(metrics, "cc_psnr"),
            "cc_ssim": _extract_float(metrics, "cc_ssim"),
            "cc_lpips": _extract_float(metrics, "cc_lpips"),
            "num_eval_images": _extract_int(metrics, "num_eval_images"),
            "sec_per_img": _extract_float(metrics, "seconds_per_image"),
        }
        rows.append(row)

    rows.sort(key=lambda x: str(x["scene"]))
    _write_summary_md(out_path=out_path, benchmark_root=benchmark_root, rows=rows)
    print(f"[ok] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
