"""Batch test for median depth rendering + backward.

Run:
  python tests/test_median_depth_batch_render.py
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


def _require_prebuilt_gsplat_cuda() -> bool:
    """Return True if a prebuilt `gsplat.csrc` module is available.

    This script intentionally avoids triggering JIT compilation. If `gsplat.csrc`
    is missing, importing `gsplat` may try to compile the CUDA extension.
    """
    return importlib.util.find_spec("gsplat.csrc") is not None


def _asset_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "assets" / "test_garden.npz")


def _results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "dn_test"


def _normalize_depth_for_vis(
    depth: np.ndarray,  # [H, W]
    alpha: np.ndarray,  # [H, W]
) -> np.ndarray:
    valid = (alpha > 1e-6) & np.isfinite(depth)
    if not np.any(valid):
        return np.zeros_like(depth, dtype=np.float32)
    vmin = np.percentile(depth[valid], 5)
    vmax = np.percentile(depth[valid], 95)
    denom = max(float(vmax - vmin), 1e-6)
    out = (depth.astype(np.float32) - float(vmin)) / denom
    return np.clip(out, 0.0, 1.0)


def _maybe_save_visualization(
    out_path: Path,
    renders: torch.Tensor,  # [B, C, H, W, 7]
    alphas: torch.Tensor,  # [B, C, H, W, 1]
    render_median: torch.Tensor,  # [B, C, H, W, 1]
) -> Path | None:
    """Save a compact PNG visualization (if matplotlib is available)."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    renders_np = renders.detach().float().cpu().numpy()
    alphas_np = alphas.detach().float().cpu().numpy()[..., 0]
    median_np = render_median.detach().float().cpu().numpy()[..., 0]

    B, C = renders_np.shape[0], renders_np.shape[1]

    fig_rows = 3 * B
    fig_cols = C
    fig, axes = plt.subplots(
        fig_rows,
        fig_cols,
        figsize=(2.0 * fig_cols, 1.35 * fig_rows),
        squeeze=False,
    )
    fig.patch.set_facecolor("black")

    for b in range(B):
        rgb = np.clip(renders_np[b, ..., :3], 0.0, 1.0)  # [C, H, W, 3]
        ed = renders_np[b, ..., -1]  # [C, H, W]
        md = median_np[b]  # [C, H, W]
        alpha_scene = alphas_np[b]  # [C, H, W]

        # Use a single scale per scene for consistent depth visualization.
        valid_scene = (alpha_scene > 1e-6) & np.isfinite(ed) & np.isfinite(md)
        if np.any(valid_scene):
            depths_cat = np.concatenate([ed[valid_scene], md[valid_scene]], axis=0)
            vmin = np.percentile(depths_cat, 5)
            vmax = np.percentile(depths_cat, 95)
        else:
            vmin, vmax = 0.0, 1.0

        for cam in range(C):
            row0 = 3 * b

            # RGB
            ax = axes[row0 + 0][cam]
            ax.imshow(rgb[cam], interpolation="nearest")
            if b == 0:
                ax.text(
                    0.02,
                    0.06,
                    f"Cam{cam}",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="white",
                    va="bottom",
                    ha="left",
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5),
                )
            if cam == 0:
                ax.text(
                    0.02,
                    0.94,
                    f"Batch{b+1} RGB",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="white",
                    va="top",
                    ha="left",
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5),
                )
            ax.axis("off")

            # Expected depth (ED)
            ax = axes[row0 + 1][cam]
            ax.imshow(
                ed[cam], cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            if cam == 0:
                ax.text(
                    0.02,
                    0.94,
                    f"Batch{b+1} ED",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="white",
                    va="top",
                    ha="left",
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5),
                )
            ax.axis("off")

            # Median depth
            ax = axes[row0 + 2][cam]
            ax.imshow(
                md[cam], cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            if cam == 0:
                ax.text(
                    0.02,
                    0.94,
                    f"Batch{b+1} Median",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="white",
                    va="top",
                    ha="left",
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5),
                )
            ax.axis("off")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0, hspace=0.0)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _run() -> int:
    print("Running median depth batch render checks...")

    if not torch.cuda.is_available():
        print("SKIP: No CUDA device")
        return 0

    if not _require_prebuilt_gsplat_cuda():
        print("SKIP: `gsplat.csrc` is not available (prebuilt CUDA extension missing).")
        print(
            "Build/install it once (e.g. `python setup.py build_ext --inplace` or `pip install -e .`) "
            "then rerun."
        )
        return 0

    try:
        from gsplat._helper import load_test_data
        from gsplat.rendering import rasterization
    except Exception as e:
        print(f"FAILED: gsplat import failed: {e}")
        traceback.print_exc()
        return 1

    torch.manual_seed(0)
    device = torch.device("cuda")

    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(
        data_path=_asset_path(),
        device=device,
        scene_grid=1,
    )

    B, C = 2, 3
    means = (
        torch.broadcast_to(means, (B, *means.shape)).contiguous().requires_grad_(True)
    )
    quats = (
        torch.broadcast_to(quats, (B, *quats.shape)).contiguous().requires_grad_(True)
    )
    scales = (
        torch.broadcast_to(scales, (B, *scales.shape)).contiguous().requires_grad_(True)
    )
    opacities = (
        torch.broadcast_to(opacities, (B, *opacities.shape))
        .contiguous()
        .requires_grad_(True)
    )
    colors = (
        torch.broadcast_to(colors, (B, *colors.shape)).contiguous().requires_grad_(True)
    )

    viewmats = torch.broadcast_to(viewmats[:C], (B, C, 4, 4)).contiguous()
    Ks = torch.broadcast_to(Ks[:C], (B, C, 3, 3)).contiguous()

    render_width, render_height = 320, 180
    Ks = Ks.clone()
    Ks[..., 0, :] *= render_width / float(width)
    Ks[..., 1, :] *= render_height / float(height)

    try:
        renders, alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=render_width,
            height=render_height,
            packed=False,
            near_plane=0.01,
            far_plane=100.0,
            radius_clip=3.0,
            render_mode="RGB+N+ED",
            with_ut=False,
            with_eval3d=False,
            distributed=False,
        )
    except Exception as e:
        print(f"FAILED: rasterization failed: {e}")
        traceback.print_exc()
        return 1

    if "render_median" not in meta:
        print("FAILED: meta['render_median'] missing")
        return 1

    render_median = meta["render_median"]
    if renders.shape != (B, C, render_height, render_width, 7):
        print(f"FAILED: unexpected renders.shape={tuple(renders.shape)}")
        return 1
    if alphas.shape != (B, C, render_height, render_width, 1):
        print(f"FAILED: unexpected alphas.shape={tuple(alphas.shape)}")
        return 1
    if render_median.shape != (B, C, render_height, render_width, 1):
        print(f"FAILED: unexpected render_median.shape={tuple(render_median.shape)}")
        return 1

    if not torch.isfinite(renders).all():
        print("FAILED: renders contains non-finite values")
        return 1
    if not torch.isfinite(alphas).all():
        print("FAILED: alphas contains non-finite values")
        return 1
    if not torch.isfinite(render_median).all():
        print("FAILED: render_median contains non-finite values")
        return 1

    # Backward: include both expected depth (last channel) and median depth.
    loss = (
        renders[..., :3].mean() * 0.1
        + renders[..., -1:].mean() * 0.2
        + render_median.mean() * 0.3
        + alphas.mean() * 0.05
    )
    try:
        loss.backward()
    except Exception as e:
        print(f"FAILED: backward failed: {e}")
        traceback.print_exc()
        return 1

    for name, tensor in [
        ("means", means),
        ("quats", quats),
        ("scales", scales),
        ("opacities", opacities),
        ("colors", colors),
    ]:
        if tensor.grad is None:
            print(f"FAILED: {name}.grad is None")
            return 1
        if not torch.isfinite(tensor.grad).all():
            print(f"FAILED: {name}.grad contains non-finite values")
            return 1

    # Save PNG: RGB / expected depth / median depth.
    out_png = _results_dir() / "rgb_ed_median_batched.png"
    saved = _maybe_save_visualization(out_png, renders, alphas, render_median)
    if saved is None:
        print(
            "OK: tests passed, but no PNG was saved. Install matplotlib to enable saving."
        )
    else:
        print(f"OK: saved {saved}")
    print("OK: median depth forward+backward look good")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
