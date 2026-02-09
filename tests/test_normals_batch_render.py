"""Tests for batched normal rendering (RGB+N+ED) using repo assets.

Run:
  pytest tests/test_normals_batch_render.py -s
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

try:
    import pytest  # type: ignore
except Exception:  # pragma: no cover
    pytest = None


def _set_torch_extensions_dir() -> None:
    # Avoid permission issues with the default ~/.cache/torch_extensions by redirecting to /tmp.
    # (This is safe for local tests; CI can override TORCH_EXTENSIONS_DIR externally if desired.)
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        str(Path(tempfile.gettempdir()) / "torch_extensions_gsplat"),
    )


def _asset_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "assets" / "test_garden.npz")


def _results_dir() -> Path:
    # Keep test artifacts in this repo.
    return Path(__file__).resolve().parents[1] / "results" / "dn_test"


def _maybe_save_visualization(
    output_dir: Path,
    renders: torch.Tensor,  # [B, C, H, W, 7]
    alphas: torch.Tensor,  # [B, C, H, W, 1]
) -> Path | None:
    """Save a lightweight PNG visualization (if matplotlib is available)."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    output_dir.mkdir(exist_ok=True, parents=True)

    renders_np = renders.detach().float().cpu().numpy()
    alphas_np = alphas.detach().float().cpu().numpy()

    B, C = renders_np.shape[0], renders_np.shape[1]

    # One PNG that includes all batches/scenes.
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
        normals = renders_np[b, ..., 3:6]  # [C, H, W, 3]
        depth = renders_np[b, ..., -1]  # [C, H, W]
        normals_vis = np.clip((normals + 1.0) * 0.5, 0.0, 1.0)

        # Use a single depth scale per scene for consistent visualization.
        alpha_scene = alphas_np[b, ..., 0]  # [C, H, W]
        valid_scene = (alpha_scene > 1e-6) & np.isfinite(depth)
        if np.any(valid_scene):
            vmin = np.percentile(depth[valid_scene], 5)
            vmax = np.percentile(depth[valid_scene], 95)
        else:
            vmin, vmax = 0.0, 1.0

        for cam in range(C):
            rgb_img = rgb[cam]
            depth_img = depth[cam]

            row0 = 3 * b

            # RGB
            ax = axes[row0 + 0][cam]
            ax.imshow(rgb_img, interpolation="nearest")
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

            # Depth
            ax = axes[row0 + 1][cam]
            ax.imshow(
                depth_img, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            if cam == 0:
                ax.text(
                    0.02,
                    0.94,
                    f"Batch{b+1} D",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="white",
                    va="top",
                    ha="left",
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5),
                )
            ax.axis("off")

            # Normals
            ax = axes[row0 + 2][cam]
            ax.imshow(normals_vis[cam], interpolation="nearest")
            if cam == 0:
                ax.text(
                    0.02,
                    0.94,
                    f"Batch{b+1} N",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="white",
                    va="top",
                    ha="left",
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5),
                )
            ax.axis("off")

    out_path = output_dir / "rgb_dn_batched.png"
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0, hspace=0.0)
    # Avoid bbox_inches="tight" to prevent extra white padding; we manage spacing manually above.
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def test_rasterization_rgb_n_ed_batched_assets():
    _set_torch_extensions_dir()

    from gsplat._helper import load_test_data
    try:
        from gsplat.rendering import rasterization
    except Exception as e:
        if pytest is not None:
            pytest.skip(f"gsplat extension unavailable: {e}")
        raise RuntimeError(f"gsplat extension unavailable: {e}") from e

    if not torch.cuda.is_available():
        if pytest is not None:
            pytest.skip("No CUDA device")
        print("SKIP: No CUDA device")
        return

    torch.manual_seed(42)

    device = torch.device("cuda")
    means, quats, scales, opacities, colors, viewmats, Ks, width, height = load_test_data(
        data_path=_asset_path(),
        device=device,
        scene_grid=1,
    )

    # Use all Gaussians from the NPZ to match real usage.

    B, C = 2, 3
    means = torch.broadcast_to(means, (B, *means.shape)).contiguous().requires_grad_(True)
    quats = torch.broadcast_to(quats, (B, *quats.shape)).contiguous().requires_grad_(True)
    scales = torch.broadcast_to(scales, (B, *scales.shape)).contiguous().requires_grad_(True)
    opacities = (
        torch.broadcast_to(opacities, (B, *opacities.shape))
        .contiguous()
        .requires_grad_(True)
    )
    colors = torch.broadcast_to(colors, (B, *colors.shape)).contiguous().requires_grad_(True)

    viewmats = torch.broadcast_to(viewmats[:C], (B, C, 4, 4)).contiguous()
    Ks = torch.broadcast_to(Ks[:C], (B, C, 3, 3)).contiguous()

    # Low resolution for speed.
    render_width, render_height = 320, 180
    Ks = Ks.clone()
    Ks[..., 0, :] *= render_width / float(width)
    Ks[..., 1, :] *= render_height / float(height)

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

    assert renders.shape == (B, C, render_height, render_width, 7)
    assert alphas.shape == (B, C, render_height, render_width, 1)
    assert "normals" in meta
    assert "render_median" in meta
    assert meta["render_median"].shape == (B, C, render_height, render_width, 1)

    assert torch.isfinite(renders).all()
    assert torch.isfinite(alphas).all()

    # Save a PNG visualization (if matplotlib is available).
    output_dir = _results_dir()
    _maybe_save_visualization(output_dir, renders, alphas)

    # Depth is normalized by alpha in RGB+N+ED mode.
    depth = renders[..., -1:]
    assert torch.isfinite(depth).all()

    # Normals are expected to be in [-1, 1] (they are accumulated under alpha compositing).
    normals = renders[..., 3:6]
    assert normals.min().item() >= -1.001
    assert normals.max().item() <= 1.001

    # Ensure gradients flow through the normals output to {means, quats, scales, opacities, colors}.
    # (This intentionally exercises the normals VJP path; do NOT use covars in this mode.)
    loss = renders.sum() * 0.1 + alphas.sum() * 0.2 + meta["render_median"].sum() * 0.05
    loss.backward()
    for name, tensor in [
        ("means", means),
        ("quats", quats),
        ("scales", scales),
        ("opacities", opacities),
        ("colors", colors),
    ]:
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name


def test_rgb_n_ed_rejects_covars_chain():
    from gsplat.rendering import rasterization

    means = torch.zeros((8, 3), dtype=torch.float32)
    covars = torch.eye(3, dtype=torch.float32).repeat(8, 1, 1)
    # `rasterization()` has required positional args `quats` and `scales` even when using
    # the `covars` projection chain. They are ignored once `covars` is provided, but must
    # be present to satisfy the signature.
    quats = torch.zeros((8, 4), dtype=torch.float32)
    scales = torch.ones((8, 3), dtype=torch.float32)
    opacities = torch.ones((8,), dtype=torch.float32)
    colors = torch.zeros((8, 3), dtype=torch.float32)
    viewmats = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    Ks = torch.eye(3, dtype=torch.float32).unsqueeze(0)

    def _call():
        rasterization(
            means=means,
            quats=quats,
            scales=scales,
            covars=covars,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=32,
            height=32,
            render_mode="RGB+N+ED",
        )

    if pytest is not None:
        with pytest.raises(ValueError, match="covars=None"):
            _call()
    else:
        try:
            _call()
        except ValueError as e:
            if "covars=None" not in str(e):
                raise
        else:
            raise AssertionError("Expected ValueError when render_mode='RGB+N+ED' with covars")


def _run_as_script() -> int:
    print("Running normals batch render checks...")
    try:
        test_rasterization_rgb_n_ed_batched_assets()
        test_rgb_n_ed_rejects_covars_chain()
    except Exception as e:
        print(f"FAILED: {e}")
        return 1

    out_dir = _results_dir()
    out_png = out_dir / "rgb_dn_batched.png"
    if out_png.exists():
        print(f"OK: saved {out_png}")
    else:
        print(
            "OK: tests passed, but no PNG was saved. "
            "Install matplotlib to enable saving."
        )
    print("OK: normals forward+backward look good")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_as_script())
