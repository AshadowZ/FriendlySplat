#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fit an image by optimizing a set of random Gaussians (3DGS) using gsplat.

This is a lightweight demo to sanity-check:
- rasterization output;
- backward pass correctness;
- basic optimization behavior.

Example:
  python tools/gsplat_demo/image_fitting.py --img-path path/to/image.jpg --iterations 2000
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path


def _load_image_rgb_float01(path: Path):
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr  # HxWx3 float32 in [0,1]


def _make_synthetic_image(height: int, width: int):
    import numpy as np

    img = np.ones((height, width, 3), dtype=np.float32)
    img[: height // 2, : width // 2, :] = np.asarray([1.0, 0.0, 0.0], np.float32)
    img[height // 2 :, width // 2 :, :] = np.asarray([0.0, 0.0, 1.0], np.float32)
    return img


def _save_gif(frames_u8, out_path: Path, *, duration_ms: int):
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(f) for f in frames_u8]
    pil_frames[0].save(
        str(out_path),
        save_all=True,
        append_images=pil_frames[1:],
        optimize=False,
        duration=int(duration_ms),
        loop=0,
    )


class SimpleImageFitter:
    def __init__(
        self,
        *,
        gt_image,
        num_points: int,
        device: str,
        seed: int,
    ):
        import torch

        if seed >= 0:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        self.device = torch.device(device)
        self.gt_image = torch.as_tensor(gt_image, dtype=torch.float32, device=self.device)
        self.num_points = int(num_points)

        fov_x = math.pi / 2.0
        self.H, self.W = int(self.gt_image.shape[0]), int(self.gt_image.shape[1])
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)

        self._init_gaussians()

    def _init_gaussians(self):
        import torch

        bd = 2.0
        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        self.rgbs = torch.rand(self.num_points, 3, device=self.device)

        # Random unit quaternions.
        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)
        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            dim=-1,
        )
        self.opacities = torch.ones((self.num_points,), device=self.device)

        # Simple fixed camera.
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        for t in (self.means, self.scales, self.rgbs, self.opacities, self.quats):
            t.requires_grad_(True)

    def _densify_topk(
        self,
        *,
        score,
        topk: int,
        noise: float,
        max_points: int,
    ) -> int:
        """Clone top-k Gaussians and jitter their means to increase capacity (demo heuristic)."""
        import torch

        if int(max_points) <= 0 or self.num_points >= int(max_points):
            return 0

        k = int(min(int(topk), int(score.numel())))
        if k <= 0:
            return 0
        # Clone each selected Gaussian once (fixed).
        idx = torch.topk(score, k=k, largest=True).indices
        remaining = int(max_points) - int(self.num_points)
        if remaining <= 0:
            return 0
        if int(idx.numel()) > remaining:
            idx = idx[:remaining]

        means = self.means[idx]
        scales = self.scales[idx]
        rgbs = self.rgbs[idx]
        opacities = self.opacities[idx]
        quats = self.quats[idx]

        # Jitter means proportionally to current scale magnitude.
        scale_mag = scales.detach().abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
        means = means + torch.randn_like(means) * (float(noise) * scale_mag)
        scales = scales * 0.7

        def _cat(old, new):
            out = torch.cat([old.detach(), new.detach()], dim=0).to(device=self.device)
            out.requires_grad_(True)
            return out

        self.means = _cat(self.means, means)
        self.scales = _cat(self.scales, scales)
        self.rgbs = _cat(self.rgbs, rgbs)
        self.opacities = _cat(self.opacities, opacities)
        self.quats = _cat(self.quats, quats)
        self.num_points = int(self.means.shape[0])
        return int(idx.numel())

    def _prune_by_opacity(self, *, threshold: float, min_points: int) -> int:
        """Drop Gaussians whose sigmoid(opacity) is too small (demo heuristic)."""
        import torch

        if self.num_points <= int(min_points):
            return 0

        keep = torch.sigmoid(self.opacities.detach()) > float(threshold)
        keep_n = int(keep.sum().item())
        if keep_n < int(min_points) or keep_n == self.num_points:
            return 0

        def _idx(x):
            out = x.detach()[keep].to(device=self.device)
            out.requires_grad_(True)
            return out

        removed = self.num_points - keep_n
        self.means = _idx(self.means)
        self.scales = _idx(self.scales)
        self.rgbs = _idx(self.rgbs)
        self.opacities = _idx(self.opacities)
        self.quats = _idx(self.quats)
        self.num_points = int(self.means.shape[0])
        return int(removed)

    def train(
        self,
        *,
        iterations: int,
        lr: float,
        save_gif: bool,
        save_every: int,
        out_dir: Path,
        gif_duration_ms: int,
        densify: bool,
        densify_from_iter: int,
        densify_to_iter: int,
        densify_interval: int,
        densify_topk: int,
        densify_noise: float,
        max_points: int,
        prune_opacity_threshold: float,
        prune_interval: int,
        min_points: int,
    ) -> None:
        import numpy as np
        import torch
        from torch import optim

        try:
            from gsplat import rasterization
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import gsplat rasterizers. Ensure the environment has gsplat + torch properly installed."
            ) from exc

        rasterize = rasterization

        def _make_optimizer():
            return optim.Adam(
                [self.rgbs, self.means, self.scales, self.opacities, self.quats],
                lr=float(lr),
            )

        optimizer = _make_optimizer()
        mse_loss = torch.nn.MSELoss()

        K = torch.tensor(
            [
                [self.focal, 0.0, self.W / 2.0],
                [0.0, self.focal, self.H / 2.0],
                [0.0, 0.0, 1.0],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        frames = []
        t_rast = 0.0
        t_bwd = 0.0
        use_cuda_sync = self.device.type == "cuda" and torch.cuda.is_available()

        out_dir.mkdir(parents=True, exist_ok=True)

        for it in range(int(iterations)):
            t0 = time.time()
            renders = rasterize(
                self.means,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.scales,
                torch.sigmoid(self.opacities),
                torch.sigmoid(self.rgbs),
                self.viewmat[None],
                K[None],
                self.W,
                self.H,
                packed=False,
            )[0]
            out_img = renders[0]
            if use_cuda_sync:
                torch.cuda.synchronize()
            t_rast += time.time() - t0

            loss = mse_loss(out_img, self.gt_image)
            optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            loss.backward()
            if use_cuda_sync:
                torch.cuda.synchronize()
            t_bwd += time.time() - t1
            optimizer.step()

            print(f"iter {it + 1:05d}/{iterations}  loss={loss.item():.6f}", flush=True)

            if bool(densify):
                in_range = (it + 1) >= int(densify_from_iter) and (
                    int(densify_to_iter) <= 0 or (it + 1) <= int(densify_to_iter)
                )
                if (
                    in_range
                    and int(densify_interval) > 0
                    and (it + 1) % int(densify_interval) == 0
                    and self.means.grad is not None
                ):
                    score = self.means.grad.detach().norm(dim=-1)
                    added = self._densify_topk(
                        score=score,
                        topk=int(densify_topk),
                        noise=float(densify_noise),
                        max_points=int(max_points),
                    )
                    if added > 0:
                        print(
                            f"[densify] iter={it+1} added={added} total={self.num_points}",
                            flush=True,
                        )
                        optimizer = _make_optimizer()

                if (
                    float(prune_opacity_threshold) > 0.0
                    and int(prune_interval) > 0
                    and (it + 1) % int(prune_interval) == 0
                ):
                    removed = self._prune_by_opacity(
                        threshold=float(prune_opacity_threshold),
                        min_points=int(min_points),
                    )
                    if removed > 0:
                        print(
                            f"[prune] iter={it+1} removed={removed} total={self.num_points}",
                            flush=True,
                        )
                        optimizer = _make_optimizer()

            if save_gif and save_every > 0 and (it % int(save_every) == 0):
                frame = (out_img.detach().clamp(0, 1).cpu().numpy() * 255.0).astype(
                    np.uint8
                )
                frames.append(frame)

        if save_gif and frames:
            _save_gif(frames, out_dir / "training.gif", duration_ms=gif_duration_ms)

        print(
            f"total(s): rasterization={t_rast:.3f} backward={t_bwd:.3f}",
            flush=True,
        )
        print(
            f"per-step(s): rasterization={t_rast/iterations:.5f} backward={t_bwd/iterations:.5f}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--img-path", type=str, default="", help="Target image path.")
    p.add_argument("--height", type=int, default=256, help="Synthetic image height.")
    p.add_argument("--width", type=int, default=256, help="Synthetic image width.")

    p.add_argument("--num-points", type=int, default=5000, help="#gaussians.")
    p.add_argument("--iterations", type=int, default=6000, help="Training iterations.")
    p.add_argument("--lr", type=float, default=0.005, help="Learning rate.")

    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device string (e.g. cuda:0, cpu).",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed; <0 disables.")

    p.add_argument("--save-gif", action="store_true", help="Save training gif.")
    p.add_argument("--save-every", type=int, default=100, help="Save frame every N iters.")
    p.add_argument(
        "--gif-duration-ms",
        type=int,
        default=5,
        help="GIF frame duration in ms.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="results/gsplat_demo",
        help="Output directory.",
    )
    p.add_argument(
        "--densify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable simple densification + pruning during training (demo heuristic).",
    )
    p.add_argument(
        "--densify-from-iter",
        type=int,
        default=1,
        help="Start densifying at this iteration (1-based).",
    )
    p.add_argument(
        "--densify-to-iter",
        type=int,
        default=4000,
        help="Stop densifying after this iteration (0 means until the end).",
    )
    p.add_argument(
        "--densify-interval",
        type=int,
        default=200,
        help="Densify every N iterations.",
    )
    p.add_argument(
        "--densify-topk",
        type=int,
        default=1500,
        help="Clone top-k Gaussians by mean-gradient magnitude (each is cloned once).",
    )
    p.add_argument(
        "--densify-noise",
        type=float,
        default=0.1,
        help="Mean jitter scale for clones (relative to per-Gaussian scale magnitude).",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=60000,
        help="Max Gaussians after densification.",
    )
    p.add_argument(
        "--prune-opacity-threshold",
        type=float,
        default=0.05,
        help="Prune Gaussians with sigmoid(opacity) below this threshold.",
    )
    p.add_argument(
        "--prune-interval",
        type=int,
        default=200,
        help="Prune every N iterations.",
    )
    p.add_argument(
        "--min-points",
        type=int,
        default=512,
        help="Never prune below this number of Gaussians.",
    )
    args = p.parse_args(argv)

    try:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            print("[warn] CUDA not available; falling back to cpu.", flush=True)
            args.device = "cpu"
    except Exception:
        pass

    img_path = Path(args.img_path) if str(args.img_path).strip() else None
    if img_path is not None:
        gt = _load_image_rgb_float01(img_path)
    else:
        gt = _make_synthetic_image(int(args.height), int(args.width))

    fitter = SimpleImageFitter(
        gt_image=gt,
        num_points=int(args.num_points),
        device=str(args.device),
        seed=int(args.seed),
    )
    fitter.train(
        iterations=int(args.iterations),
        lr=float(args.lr),
        save_gif=bool(args.save_gif),
        save_every=int(args.save_every),
        out_dir=Path(args.out_dir),
        gif_duration_ms=int(args.gif_duration_ms),
        densify=bool(args.densify),
        densify_from_iter=int(args.densify_from_iter),
        densify_to_iter=int(args.densify_to_iter),
        densify_interval=int(args.densify_interval),
        densify_topk=int(args.densify_topk),
        densify_noise=float(args.densify_noise),
        max_points=int(args.max_points),
        prune_opacity_threshold=float(args.prune_opacity_threshold),
        prune_interval=int(args.prune_interval),
        min_points=int(args.min_points),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
