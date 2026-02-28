from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from friendly_splat.data.dataset import InputDataset
from friendly_splat.modules.gaussian import GaussianModel
from friendly_splat.trainer.configs import HardPruneConfig

from gsplat.rendering import rasterization
from gsplat.strategy.ops import remove


def _train_step_due(*, step: int, every_n: int) -> bool:
    train_step = int(step) + 1
    return (train_step % int(every_n)) == 0


def _is_prune_window_active(
    *,
    step: int,
    hard_prune_cfg: HardPruneConfig,
) -> bool:
    if not bool(hard_prune_cfg.enable):
        return False
    train_step = int(step) + 1
    if train_step < int(hard_prune_cfg.start_step):
        return False
    if not _train_step_due(step=int(step), every_n=int(hard_prune_cfg.every_n)):
        return False
    if train_step > int(hard_prune_cfg.stop_step):
        return False
    return True


def _hard_prune_event_bounds(
    *,
    start_step: int,
    stop_step: int,
    every_n: int,
) -> tuple[int, int, int]:
    """Return (first_due_step, last_due_step, num_events) in 1-based train-step space."""
    start_step = int(start_step)
    stop_step = int(stop_step)
    every_n = int(every_n)
    if start_step <= 0 or stop_step <= 0 or every_n <= 0:
        raise ValueError("start_step/stop_step/every_n must be positive.")
    if stop_step < start_step:
        raise ValueError("stop_step must be >= start_step.")

    # First multiple of every_n that is >= start_step.
    first_due = ((start_step + every_n - 1) // every_n) * every_n
    # Last multiple of every_n that is <= stop_step.
    last_due = (stop_step // every_n) * every_n
    if last_due < first_due:
        return first_due, last_due, 0
    num_events = ((last_due - first_due) // every_n) + 1
    return int(first_due), int(last_due), int(num_events)


def _sample_train_image_indices(
    *,
    parsed_scene: Any,
    max_views: Optional[int],
    seed: int,
    step: int,
) -> np.ndarray:
    indices = np.asarray(parsed_scene.indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size <= 0:
        raise ValueError("parsed_scene.indices must be a non-empty 1D array.")
    if max_views is None or int(max_views) >= int(indices.size):
        return indices
    g = torch.Generator(device="cpu")
    # Deterministic per-step sampling.
    g.manual_seed(int(seed) ^ (int(step) + 1))
    perm = torch.randperm(int(indices.size), generator=g)
    sel = perm[: int(max_views)].cpu().numpy()
    return indices[sel]


def _get_image_size_for_index(parsed_scene: Any, image_index: int) -> tuple[int, int]:
    meta = getattr(parsed_scene, "metadata", None)
    if isinstance(meta, dict):
        sizes = meta.get("image_sizes")
        if sizes is not None:
            arr = np.asarray(sizes)
            if (
                arr.ndim == 2
                and arr.shape[1] == 2
                and int(image_index) < int(arr.shape[0])
            ):
                w = int(arr[int(image_index), 0])
                h = int(arr[int(image_index), 1])
                if w > 0 and h > 0:
                    return w, h
    raise KeyError(
        "Missing parsed_scene.metadata['image_sizes']; required for hard-prune scoring without image I/O."
    )


def _compute_opacity_score_grad(
    *,
    gaussian_model: GaussianModel,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int,
    use_sqgrad: bool,
) -> torch.Tensor:
    """Compute one-view per-Gaussian sensitivity score.

    This computes gradients w.r.t. a per-Gaussian "score" scalar that *does not*
    change the forward render when initialized at zeros:

        opacity_used = opacity_detached * (1 + (s - s.detach()))

    Forward: identical to `opacity_detached` (since (s - s.detach()) == 0).
    Backward: exposes d(render)/ds, proportional to opacity sensitivity.
    """
    device = gaussian_model.device
    n = int(gaussian_model.num_gaussians)
    if n <= 0:
        return torch.empty((0,), device=device)

    means = gaussian_model.means.detach()
    quats = gaussian_model.quats.detach()
    scales = gaussian_model.scales.detach()
    colors = gaussian_model.sh_coeffs(sh_degree=int(sh_degree)).detach()
    opacities = gaussian_model.opacities.detach()

    scores = torch.zeros((n,), device=device, dtype=opacities.dtype, requires_grad=True)
    # Forward is unchanged; backward exposes gradients to `scores`.
    factor = 1.0 + (scores - scores.detach())
    opacities_used = opacities * factor

    renders, _alphas, _meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities_used,
        colors=colors,
        viewmats=viewmat[None, ...],
        Ks=K[None, ...],
        width=int(width),
        height=int(height),
        sh_degree=int(sh_degree),
        packed=False,
        rasterize_mode="classic",
        render_mode="RGB",
    )
    # Normalize by pixel count so different resolutions don't dominate the sum.
    loss = renders[..., 0:3].sum() / float(int(height) * int(width) * 3)
    (grad,) = torch.autograd.grad(loss, scores, retain_graph=False, create_graph=False)
    grad = grad.detach()
    if bool(use_sqgrad):
        return grad.square()
    return grad.abs()


def maybe_hard_prune_after_densify(
    *,
    step: int,
    hard_prune_cfg: HardPruneConfig,
    io_seed: int,
    active_sh_degree: int,
    train_dataset: InputDataset,
    gaussian_model: GaussianModel,
    splat_optimizers: Dict[str, torch.optim.Optimizer],
    strategy_state: Dict[str, Any],
) -> None:
    if not _is_prune_window_active(
        step=int(step),
        hard_prune_cfg=hard_prune_cfg,
    ):
        return

    n_before = int(gaussian_model.num_gaussians)
    if n_before <= 1:
        return

    target_budget = int(hard_prune_cfg.final_budget)
    train_step = int(step) + 1
    first_due, _last_due, num_events = _hard_prune_event_bounds(
        start_step=int(hard_prune_cfg.start_step),
        stop_step=int(hard_prune_cfg.stop_step),
        every_n=int(hard_prune_cfg.every_n),
    )
    if num_events <= 0:
        return
    event_index = (int(train_step) - int(first_due)) // int(hard_prune_cfg.every_n)
    if event_index < 0:
        return

    # Track the Gaussian count at the first pruning event so we can linearly
    # schedule pruning to `final_budget` across the configured window.
    start_count_key = "hard_prune_start_count"
    start_count_obj = strategy_state.get(start_count_key)
    if start_count_obj is None:
        strategy_state[start_count_key] = int(n_before)
        start_count = int(n_before)
    else:
        start_count = int(start_count_obj)

    if start_count <= target_budget:
        return

    # Linear target: reach final_budget at the last pruning event.
    frac = float(event_index + 1) / float(num_events)
    target_count = int(
        round(float(start_count) - float(start_count - target_budget) * frac)
    )
    target_count = max(int(target_budget), min(int(start_count), int(target_count)))

    n_prune = int(max(0, int(n_before) - int(target_count)))
    n_prune = min(n_prune, n_before - 1)
    if n_prune <= 0:
        return

    device = gaussian_model.device
    parsed_scene = train_dataset.parsed_scene
    selected = _sample_train_image_indices(
        parsed_scene=parsed_scene,
        max_views=hard_prune_cfg.score_num_views,
        seed=int(io_seed),
        step=int(step),
    )
    num_views = int(selected.size)
    if num_views <= 0:
        return

    # Scoring needs gradients (only for the synthetic `scores` leaf).
    score_accum = torch.zeros((n_before,), device=device, dtype=torch.float32)
    for global_image_idx in selected.tolist():
        idx = int(global_image_idx)
        c2w = torch.from_numpy(np.asarray(parsed_scene.camtoworlds[idx])).to(
            device=device, dtype=torch.float32
        )
        K = torch.from_numpy(np.asarray(parsed_scene.Ks[idx])).to(
            device=device, dtype=torch.float32
        )
        width, height = _get_image_size_for_index(parsed_scene, idx)
        viewmat = torch.linalg.inv(c2w)
        grad_score = _compute_opacity_score_grad(
            gaussian_model=gaussian_model,
            viewmat=viewmat,
            K=K,
            width=int(width),
            height=int(height),
            sh_degree=int(active_sh_degree),
            use_sqgrad=bool(hard_prune_cfg.score_use_sqgrad),
        )
        if int(grad_score.numel()) != int(score_accum.numel()):
            raise RuntimeError(
                "Hard prune scoring produced unexpected score size: "
                f"{grad_score.shape} vs expected {(score_accum.shape,)}."
            )
        score_accum.add_(grad_score.to(dtype=score_accum.dtype))

    # Remove the lowest-score Gaussians.
    remove_ids = torch.topk(score_accum, k=int(n_prune), largest=False).indices
    mask_remove = torch.zeros((n_before,), device=device, dtype=torch.bool)
    mask_remove[remove_ids] = True

    with torch.no_grad():
        remove(
            params=gaussian_model.splats,
            optimizers=splat_optimizers,
            state=strategy_state,
            mask=mask_remove,
        )
    torch.cuda.empty_cache()

    n_after = int(gaussian_model.num_gaussians)
    print(
        "[HardPrune] "
        f"step={int(step) + 1} "
        f"pruned={int(n_prune)} "
        f"num_gs={int(n_after)} "
        f"budget={int(target_budget)} "
        f"target={int(target_count)}",
        flush=True,
    )
    return
