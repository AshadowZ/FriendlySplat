"""Tests for the FastGS integration (`--fast`).

Run:
  pytest tests/test_fastgs.py -s
  python tests/test_fastgs.py        # same checks, no pytest needed
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from friendly_splat.fastgs import (  # noqa: E402
    FastGSConfig,
    FastGSStepGate,
    FastGSStrategy,
    ViewCache,
    apply_fastgs_presets,
    compute_multiview_scores,
    resolve_importance_thresh,
)
from friendly_splat.trainer.configs import (  # noqa: E402
    IOConfig,
    TrainConfig,
    apply_steps_scaler,
    validate_train_config,
)


def _base_cfg(**kwargs) -> TrainConfig:
    return dataclasses.replace(
        TrainConfig(io=IOConfig(data_dir="unused")), fast=True, **kwargs
    )


# ---------------------------------------------------------------------------
# Config: toggle, presets, validation, step scaling
# ---------------------------------------------------------------------------


def test_fast_is_off_by_default() -> None:
    cfg = TrainConfig(io=IOConfig(data_dir="unused"))
    assert cfg.fast is False
    # Nothing FastGS-related may be applied while the flag is off.
    assert apply_fastgs_presets(cfg) is cfg
    validate_train_config(cfg)


def test_presets_only_touch_fastgs_values() -> None:
    cfg = _base_cfg()
    out = apply_fastgs_presets(cfg)
    assert out.strategy.refine_every == cfg.fastgs.densify_every
    assert out.strategy.absgrad is True
    assert out.optim.optimizers.opacities.optimizer.lr == cfg.fastgs.opacity_lr
    assert out.optim.optimizers.shN.optimizer.lr == cfg.fastgs.shN_lr / 20.0
    # Untouched groups keep FriendlySplat's values.
    assert (
        out.optim.optimizers.means.optimizer.lr
        == cfg.optim.optimizers.means.optimizer.lr
    )
    assert out.optim.max_steps == cfg.optim.max_steps
    # Opt-out is honored.
    off = dataclasses.replace(
        cfg, fastgs=dataclasses.replace(cfg.fastgs, apply_presets=False)
    )
    assert apply_fastgs_presets(off) is off


def test_validation_rejects_incompatible_switches() -> None:
    cases = [
        dict(strategy=dataclasses.replace(_base_cfg().strategy, impl="mcmc")),
        dict(optim=dataclasses.replace(_base_cfg().optim, mu_enable=True)),
        dict(
            optim=dataclasses.replace(_base_cfg().optim, sparse_grad=True, packed=True)
        ),
        dict(fastgs=FastGSConfig(loss_thresh=1.5)),
        dict(fastgs=FastGSConfig(score_num_views=0)),
        dict(fastgs=FastGSConfig(prune_budget_ratio=0.0)),
        # The pixel tracker cannot run with packed rasterization.
        dict(
            fastgs=FastGSConfig(score_backend="tracked"),
            optim=dataclasses.replace(_base_cfg().optim, packed=True),
        ),
        # Late pruning must start after densification stops.
        dict(fastgs=FastGSConfig(final_prune_start_step=1000)),
        # Phase 1 must cover the whole densification window.
        dict(fastgs=FastGSConfig(optim_phase1_end_step=1000)),
        # Absolute gradients are required; presets normally enable them, so this
        # can only trip when the user opts out of the presets.
        dict(
            strategy=dataclasses.replace(_base_cfg().strategy, absgrad=False),
            fastgs=FastGSConfig(apply_presets=False),
        ),
    ]
    for overrides in cases:
        cfg = apply_fastgs_presets(_base_cfg(**overrides))
        try:
            validate_train_config(cfg)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {overrides}")

    # Presets repair a missing `strategy.absgrad` on their own.
    repaired = apply_fastgs_presets(
        _base_cfg(strategy=dataclasses.replace(_base_cfg().strategy, absgrad=False))
    )
    validate_train_config(repaired)

    # The default fast config must validate.
    validate_train_config(apply_fastgs_presets(_base_cfg()))


def test_steps_scaler_scales_fastgs_schedules() -> None:
    cfg = _base_cfg()
    scaled = apply_steps_scaler(cfg=cfg, steps_scaler=0.5)
    assert scaled.fastgs.densify_every == cfg.fastgs.densify_every // 2
    assert (
        scaled.fastgs.final_prune_start_step == cfg.fastgs.final_prune_start_step // 2
    )
    assert scaled.fastgs.optim_phase1_end_step == cfg.fastgs.optim_phase1_end_step // 2
    assert scaled.fastgs.shN_every == cfg.fastgs.shN_every // 2
    # Scaled schedules must still be internally consistent.
    validate_train_config(apply_fastgs_presets(scaled))


def test_importance_threshold_defaults_per_backend() -> None:
    assert resolve_importance_thresh(FastGSConfig(score_backend="weighted")) == 0.15
    assert resolve_importance_thresh(FastGSConfig(score_backend="tracked")) == 0.6
    assert resolve_importance_thresh(FastGSConfig(importance_thresh=3.0)) == 3.0


# ---------------------------------------------------------------------------
# Optimizer gate
# ---------------------------------------------------------------------------


def test_step_gate_phases() -> None:
    gate = FastGSStepGate(
        phase1_end_step=100,
        phase2_end_step=200,
        shN_every=16,
        phase2_every=32,
        phase3_every=64,
        max_steps=300,
    )
    # Phase 1: everything but shN steps every iteration.
    assert gate.should_step(group="means", step=5)
    assert not gate.should_step(group="shN", step=5)
    assert gate.should_step(group="shN", step=15)  # 1-based step 16
    # Phase 2: every group throttled to phase2_every.
    assert not gate.should_step(group="means", step=100)  # 1-based 101
    assert gate.should_step(group="means", step=127)  # 1-based 128
    # Phase 3.
    assert gate.update_every(group="means", step=250) == 64
    assert gate.should_step(group="means", step=255)  # 1-based 256
    # The final step always flushes accumulated gradients.
    assert gate.should_step(group="means", step=299)


# ---------------------------------------------------------------------------
# View cache
# ---------------------------------------------------------------------------


def test_view_cache_is_bounded_and_lossless() -> None:
    cache = ViewCache(capacity=3)
    for i in range(5):
        pixels = torch.rand(1, 4, 6, 3)
        cache.observe(
            pixels=pixels,
            camtoworlds=torch.eye(4)[None] * (i + 1),
            Ks=torch.eye(3)[None],
            image_ids=torch.tensor([i]),
        )
        last = pixels
    assert len(cache) == 3
    views = cache.sample(num_views=10, generator=torch.Generator().manual_seed(0))
    assert len(views) == 3  # sampling is capped by what is cached
    # uint8 round-trip keeps the image faithful to ~1/255.
    newest = [v for v in views if int(v.image_id) == 4][0]
    assert torch.allclose(newest.pixels_u8.float() / 255.0, last[0], atol=1.0 / 255.0)


# ---------------------------------------------------------------------------
# Densification mechanics (CPU: gsplat's param ops are pure torch)
# ---------------------------------------------------------------------------


def _toy_params(n: int) -> tuple[dict, dict]:
    torch.manual_seed(0)
    params = {
        "means": torch.nn.Parameter(torch.randn(n, 3)),
        "scales": torch.nn.Parameter(torch.full((n, 3), math.log(0.02))),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.5))),
        "sh0": torch.nn.Parameter(torch.zeros(n, 1, 3)),
        "shN": torch.nn.Parameter(torch.zeros(n, 15, 3)),
    }
    optimizers = {name: torch.optim.Adam([p], lr=1e-3) for name, p in params.items()}
    return params, optimizers


def test_grow_and_prune_respect_scores() -> None:
    from friendly_splat.fastgs.scoring import MultiViewScores

    n = 8
    params, optimizers = _toy_params(n)
    strategy = FastGSStrategy(
        grow_grad2d=0.1,
        grad_abs_thresh=0.1,
        percent_dense=1.0,  # every Gaussian counts as "small" -> clone path
        importance_thresh=0.5,
        prune_budget_ratio=1.0,
        verbose=False,
    )
    state = strategy.initialize_state(scene_scale=1.0)
    state.update(
        {
            "grad2d": torch.zeros(n),
            "grad2d_abs": torch.zeros(n),
            "count": torch.ones(n),
            "radii_norm": torch.zeros(n),
        }
    )
    # Gaussians 0..3 have high gradients; only 0 and 1 are blamed for real error.
    state["grad2d"][:4] = 1.0
    state["grad2d_abs"][:4] = 1.0
    importance = torch.zeros(n)
    importance[:2] = 1.0
    scores = MultiViewScores(importance=importance, pruning=torch.zeros(n), num_views=4)

    n_dupli, n_split = strategy._grow_gs(params, optimizers, state, scores)
    assert (n_dupli, n_split) == (2, 0)
    assert int(params["means"].shape[0]) == n + 2
    # Cloned Gaussians are copies appended at the end.
    assert torch.allclose(params["means"][-2:], params["means"][:2])
    # Optimizer state stays in sync with the parameters.
    for name, opt in optimizers.items():
        assert opt.param_groups[0]["params"][0] is params[name]


def test_prune_keeps_scores_aligned_with_params() -> None:
    from friendly_splat.fastgs.scoring import MultiViewScores
    from friendly_splat.fastgs.strategy import _reindex_scores

    n = 6
    params, optimizers = _toy_params(n)
    with torch.no_grad():
        params["opacities"][:2] = torch.logit(torch.tensor(1e-4))  # prune candidates
    strategy = FastGSStrategy(prune_opa=0.005, prune_budget_ratio=1.0, verbose=False)
    state = strategy.initialize_state(scene_scale=1.0)
    state.update(
        {
            "grad2d": torch.arange(n, dtype=torch.float32),
            "grad2d_abs": torch.arange(n, dtype=torch.float32),
            "count": torch.ones(n),
            "radii_norm": torch.zeros(n),
        }
    )
    scores = MultiViewScores(
        importance=torch.arange(n, dtype=torch.float32),
        pruning=torch.zeros(n),
        num_views=4,
    )
    n_prune, keep = strategy._prune_gs(params, optimizers, state, 0, scores)
    assert n_prune == 2
    assert keep is not None and int(keep.sum()) == n - 2
    # `remove` reindexes the running state; scores must follow the same order so
    # the multi-view filter keeps pointing at the right Gaussians.
    scores = _reindex_scores(scores=scores, keep=keep)
    assert int(scores.importance.numel()) == int(params["means"].shape[0])
    assert torch.allclose(scores.importance, state["grad2d"])


def test_prune_budget_removes_the_worst_scored_candidates() -> None:
    from friendly_splat.fastgs.scoring import MultiViewScores

    n = 6
    params, optimizers = _toy_params(n)
    with torch.no_grad():
        params["opacities"][:4] = torch.logit(torch.tensor(1e-4))
    strategy = FastGSStrategy(prune_opa=0.005, prune_budget_ratio=0.5, verbose=False)
    state = strategy.initialize_state(scene_scale=1.0)
    state.update(
        {
            "grad2d": torch.zeros(n),
            "grad2d_abs": torch.zeros(n),
            "count": torch.ones(n),
            "radii_norm": torch.zeros(n),
        }
    )
    pruning = torch.tensor([0.1, 0.9, 0.2, 0.8, 1.0, 1.0])
    scores = MultiViewScores(importance=torch.zeros(n), pruning=pruning, num_views=4)
    means_before = params["means"].detach().clone()
    n_prune, keep = strategy._prune_gs(params, optimizers, state, 0, scores)
    # Half of the 4 candidates, the two the score blames most (indices 1 and 3).
    assert n_prune == 2
    assert keep.tolist() == [True, False, True, False, True, True]
    assert torch.allclose(params["means"], means_before[keep])


# ---------------------------------------------------------------------------
# Scoring (CUDA)
# ---------------------------------------------------------------------------


def test_weighted_scores_match_accumulated_alpha() -> None:
    """The weighted backend must return `sum_{p in mask} alpha_p * T_p`.

    Summing over Gaussians collapses that to the rendered alpha of the masked
    pixels (`sum_i alpha_ip * T_ip == 1 - T_final,p`), which gives an independent
    reference for the autograd trick.
    """
    if not torch.cuda.is_available():
        print("SKIP: test_weighted_scores_match_accumulated_alpha (no CUDA)")
        return

    from friendly_splat.fastgs.scoring import CachedView
    from friendly_splat.modules.gaussian import GaussianModel
    from gsplat.rendering import rasterization

    device = torch.device("cuda")
    torch.manual_seed(0)
    model = GaussianModel.from_random(
        num_points=4000,
        scene_scale=1.0,
        init_extent=1.0,
        sh_degree=0,
        init_scale=1.0,
        init_opacity=0.3,
        device=device,
    )
    width, height = 64, 48
    focal = 60.0
    K = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    camtoworld = torch.eye(4, device=device)
    camtoworld[2, 3] = -4.0  # pull the camera back so the cloud is in frame
    view = CachedView(
        pixels_u8=torch.randint(
            0, 256, (height, width, 3), dtype=torch.uint8, device=device
        ),
        camtoworld=camtoworld,
        K=K,
    )

    scores = compute_multiview_scores(
        gaussian_model=model,
        views=[view],
        fast_cfg=FastGSConfig(loss_thresh=0.05, score_backend="weighted"),
        sh_degree=0,
        ssim_lambda=0.2,
        rasterize_mode="classic",
        need_importance=True,
    )
    assert scores is not None and scores.importance is not None

    tensors = model.to_render_tensors(sh_degree=0)
    with torch.no_grad():
        renders, alphas, _ = rasterization(
            means=tensors["means"],
            quats=tensors["quats"],
            scales=tensors["scales"],
            opacities=tensors["opacities"],
            colors=tensors["colors"],
            viewmats=torch.linalg.inv(camtoworld)[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=0,
            packed=False,
            render_mode="RGB",
        )
        pred = renders[..., 0:3].clamp(0.0, 1.0)
        gt = (view.pixels_u8.float() / 255.0)[None]
        l1 = (pred - gt).abs().mean(dim=-1)[0]
        l1_norm = (l1 - l1.min()) / (l1.max() - l1.min()).clamp_min(1e-12)
        mask = l1_norm > 0.05
        reference = float(alphas[0, ..., 0][mask].sum())

    total = float(scores.importance.sum())
    assert mask.any(), "the test image should flag some pixels"
    assert reference > 0.0
    rel = abs(total - reference) / reference
    print(
        f"weighted scores: sum={total:.3f} accumulated alpha={reference:.3f} "
        f"rel_err={rel:.2e}"
    )
    assert rel < 1e-3, f"weighted scores off by {rel:.3e}"


def _run_as_script() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"OK: {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAILED: {fn.__name__}: {exc}")
            import traceback

            traceback.print_exc()
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_as_script())
