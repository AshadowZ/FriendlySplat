# FastGS (`--fast`)

Integration of [FastGS: Training 3D Gaussian Splatting in 100 Seconds](https://github.com/fastgs/FastGS)
(Ren et al., CVPR 2026) into FriendlySplat.

FastGS is an *acceleration recipe* rather than a new representation: the splats it
produces are ordinary 3D Gaussians. It gets its speed from three ideas:

1. **Multi-view consistent densification.** Every densification event renders a
   handful of views, finds the badly reconstructed pixels, and measures how much
   each Gaussian is responsible for them. Only Gaussians that are both
   high-gradient *and* responsible for real error are allowed to grow, which
   keeps the Gaussian count (and therefore every later iteration) small.
2. **Multi-view consistent pruning.** The same score decides which Gaussians to
   drop, both during densification and in an aggressive pass after it.
3. **A sparse-in-time optimizer schedule.** Once the geometry has settled,
   parameters stop being updated on every iteration (gradients accumulate
   instead), which makes the tail of training nearly free.

Everything is inert unless `--fast` is passed.

## Usage

```bash
# FastGS "base" (fastest; densify every 500 steps -- the default here)
fs-train \
  --io.data-dir /path/to/data-dir \
  --io.result-dir /path/to/result-dir \
  --fast

# FastGS "big" (higher quality; densify every 100 steps)
fs-train \
  --io.data-dir /path/to/data-dir \
  --io.result-dir /path/to/result-dir \
  --fast \
  --fastgs.densify-every 100 \
  --fastgs.grad-abs-thresh 0.0008
```

`--fast` replaces `--strategy.impl`, so it composes with the rest of
FriendlySplat (bilateral grid, pose optimization, priors, exporters, viewer,
evaluation) but not with `--strategy.impl mcmc`. Every knob lives under
`--fastgs.*` — see the [parameter reference](#parameter-reference) below, or run
`fs-train --help` for the raw list.

## What `--fast` changes

| Area | Effect |
| --- | --- |
| Densification | `FastGSStrategy` replaces the configured strategy: clone on mean 2D gradient (`--strategy.grow-grad2d`), split on absolute gradient (`--fastgs.grad-abs-thresh`), both gated by the multi-view importance score |
| Pruning | per-event opacity/size pruning limited to `--fastgs.prune-budget-ratio` of the candidates (worst score first), plus a post-densification pass (`--fastgs.final-prune-*`) |
| Optimizers | per-group step cadence (`--fastgs.optim-*`, `--fastgs.shN-every`, `--fastgs.phase{2,3}-every`); skipped iterations accumulate gradients |
| Presets | `strategy.refine_every`, `strategy.absgrad` and the opacity/SH learning rates are set to FastGS's values (printed at startup; disable with `--fastgs.no-apply-presets`) |

## Parameter reference

Every FastGS decision is a flag — nothing about the recipe is hard-coded. `--fast`
switches the whole thing on; everything below lives under `--fastgs.*` and is
only read when `--fast` is set. Boolean flags also have a `--fastgs.no-*` form.
Where FastGS reuses a FriendlySplat knob, that is called out instead of being
duplicated.

### 1. Multi-view consistency scoring

Which pixels count as badly reconstructed, and which Gaussians are blamed for
them. This is the part that makes FastGS more than AbsGS.

| flag | default | what it does | trade-off |
| --- | --- | --- | --- |
| `--fastgs.score-num-views` | `10` | views scored per densification event | higher = steadier scores, but each view costs 1–2 extra renders per event; lower = cheaper and noisier |
| `--fastgs.view-cache-size` | `None` → `2 x score_num_views` | size of the recent-view pool sampled from | higher = more view diversity, more GPU memory (one uint8 `H x W x 3` per view); lower = less memory, more correlated samples |
| `--fastgs.loss-thresh` | `0.1` | threshold on the min-max normalized per-pixel L1 error that defines the "bad pixel" map | lower = flags more pixels, so more Gaussians clear the filter and the model grows; higher = focuses only on the worst regions. Upstream tunes this per scene (garden uses `0.06`) |
| `--fastgs.score-backend` | `weighted` | how per-Gaussian responsibility is measured | `weighted`: complete, no thresholds, bounded memory, 2 renders/view. `tracked`: hard counts, 1 render/view, but allocates `H x W x ceil(1/tracked_vis_thresh)` index pairs and needs `--optim.no-packed` |
| `--fastgs.tracked-vis-thresh` | `0.05` | (`tracked` only) minimum `alpha * T` for a Gaussian to be counted at a pixel | higher = cheaper and sparser counts; lower = closer to upstream's "any contribution" counter but memory grows as `1/thresh` per pixel |
| `--fastgs.importance-thresh` | `None` → `0.15` (`weighted`) / `0.6` (`tracked`) | minimum per-view responsibility required to densify a Gaussian | higher = fewer Gaussians grow (at `0.5` the count actually *shrank* over training); lower = more growth. **Not** a capacity control at `densify-every 100` — the gradient criteria reach the budget regardless, so cap the count with `--strategy.densification-budget` |

### 2. Densification

| flag | default | what it does | trade-off |
| --- | --- | --- | --- |
| `--fastgs.densify-every` | `100` | densification interval; FastGS calls `100` "big" and `500` "base" | shorter = more events, better placement, more scoring overhead; longer = faster training, coarser placement |
| `--fastgs.grad-abs-thresh` | `0.0012` | AbsGS absolute-gradient threshold for **splitting** | lower = more splits, more detail, more Gaussians. Upstream tunes `0.0002`–`0.002` per scene (garden uses `0.0003`) |
| `--fastgs.percent-dense` | `0.001` | scale boundary (fraction of scene extent) between clone and split | higher = more Gaussians take the clone path; lower = split-dominated growth |

Reused from FriendlySplat: `--strategy.grow-grad2d` (the *clone* gradient
threshold), `--strategy.densification-budget` (hard cap on the Gaussian count —
the real capacity control), `--strategy.refine-start-iter`,
`--strategy.refine-stop-iter`, `--strategy.reset-every`, `--strategy.absgrad`.

### 3. Pruning during densification

| flag | default | what it does | trade-off |
| --- | --- | --- | --- |
| `--fastgs.prune-budget-ratio` | `0.5` | fraction of the opacity/size prune candidates removed per event, worst multi-view score first | `1.0` prunes every candidate (vanilla 3DGS behaviour) and keeps the model leaner; lower keeps borderline Gaussians alive for another interval |
| `--fastgs.opacity-clamp` | `0.8` | opacities are clamped to this after every event; `1.0` disables | lower = more conservative opacities and more churn, which can help escape bad local minima but slows convergence |
| `--fastgs.opacity-reset-multiplier` | `2.0` | periodic reset target = this x `--strategy.prune-opa` | FastGS and vanilla 3DGS use `2.0` (reset 0.01 vs prune 0.005); gsplat's `improved` uses `10.0`. Higher leaves more headroom above the prune threshold, so fewer Gaussians are culled right after a reset |

Reused from FriendlySplat: `--strategy.prune-opa`, `--strategy.prune-scale3d`,
`--strategy.prune-scale2d`, `--strategy.refine-scale2d-stop-iter` (see deviation
4 below for why the 2D rule is normalized rather than a pixel count).

### 4. Post-densification pruning

FastGS's aggressive late pass: once the model has converged, Gaussians the
multi-view score still blames are removed outright.

| flag | default | what it does | trade-off |
| --- | --- | --- | --- |
| `--fastgs.final-prune-enable` | `true` | enable the late pass | on = a smaller final model; measured cost on garden was ~0.06 dB for 11% fewer Gaussians. Turn it off for maximum quality |
| `--fastgs.final-prune-start-step` | `18000` | first event (1-based); must be after densification stops | earlier = more time to recover from the prune, but the model is less converged when it is judged |
| `--fastgs.final-prune-every` | `3000` | cadence | more events = more compaction and more scoring cost |
| `--fastgs.final-prune-stop-step` | `27000` | last event | later = compaction closer to the final model, with less time left to recover |
| `--fastgs.final-prune-min-opacity` | `0.1` | opacity below this is removed | higher = far more aggressive (this is 20x the densification-phase threshold) |
| `--fastgs.final-prune-score-thresh` | `0.9` | normalized pruning score above which a Gaussian is removed | lower = removes more of the "blamed" tail; `1.0` effectively disables the score criterion |

### 5. Sparse-in-time optimizer schedule

Skipped iterations accumulate gradients rather than dropping them, so this trades
update frequency for wall time.

| flag | default | what it does | trade-off |
| --- | --- | --- | --- |
| `--fastgs.optim-gate-enable` | `true` | enable the schedule | off = every group steps every iteration (slower, more updates) |
| `--fastgs.optim-phase1-end-step` | `15000` | end of phase 1; must cover the densification window | — |
| `--fastgs.phase1-gated-groups` | `("shN",)` | which groups are throttled during phase 1 | throttling more groups saves time but needs their learning rates raised to compensate, which is exactly why FastGS raises the SH rate |
| `--fastgs.shN-every` | `16` | phase-1 period for the gated groups | larger = cheaper, but fewer effective SH updates |
| `--fastgs.optim-phase2-end-step` | `20000` | end of phase 2 | — |
| `--fastgs.phase2-every` | `32` | phase-2 period for **all** groups | larger = faster tail, less refinement |
| `--fastgs.phase3-every` | `64` | phase-3 period for all groups | as above, more so |

### 6. Learning-rate presets

| flag | default | what it does | trade-off |
| --- | --- | --- | --- |
| `--fastgs.apply-presets` | `true` | apply FastGS's LR/cadence values over FriendlySplat's defaults (printed at startup) | off = keep FriendlySplat's own LRs and take only FastGS's densification/pruning/optimizer schedule |
| `--fastgs.opacity-lr` | `0.025` | opacity LR (FriendlySplat default is `0.05`) | lower = steadier opacities, slower recovery after a reset |
| `--fastgs.sh0-lr` | `0.0025` | SH DC LR | — |
| `--fastgs.shN-lr` | `0.005` | high-order SH LR; divided by 20 into the optimizer, as upstream does | must go up when `shN-every` goes up. Upstream uses `0.02` for garden and the indoor scenes, `0.04` for `truck` |

### 7. Logging

| flag | default | what it does |
| --- | --- | --- |
| `--fastgs.verbose` | `true` | print per-event densify/prune counts and the score distribution — the fastest way to calibrate `importance-thresh` |

## Recipes

Measured on garden (see the table further down); pick by what you are optimizing.

**Highest quality** — beats the `improved` baseline at equal capacity:

```bash
--fast --fastgs.importance-thresh 0.15 --fastgs.no-final-prune-enable \
  --fastgs.grad-abs-thresh 0.0003 --fastgs.loss-thresh 0.06 --fastgs.shN-lr 0.02
```

**Balanced** — the defaults, no extra flags:

```bash
--fast
```

**Smallest/fastest** — roughly 6x fewer Gaussians and 3x less time for about
1.3 dB on garden:

```bash
--fast --fastgs.densify-every 500 --fastgs.importance-thresh 0.3
```

**Fixed Gaussian budget** — the honest way to compare against another strategy,
since capacity dominates quality:

```bash
--fast --strategy.densification-budget 1000000
```

## Scoring backends

Upstream FastGS gets the per-Gaussian counts from a patched CUDA rasterizer
(`diff_gaussian_rasterization_fastgs` increments a per-Gaussian counter whenever
it composites a flagged pixel). FriendlySplat renders with gsplat, so the same
quantity is reconstructed through gsplat's public API. Two backends are
available via `--fastgs.score-backend`:

- **`weighted`** (default) renders a 1-channel per-Gaussian feature and reads the
  gradient of the masked sum. Because `out_p = sum_i s_i * alpha_ip * T_ip`, the
  gradient w.r.t. `s_i` is exactly `sum_{p in mask} alpha_ip * T_ip`: the
  *expected number* of badly reconstructed pixels Gaussian `i` accounts for.
  Complete (no threshold, no truncation), bounded memory, two renders per view.
- **`tracked`** uses gsplat's `track_pixel_gaussians` to get hard pixel counts
  for contributions above `--fastgs.tracked-vis-thresh`, which is closest to the
  upstream counter and needs only one render per view. It allocates
  `H * W * ceil(1 / tracked_vis_thresh)` index pairs per view and requires
  `--optim.no-packed`.

Both are min-max normalized identically for the pruning score, and the default
thresholds put them on comparable footing: on `bonsai` at 1/4 resolution, 120
steps in, the mean importance was `0.71` (`weighted`) vs `2.64` (`tracked`), and
the defaults selected `8.8%` vs `8.0%` of the Gaussians.

## Measured behaviour

`garden` from MipNeRF360 under the community protocol (`images_4`, every 8th
image held out of training, 30k steps, inria PSNR/SSIM + VGG LPIPS, no bilateral
grid, no geometry priors), Tesla T4, one seed per cell. `--fast` uses
`--fastgs.densify-every 100 --fastgs.grad-abs-thresh 0.0003
--fastgs.loss-thresh 0.06 --fastgs.shN-lr 0.02 --fastgs.importance-thresh 0.15
--fastgs.no-final-prune-enable`; the baseline is `--strategy.impl improved`.
Both sides get the same hard `--strategy.densification-budget`.

| budget | strategy | PSNR | SSIM | LPIPS | wall |
| --- | --- | --- | --- | --- | --- |
| 400k | `improved` | 25.754 | 0.7916 | 0.2290 | 1273 s |
| 400k | `--fast` | **26.211** | **0.8055** | **0.2181** | **1243 s** |
| 1M | `improved` | **26.860** | 0.8445 | 0.1463 | 2085 s |
| 1M | `--fast` | 26.825 | **0.8505** | **0.1360** | **2013 s** |
| 2.5M | `improved` | 27.428 | 0.8684 | 0.1043 | 3816 s |
| 2.5M | `--fast` | **27.571** | **0.8706** | **0.1005** | **3586 s** |

`--fast` wins SSIM and LPIPS at every budget and is 2-6% faster at every budget.
PSNR is +0.457 at 400k, -0.035 at 1M and +0.143 at 2.5M: it wins two of three
and ties the middle, so there is no monotonic trend to claim. Everything here is
single-seed and 3DGS run-to-run variance is typically 0.05-0.10 dB, so treat the
1M PSNR gap as a tie and the 2.5M one as marginal.

Two component measurements on the same protocol (garden, 1M budget):

| variant | PSNR | SSIM | LPIPS | wall |
| --- | --- | --- | --- | --- |
| `--fast` (gate on, `weighted`) | 26.825 | 0.8505 | 0.1360 | 2013 s |
| `--fastgs.no-optim-gate-enable` | 27.100 | 0.8502 | 0.1360 | 2394 s |
| `--fastgs.score-backend tracked` | 27.076 | 0.8509 | 0.1355 | 1979 s |

The optimizer gate buys ~19% wall time for ~0.27 dB PSNR (SSIM and LPIPS are
unchanged), so turn it off with `--fastgs.no-optim-gate-enable` when quality
matters more than speed. The `tracked` backend matched or slightly beat
`weighted` while being marginally faster — but the two ran at their own
importance thresholds (0.6 vs 0.15) rather than at matched selectivity, so part
of that gap may be the threshold rather than the backend.

For reference, published 3DGS on garden at 30k is usually quoted near
27.3-27.4 PSNR / 0.868 SSIM, so the baseline is a credible target rather than a
weak one. Full protocol, commands and the ablation behind the `prune_scale2d` decision
below: [`benchmarks/fastgs/`](../../benchmarks/fastgs/README.md).

## Deliberate differences from upstream

These are worth knowing when comparing numbers with the FastGS paper:

1. **Scored views.** Upstream samples 10 cameras from the full training set at
   every event, which it can do for free because it keeps all images resident on
   the GPU. FriendlySplat streams images through a DataLoader, so we score the
   last `--fastgs.view-cache-size` *trained* views (kept as uint8) instead. The
   trainer samples views uniformly at random, so this is still a uniform sample,
   and it keeps scoring off the I/O path.
2. **Prune before grow.** gsplat's clone/split ops append and reorder Gaussians,
   which would invalidate the score vector. We therefore prune first (scores
   aligned), reindex the scores with the survivors, then grow. Upstream grows
   first and prunes afterwards; a Gaussian about to be pruned is not worth
   cloning anyway.
3. **Prune-budget selection.** Upstream samples the removal budget stochastically
   with weights `1 / (1 - pruning_score)` over *all* Gaussians and intersects
   that with the candidate set. We deterministically remove the
   `prune_budget_ratio` fraction of candidates with the highest pruning score,
   which expresses the same intent (drop the Gaussians the multi-view score
   blames most) without depending on the sampling overlap.
4. **Projected-size pruning is resolution-normalized.** FastGS inherits vanilla
   3DGS's `max_radii2D > 20 px`, calibrated for ~1.6K-wide images. Ported
   literally it deletes the coarse structure of the scene at lower resolutions:
   on `bonsai` at 1/4 resolution (7.5k steps, `steps_scaler 0.25`) it cost
   **10.4 dB** — 18.09 dB with the 20 px rule vs 28.49 dB without it, while the
   `improved` baseline scored 30.93 dB. `--fast` therefore uses FriendlySplat's
   own `strategy.prune_scale2d` (a fraction of the larger image side, default
   0.15) gated by `strategy.refine_scale2d_stop_iter`, exactly like the other
   strategies in this repo.
5. **Gradient bookkeeping** uses gsplat's screen-space normalization
   (`grad * width / 2 * n_cameras`) rather than the original NDC accumulation.
   The thresholds are numerically comparable, but not bit-identical.
6. **No rasterizer changes.** Upstream also ships Speedy-Splat's tighter tile
   bounds (its `--mult` flag). That lives inside its CUDA rasterizer and has no
   equivalent knob in gsplat, so it is not part of this integration; the speedup
   here comes from the densification, pruning and optimizer schedule only.
7. **`--optim.visible-adam`** composes with the gate: because gradients
   accumulate over several renders, `OptimizerCoordinator` feeds `SelectiveAdam`
   the union of the visibility masks seen since that group last stepped, rather
   than the last render's mask alone.
8. **Learning rates** follow FriendlySplat's (gsplat-style) parameterization, so
   only the values FastGS actually retunes are overridden (opacity, SH DC, SH
   rest); the position LR schedule stays FriendlySplat's.

## Layout

| File | Contents |
| --- | --- |
| `config.py` | `FastGSConfig` (everything under `--fastgs.*`) |
| `presets.py` | FastGS's LR / cadence overrides applied when `--fast` is set |
| `validate.py` | validation for the `--fast` code path |
| `scoring.py` | view cache + multi-view consistency scores (upstream `utils/fast_utils.py`) |
| `strategy.py` | `FastGSStrategy`, a gsplat `Strategy` (upstream `densify_and_prune_fastgs`) |
| `optim_schedule.py` | `FastGSStepGate` (upstream `GaussianModel.optimizer_step`) |
| `runtime.py` | `FastGSRuntime`: wiring, scoring events, post-densification pruning (upstream `final_prune_fastgs`) |

Integration points outside this folder are intentionally minimal and all guarded
by `cfg.fast`:

- `trainer/configs.py` — `fast: bool` / `fastgs: FastGSConfig` fields, step
  scaling, one validation call.
- `trainer/builder.py` — build the runtime and use its strategy.
- `trainer/optimizer_coordinator.py` — optional per-group step gate.
- `train_app.py` — apply presets, record each batch, run the late prune.

Tests live in [`tests/test_fastgs.py`](../../tests/test_fastgs.py) (`pytest
tests/test_fastgs.py -s`, or run the file directly). They cover the config
toggle/presets/validation, the optimizer gate, the view cache, the
densification/pruning mechanics including score realignment, and — on CUDA — a
reference check that the `weighted` backend really returns
`sum_{p in mask} alpha_p * T_p` (summed over Gaussians it must equal the
rendered alpha of the flagged pixels; it matches to ~4e-6 relative error).

## Citation

```bibtex
@article{ren2025fastgs,
  title={FastGS: Training 3D Gaussian Splatting in 100 Seconds},
  author={Ren, Shiwei and Wen, Tianci and Fang, Yongchun and Lu, Biao},
  journal={arXiv preprint arXiv:2511.04283},
  year={2025}
}
```

FastGS itself builds on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting),
[Taming-3DGS](https://github.com/humansensinglab/taming-3dgs),
[Speedy-Splat](https://github.com/j-alex-hanson/speedy-splat) and
[AbsGS](https://github.com/TY424/AbsGS).
