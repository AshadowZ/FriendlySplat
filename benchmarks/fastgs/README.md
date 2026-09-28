# FastGS benchmark (`--fast`)

Measurements behind the claims in
[`friendly_splat/fastgs/README.md`](../../friendly_splat/fastgs/README.md).
Every number here is a single seed on a single **NVIDIA Tesla T4**; 3DGS
run-to-run variance is typically 0.05–0.10 dB, so differences below ~0.1 dB
should be read as ties.

## Protocol

The standard MipNeRF360 / 3DGS evaluation protocol, matching
[3DGS's `metrics.py`](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/metrics.py)
and [FastGS's `full_eval.py`](https://github.com/fastgs/FastGS/blob/main/full_eval.py):

- `garden`, `images_4` (1297x840) — outdoor MipNeRF360 scenes use factor 4
- every 8th image **held out of training** (161 train / 24 test)
- 30 000 steps
- inria PSNR + inria SSIM (11x11 Gaussian window, sigma 1.5) + VGG LPIPS
- no bilateral grid, no geometry priors, no `visible_adam`
- both strategies given the same hard `--strategy.densification-budget`

```bash
# baseline
fs-train --io.data-dir ./data/garden --io.result-dir ./output/garden-360base \
  --data.data-factor 4 --data.test-every 8 --data.benchmark-train-split \
  --optim.max-steps 30000 --strategy.impl improved \
  --strategy.densification-budget 2500000 \
  --eval.enable --eval.eval-every-n 30000 \
  --eval.metrics-backend inria --eval.lpips-net vgg --viewer.disable-viewer

# FastGS (same flags, plus)
  --fast --fastgs.densify-every 100 --fastgs.grad-abs-thresh 0.0003 \
  --fastgs.loss-thresh 0.06 --fastgs.shN-lr 0.02 \
  --fastgs.importance-thresh 0.15 --fastgs.no-final-prune-enable
```

## Budget-matched results

| Gaussians | strategy | PSNR | SSIM | LPIPS | wall |
| --- | --- | --- | --- | --- | --- |
| 400k | `improved` | 25.754 | 0.7916 | 0.2290 | 1272.8 s |
| 400k | `--fast` | **26.211** | **0.8055** | **0.2181** | **1243.2 s** |
| 1M | `improved` | **26.860** | 0.8445 | 0.1463 | 2085.1 s |
| 1M | `--fast` | 26.825 | **0.8505** | **0.1360** | **2013.3 s** |
| 2.5M | `improved` | 27.428 | 0.8684 | 0.1043 | 3816.2 s |
| 2.5M | `--fast` | **27.571** | **0.8706** | **0.1005** | **3586.0 s** |

What holds at every budget: `--fast` wins SSIM (+0.0139 / +0.0060 / +0.0022),
wins LPIPS (−0.0109 / −0.0103 / −0.0038) and is faster (−2.3% / −3.4% / −6.0%).

What does **not** hold: PSNR is +0.457 at 400k, −0.035 at 1M and +0.143 at 2.5M.
That is not monotonic, so "the advantage grows as capacity shrinks" is not a
supported claim. A consistent LPIPS/SSIM edge with a wobbly PSNR edge is
consistent with better perceptual placement of detail rather than better pixel
fidelity.

For reference, published 3DGS on `garden` at 30k is usually quoted near
27.3–27.4 PSNR / 0.868 SSIM, so the baseline here is a credible target.

## Component measurements

Same protocol, 1M budget, FastGS knobs as above.

| variant | PSNR | SSIM | LPIPS | wall |
| --- | --- | --- | --- | --- |
| `--fast` (gate on, `weighted`) | 26.825 | 0.8505 | 0.1360 | 2013.3 s |
| `--fastgs.no-optim-gate-enable` | 27.100 | 0.8502 | 0.1360 | 2394.2 s |
| `--fastgs.score-backend tracked` | 27.076 | 0.8509 | 0.1355 | 1979.2 s |

- **Optimizer gate**: worth ~19% wall time for ~0.27 dB PSNR (SSIM/LPIPS
  unchanged). Keep it on when speed is the point; pass
  `--fastgs.no-optim-gate-enable` when it is not.
- **Scoring backend**: `tracked` matched or slightly beat `weighted` and was
  1.7% faster. Caveat: the two ran at their own importance thresholds (0.6 vs
  0.15) rather than at matched selectivity, so part of that gap may be the
  threshold rather than the backend.

## Projected-size pruning (why `prune_scale2d` replaces FastGS's 20 px rule)

Ablation on `bonsai` at 1/4 resolution, `--optim.steps-scaler 0.25` (7500 steps),
test views held out:

| run | PSNR @3750 | PSNR @7500 | final Gaussians | wall |
| --- | --- | --- | --- | --- |
| `improved` baseline | 29.461 | 30.928 | ~1 000 000 | 825.3 s |
| `--fast`, 20 px screen prune | 11.428 | 18.092 | 78 715 | 118.1 s |
| `--fast`, rule disabled | 26.377 | 28.494 | 141 948 | 152.2 s |
| `--fast`, resolution-normalized (shipped) | 25.017 | 28.543 | 141 376 | 152.3 s |

Porting FastGS/vanilla-3DGS's absolute `max_radii2D > 20 px` literally costs
**10.4 dB** here. That threshold is calibrated for ~1.6K-wide images; at 780x520
it deletes the large Gaussians carrying the coarse scene structure. `--fast`
therefore uses FriendlySplat's resolution-normalized `strategy.prune_scale2d`
(fraction of the larger image side, default 0.15) gated by
`strategy.refine_scale2d_stop_iter`, like the repo's other strategies.
