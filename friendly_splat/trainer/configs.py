from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class IOConfig:
    # Path to the dataset root directory.
    data_dir: str
    # Directory to save outputs (checkpoints, PLY exports, etc.).
    result_dir: str = "results"
    # Torch device string, e.g. "cuda", "cuda:0", or "cpu".
    device: str = "cuda"
    # Global random seed (data sampling + initialization).
    seed: int = 42

    # Whether to export splats as PLY files during training.
    export_ply: bool = False
    # PLY export format: "ply" or "ply_compressed".
    ply_format: str = "ply"
    # 1-based training step numbers to export PLY files. e.g. (15000, 30000)
    ply_steps: Tuple[int, ...] = (30000,)

    # Whether to save checkpoint(s) during training.
    save_ckpt: bool = False
    # 1-based training step numbers to save checkpoints. e.g. (15000, 30000)
    save_steps: Tuple[int, ...] = (30_000,)


@dataclass(frozen=True)
class DataConfig:
    # Downsample factor for the dataset images.
    data_factor: float = 1.0
    # Normalize the world space based on COLMAP points/cameras.
    normalize_world_space: bool = True
    # Whether to align world axes during normalization (z-up + PCA principal axes).
    # When False, normalization becomes translation+uniform scale only (no rotation).
    # This keeps the world axes aligned with COLMAP and avoids SH coefficient rotation
    # when exporting/evaluating in COLMAP coordinates.
    align_world_axes: bool = False
    # Every N images there is a test image.
    test_every: int = 8
    # Benchmark split mode for training:
    # - False: `split="train"` uses all images.
    # - True: `split="train"` excludes every `test_every`-th image (train/test are disjoint).
    benchmark_train_split: bool = False
    # DataLoader preload mode: "none" or "cuda".
    # - "none": load samples on-demand on CPU; batches are moved to the training device each step.
    # - "cuda": preload the entire dataset to CUDA once inside DataLoader (uses GPU memory).
    preload: str = "none"
    # Batch size for training.
    batch_size: int = 1
    # Number of DataLoader workers.
    num_workers: Optional[int] = 8
    # Use an infinite sampler to avoid DataLoader epoch-boundary stalls.
    infinite_sampler: bool = True
    # Asynchronously prefetch the next batch to GPU (hides H2D latency for non-cuda preload).
    prefetch_to_gpu: bool = True

    ## Optional priors / masks (resolved relative to dataset root).

    # Folder name for dense depth priors (sibling of "images"); .npy files.
    depth_dir_name: Optional[str] = None
    # Folder name for dense normal priors (sibling of "images"); .png files.
    normal_dir_name: Optional[str] = None
    # Folder name for dynamic object masks (optional).
    dynamic_mask_dir_name: Optional[str] = None
    # Folder name for sky masks (optional).
    sky_mask_dir_name: Optional[str] = None


@dataclass(frozen=True)
class InitConfig:
    # Initialization strategy: "sfm" (COLMAP points) or "random".
    init_type: str = "sfm"
    # Initial number of Gaussians (ignored when init_type="sfm").
    init_num_pts: int = 100_000
    # Initial extent for random init, as a multiple of scene scale (ignored for "sfm").
    init_extent: float = 3.0
    # Initial opacity for Gaussians (before logit).
    init_opacity: float = 0.1
    # Initial scale multiplier for Gaussians (applied to KNN-based scale estimate).
    init_scale: float = 1.0


@dataclass(frozen=True)
class PoseConfig:
    ## Pose optimization (experimental).

    # Whether to optimize per-image camera pose adjustments during training.
    pose_opt: bool = False


@dataclass(frozen=True)
class PostprocessConfig:
    ## Post-processing modules (experimental).
    # Postprocess modules apply to rendered RGB and help absorb cross-frame photometric drift.

    # Whether to enable fused bilateral grid post-processing (requires `fused_bilagrid`).
    use_bilateral_grid: bool = False
    # Bilateral grid resolution (X, Y, W).
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    # TV regularization weight for bilateral grid.
    bilateral_grid_tv_weight: float = 10.0


@dataclass(frozen=True)
class ViewerConfig:
    # Whether to disable the online viewer (viser/nerfview).
    # When enabled, training can be paused/resumed and rendered interactively.
    disable_viewer: bool = False
    # Port for the viewer server.
    port: int = 8080
    # Keep the viewer process alive after training finishes (Ctrl+C to exit).
    keep_alive_after_train: bool = True


@dataclass(frozen=True)
class TensorBoardConfig:
    # Enable TensorBoard scalar logging.
    enable: bool = False
    # Log training scalars every N training steps (1-based).
    every_n: int = 100
    # Flush TensorBoard event file every N logged training steps.
    flush_every_n: int = 500


@dataclass(frozen=True)
class EvalConfig:
    # Enable periodic evaluation on holdout images.
    enable: bool = False
    # Split name to evaluate. ColmapDataParser treats non-"train" as holdout split.
    split: str = "test"
    # Run evaluation every N training steps (1-based, e.g. 1000 means 1000,2000,...).
    eval_every_n: int = 1000
    # Optional cap on evaluated images (None means full split).
    max_images: Optional[int] = None
    # LPIPS backbone for evaluation.
    lpips_net: Literal["alex", "vgg"] = "alex"
    # Metric implementation backend:
    # - "gsplat": torchmetrics-based metrics (current default)
    # - "inria": inria/gaussian-splatting-style metrics
    metrics_backend: Literal["gsplat", "inria"] = "gsplat"
    # Whether to compute color-corrected metrics (cc_psnr/cc_ssim/cc_lpips).
    # Effective only when postprocess uses a photometric adapter (e.g. bilateral grid).
    compute_cc_metrics: bool = True


@dataclass(frozen=True)
class AdamOptimizerConfig:
    lr: float
    eps: float = 1e-15
    weight_decay: float = 0.0


@dataclass(frozen=True)
class ExponentialDecaySchedulerConfig:
    lr_final: float
    # If None, use `OptimConfig.max_steps`.
    max_steps: Optional[int] = None
    warmup_steps: int = 0


@dataclass(frozen=True)
class OptimizerConfigEntry:
    optimizer: AdamOptimizerConfig
    scheduler: Optional[ExponentialDecaySchedulerConfig] = None
    lr_mult_scene_scale: bool = False


@dataclass(frozen=True)
class OptimizersConfig:
    means: OptimizerConfigEntry = field(
        default_factory=lambda: OptimizerConfigEntry(
            optimizer=AdamOptimizerConfig(lr=4e-5),
            scheduler=ExponentialDecaySchedulerConfig(lr_final=4e-7),
            lr_mult_scene_scale=True,
        )
    )
    scales: OptimizerConfigEntry = field(
        default_factory=lambda: OptimizerConfigEntry(
            optimizer=AdamOptimizerConfig(lr=5e-3),
            scheduler=None,
        )
    )
    quats: OptimizerConfigEntry = field(
        default_factory=lambda: OptimizerConfigEntry(
            optimizer=AdamOptimizerConfig(lr=1e-3),
            scheduler=None,
        )
    )
    opacities: OptimizerConfigEntry = field(
        default_factory=lambda: OptimizerConfigEntry(
            optimizer=AdamOptimizerConfig(lr=5e-2),
            scheduler=None,
        )
    )
    sh0: OptimizerConfigEntry = field(
        default_factory=lambda: OptimizerConfigEntry(
            optimizer=AdamOptimizerConfig(lr=2.5e-3),
            scheduler=None,
        )
    )
    shN: OptimizerConfigEntry = field(
        default_factory=lambda: OptimizerConfigEntry(
            optimizer=AdamOptimizerConfig(lr=2.5e-3 / 20.0),
            scheduler=None,
        )
    )
    pose_opt: OptimizerConfigEntry = field(
        default_factory=lambda: OptimizerConfigEntry(
            optimizer=AdamOptimizerConfig(lr=1e-5, weight_decay=1e-6),
            scheduler=ExponentialDecaySchedulerConfig(lr_final=1e-7),
        )
    )
    bilateral_grid: OptimizerConfigEntry = field(
        default_factory=lambda: OptimizerConfigEntry(
            optimizer=AdamOptimizerConfig(lr=2e-3),
            scheduler=ExponentialDecaySchedulerConfig(
                lr_final=1e-4,
                warmup_steps=1000,
            ),
        )
    )

    def as_dict(self) -> dict[str, OptimizerConfigEntry]:
        return {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
            "sh0": self.sh0,
            "shN": self.shN,
            "pose_opt": self.pose_opt,
            "bilateral_grid": self.bilateral_grid,
        }


@dataclass(frozen=True)
class OptimConfig:
    # Scale training schedules by this factor (e.g. 0.25 for 4 GPUs with 4x effective batch).
    # This is applied by `apply_steps_scaler(...)` in the training entrypoint.
    steps_scaler: float = 1.0
    # Number of training steps.
    max_steps: int = 30_000

    # Spherical harmonics configuration.
    # Maximum degree of spherical harmonics for color.
    sh_degree: int = 3
    # Turn on another SH degree every this many steps (0 disables progressive SH).
    sh_degree_interval: int = 1000

    # Use random backgrounds during training to discourage transparency.
    random_bkgd: bool = False
    # Use packed rasterization mode (lower memory, slightly slower).
    packed: bool = False
    # Enable anti-aliasing in rasterization (may affect quantitative metrics).
    antialiased: bool = False

    # Convert dense grads to sparse grads and use SparseAdam (requires packed=True).
    sparse_grad: bool = False
    # Use SelectiveAdam to update only visible Gaussians (experimental).
    visible_adam: bool = False

    # MU-style splat optimizer step schedule (optional).
    # After densification, step splat optimizers less frequently and accumulate gradients in between.
    # Schedule: [1..mu_start_iter] every step; (mu_start_iter..mu_end_iter] every 5 steps; (mu_end_iter..] every 20 steps.

    # Enable MU schedule.
    mu_enable: bool = False
    # Phase-1 end iteration (1-based, inclusive).
    mu_start_iter: int = 15_000
    # Phase-2 end iteration (1-based, inclusive).
    mu_end_iter: int = 30_000

    # Per-parameter-group optimizer/scheduler configuration (nerfstudio-style).
    optimizers: OptimizersConfig = field(default_factory=OptimizersConfig)


@dataclass(frozen=True)
class RegConfig:
    ## Photometric loss

    # Weight for SSIM in the RGB loss (rgb = (1-ssim)*L1 + ssim*SSIM).
    ssim_lambda: float = 0.2

    ## MCMC-style regularizers (optional).

    # Mean opacity penalty (MCMC-3DGS-style): `mean(sigmoid(opacity_logits))`.
    opacity_reg_weight: float = 0.0
    # Mean scale penalty (MCMC-3DGS-style): `mean(exp(log_scales))` (not the scale-ratio reg below).
    scale_l1_reg_weight: float = 0.0

    # PhysGauss-style scale regularizations (optional).
    # - Flatness: encourages each Gaussian to have a small minimum axis (disk-/sheet-like).
    # - Scale ratio: suppresses spiky Gaussians where max(scale) >> median(scale).

    # Weight of flatness regularization (encourage Gaussians to be flat/disc-like).
    flat_reg_weight: float = 0.0
    # Weight of scale ratio regularization (PhysGauss-style, adapted to max/median ratio).
    scale_ratio_reg_weight: float = 0.0
    # Threshold of max/median scale ratio before applying regularization.
    max_gauss_ratio: float = 6.0
    # Apply scale ratio regularization once every N steps.
    scale_ratio_reg_every_n: int = 10

    ## Regularizations using priors / masks.

    # Weight for sky supervision loss (encourage transparency in sky pixels).
    sky_loss_weight: float = 0.05
    # Apply sky supervision loss once every N steps.
    sky_loss_every_n: int = 10

    # Apply depth regularization once every N steps.
    depth_reg_every_n: int = 4
    # Weight of the depth loss.
    depth_loss_weight: float = 0.25
    # Starting step for depth regularization.
    depth_loss_activation_step: int = 1000
    # Stop applying depth regularization at this step.
    depth_loss_stop_step: int = 15_000

    # Normal supervision weights:
    # - `normal_loss_*`: supervise rendered normals (from gsplat) w.r.t. the normal prior.
    # - `surf_normal_loss_*`: supervise normals implied by depth w.r.t. the normal prior.
    # - `consistency_normal_loss_*`: encourage rendered normals to match depth-implied normals.

    # Apply normal-prior regularizations once every N steps.
    prior_normal_reg_every_n: int = 8
    # Weight of the rendered-normal loss.
    normal_loss_weight: float = 0.1
    # Starting step for rendered-normal regularization.
    normal_loss_activation_step: int = 7000
    # Weight of the depth-implied surface-normal loss.
    surf_normal_loss_weight: float = 0.1
    # Starting step for surface-normal regularization.
    surf_normal_loss_activation_step: int = 7000
    # Apply normal-consistency regularization once every N steps.
    consistency_normal_reg_every_n: int = 1
    # Weight of the normal consistency loss (rendered normals vs depth-implied normals).
    consistency_normal_loss_weight: float = 0.0
    # Starting step for normal consistency regularization.
    consistency_normal_loss_activation_step: int = 15000


@dataclass(frozen=True)
class GNSConfig:
    ## Natural Selection (GNS) pruning (optional).

    # Whether to enable the Natural Selection pruning phase.
    gns_enable: bool = False
    # Step (1-based) to start Natural Selection (post-densification).
    reg_start: int = 15_001
    # Step (1-based) to end Natural Selection (inclusive).
    reg_end: int = 23_000
    # Final target Gaussian count (budget).
    final_budget: int = 1_000_000
    # Base regularization strength during Natural Selection (adjusted dynamically).
    opacity_reg_weight: float = 2e-5


@dataclass(frozen=True)
class HardPruneConfig:
    """Hard pruning (score-based).

    At configured events, compute per-Gaussian scores and remove the lowest ones.
    Cadence uses 1-based training steps.
    """

    # Enable hard pruning.
    enable: bool = False

    # How to choose the per-event prune count.
    policy: Literal["uniform_to_budget", "fixed_percent"] = "uniform_to_budget"
    # For `policy="fixed_percent"` only.
    percent_per_event: float = 0.1

    # First (1-based) training step at which hard pruning is allowed.
    # For policy="uniform_to_budget", must satisfy start_step > strategy.refine_stop_iter.
    start_step: int = 15_001
    # Final target count for policy="uniform_to_budget" only.
    # Ignored when policy="fixed_percent".
    final_budget: int = 1_000_000
    # Run hard pruning every N (1-based) training steps.
    every_n: int = 2500
    # Last (1-based) training step to allow pruning (inclusive).
    stop_step: int = 25000

    # Scoring controls (trade accuracy for speed).
    # Use at most this many training views per pruning event (None means use all train views).
    score_num_views: Optional[int] = None
    # Accumulate squared gradient magnitude (more Speedy-Splat-like). When False, use abs-grad.
    score_use_sqgrad: bool = True


@dataclass(frozen=True)
class StrategyConfig:
    """Densification strategy config.

    The implementation lives in `gsplat.strategy.*`. FriendlySplat selects the
    concrete strategy via `impl`.
    """

    # Strategy implementation.
    impl: Literal["improved", "default", "mcmc"] = "improved"

    # ---------------------------
    # Shared across strategies
    # ---------------------------

    # Opacity threshold for pruning.
    prune_opa: float = 0.005
    # Gradient threshold (2D) for splitting/growing Gaussians.
    grow_grad2d: float = 0.0002
    # 3D scale threshold for pruning.
    prune_scale3d: float = 0.1
    # 2D projected scale threshold for pruning.
    prune_scale2d: float = 0.15
    # Stop refining 2D scale after this step.
    refine_scale2d_stop_iter: int = 4000
    # Start densification/refinement at this step.
    refine_start_iter: int = 500
    # Stop densification/refinement at this step.
    refine_stop_iter: int = 15_000
    # Reset opacities every N steps (during densification window).
    reset_every: int = 3000
    # Run densification refine step every N steps.
    refine_every: int = 100
    # Use absolute gradients for splitting (requires rasterization absgrad enabled).
    absgrad: bool = True
    # Verbosity for densification logs.
    verbose: bool = True
    # Which meta key to use for gradients ("means2d" for 3DGS; "gradient_2dgs" for 2DGS).
    key_for_gradient: str = "means2d"

    # ---------------------------
    # DefaultStrategy-only knobs
    # ---------------------------

    # 3D scale threshold for grow mode (duplicate vs split decision).
    grow_scale3d: float = 0.01
    # 2D projected scale threshold for grow mode.
    grow_scale2d: float = 0.05
    # Pause refine for N steps after each opacity reset.
    pause_refine_after_reset: int = 0
    # Experimental revised opacity heuristic.
    revised_opacity: bool = False

    # ---------------------------
    # ImprovedStrategy-only knobs
    # ---------------------------

    # Maximum number of Gaussians allowed during densification.
    densification_budget: int = 2_500_000

    # ---------------------------
    # MCMCStrategy-only knobs
    # ---------------------------

    # Maximum number of Gaussians.
    mcmc_cap_max: int = 1_000_000
    # Noise scale multiplier (position noise ~ lr_means * mcmc_noise_lr).
    mcmc_noise_lr: float = 5e5
    # Stop injecting noise after this step (-1 means never stop).
    mcmc_noise_injection_stop_iter: int = -1
    # Minimum opacity (post-sigmoid) for "alive" Gaussians.
    mcmc_min_opacity: float = 0.005


@dataclass(frozen=True)
class TrainConfig:
    # I/O and runtime paths (dataset, output directory, device, etc.).
    io: IOConfig
    # Dataset / DataLoader configuration.
    data: DataConfig = field(default_factory=DataConfig)
    # Initialization configuration (sfm/random init).
    init: InitConfig = field(default_factory=InitConfig)
    # Optimizer and training loop configuration (includes max_steps).
    optim: OptimConfig = field(default_factory=OptimConfig)
    # Loss / regularization configuration.
    reg: RegConfig = field(default_factory=RegConfig)
    # Densification strategy configuration.
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    # Natural Selection (GNS) configuration.
    gns: GNSConfig = field(default_factory=GNSConfig)
    # Speedy-Splat-style hard pruning (score-based budget pruning).
    hard_prune: HardPruneConfig = field(default_factory=HardPruneConfig)
    # Pose optimization / noise configuration.
    pose: PoseConfig = field(default_factory=PoseConfig)
    # Post-processing configuration (photometric adapters).
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    # Online viewer configuration (viser/nerfview).
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    # TensorBoard logging configuration.
    tb: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    # Evaluation configuration (periodic holdout metrics).
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Reserved for future multi-GPU training (currently not implemented / not tested here).
    # Whether to enable distributed training (not implemented in this repo).
    distributed: bool = False
    # Distributed rank (reserved; must be 0).
    world_rank: int = 0
    # Distributed world size (reserved; must be 1).
    world_size: int = 1


def apply_steps_scaler(*, cfg: TrainConfig, steps_scaler: float) -> TrainConfig:
    """Return a new config with step-based schedules scaled by `steps_scaler`.

    This intentionally scales only a curated set of *time-based* hyperparameters
    (steps/iters/intervals). It does not scale learning rates, weights, or budgets.
    """
    s = float(steps_scaler)
    if not (s > 0.0):
        raise ValueError(f"steps_scaler must be > 0, got {steps_scaler}")
    if abs(s - 1.0) <= 1e-12:
        return cfg

    def scale(
        value: int,
        *,
        min_value: int = 0,
        keep_minus_one: bool = False,
    ) -> int:
        if keep_minus_one and int(value) == -1:
            return -1
        return max(int(min_value), int(round(float(value) * s)))

    def scale_steps(values: Tuple[int, ...], *, max_step: int) -> Tuple[int, ...]:
        scaled = {
            max(1, min(int(max_step), scale(int(v), min_value=1)))
            for v in values
            if int(v) > 0
        }
        return tuple(sorted(scaled))

    def scale_scheduler(
        sch: ExponentialDecaySchedulerConfig,
    ) -> ExponentialDecaySchedulerConfig:
        return replace(
            sch,
            warmup_steps=scale(int(sch.warmup_steps), min_value=0),
            max_steps=scale(int(sch.max_steps), min_value=1)
            if sch.max_steps is not None
            else None,
        )

    # 1) Scale `optim.max_steps` first (others may clamp against it).
    new_max_steps = scale(int(cfg.optim.max_steps), min_value=1)

    # 2) IO steps (1-based).
    io_cfg = replace(
        cfg.io,
        ply_steps=scale_steps(cfg.io.ply_steps, max_step=new_max_steps),
        save_steps=scale_steps(cfg.io.save_steps, max_step=new_max_steps),
    )

    # 3) Optim schedules.
    optim_cfg = cfg.optim
    sh_degree_interval = int(optim_cfg.sh_degree_interval)
    if sh_degree_interval > 0:
        sh_degree_interval = scale(sh_degree_interval, min_value=1)
    optim_cfg = replace(
        optim_cfg,
        max_steps=int(new_max_steps),
        sh_degree_interval=int(sh_degree_interval),
        mu_start_iter=scale(int(optim_cfg.mu_start_iter), min_value=1),
        mu_end_iter=scale(int(optim_cfg.mu_end_iter), min_value=1),
    )

    # 4) Per-optimizer warmup (and explicit scheduler max_steps when set).
    optim_cfg = replace(
        optim_cfg,
        optimizers=OptimizersConfig(
            **{
                name: (
                    replace(entry, scheduler=scale_scheduler(entry.scheduler))
                    if entry.scheduler is not None
                    else entry
                )
                for name, entry in optim_cfg.optimizers.as_dict().items()
            }
        ),
    )

    # 5) Regularizer activation/stop windows (0-based; keep -1 sentinels).
    reg_cfg = replace(
        cfg.reg,
        depth_loss_activation_step=scale(
            int(cfg.reg.depth_loss_activation_step),
            min_value=0,
        ),
        depth_loss_stop_step=scale(
            int(cfg.reg.depth_loss_stop_step),
            min_value=0,
            keep_minus_one=True,
        ),
        normal_loss_activation_step=scale(
            int(cfg.reg.normal_loss_activation_step),
            min_value=0,
        ),
        surf_normal_loss_activation_step=scale(
            int(cfg.reg.surf_normal_loss_activation_step),
            min_value=0,
        ),
        consistency_normal_loss_activation_step=scale(
            int(cfg.reg.consistency_normal_loss_activation_step),
            min_value=0,
        ),
    )

    # 6) Strategy time-based params (mostly 0-based; keep -1 sentinel for mcmc stop).
    strategy_cfg = replace(
        cfg.strategy,
        refine_scale2d_stop_iter=scale(int(cfg.strategy.refine_scale2d_stop_iter)),
        refine_start_iter=scale(int(cfg.strategy.refine_start_iter)),
        refine_stop_iter=scale(int(cfg.strategy.refine_stop_iter)),
        reset_every=scale(int(cfg.strategy.reset_every), min_value=1),
        refine_every=scale(int(cfg.strategy.refine_every), min_value=1),
        pause_refine_after_reset=scale(int(cfg.strategy.pause_refine_after_reset)),
        mcmc_noise_injection_stop_iter=scale(
            int(cfg.strategy.mcmc_noise_injection_stop_iter),
            min_value=0,
            keep_minus_one=True,
        ),
    )

    # 7) Pruning schedules (1-based).
    gns_cfg = replace(
        cfg.gns,
        reg_start=scale(int(cfg.gns.reg_start), min_value=1),
        reg_end=scale(int(cfg.gns.reg_end), min_value=1),
    )
    hard_prune_cfg = replace(
        cfg.hard_prune,
        start_step=scale(int(cfg.hard_prune.start_step), min_value=1),
        every_n=scale(int(cfg.hard_prune.every_n), min_value=1),
        stop_step=scale(int(cfg.hard_prune.stop_step), min_value=1),
    )

    return replace(
        cfg,
        io=io_cfg,
        optim=optim_cfg,
        reg=reg_cfg,
        strategy=strategy_cfg,
        gns=gns_cfg,
        hard_prune=hard_prune_cfg,
    )


def validate_train_config(cfg: TrainConfig) -> None:
    if cfg.distributed or cfg.world_size != 1 or cfg.world_rank != 0:
        raise NotImplementedError(
            "Distributed/multi-GPU training is reserved for future work in this repo. "
            "Run with distributed=False, world_size=1, world_rank=0."
        )

    if cfg.data.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {cfg.data.batch_size}")
    if cfg.data.test_every <= 0:
        raise ValueError(f"data.test_every must be > 0, got {cfg.data.test_every}")
    if cfg.data.benchmark_train_split and cfg.data.test_every <= 1:
        raise ValueError(
            "data.benchmark_train_split=True requires data.test_every > 1, "
            f"got data.test_every={cfg.data.test_every}"
        )
    if cfg.world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {cfg.world_size}")
    if cfg.optim.max_steps <= 0:
        raise ValueError(f"optim.max_steps must be > 0, got {cfg.optim.max_steps}")
    if (
        int(cfg.reg.depth_loss_stop_step) != -1
        and int(cfg.reg.depth_loss_stop_step) < 0
    ):
        raise ValueError(
            f"reg.depth_loss_stop_step must be -1 or >= 0, got {cfg.reg.depth_loss_stop_step}"
        )
    if cfg.eval.max_images is not None and int(cfg.eval.max_images) <= 0:
        raise ValueError(
            f"eval.max_images must be > 0 or None, got {cfg.eval.max_images}"
        )
    if str(cfg.eval.lpips_net) not in ("alex", "vgg"):
        raise ValueError(
            f"eval.lpips_net must be 'alex' or 'vgg', got {cfg.eval.lpips_net!r}"
        )
    if str(cfg.eval.split).strip().lower() == "train":
        raise ValueError(
            "eval.split must be a holdout split (e.g. 'test' or 'val'), not 'train'."
        )
    if int(cfg.eval.eval_every_n) <= 0:
        raise ValueError(f"eval.eval_every_n must be > 0, got {cfg.eval.eval_every_n}")

    if cfg.data.preload not in ("none", "cuda"):
        raise ValueError(f"preload must be 'none' or 'cuda', got {cfg.data.preload!r}")
    if cfg.data.preload == "cuda":
        if not str(cfg.io.device).startswith("cuda"):
            raise ValueError(
                f"preload='cuda' requires io.device to be CUDA (e.g. 'cuda' or 'cuda:0'), got {cfg.io.device!r}"
            )

    if cfg.optim.sparse_grad and cfg.optim.visible_adam:
        raise ValueError(
            "optim.sparse_grad and optim.visible_adam are mutually exclusive."
        )
    if cfg.optim.sparse_grad and not cfg.optim.packed:
        raise ValueError(
            "optim.sparse_grad=True requires optim.packed=True (packed rasterization)."
        )

    # GNS config validation (pruning window + budget).
    if bool(cfg.gns.gns_enable):
        if int(cfg.gns.reg_start) <= 0:
            raise ValueError(
                f"gns.reg_start must be > 0 (1-based), got {cfg.gns.reg_start}"
            )
        if int(cfg.gns.reg_end) < int(cfg.gns.reg_start):
            raise ValueError(
                f"gns.reg_end must be >= gns.reg_start (1-based), got {cfg.gns.reg_end} < {cfg.gns.reg_start}"
            )
        if int(cfg.gns.final_budget) <= 0:
            raise ValueError(
                f"gns.final_budget must be > 0, got {cfg.gns.final_budget}"
            )
        if float(cfg.gns.opacity_reg_weight) <= 0.0:
            raise ValueError(
                f"gns.opacity_reg_weight must be > 0, got {cfg.gns.opacity_reg_weight}"
            )
        if cfg.optim.sparse_grad:
            raise ValueError(
                "gns.gns_enable=True is incompatible with optim.sparse_grad=True "
                "(packed+sparse mode). Disable either GNS or sparse_grad."
            )
        # `strategy.refine_stop_iter` uses 0-based `step` units. GNS uses 1-based
        # training steps, so the strict "after densification" condition is:
        #   reg_start >= refine_stop_iter + 1  <=>  reg_start > refine_stop_iter
        if int(cfg.gns.reg_start) <= int(cfg.strategy.refine_stop_iter):
            raise ValueError(
                "gns.reg_start must be strictly after densification. "
                f"Got gns.reg_start={int(cfg.gns.reg_start)} (1-based) and "
                f"strategy.refine_stop_iter={int(cfg.strategy.refine_stop_iter)} (0-based)."
            )

    # Hard prune config.
    hp = cfg.hard_prune
    if bool(hp.enable):
        policy = str(hp.policy)
        if int(hp.start_step) <= 0:
            raise ValueError(f"hard_prune.start_step must be > 0, got {hp.start_step}")
        # `strategy.refine_stop_iter` uses 0-based `step` units, while hard-prune
        # config uses 1-based train-step numbers.
        if policy == "uniform_to_budget" and int(hp.start_step) <= int(
            cfg.strategy.refine_stop_iter
        ):
            raise ValueError(
                "hard_prune.start_step must be strictly after densification when "
                "hard_prune.policy='uniform_to_budget'. "
                f"Got hard_prune.start_step={int(hp.start_step)} (1-based) and "
                f"strategy.refine_stop_iter={int(cfg.strategy.refine_stop_iter)} (0-based)."
            )
        if policy == "uniform_to_budget" and int(hp.final_budget) <= 0:
            raise ValueError(
                f"hard_prune.final_budget must be > 0 when policy='uniform_to_budget', got {hp.final_budget}"
            )
        if policy == "fixed_percent":
            if not (0.0 < float(hp.percent_per_event) < 1.0):
                raise ValueError(
                    "hard_prune.percent_per_event must be in (0, 1) when "
                    "hard_prune.policy='fixed_percent', "
                    f"got {hp.percent_per_event}"
                )
        if int(hp.every_n) <= 0:
            raise ValueError(f"hard_prune.every_n must be > 0, got {hp.every_n}")
        if int(hp.stop_step) <= 0:
            raise ValueError(f"hard_prune.stop_step must be > 0, got {hp.stop_step}")
        if int(hp.stop_step) < int(hp.start_step):
            raise ValueError(
                f"hard_prune.stop_step must be >= hard_prune.start_step, got stop_step={hp.stop_step}, start_step={hp.start_step}"
            )
        if hp.score_num_views is not None and int(hp.score_num_views) <= 0:
            raise ValueError(
                f"hard_prune.score_num_views must be > 0 or None, got {hp.score_num_views}"
            )
    if cfg.strategy.key_for_gradient not in ("means2d", "gradient_2dgs"):
        raise ValueError(
            "strategy.key_for_gradient must be 'means2d' (3DGS) or 'gradient_2dgs' (2DGS), "
            f"got {cfg.strategy.key_for_gradient!r}"
        )

    if cfg.optim.mu_enable:
        if cfg.gns.gns_enable:
            raise ValueError(
                "optim.mu_enable=True is incompatible with gns.gns_enable=True. "
                "Disable GNS or disable MU."
            )
        if str(cfg.strategy.impl).strip().lower() == "mcmc":
            raise ValueError(
                "optim.mu_enable=True is incompatible with strategy.impl='mcmc'."
            )
        if int(cfg.optim.mu_start_iter) <= 0:
            raise ValueError(
                f"optim.mu_start_iter must be > 0, got {cfg.optim.mu_start_iter}"
            )
        if int(cfg.optim.mu_end_iter) < int(cfg.optim.mu_start_iter):
            raise ValueError(
                "optim.mu_end_iter must be >= optim.mu_start_iter, "
                f"got {cfg.optim.mu_end_iter} < {cfg.optim.mu_start_iter}."
            )
        if int(cfg.optim.mu_start_iter) < int(cfg.strategy.refine_stop_iter):
            raise ValueError(
                "optim.mu requires phase 1 (per-step updates) to cover the entire densification window. "
                "Set optim.mu_start_iter >= strategy.refine_stop_iter, "
                f"got {cfg.optim.mu_start_iter} < {cfg.strategy.refine_stop_iter}."
            )
        if int(cfg.optim.mu_end_iter) > int(cfg.optim.max_steps):
            raise ValueError(
                "optim.mu_end_iter must be <= optim.max_steps, "
                f"got {cfg.optim.mu_end_iter} > {cfg.optim.max_steps}."
            )

    if str(cfg.strategy.impl).strip().lower() == "mcmc" and cfg.gns.gns_enable:
        raise ValueError(
            "strategy.impl='mcmc' is incompatible with gns.gns_enable=True. "
            "Disable GNS or switch to a densification strategy ('default'/'improved')."
        )

    if not cfg.viewer.disable_viewer:
        if not (1 <= cfg.viewer.port <= 65535):
            raise ValueError(
                f"viewer.port must be in [1, 65535], got {cfg.viewer.port}"
            )
    if cfg.tb.enable:
        if int(cfg.tb.every_n) <= 0:
            raise ValueError(f"tb.every_n must be > 0, got {cfg.tb.every_n}")
        if int(cfg.tb.flush_every_n) <= 0:
            raise ValueError(
                f"tb.flush_every_n must be > 0, got {cfg.tb.flush_every_n}"
            )

    if cfg.io.export_ply:
        if len(cfg.io.ply_steps) == 0:
            raise ValueError(
                "export_ply=True requires non-empty ply_steps (e.g. --io.ply_steps 15000 30000)."
            )
        if cfg.io.ply_format not in ("ply", "ply_compressed"):
            raise ValueError(
                f"ply_format must be 'ply' or 'ply_compressed', got {cfg.io.ply_format!r}"
            )

    # (GNS validation handled above.)

    if cfg.eval.enable and int(cfg.eval.eval_every_n) <= 0:
        raise ValueError(
            f"eval.enable=True requires eval.eval_every_n > 0, got {cfg.eval.eval_every_n}"
        )

    for name, entry in cfg.optim.optimizers.as_dict().items():
        opt = entry.optimizer
        if float(opt.lr) <= 0.0:
            raise ValueError(
                f"optim.optimizers.{name}.optimizer.lr must be > 0, got {opt.lr}"
            )
        if float(opt.eps) <= 0.0:
            raise ValueError(
                f"optim.optimizers.{name}.optimizer.eps must be > 0, got {opt.eps}"
            )
        if float(opt.weight_decay) < 0.0:
            raise ValueError(
                f"optim.optimizers.{name}.optimizer.weight_decay must be >= 0, got {opt.weight_decay}"
            )
        sch = entry.scheduler
        if sch is not None:
            if float(sch.lr_final) <= 0.0:
                raise ValueError(
                    f"optim.optimizers.{name}.scheduler.lr_final must be > 0, got {sch.lr_final}"
                )
            if sch.max_steps is not None and int(sch.max_steps) <= 0:
                raise ValueError(
                    f"optim.optimizers.{name}.scheduler.max_steps must be > 0, got {sch.max_steps}"
                )
            if int(sch.warmup_steps) < 0:
                raise ValueError(
                    f"optim.optimizers.{name}.scheduler.warmup_steps must be >= 0, got {sch.warmup_steps}"
                )
