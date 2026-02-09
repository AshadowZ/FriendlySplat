from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from friendly_splat.data import DataLoader, InputDataset
from friendly_splat.data.colmap_dataparser import ColmapDataParser
from friendly_splat.models.bilateral_grid import BilateralGridPostProcessor
from friendly_splat.models.camera_opt import CameraOptModule
from friendly_splat.models.gaussian import GaussianModel
from friendly_splat.models.ppisp import PPISPPostProcessor
from friendly_splat.trainer.configs import (
    DataConfig,
    EvalConfig,
    InitConfig,
    IOConfig,
    OptimConfig,
    PoseConfig,
    PostprocessConfig,
    TrainConfig,
)
from friendly_splat.trainer.optimizer_coordinator import (
    OptimizerBundle,
    OptimizerCoordinator,
)
from gsplat.strategy import ImprovedStrategy
from gsplat.strategy.natural_selection import (
    NaturalSelectionPolicy,
    auto_gns_reg_interval,
)


@dataclass
class TrainingContext:
    cfg: TrainConfig
    device: torch.device
    dataset: InputDataset
    loader: DataLoader
    eval_dataset: Optional[InputDataset]
    eval_loader: Optional[DataLoader]
    gaussian_model: GaussianModel
    splats: torch.nn.ParameterDict
    pose_adjust: Optional[CameraOptModule]
    bilagrid: Optional[BilateralGridPostProcessor]
    ppisp: Optional[PPISPPostProcessor]
    natural_selection_policy: Optional[NaturalSelectionPolicy]
    strategy: ImprovedStrategy
    strategy_state: Dict[str, Any]
    optimizer_coordinator: OptimizerCoordinator


def build_dataset_and_loader(
    *,
    io_cfg: IOConfig,
    data_cfg: DataConfig,
) -> tuple[InputDataset, DataLoader]:
    device = torch.device(io_cfg.device)
    dataparser = ColmapDataParser(
        data_dir=io_cfg.data_dir,
        factor=data_cfg.data_factor,
        normalize_world_space=data_cfg.normalize_world_space,
        test_every=data_cfg.test_every,
        benchmark_train_split=data_cfg.benchmark_train_split,
        depth_dir_name=data_cfg.depth_dir_name,
        normal_dir_name=data_cfg.normal_dir_name,
        dynamic_mask_dir_name=data_cfg.dynamic_mask_dir_name,
        sky_mask_dir_name=data_cfg.sky_mask_dir_name,
    )
    parsed_scene = dataparser.get_dataparser_outputs(split="train")
    dataset = InputDataset(parsed_scene)
    loader = DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        device=device,
        infinite_sampler=data_cfg.infinite_sampler,
        prefetch_to_gpu=data_cfg.prefetch_to_gpu,
        preload=data_cfg.preload,  # type: ignore[arg-type]
        seed=io_cfg.seed,
    )
    return dataset, loader


def build_eval_dataset_and_loader(
    *,
    io_cfg: IOConfig,
    data_cfg: DataConfig,
    eval_cfg: EvalConfig,
) -> tuple[InputDataset, DataLoader]:
    device = torch.device(io_cfg.device)
    dataparser = ColmapDataParser(
        data_dir=io_cfg.data_dir,
        factor=data_cfg.data_factor,
        normalize_world_space=data_cfg.normalize_world_space,
        test_every=data_cfg.test_every,
        benchmark_train_split=data_cfg.benchmark_train_split,
        depth_dir_name=data_cfg.depth_dir_name,
        normal_dir_name=data_cfg.normal_dir_name,
        dynamic_mask_dir_name=data_cfg.dynamic_mask_dir_name,
        sky_mask_dir_name=data_cfg.sky_mask_dir_name,
    )
    parsed_scene = dataparser.get_dataparser_outputs(split=str(eval_cfg.split))
    dataset = InputDataset(parsed_scene)
    loader = DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        device=device,
        infinite_sampler=False,
        prefetch_to_gpu=data_cfg.prefetch_to_gpu,
        preload=data_cfg.preload,  # type: ignore[arg-type]
        seed=io_cfg.seed,
    )
    return dataset, loader


def build_gaussian_model(
    *,
    dataset: InputDataset,
    init_cfg: InitConfig,
    optim_cfg: OptimConfig,
    device: torch.device,
) -> GaussianModel:
    parsed_scene = dataset.parsed_scene
    sh_degree = int(optim_cfg.sh_degree)
    init_scale = float(init_cfg.init_scale)
    init_opacity = float(init_cfg.init_opacity)

    if init_cfg.init_type == "sfm" and int(parsed_scene.points.shape[0]) > 0:
        return GaussianModel.from_sfm(
            points=torch.from_numpy(parsed_scene.points),
            points_rgb=torch.from_numpy(parsed_scene.points_rgb),
            sh_degree=sh_degree,
            init_scale=init_scale,
            init_opacity=init_opacity,
            device=device,
        )

    if init_cfg.init_type == "random":
        return GaussianModel.from_random(
            num_points=int(init_cfg.init_num_pts),
            scene_scale=float(parsed_scene.scene_scale),
            init_extent=float(init_cfg.init_extent),
            sh_degree=sh_degree,
            init_scale=init_scale,
            init_opacity=init_opacity,
            device=device,
        )

    raise ValueError(
        f"Unsupported init_type={init_cfg.init_type!r} (sfm requires COLMAP points3D)."
    )


def build_optimizer_bundle(
    *,
    optim_cfg: OptimConfig,
    data_cfg: DataConfig,
    pose_cfg: PoseConfig,
    postprocess_cfg: PostprocessConfig,
    world_size: int,
    device: torch.device,
    scene_scale: float,
    splats: torch.nn.ParameterDict,
    pose_adjust: Optional[CameraOptModule],
    bilagrid: Optional[BilateralGridPostProcessor],
    ppisp: Optional[PPISPPostProcessor],
) -> OptimizerBundle:
    means_lr_final_ratio = 0.01
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = int(data_cfg.batch_size) * int(world_size)
    if BS <= 0:
        raise ValueError(f"batch_size*world_size must be > 0, got {BS}")
    lr_scale = math.sqrt(float(BS))
    eps = 1e-15 / lr_scale
    beta1 = max(0.0, 1.0 - float(BS) * (1.0 - 0.9))
    beta2 = max(0.0, 1.0 - float(BS) * (1.0 - 0.999))
    betas = (beta1, beta2)

    def _make_optimizer(param: torch.nn.Parameter, lr: float) -> torch.optim.Optimizer:
        lr_scaled = float(lr) * lr_scale
        if optim_cfg.sparse_grad:
            return torch.optim.SparseAdam(
                [{"params": param, "lr": lr_scaled}], betas=betas, eps=eps
            )
        if optim_cfg.visible_adam:
            from gsplat.optimizers import SelectiveAdam  # noqa: WPS433

            return SelectiveAdam(
                [{"params": param, "lr": lr_scaled}], eps=eps, betas=betas
            )

        # Default: Adam with fused implementation when available.
        if device.type == "cuda":
            try:
                return torch.optim.Adam(
                    [{"params": param, "lr": lr_scaled}],
                    eps=eps,
                    betas=betas,
                    fused=True,
                )
            except TypeError:
                pass
        return torch.optim.Adam(
            [{"params": param, "lr": lr_scaled}], eps=eps, betas=betas
        )

    scene_scale = float(scene_scale)
    splat_optimizers: Dict[str, torch.optim.Optimizer] = {
        name: _make_optimizer(param, lr)
        for name, param, lr in [
            ("means", splats["means"], optim_cfg.means_lr * scene_scale),
            ("scales", splats["scales"], optim_cfg.scales_lr),
            ("quats", splats["quats"], optim_cfg.quats_lr),
            ("opacities", splats["opacities"], optim_cfg.opacities_lr),
        ]
    }
    splat_optimizers["sh0"] = _make_optimizer(splats["sh0"], optim_cfg.sh0_lr)
    splat_optimizers["shN"] = _make_optimizer(splats["shN"], optim_cfg.shN_lr)

    pose_optimizers: list[torch.optim.Optimizer] = []
    if pose_cfg.pose_opt:
        if pose_adjust is None:
            raise RuntimeError("pose_opt=True but pose_adjust is not initialized.")
        pose_optimizers = [
            torch.optim.Adam(
                pose_adjust.parameters(),
                lr=float(pose_cfg.pose_opt_lr) * math.sqrt(float(data_cfg.batch_size)),
                weight_decay=float(pose_cfg.pose_opt_reg),
            )
        ]

    bilagrid_optimizers: list[torch.optim.Optimizer] = []
    if postprocess_cfg.use_bilateral_grid:
        if bilagrid is None:
            raise RuntimeError(
                "use_bilateral_grid=True but bilagrid module is not initialized."
            )
        bilagrid_optimizers = [
            torch.optim.Adam(
                bilagrid.parameters(),
                lr=float(postprocess_cfg.bilateral_grid_lr)
                * math.sqrt(float(data_cfg.batch_size)),
                eps=1e-15,
            )
        ]

    ppisp_optimizers: list[torch.optim.Optimizer] = []
    if postprocess_cfg.use_ppisp:
        if ppisp is None:
            raise RuntimeError("use_ppisp=True but ppisp module is not initialized.")
        ppisp_optimizers = list(ppisp.optimizers)

    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
    gamma = float(means_lr_final_ratio) ** (1.0 / float(optim_cfg.max_steps))
    schedulers.append(
        torch.optim.lr_scheduler.ExponentialLR(splat_optimizers["means"], gamma=gamma)
    )
    if pose_cfg.pose_opt:
        # Pose optimization ends at 1% of the initial value.
        if len(pose_optimizers) == 0:
            raise RuntimeError("pose_opt=True but pose optimizer is not initialized.")
        gamma = 0.01 ** (1.0 / float(optim_cfg.max_steps))
        schedulers.append(
            torch.optim.lr_scheduler.ExponentialLR(pose_optimizers[0], gamma=gamma)
        )
    if postprocess_cfg.use_bilateral_grid:
        # Linear warmup (1000 iters) then exponential decay to 1% at max_steps.
        if len(bilagrid_optimizers) == 0:
            raise RuntimeError(
                "use_bilateral_grid=True but bilagrid optimizer is not initialized."
            )
        gamma = 0.01 ** (1.0 / float(optim_cfg.max_steps))
        schedulers.append(
            torch.optim.lr_scheduler.ChainedScheduler(
                [
                    torch.optim.lr_scheduler.LinearLR(
                        bilagrid_optimizers[0],
                        start_factor=0.01,
                        total_iters=1000,
                    ),
                    torch.optim.lr_scheduler.ExponentialLR(
                        bilagrid_optimizers[0],
                        gamma=gamma,
                    ),
                ]
            )
        )
    if postprocess_cfg.use_ppisp:
        # Let PPISP define its own schedulers.
        if ppisp is None:
            raise RuntimeError("use_ppisp=True but ppisp module is not initialized.")
        if hasattr(ppisp.module, "create_schedulers"):
            try:
                ppisp_schedulers = ppisp.module.create_schedulers(  # type: ignore[attr-defined]
                    ppisp.optimizers,
                    max_optimization_iters=int(optim_cfg.max_steps),
                )
            except TypeError:
                ppisp_schedulers = ppisp.module.create_schedulers(  # type: ignore[attr-defined]
                    ppisp.optimizers,
                    int(optim_cfg.max_steps),
                )
            schedulers.extend(list(ppisp_schedulers))

    return OptimizerBundle(
        splat_optimizers=splat_optimizers,
        pose_optimizers=pose_optimizers,
        bilagrid_optimizers=bilagrid_optimizers,
        ppisp_optimizers=ppisp_optimizers,
        schedulers=schedulers,
    )


def build_training_context(cfg: TrainConfig) -> TrainingContext:
    device = torch.device(cfg.io.device)
    dataset, loader = build_dataset_and_loader(io_cfg=cfg.io, data_cfg=cfg.data)
    eval_dataset: Optional[InputDataset] = None
    eval_loader: Optional[DataLoader] = None
    if cfg.eval.enable:
        eval_dataset, eval_loader = build_eval_dataset_and_loader(
            io_cfg=cfg.io,
            data_cfg=cfg.data,
            eval_cfg=cfg.eval,
        )
    gaussian_model = build_gaussian_model(
        dataset=dataset,
        init_cfg=cfg.init,
        optim_cfg=cfg.optim,
        device=device,
    )
    splats = gaussian_model.as_parameter_dict()
    parsed_scene = dataset.parsed_scene
    n_images = int(len(parsed_scene.image_names))

    bilagrid: Optional[BilateralGridPostProcessor] = None
    if cfg.postprocess.use_bilateral_grid:
        grid_shape = tuple(int(x) for x in cfg.postprocess.bilateral_grid_shape)
        bilagrid = BilateralGridPostProcessor.create(
            num_frames=n_images,
            grid_shape=grid_shape,  # type: ignore[arg-type]
            device=device,
        )

    ppisp: Optional[PPISPPostProcessor] = None
    if cfg.postprocess.use_ppisp:
        ppisp = PPISPPostProcessor.create(
            num_frames=n_images,
            device=device,
        )

    pose_adjust: Optional[CameraOptModule] = None
    if cfg.pose.pose_opt:
        pose_adjust = CameraOptModule(n_images).to(device)
        pose_adjust.zero_init()

    optimizer_bundle = build_optimizer_bundle(
        optim_cfg=cfg.optim,
        data_cfg=cfg.data,
        pose_cfg=cfg.pose,
        postprocess_cfg=cfg.postprocess,
        world_size=cfg.world_size,
        device=device,
        scene_scale=float(parsed_scene.scene_scale),
        splats=splats,
        pose_adjust=pose_adjust,
        bilagrid=bilagrid,
        ppisp=ppisp,
    )

    natural_selection_policy: Optional[NaturalSelectionPolicy] = None
    if cfg.gns.gns_enable:
        gns_reg_interval = auto_gns_reg_interval(num_train_images=len(dataset))
        natural_selection_policy = NaturalSelectionPolicy(
            enable=True,
            densify_stop_step=int(cfg.strategy.refine_stop_iter),
            reg_start=int(cfg.gns.reg_start),
            reg_end=int(cfg.gns.reg_end),
            reg_interval=int(gns_reg_interval),
            final_budget=int(cfg.gns.final_budget),
            opacity_reg_weight=float(cfg.gns.opacity_reg_weight),
            verbose=bool(cfg.strategy.verbose),
        )

    strategy = ImprovedStrategy(
        prune_opa=float(cfg.strategy.prune_opa),
        grow_grad2d=float(cfg.strategy.grow_grad2d),
        prune_scale3d=float(cfg.strategy.prune_scale3d),
        prune_scale2d=float(cfg.strategy.prune_scale2d),
        refine_scale2d_stop_iter=int(cfg.strategy.refine_scale2d_stop_iter),
        refine_start_iter=int(cfg.strategy.refine_start_iter),
        refine_stop_iter=int(cfg.strategy.refine_stop_iter),
        reset_every=int(cfg.strategy.reset_every),
        refine_every=int(cfg.strategy.refine_every),
        max_steps=int(cfg.optim.max_steps),
        absgrad=bool(cfg.strategy.absgrad),
        verbose=bool(cfg.strategy.verbose),
        key_for_gradient=str(cfg.strategy.key_for_gradient),
        budget=int(cfg.strategy.densification_budget),
    )
    strategy.check_sanity(splats, optimizer_bundle.splat_optimizers)
    strategy_state = strategy.initialize_state(
        scene_scale=float(parsed_scene.scene_scale)
    )

    optimizer_coordinator = OptimizerCoordinator(
        optim_cfg=cfg.optim,
        device=device,
        splats=splats,
        optimizers=optimizer_bundle,
        gns=natural_selection_policy,
    )

    return TrainingContext(
        cfg=cfg,
        device=device,
        dataset=dataset,
        loader=loader,
        eval_dataset=eval_dataset,
        eval_loader=eval_loader,
        gaussian_model=gaussian_model,
        splats=splats,
        pose_adjust=pose_adjust,
        bilagrid=bilagrid,
        ppisp=ppisp,
        natural_selection_policy=natural_selection_policy,
        strategy=strategy,
        strategy_state=strategy_state,
        optimizer_coordinator=optimizer_coordinator,
    )
