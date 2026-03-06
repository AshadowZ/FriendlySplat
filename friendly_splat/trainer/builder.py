from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from friendly_splat.data import DataLoader, InputDataset
from friendly_splat.data.colmap_dataparser import ColmapDataParser
from friendly_splat.modules.bilateral_grid import BilateralGridPostProcessor
from friendly_splat.modules.gaussian import GaussianModel
from friendly_splat.modules.pose_opt import PoseOptModule
from friendly_splat.trainer.configs import (
    DataConfig,
    InitConfig,
    IOConfig,
    OptimConfig,
    TrainConfig,
)
from friendly_splat.trainer.optimizer_coordinator import (
    OptimizerBundle,
    OptimizerCoordinator,
)
from friendly_splat.trainer.gns_pruning import (
    NaturalSelectionPolicy,
    auto_gns_reg_interval,
)
from gsplat.strategy import DefaultStrategy, ImprovedStrategy, MCMCStrategy, Strategy


@dataclass
class TrainingContext:
    cfg: TrainConfig
    device: torch.device
    dataset: InputDataset
    loader: DataLoader
    eval_dataset: Optional[InputDataset]
    eval_loader: Optional[DataLoader]
    gaussian_model: GaussianModel
    pose_adjust: Optional[PoseOptModule]
    bilateral_grid: Optional[BilateralGridPostProcessor]
    natural_selection_policy: Optional[NaturalSelectionPolicy]
    strategy: Strategy
    strategy_state: Dict[str, Any]
    optimizer_coordinator: OptimizerCoordinator


def build_dataset_and_loader(
    *,
    io_cfg: IOConfig,
    data_cfg: DataConfig,
    split: str = "train",
) -> tuple[InputDataset, DataLoader]:
    """Build dataparser outputs, dataset, and loader for a given split.

    Infinite sampling is enabled only for the training split.
    """
    device = torch.device(io_cfg.device)
    # DataParser defines the "scene contract": cameras, points, image paths, and scale.
    dataparser = ColmapDataParser(
        data_dir=io_cfg.data_dir,
        factor=data_cfg.data_factor,
        normalize_world_space=data_cfg.normalize_world_space,
        align_world_axes=data_cfg.align_world_axes,
        test_every=data_cfg.test_every,
        benchmark_train_split=data_cfg.benchmark_train_split,
        train_image_list_file=data_cfg.train_image_list_file,
        depth_dir_name=data_cfg.depth_dir_name,
        normal_dir_name=data_cfg.normal_dir_name,
        dynamic_mask_dir_name=data_cfg.dynamic_mask_dir_name,
        sky_mask_dir_name=data_cfg.sky_mask_dir_name,
    )
    parsed_scene = dataparser.get_dataparser_outputs(split=str(split))
    dataset = InputDataset(parsed_scene)
    is_train_split = str(split).strip().lower() == "train"
    loader = DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        device=device,
        infinite_sampler=bool(data_cfg.infinite_sampler) if is_train_split else False,
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
    init_type = str(init_cfg.init_type).strip().lower()

    if init_type == "from_ckpt":
        return GaussianModel.from_ckpt(
            ckpt_path=str(init_cfg.init_ckpt_path),
            device=device,
        )

    # Prefer SFM (COLMAP points) init when available; otherwise fall back to random.
    if init_type == "sfm" and int(parsed_scene.points.shape[0]) > 0:
        return GaussianModel.from_sfm(
            points=torch.from_numpy(parsed_scene.points),
            points_rgb=torch.from_numpy(parsed_scene.points_rgb),
            sh_degree=sh_degree,
            init_scale=init_scale,
            init_opacity=init_opacity,
            device=device,
        )

    if init_type == "random":
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
        f"Unsupported init_type={init_cfg.init_type!r} "
        "(expected 'sfm', 'random', or 'from_ckpt'; "
        "and 'sfm' requires COLMAP points3D)."
    )


def build_training_context(cfg: TrainConfig) -> TrainingContext:
    device = torch.device(cfg.io.device)
    dataset, loader = build_dataset_and_loader(
        io_cfg=cfg.io,
        data_cfg=cfg.data,
        split="train",
    )
    eval_dataset: Optional[InputDataset] = None
    eval_loader: Optional[DataLoader] = None
    if cfg.eval.enable:
        # Eval split is dataparser-defined (typically "test" / holdout).
        eval_dataset, eval_loader = build_dataset_and_loader(
            io_cfg=cfg.io,
            data_cfg=cfg.data,
            split=str(cfg.eval.split),
        )
    gaussian_model = build_gaussian_model(
        dataset=dataset,
        init_cfg=cfg.init,
        optim_cfg=cfg.optim,
        device=device,
    )
    parsed_scene = dataset.parsed_scene
    n_images = int(len(parsed_scene.image_names))

    bilateral_grid: Optional[BilateralGridPostProcessor] = None
    if bool(cfg.postprocess.use_bilateral_grid):
        # Optional post-process module: contributes its own param group(s).
        bilateral_grid = BilateralGridPostProcessor.create(
            num_frames=int(n_images),
            grid_shape=tuple(int(x) for x in cfg.postprocess.bilateral_grid_shape),
            device=device,
        )

    pose_adjust: Optional[PoseOptModule] = None
    if cfg.pose.pose_opt:
        # Optional pose module: per-frame learnable adjustment (initialized at identity).
        pose_adjust = PoseOptModule(n_images).to(device)
        pose_adjust.zero_init()

    # Core boundary contract: modules expose *named parameter groups*; trainer owns optimizer policy.
    param_groups: Dict[str, list[torch.nn.Parameter]] = {}
    param_groups.update(gaussian_model.get_param_groups())
    if pose_adjust is not None:
        param_groups.update(pose_adjust.get_param_groups())
    if bilateral_grid is not None:
        param_groups.update(bilateral_grid.get_param_groups())

    # Build optimizers/schedulers from the merged param groups (nerfstudio-style config).
    optimizer_bundle = OptimizerBundle.build_from_param_groups(
        optim_cfg=cfg.optim,
        batch_size=int(cfg.data.batch_size),
        world_size=cfg.world_size,
        device=device,
        scene_scale=float(parsed_scene.scene_scale),
        param_groups=param_groups,
        splat_group_names=set(gaussian_model.get_param_groups().keys()),
    )

    natural_selection_policy: Optional[NaturalSelectionPolicy] = None
    if cfg.gns.gns_enable:
        # GNS regularizes opacity and influences which Gaussians survive/densify.
        base_gns_reg_interval = auto_gns_reg_interval(num_train_images=len(dataset))
        gns_reg_interval = int(base_gns_reg_interval)
        gns_reg_interval = max(
            1,
            int(round(float(gns_reg_interval) * float(cfg.optim.steps_scaler))),
        )
        natural_selection_policy = NaturalSelectionPolicy(
            cfg=cfg.gns,
            densify_stop_step=int(cfg.strategy.refine_stop_iter),
            reg_interval=int(gns_reg_interval),
            verbose=bool(cfg.strategy.verbose),
        )

    impl = str(cfg.strategy.impl).strip().lower()
    if impl == "improved":
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
    elif impl == "default":
        strategy = DefaultStrategy(
            prune_opa=float(cfg.strategy.prune_opa),
            grow_grad2d=float(cfg.strategy.grow_grad2d),
            grow_scale3d=float(cfg.strategy.grow_scale3d),
            grow_scale2d=float(cfg.strategy.grow_scale2d),
            prune_scale3d=float(cfg.strategy.prune_scale3d),
            prune_scale2d=float(cfg.strategy.prune_scale2d),
            refine_scale2d_stop_iter=int(cfg.strategy.refine_scale2d_stop_iter),
            refine_start_iter=int(cfg.strategy.refine_start_iter),
            refine_stop_iter=int(cfg.strategy.refine_stop_iter),
            reset_every=int(cfg.strategy.reset_every),
            refine_every=int(cfg.strategy.refine_every),
            pause_refine_after_reset=int(cfg.strategy.pause_refine_after_reset),
            absgrad=bool(cfg.strategy.absgrad),
            revised_opacity=bool(cfg.strategy.revised_opacity),
            verbose=bool(cfg.strategy.verbose),
            key_for_gradient=str(cfg.strategy.key_for_gradient),  # type: ignore[arg-type]
        )
    elif impl == "mcmc":
        strategy = MCMCStrategy(
            cap_max=int(cfg.strategy.mcmc_cap_max),
            noise_lr=float(cfg.strategy.mcmc_noise_lr),
            refine_start_iter=int(cfg.strategy.refine_start_iter),
            refine_stop_iter=int(cfg.strategy.refine_stop_iter),
            noise_injection_stop_iter=int(cfg.strategy.mcmc_noise_injection_stop_iter),
            refine_every=int(cfg.strategy.refine_every),
            min_opacity=float(cfg.strategy.mcmc_min_opacity),
            verbose=bool(cfg.strategy.verbose),
        )
    else:
        raise ValueError(
            "Unknown strategy.impl="
            f"{cfg.strategy.impl!r} (expected 'improved', 'default', or 'mcmc')."
        )
    # Strategy sanity requires access to raw splat tensors + the per-splat optimizers.
    strategy.check_sanity(gaussian_model.splats, optimizer_bundle.splat_optimizers)
    strategy_state = strategy.initialize_state(
        scene_scale=float(parsed_scene.scene_scale)
    )

    optimizer_coordinator = OptimizerCoordinator(
        optim_cfg=cfg.optim,
        device=device,
        gaussian_model=gaussian_model,
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
        pose_adjust=pose_adjust,
        bilateral_grid=bilateral_grid,
        natural_selection_policy=natural_selection_policy,
        strategy=strategy,
        strategy_state=strategy_state,
        optimizer_coordinator=optimizer_coordinator,
    )
