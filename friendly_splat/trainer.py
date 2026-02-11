from __future__ import annotations

"""Training entrypoint for FriendlySplat.

This file intentionally stays thin and mostly orchestrates:
- `friendly_splat.trainer.builder`: builds the training context (data/modules/strategy/optimizers)
- `friendly_splat.trainer.step_runtime`: per-step batch prep/schedule/render/loss helpers
- `friendly_splat.trainer.logger`: TensorBoard writing + (optional) viewer log payloads
- `friendly_splat.trainer.io_utils`: output folders, checkpoints, and exports
- `friendly_splat.viewer.viewer_runtime`: optional online viewer integration
"""

import torch
import tyro
import tqdm

import random
import numpy as np

from friendly_splat.viewer.viewer_runtime import ViewerRuntime

from friendly_splat.trainer.builder import build_training_context
from friendly_splat.trainer.logger import TensorBoardWriter, maybe_handle_step_logging

from friendly_splat.trainer.step_runtime import (
    build_step_schedule_from_prepared_batch,
    compute_losses_from_prepared_batch_and_render,
    maybe_run_evaluation_for_step,
    prepare_training_batch,
    render_from_prepared_batch,
)

from friendly_splat.trainer.configs import TrainConfig, validate_train_config
from friendly_splat.trainer.io_utils import (
    init_output_paths,
    save_train_config_snapshot,
    maybe_save_outputs,
)


def set_seed(seed: int) -> None:
    # Local, explicit seeding (kept in trainer to avoid a shared utils module).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

        # Validate configuration and runtime prerequisites.
        validate_train_config(cfg)

        # Device + reproducibility.
        set_seed(cfg.io.seed)
        self.device = torch.device(cfg.io.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

        # Prepare output folders (checkpoints / PLY exports).
        init_output_paths(io_cfg=cfg.io)
        save_train_config_snapshot(io_cfg=cfg.io, train_cfg=cfg)

        context = build_training_context(cfg)
        self.dataset = context.dataset
        self.loader = context.loader
        self.eval_dataset = context.eval_dataset
        self.eval_loader = context.eval_loader
        self.gaussian_model = context.gaussian_model
        self.bilateral_grid = context.bilateral_grid
        self.pose_adjust = context.pose_adjust
        self.natural_selection_policy = context.natural_selection_policy
        self.strategy = context.strategy
        self.strategy_state = context.strategy_state
        self.optimizer_coordinator = context.optimizer_coordinator
        print(f"Initialized {self.gaussian_model.num_gaussians} gaussians.")

        torch.backends.cudnn.benchmark = True

        # Viewer runtime is initialized in `train()` (after DataLoader workers are
        # started) to avoid fork-after-threads stalls when num_workers>0.
        self.viewer_runtime = None

    def train(self) -> None:
        cfg = self.cfg
        gaussian_model = self.gaussian_model
        eval_loader = self.eval_loader
        bilateral_grid = self.bilateral_grid
        pose_adjust = self.pose_adjust
        strategy = self.strategy
        strategy_state = self.strategy_state
        gns = self.natural_selection_policy
        optimizer_coordinator = self.optimizer_coordinator
        loader_iter = iter(self.loader)
        tb_writer = TensorBoardWriter(io_cfg=cfg.io, tb_cfg=cfg.tb)

        viewer_runtime = ViewerRuntime(
            disable_viewer=cfg.viewer.disable_viewer,
            port=cfg.viewer.port,
            device=self.device,
            gaussian_model=gaussian_model,
            output_dir=cfg.io.result_dir,
            train_dataset=self.dataset,
        )
        self.viewer_runtime = viewer_runtime
        tqdm_update_every = 10
        pbar = tqdm.tqdm(
            range(cfg.optim.max_steps),
            miniters=int(tqdm_update_every),
        )

        for step in pbar:
            # Viewer sync (pause/step-lock).
            tic = viewer_runtime.before_step()

            # Per-step policy hooks (LR updates, etc.).
            optimizer_coordinator.prepare_step(step=int(step))

            # Data.
            prepared_batch = next(loader_iter)

            # Optional pose optimization (applies to this batch only).
            prepared_batch = prepare_training_batch(
                prepared_batch=prepared_batch,
                pose_opt=bool(cfg.pose.pose_opt),
                pose_adjust=pose_adjust,
            )

            # Step schedule (which regularizers/aux outputs are active).
            schedule = build_step_schedule_from_prepared_batch(
                step=step,
                optim_cfg=cfg.optim,
                reg_cfg=cfg.reg,
                prepared_batch=prepared_batch,
            )

            # Render.
            render_out = render_from_prepared_batch(
                prepared_batch=prepared_batch,
                gaussian_model=gaussian_model,
                optim_cfg=cfg.optim,
                schedule=schedule,
                absgrad=bool(cfg.strategy.absgrad),
                bilateral_grid=bilateral_grid,
            )
            meta = render_out.meta
            active_sh_degree = render_out.active_sh_degree

            # Losses.
            loss_output = compute_losses_from_prepared_batch_and_render(
                reg_cfg=cfg.reg,
                schedule=schedule,
                step=step,
                prepared_batch=prepared_batch,
                render_out=render_out,
                gaussian_model=gaussian_model,
                bilateral_grid=bilateral_grid,
                bilateral_grid_tv_weight=float(cfg.postprocess.bilateral_grid_tv_weight),
                gns=gns,
            )
            loss = loss_output.total

            # Strategy hooks may modify params/optimizers (e.g., densification).
            strategy.step_pre_backward(
                gaussian_model.splats,
                optimizer_coordinator.splat_optimizers,
                strategy_state,
                step,
                meta,
            )

            # Backward + optimizer step.
            optimizer_coordinator.zero_grad()
            loss.backward()

            optimizer_coordinator.step_all(
                step=int(step),
                meta=meta,
                batch_size=int(prepared_batch.pixels.shape[0]),
            )

            # Strategy post-update (densify/prune).
            strategy.step_post_backward(
                params=gaussian_model.splats,
                optimizers=optimizer_coordinator.splat_optimizers,
                state=strategy_state,
                step=step,
                info=meta,
                packed=cfg.optim.packed,
            )

            if gns is not None:
                gns.step_post_update(
                    step=step,
                    params=gaussian_model.splats,
                    optimizers=optimizer_coordinator.splat_optimizers,
                    strategy_state=strategy_state,
                )

            # Update viewer stats + release step lock.
            viewer_runtime.after_step(
                step=int(step),
                tic=tic,
                batch_size=int(prepared_batch.pixels.shape[0]),
                height=int(prepared_batch.height),
                width=int(prepared_batch.width),
                meta=meta,
            )

            # Periodic evaluation.
            eval_output = maybe_run_evaluation_for_step(
                step=int(step),
                train_cfg=cfg,
                eval_loader=eval_loader,
                gaussian_model=gaussian_model,
                bilateral_grid=bilateral_grid,
            )
            # Logging is cadence-driven; when nothing is due, payload is None.
            log_payload = maybe_handle_step_logging(
                step=int(step),
                device=self.device,
                num_gs=int(gaussian_model.num_gaussians),
                train_loss_items=loss_output.items,
                eval_stats=eval_output.stats if eval_output is not None else None,
                tb_writer=tb_writer,
                emit_payload=not bool(cfg.viewer.disable_viewer),
            )
            if not bool(cfg.viewer.disable_viewer):
                viewer_runtime.log_payload(payload=log_payload)

            # Keep tqdm refresh frequency low to reduce terminal overhead.
            if (int(step) % int(tqdm_update_every) == 0) or (
                int(step) == int(cfg.optim.max_steps) - 1
            ):
                pbar.set_description(f"sh degree={active_sh_degree}| ")

            # Save configured artifacts (checkpoint / exports).
            maybe_save_outputs(
                io_cfg=cfg.io,
                pose_cfg=cfg.pose,
                train_cfg=cfg,
                step=int(step),
                max_steps=int(cfg.optim.max_steps),
                gaussian_model=gaussian_model,
                active_sh_degree=int(active_sh_degree),
                pose_adjust=pose_adjust,
                bilateral_grid=bilateral_grid,
            )

        tb_writer.close()
        viewer_runtime.complete()
        if (not cfg.viewer.disable_viewer) and cfg.viewer.keep_alive_after_train:
            viewer_runtime.keep_alive()


def train(cfg: TrainConfig) -> None:
    Trainer(cfg).train()


def _parse_args() -> TrainConfig:
    return tyro.cli(TrainConfig)


if __name__ == "__main__":
    Trainer(_parse_args()).train()
