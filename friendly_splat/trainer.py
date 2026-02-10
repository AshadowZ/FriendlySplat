from __future__ import annotations

"""Training entrypoint for FriendlySplat.

This file intentionally stays thin and delegates most logic to:
- `friendly_splat.trainer.builder`: context construction (data/model/components/optimizers)
- `friendly_splat.trainer.step_runtime`: per-step schedule/render/loss helpers
- `friendly_splat.trainer.io_utils`: checkpoints / PLY export
- `friendly_splat.viewer.viewer_runtime`: optional online viewer (viser/nerfview)
"""

import torch
import tyro
import tqdm

from friendly_splat.viewer.viewer_runtime import ViewerRuntime

from friendly_splat.trainer.builder import build_training_context
from friendly_splat.trainer.logger import TensorBoardWriter, handle_step_logging

from friendly_splat.trainer.step_runtime import (
    build_step_schedule_from_prepared_batch,
    compute_losses_from_prepared_batch_and_render,
    maybe_run_evaluation_for_step,
    prepare_training_batch,
    render_from_prepared_batch,
)

from friendly_splat.trainer.configs import TrainConfig, validate_train_config
from friendly_splat.utils.common_utils import set_seed
from friendly_splat.trainer.io_utils import (
    init_output_paths,
    save_train_config_snapshot,
    maybe_save_outputs,
)



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
        init_output_paths(io_cfg=cfg.io, eval_cfg=cfg.eval)
        save_train_config_snapshot(io_cfg=cfg.io, train_cfg=cfg)

        context = build_training_context(cfg)
        self.dataset = context.dataset
        self.loader = context.loader
        self.eval_dataset = context.eval_dataset
        self.eval_loader = context.eval_loader
        self.gaussian_model = context.gaussian_model
        self.splats = context.splats
        self.bilagrid = context.bilagrid
        self.ppisp = context.ppisp
        self.pose_adjust = context.pose_adjust
        self.natural_selection_policy = context.natural_selection_policy
        self.strategy = context.strategy
        self.strategy_state = context.strategy_state
        self.optimizer_coordinator = context.optimizer_coordinator
        print(f"Initialized {self.splats['means'].shape[0]} gaussians.")

        torch.backends.cudnn.benchmark = True

        # Viewer runtime is initialized lazily in `train()` after DataLoader workers
        # are started, to avoid fork-after-threads stalls with num_workers>0.
        self.viewer_runtime = None

    def train(self) -> None:
        cfg = self.cfg
        splats = self.splats
        eval_loader = self.eval_loader
        bilagrid = self.bilagrid
        ppisp = self.ppisp
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
            splats=splats,
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
            # /*-------------------- Viewer / Step Preamble --------------------*/
            # Respect viewer pause state and take the step lock.
            tic = viewer_runtime.before_step()

            # Apply per-step optimizer/GNS policy updates.
            optimizer_coordinator.prepare_step(step=int(step))

            # /*-------------------- Data + Schedule --------------------*/
            # Load one training batch.
            prepared_batch = next(loader_iter)

            # Optionally adjust camera poses for this batch.
            prepared_batch = prepare_training_batch(
                prepared_batch=prepared_batch,
                pose_opt=bool(cfg.pose.pose_opt),
                pose_adjust=pose_adjust,
            )

            # Build the step schedule (active regs + render mode).
            schedule = build_step_schedule_from_prepared_batch(
                step=step,
                optim_cfg=cfg.optim,
                reg_cfg=cfg.reg,
                prepared_batch=prepared_batch,
            )

            # /*-------------------- Forward Render + Loss --------------------*/
            # Render with optional postprocessing.
            render_out = render_from_prepared_batch(
                prepared_batch=prepared_batch,
                splats=splats,
                optim_cfg=cfg.optim,
                postprocess_cfg=cfg.postprocess,
                schedule=schedule,
                absgrad=bool(cfg.strategy.absgrad),
                bilagrid=bilagrid,
                ppisp=ppisp,
            )
            meta = render_out.meta
            active_sh_degree = render_out.active_sh_degree

            # Assemble total loss and per-term loss items.
            loss_output = compute_losses_from_prepared_batch_and_render(
                reg_cfg=cfg.reg,
                postprocess_cfg=cfg.postprocess,
                schedule=schedule,
                step=step,
                prepared_batch=prepared_batch,
                render_out=render_out,
                splats=splats,
                bilagrid=bilagrid,
                ppisp=ppisp,
                gns=gns,
            )
            loss = loss_output.total

            # /*-------------------- Backward + Optimizer Step --------------------*/
            # Run pre-backward strategy hooks.
            strategy.step_pre_backward(
                splats,
                optimizer_coordinator.splat_optimizers,
                strategy_state,
                step,
                meta,
            )

            # Backpropagate and apply optimizer/scheduler updates.
            optimizer_coordinator.zero_grad()
            loss.backward()

            optimizer_coordinator.step_all(
                step=int(step),
                meta=meta,
                batch_size=int(prepared_batch.pixels.shape[0]),
            )

            # /*-------------------- Strategy Post Update --------------------*/
            # Run post-update densification/pruning hooks.
            strategy.step_post_backward(
                params=splats,
                optimizers=optimizer_coordinator.splat_optimizers,
                state=strategy_state,
                step=step,
                info=meta,
                packed=cfg.optim.packed,
            )

            if gns is not None:
                gns.step_post_update(
                    step=step,
                    params=splats,
                    optimizers=optimizer_coordinator.splat_optimizers,
                    strategy_state=strategy_state,
                )

            # Update viewer statistics and release the step lock.
            viewer_runtime.after_step(
                step=int(step),
                tic=tic,
                batch_size=int(prepared_batch.pixels.shape[0]),
                height=int(prepared_batch.height),
                width=int(prepared_batch.width),
                meta=meta,
            )

            # /*-------------------- Eval / Logging / Outputs --------------------*/
            # Run periodic evaluation and report summary in the progress bar.
            eval_output = maybe_run_evaluation_for_step(
                step=int(step),
                train_cfg=cfg,
                eval_loader=eval_loader,
                splats=splats,
                bilagrid=bilagrid,
                ppisp=ppisp,
            )
            log_payload = handle_step_logging(
                step=int(step),
                device=self.device,
                num_gs=int(splats["means"].shape[0]),
                train_loss_items=loss_output.items,
                eval_stats=eval_output.stats if eval_output is not None else None,
                tb_writer=tb_writer,
            )
            viewer_runtime.log_payload(payload=log_payload)
            # Keep tqdm refresh frequency low to reduce terminal overhead.
            if (int(step) % int(tqdm_update_every) == 0) or (
                int(step) == int(cfg.optim.max_steps) - 1
            ):
                pbar.set_description(f"sh degree={active_sh_degree}| ")

            # Save configured artifacts (checkpoint / PLY).
            maybe_save_outputs(
                io_cfg=cfg.io,
                pose_cfg=cfg.pose,
                postprocess_cfg=cfg.postprocess,
                train_cfg=cfg,
                step=int(step),
                max_steps=int(cfg.optim.max_steps),
                splats=splats,
                active_sh_degree=int(active_sh_degree),
                pose_adjust=pose_adjust,
                bilagrid=bilagrid,
                ppisp=ppisp,
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
