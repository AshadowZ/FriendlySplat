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

from friendly_splat.trainer.step_runtime import (
    build_step_schedule_from_prepared_batch,
    compute_losses_from_prepared_batch_and_render,
    prepare_training_batch,
    render_from_prepared_batch,
)

from friendly_splat.trainer.configs import TrainConfig, validate_train_config
from friendly_splat.utils.common_utils import set_seed
from friendly_splat.trainer.io_utils import (
    init_output_paths,
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
        init_output_paths(io_cfg=cfg.io)

        context = build_training_context(cfg)
        self.dataset = context.dataset
        self.loader = context.loader
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

        # Optional online viewer runtime (pause/resume + interactive renders).
        self.viewer_runtime = ViewerRuntime(
            disable_viewer=bool(cfg.viewer.disable_viewer),
            port=int(cfg.viewer.port),
            device=self.device,
            splats=self.splats,
            output_dir=cfg.io.result_dir,
        )

    def train(self) -> None:
        cfg = self.cfg
        splats = self.splats
        bilagrid = self.bilagrid
        ppisp = self.ppisp
        pose_adjust = self.pose_adjust
        strategy = self.strategy
        strategy_state = self.strategy_state
        gns = self.natural_selection_policy
        optimizer_coordinator = self.optimizer_coordinator
        viewer_runtime = self.viewer_runtime

        loader_iter = iter(self.loader)
        tqdm_update_every = 10
        pbar = tqdm.tqdm(
            range(cfg.optim.max_steps),
            miniters=int(tqdm_update_every),
        )
        
        for step in pbar:
            # Viewer runtime may pause training and holds a lock during the step.
            tic = viewer_runtime.before_step()

            # Step 0: Prepare per-step optimizer/GNS state.
            # If GNS is enabled, opacity LR is boosted during GNS (default 4x).
            optimizer_coordinator.prepare_step(step=int(step))

            # Step 1: Fetch the next training batch.
            prepared_batch = next(loader_iter)

            # Step 2: Apply optional pose optimization adjustment.
            prepared_batch = prepare_training_batch(
                prepared_batch=prepared_batch,
                pose_opt=bool(cfg.pose.pose_opt),
                pose_adjust=pose_adjust,
            )

            # Step 3: Build the per-step schedule (active regs + render mode).
            schedule = build_step_schedule_from_prepared_batch(
                step=step,
                optim_cfg=cfg.optim,
                reg_cfg=cfg.reg,
                prepared_batch=prepared_batch,
            )

            # Step 4: Render and apply optional postprocessing.
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

            # Step 5: Compute total training loss.
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

            # Step 6: Run pre-backward strategy hook.
            strategy.step_pre_backward(
                splats,
                optimizer_coordinator.splat_optimizers,
                strategy_state,
                step,
                meta,
            )

            # Step 7: Backpropagate and update optimizers/schedulers.
            optimizer_coordinator.zero_grad()
            loss.backward()

            # Limit tqdm refresh frequency to reduce terminal overhead.
            if (int(step) % int(tqdm_update_every) == 0) or (int(step) == int(cfg.optim.max_steps) - 1):
                pbar.set_description(f"sh degree={active_sh_degree}| ")

            optimizer_coordinator.step_all(
                step=int(step),
                meta=meta,
                batch_size=int(prepared_batch.pixels.shape[0]),
            )

            # Step 8: Run post-update densification / pruning hooks.
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

            # Step 9: Optionally save checkpoint / PLY outputs.
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

            # Finalize viewer step (update counters and release lock).
            viewer_runtime.after_step(
                step=int(step),
                tic=tic,
                batch_size=int(prepared_batch.pixels.shape[0]),
                height=int(prepared_batch.height),
                width=int(prepared_batch.width),
                meta=meta,
            )

        viewer_runtime.complete()
        if (not bool(cfg.viewer.disable_viewer)) and bool(cfg.viewer.keep_alive_after_train):
            viewer_runtime.keep_alive()


def train(cfg: TrainConfig) -> None:
    Trainer(cfg).train()


def _parse_args() -> TrainConfig:
    return tyro.cli(TrainConfig)


if __name__ == "__main__":
    Trainer(_parse_args()).train()
