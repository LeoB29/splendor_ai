import os
import torch

from alpha_zero import az_train, DML_DEVICE


if __name__ == "__main__":
    if DML_DEVICE is None:
        raise SystemExit("DirectML not available. Install torch-directml and use the DirectML env.")

    dev = DML_DEVICE
    print("[Device] Using DirectML (AMD/Intel GPU)")

    # Attention-branch profiles tuned for the attention encoder.
    # Set to one of: "ramp", "stable", "progress"
    PROFILE = "ramp"

    # Reset stale gate anchors to avoid inheriting weak champions.
    RESET_ANCHORS_ON_START = True
    if RESET_ANCHORS_ON_START:
        for p in [
            os.path.join("checkpoints", "champion.pt"),
            os.path.join("logs", "elo_pool.json"),
            os.path.join("logs", "elo_log.csv"),
            os.path.join("logs", "gating_log.csv"),
        ]:
            try:
                if os.path.exists(p):
                    os.remove(p)
                    print(f"[Init] Removed stale anchor/log: {p}")
            except Exception as e:
                print(f"[Init] Could not remove {p}: {e}")

    if PROFILE == "ramp":
        cfg = dict(
            iterations=30,
            games_per_iter=48,
            mcts_simulations=256,
            mcts_batch=32,
            eval_mcts_batch=64,
            eval_workers=4,
            eval_no_legal_sample_cap=2,
            lr=1.0e-5,
            lr_min=1e-6,
            device=dev,
            replay_capacity=50000,
            batch_size=512,
            train_batches_per_iter=10,
            eval_games=40,
            resume=False,
            resume_optimizer_state=False,
            weight_decay=1e-4,
            grad_clip=0.20,
            policy_weight=1.0,
            value_weight=1.0,
            return_weight=0.25,
            use_return_loss=True,
            temp_init=1.0,
            temp_final=0.0,
            temp_moves=30,
            entropy_init=0.015,
            entropy_anneal_iters=30,
            warmup_iters=0,
            compile_model=False,
            res_blocks=6,
            width=512,
            use_elo_ladder=True,
            gate_pool=True,
            gate_games=128,
            gate_threshold=0.52,
            gate_start_iter=6,
            gate_revert_on_reject=False,
            divergence_policy_loss=4.2,
            divergence_total_loss=5.8,
            divergence_lr_backoff=0.5,
            divergence_min_lr=1e-6,
            divergence_max_skips_per_iter=1,
            divergence_rollback=True,
            stability_ramp=True,
            ramp_min_buffer=20000,
            ramp_iters=12,
            ramp_batch_frac_start=0.10,
            ramp_divergence_policy_start=8.0,
            ramp_divergence_total_start=10.5,
            ramp_max_skips_start=1,
        )
        print("[Profile] Using ATTENTION-RAMP profile")
    elif PROFILE == "stable":
        cfg = dict(
            iterations=30,
            games_per_iter=48,
            mcts_simulations=256,
            mcts_batch=32,
            eval_mcts_batch=64,
            eval_workers=4,
            eval_no_legal_sample_cap=2,
            lr=2.5e-5,
            lr_min=1e-6,
            device=dev,
            replay_capacity=50000,
            batch_size=512,
            train_batches_per_iter=10,
            eval_games=40,
            resume=False,
            resume_optimizer_state=False,
            weight_decay=1e-4,
            grad_clip=0.35,
            policy_weight=1.0,
            value_weight=1.0,
            return_weight=0.25,
            use_return_loss=True,
            temp_init=1.0,
            temp_final=0.0,
            temp_moves=30,
            entropy_init=0.015,
            entropy_anneal_iters=30,
            warmup_iters=0,
            compile_model=False,
            res_blocks=6,
            width=512,
            use_elo_ladder=True,
            gate_pool=True,
            gate_games=128,
            gate_threshold=0.52,
            gate_start_iter=6,
            gate_revert_on_reject=False,
            divergence_policy_loss=4.0,
            divergence_total_loss=5.6,
            divergence_lr_backoff=0.5,
            divergence_min_lr=1e-6,
            divergence_max_skips_per_iter=1,
            divergence_rollback=True,
        )
        print("[Profile] Using ATTENTION-STABLE profile")
    else:
        cfg = dict(
            iterations=30,
            games_per_iter=48,
            mcts_simulations=256,
            mcts_batch=32,
            eval_mcts_batch=64,
            eval_workers=4,
            eval_no_legal_sample_cap=2,
            lr=4.0e-5,
            lr_min=1e-6,
            device=dev,
            replay_capacity=50000,
            batch_size=512,
            train_batches_per_iter=12,
            eval_games=40,
            resume=False,
            resume_optimizer_state=False,
            weight_decay=1e-4,
            grad_clip=0.40,
            policy_weight=1.0,
            value_weight=1.0,
            return_weight=0.25,
            use_return_loss=True,
            temp_init=1.0,
            temp_final=0.0,
            temp_moves=30,
            entropy_init=0.02,
            entropy_anneal_iters=25,
            warmup_iters=1,
            compile_model=False,
            res_blocks=6,
            width=512,
            use_elo_ladder=True,
            gate_pool=True,
            gate_games=128,
            gate_threshold=0.52,
            gate_start_iter=6,
            gate_revert_on_reject=False,
            divergence_policy_loss=4.6,
            divergence_total_loss=6.2,
            divergence_lr_backoff=0.5,
            divergence_min_lr=1e-6,
            divergence_max_skips_per_iter=2,
            divergence_rollback=True,
        )
        print("[Profile] Using ATTENTION-PROGRESS profile")

    print(
        "[Config] "
        f"profile={PROFILE} "
        f"resume={cfg.get('resume')} "
        f"lr={cfg.get('lr'):.2e} "
        f"lr_min={cfg.get('lr_min'):.2e} "
        f"batches={cfg.get('train_batches_per_iter')} "
        f"eval_games={cfg.get('eval_games')} "
        f"eval_mcts_batch={cfg.get('eval_mcts_batch', cfg.get('mcts_batch'))} "
        f"eval_workers={cfg.get('eval_workers', 0)} "
        f"div_pol={cfg.get('divergence_policy_loss')} "
        f"div_tot={cfg.get('divergence_total_loss')} "
        f"gate_start_iter={cfg.get('gate_start_iter')} "
        f"ramp={cfg.get('stability_ramp', False)} "
        f"ramp_min_buffer={cfg.get('ramp_min_buffer', 0)}"
    )

    az_train(**cfg)
