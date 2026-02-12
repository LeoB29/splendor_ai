import torch
import os

from alpha_zero import az_train, DML_DEVICE


if __name__ == "__main__":
    if DML_DEVICE is None:
        raise SystemExit("DirectML not available. Install torch-directml and use the DirectML env.")

    dev = DML_DEVICE
    print("[Device] Using DirectML (AMD/Intel GPU)")

    # -----------------------------
    # Profile selection
    # -----------------------------
    # Set PROFILE to one of: "recovery", "progress", "default"
    # recovery: safest (post-instability stabilization)
    # progress: slightly more aggressive while keeping guardrails
    # default: baseline long-run profile
    PROFILE = "recovery"

    # Reset weak gating anchors before a new run.
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

    if PROFILE == "recovery":
        cfg = dict(
            iterations=20,
            games_per_iter=48,
            mcts_simulations=256,
            mcts_batch=32,
            lr=3e-5,
            lr_min=1e-6,
            device=dev,
            replay_capacity=50000,
            batch_size=512,
            train_batches_per_iter=8,
            eval_games=40,
            resume=False,
            #resume_path=r"checkpoints\az_iter_10.pt",
            resume_optimizer_state=False,
            weight_decay=1e-4,
            grad_clip=0.4,
            policy_weight=1.0,
            value_weight=1.0,
            return_weight=0.25,
            use_return_loss=True,
            temp_init=1.0,
            temp_final=0.0,
            temp_moves=30,
            entropy_init=0.015,
            entropy_anneal_iters=20,
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
            divergence_policy_loss=4.3,
            divergence_total_loss=5.8,
            divergence_lr_backoff=0.5,
            divergence_min_lr=1e-6,
            divergence_max_skips_per_iter=1,
            divergence_rollback=True,
        )
        print("[Profile] Using RECOVERY profile")
    elif PROFILE == "progress":
        cfg = dict(
            iterations=20,
            games_per_iter=48,
            mcts_simulations=256,
            mcts_batch=32,
            lr=7.5e-5,
            lr_min=1e-6,
            device=dev,
            replay_capacity=50000,
            batch_size=512,
            train_batches_per_iter=12,
            eval_games=40,
            resume=False,
            resume_path=r"checkpoints\az_iter_11.pt",
            resume_optimizer_state=False,
            weight_decay=1e-4,
            grad_clip=0.45,
            policy_weight=1.0,
            value_weight=1.0,
            return_weight=0.25,
            use_return_loss=True,
            temp_init=1.0,
            temp_final=0.0,
            temp_moves=30,
            entropy_init=0.02,
            entropy_anneal_iters=20,
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
            divergence_policy_loss=8.0,
            divergence_total_loss=12.0,
            divergence_lr_backoff=0.5,
            divergence_min_lr=1e-6,
            divergence_max_skips_per_iter=3,
            divergence_rollback=True,
        )
        print("[Profile] Using PROGRESS profile")
    else:
        cfg = dict(
            iterations=20,
            games_per_iter=48,
            mcts_simulations=256,
            mcts_batch=32,
            # DirectML long-run profile: baseline
            lr=1e-4,
            lr_min=1e-6,
            device=dev,
            replay_capacity=50000,
            batch_size=512,
            train_batches_per_iter=10,
            eval_games=12,
            resume=False,
            resume_path=r"checkpoints\az_iter_10.pt",
            resume_optimizer_state=False,
            weight_decay=1e-4,
            grad_clip=0.5,
            policy_weight=1.0,
            value_weight=1.0,
            return_weight=0.25,
            use_return_loss=True,
            temp_init=1.0,
            temp_final=0.0,
            temp_moves=30,
            entropy_init=0.02,
            entropy_anneal_iters=20,
            compile_model=False,
            res_blocks=6,
            width=512,
            use_elo_ladder=True,
            gate_pool=True,
            gate_games=128,
            gate_threshold=0.52,
            gate_start_iter=6,
            gate_revert_on_reject=False,
            divergence_policy_loss=50.0,
            divergence_total_loss=200.0,
            divergence_lr_backoff=0.5,
            divergence_min_lr=1e-6,
            divergence_max_skips_per_iter=3,
            divergence_rollback=True,
        )
        print("[Profile] Using DEFAULT profile")

    print(
        "[Config] "
        f"resume={cfg.get('resume')} "
        f"resume_path={cfg.get('resume_path')} "
        f"resume_optimizer_state={cfg.get('resume_optimizer_state')} "
        f"lr={cfg.get('lr'):.2e} "
        f"lr_min={cfg.get('lr_min'):.2e} "
        f"batches={cfg.get('train_batches_per_iter')} "
        f"eval_games={cfg.get('eval_games')} "
        f"div_pol={cfg.get('divergence_policy_loss')} "
        f"div_tot={cfg.get('divergence_total_loss')} "
        f"gate_start_iter={cfg.get('gate_start_iter')}"
    )

    az_train(**cfg)
