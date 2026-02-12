import torch

from alpha_zero import az_train


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Install CUDA build and ensure an NVIDIA GPU is present.")

    dev = torch.device("cuda")
    print("[Device] Using CUDA GPU")

    az_train(
        iterations=20,
        games_per_iter=64,
        mcts_simulations=384,
        mcts_batch=96,
        lr=5e-4,
        device=dev,
        replay_capacity=75000,
        batch_size=512,
        train_batches_per_iter=12,
        eval_games=24,
        resume=True,
        weight_decay=1e-4,
        grad_clip=1.0,
        policy_weight=1.0,
        value_weight=1.0,
        use_return_loss=True,
        temp_init=1.0,
        temp_final=0.0,
        temp_moves=30,
        compile_model=False,
        res_blocks=6,
        width=512,
        use_elo_ladder=True,
    )
