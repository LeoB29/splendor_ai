import torch

from alpha_zero import az_train, DML_DEVICE


def pick_device():
    if torch.cuda.is_available():
        print("[Device] Using CUDA GPU")
        return torch.device("cuda")
    elif DML_DEVICE is not None:
        print("[Device] Using DirectML (AMD/Intel GPU)")
        return DML_DEVICE
    else:
        print("[Device] Using CPU")
        return "cpu"


if __name__ == "__main__":
    dev = pick_device()

    # Small, fast config intended to run quickly even on CPU
    # Adjusts mcts_batch based on device for better throughput.
    is_cuda = isinstance(dev, torch.device) and dev.type == "cuda"
    mcts_batch = 32 if is_cuda else 8

    az_train(
        iterations=1,
        games_per_iter=2,
        mcts_simulations=64,
        mcts_batch=mcts_batch,
        lr=1e-3,
        device=dev,
        replay_capacity=5000,
        batch_size=128,
        train_batches_per_iter=10,
        eval_games=5,
        resume=False,
        weight_decay=1e-4,
        grad_clip=1.0,
        policy_weight=1.0,
        value_weight=1.0,
        temp_init=1.0,
        temp_final=0.0,
        temp_moves=15,
        compile_model=False,
        res_blocks=6,
        width=512,
        big_eval_every=0,  # disable heavy eval in smoke run
    )
