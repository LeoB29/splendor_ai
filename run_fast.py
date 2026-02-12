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
    is_dml = (not is_cuda) and (dev != "cpu")
    mcts_batch = 32 if is_cuda else (16 if is_dml else 8)
    batch_size = 256 if is_dml else (256 if is_cuda else 128)
    # Conservative optimizer settings on DML improve stability with the new return head.
    lr = 2.5e-4 if is_dml else (5e-4 if is_cuda else 1e-3)
    grad_clip = 0.7 if is_dml else 1.0
    return_weight = 0.25 if is_dml else 0.5

    az_train(
        iterations=2,
        games_per_iter=6,
        mcts_simulations=96,
        mcts_batch=mcts_batch,
        lr=lr,
        device=dev,
        replay_capacity=5000,
        batch_size=batch_size,
        train_batches_per_iter=8,
        eval_games=5,
        resume=False,
        weight_decay=1e-4,
        grad_clip=grad_clip,
        policy_weight=1.0,
        value_weight=1.0,
        return_weight=return_weight,
        use_return_loss=True,
        temp_init=1.0,
        temp_final=0.0,
        temp_moves=15,
        compile_model=False,
        res_blocks=6,
        width=512,
        big_eval_every=0,  # disable heavy eval in smoke run
        use_elo_ladder=True,
    )
