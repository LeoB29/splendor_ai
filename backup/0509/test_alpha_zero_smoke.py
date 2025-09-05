import random
import numpy as np
import torch
import torch.optim as optim

from cards_init import setup_game
from nn_input_output import flatten_game_state, legal_actions_mask
from alpha_zero import PolicyValueNet, AlphaZeroMCTS, self_play_episode, compute_targets, train_on_batch


def main():
    # Determinism (best-effort)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # 1) Instantiate model and MCTS
    state0 = setup_game(num_players=2)
    in_size = len(flatten_game_state(state0))
    model = PolicyValueNet(input_size=in_size, action_size=43)
    mcts = AlphaZeroMCTS(model, n_simulations=8)

    # 2) Single MCTS run on initial state
    pi, a_idx = mcts.run(state0, temperature=1.0)
    assert pi.shape == (43,), f"pi shape mismatch: {pi.shape}"
    assert (pi >= 0).all() and pi.sum() > 0, "pi must be a valid distribution"
    legal = legal_actions_mask(state0)
    assert legal[a_idx] == 1, "Selected action should be legal"
    print("MCTS run OK. Sum(pi)=", float(pi.sum()))

    # 3) Self-play one episode
    traj, winner = self_play_episode(model, mcts_simulations=8, temperature=1.0)
    assert len(traj) > 0, "Trajectory should have at least one step"
    print(f"Self-play OK. Steps={len(traj)}, winner={winner}")

    # 4) Build training targets and train one batch
    X, P, Z = compute_targets(traj, winner)
    assert X.ndim == 2 and P.ndim == 2 and Z.ndim == 1, "Targets shapes are incorrect"
    assert X.size(0) == P.size(0) == Z.size(0), "Batch dims mismatch"

    opt = optim.Adam(model.parameters(), lr=1e-3)
    stats = train_on_batch(model, opt, (X, P, Z))
    print("Train step OK.", stats)


if __name__ == "__main__":
    main()

