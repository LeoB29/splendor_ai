import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_state import GameState
from nn_input_output import flatten_game_state, legal_actions_mask, index_to_action
from cards_init import setup_game


# -----------------------------
# Policy + Value Network (MLP)
# -----------------------------
class PolicyValueNet(nn.Module):
    def __init__(self, input_size: int, action_size: int = 43, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_size)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # value in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.policy_head(h)  # (B, A)
        value = self.value_head(h).squeeze(-1)  # (B,)
        return logits, value


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mask: 1 for legal, 0 for illegal
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask > 0, logits, torch.full_like(logits, neg_inf))
    probs = torch.softmax(masked_logits, dim=-1)
    # if all masked (rare), fallback to uniform over mask or logits
    zero_rows = probs.sum(dim=-1) == 0
    if zero_rows.any():
        # fallback to uniform over legal actions
        legal_counts = mask[zero_rows].sum(dim=-1, keepdim=True).clamp(min=1)
        probs[zero_rows] = mask[zero_rows] / legal_counts
    return probs


# -----------------------------
# AlphaZero-style MCTS
# -----------------------------
@dataclass
class EdgeStats:
    P: float  # prior
    N: int = 0
    W: float = 0.0

    @property
    def Q(self) -> float:
        return 0.0 if self.N == 0 else self.W / self.N


class AZNode:
    def __init__(self, state: GameState, priors: np.ndarray | None = None):
        self.state = state
        self.children: Dict[int, "AZNode"] = {}
        self.edges: Dict[int, EdgeStats] = {}  # keyed by action index
        self.priors = priors  # numpy array over actions (optional)

    def is_expanded(self) -> bool:
        return len(self.edges) > 0 or self.state.is_terminal

    def total_N(self) -> int:
        return sum(e.N for e in self.edges.values())


class AlphaZeroMCTS:
    def __init__(self, model: PolicyValueNet, device: str = "cpu", c_puct: float = 1.5, n_simulations: int = 100):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.n_sim = n_simulations

    @torch.no_grad()
    def _evaluate(self, state: GameState) -> Tuple[np.ndarray, float]:
        x = torch.tensor(flatten_game_state(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(x)
        mask = torch.tensor(legal_actions_mask(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        priors = masked_softmax(logits, mask)[0].detach().cpu().numpy()  # (A,)
        v = float(value.item())  # scalar in [-1, 1]
        return priors, v

    def _select(self, node: AZNode) -> int:
        # Pick action index maximizing PUCT
        sqrt_sum = math.sqrt(node.total_N() + 1)
        best_score = -1e30
        best_a = None
        for a, e in node.edges.items():
            u = self.c_puct * e.P * (sqrt_sum / (1 + e.N))
            score = e.Q + u
            if score > best_score:
                best_score = score
                best_a = a
        assert best_a is not None
        return best_a

    def _expand(self, node: AZNode) -> float:
        # If terminal, from current player's perspective value is -1 (lost) in 2-player case
        if node.state.is_terminal:
            return -1.0

        priors, v = self._evaluate(node.state)
        node.priors = priors

        # Initialize edges only for legal actions (non-zero priors)
        legal_mask = legal_actions_mask(node.state)
        for a_idx, legal in enumerate(legal_mask):
            if legal:
                p = float(priors[a_idx])
                node.edges[a_idx] = EdgeStats(P=p)
        # Note: children are created lazily when traversed the first time
        return v

    def _step(self, state: GameState, a_idx: int) -> GameState:
        action = index_to_action(a_idx, state)
        return state.apply_action(action)

    def run(self, root_state: GameState, temperature: float = 1.0) -> Tuple[np.ndarray, int]:
        root = AZNode(root_state)
        # Expand root
        self._expand(root)

        # Simulations
        for _ in range(self.n_sim):
            node = root
            path: List[Tuple[AZNode, int]] = []

            # Selection down to a leaf
            while node.is_expanded() and not node.state.is_terminal and node.edges:
                a = self._select(node)
                path.append((node, a))
                # Lazily create child
                if a not in node.children:
                    next_state = self._step(node.state, a)
                    node.children[a] = AZNode(next_state)
                node = node.children[a]

            # Expansion/Evaluation
            v = self._expand(node)

            # Backup with sign flip at each step
            for parent, a in reversed(path):
                e = parent.edges[a]
                e.N += 1
                e.W += v
                v = -v

        # Build policy target from visit counts
        pi = np.zeros(43, dtype=np.float32)
        total = 0
        for a, e in root.edges.items():
            pi[a] = e.N
            total += e.N
        if total > 0:
            if temperature != 1.0:
                # Apply temperature to visits
                visits = np.power(pi, 1.0 / max(1e-3, temperature))
                s = visits.sum()
                pi = visits / s if s > 0 else pi
            else:
                pi = pi / total

        # Choose an action index, preferring legal actions
        legal_mask = legal_actions_mask(root_state)
        legal_idxs = [i for i, v in enumerate(legal_mask) if v]
        if pi.sum() > 0:
            a_idx = int(np.argmax(pi)) if temperature == 0 else int(np.random.choice(len(pi), p=pi))
        else:
            a_idx = random.choice(legal_idxs) if legal_idxs else -1
        return pi, a_idx


# -----------------------------
# Self-play and Training
# -----------------------------
@dataclass
class Sample:
    state: np.ndarray  # flattened state (float32)
    pi: np.ndarray     # policy target over actions (float32, sum=1)
    player: int        # player to move at this state


def self_play_episode(model: PolicyValueNet, mcts_simulations: int = 100, device: str = "cpu", temperature: float = 1.0) -> Tuple[List[Sample], int]:
    state = setup_game(num_players=2)
    mcts = AlphaZeroMCTS(model, device=device, n_simulations=mcts_simulations)

    trajectory: List[Sample] = []
    consecutive_passes = 0

    while not state.is_terminal:
        # If no legal actions, pass the turn (as in game_sim)
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            state.current_player = (state.current_player + 1) % len(state.players)
            consecutive_passes += 1
            # End if both players consecutively have no moves
            if consecutive_passes >= len(state.players):
                break
            continue
        consecutive_passes = 0

        pi, _ = mcts.run(state, temperature=temperature)
        s = flatten_game_state(state)
        trajectory.append(Sample(state=s, pi=pi.astype(np.float32), player=state.current_player))

        # Sample an action: restrict to legal actions and renormalize
        legal_mask = legal_actions_mask(state)
        legal_idxs = [i for i, v in enumerate(legal_mask) if v]
        probs = pi.copy()
        probs *= np.array(legal_mask, dtype=np.float32)
        if probs.sum() <= 0 or not legal_idxs:
            a_idx = random.choice(legal_idxs)
        else:
            probs = probs / probs.sum()
            a_idx = int(np.random.choice(len(probs), p=probs))

        action = index_to_action(a_idx, state)
        state = state.apply_action(action)

    winner = state.winner if state.winner is not None else -1
    return trajectory, winner


def compute_targets(trajectory: List[Sample], winner: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.tensor(np.stack([t.state for t in trajectory]), dtype=torch.float32)
    P = torch.tensor(np.stack([t.pi for t in trajectory]), dtype=torch.float32)
    if winner < 0:
        Z = torch.zeros(len(trajectory), dtype=torch.float32)
    else:
        z = [1.0 if t.player == winner else -1.0 for t in trajectory]
        Z = torch.tensor(z, dtype=torch.float32)
    return X, P, Z


def train_on_batch(model: PolicyValueNet, optimizer: optim.Optimizer, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: str = "cpu") -> Dict[str, float]:
    X, P, Z = batch
    X, P, Z = X.to(device), P.to(device), Z.to(device)
    logits, values = model(X)

    # Policy loss: cross-entropy between target pi and predicted log-probs (masked by pi)
    log_probs = torch.log_softmax(logits, dim=-1)
    policy_loss = -(P * log_probs).sum(dim=-1).mean()

    # Value loss: MSE
    value_loss = nn.functional.mse_loss(values, Z)

    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
    }


def az_train(iterations: int = 5, games_per_iter: int = 4, mcts_simulations: int = 64, lr: float = 1e-3, device: str = "cpu"):
    # Infer input size from a fresh game
    dummy_state = setup_game(num_players=2)
    input_size = len(flatten_game_state(dummy_state))
    action_size = 43

    model = PolicyValueNet(input_size=input_size, action_size=action_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for it in range(1, iterations + 1):
        all_X, all_P, all_Z = [], [], []
        for g in range(games_per_iter):
            traj, winner = self_play_episode(model, mcts_simulations=mcts_simulations, device=device, temperature=1.0)
            X, P, Z = compute_targets(traj, winner)
            all_X.append(X)
            all_P.append(P)
            all_Z.append(Z)

        # Concatenate and train for a few mini-batches
        X = torch.cat(all_X, dim=0)
        P = torch.cat(all_P, dim=0)
        Z = torch.cat(all_Z, dim=0)

        # Shuffle
        idx = torch.randperm(X.size(0))
        X, P, Z = X[idx], P[idx], Z[idx]

        # Train
        stats = train_on_batch(model, optimizer, (X, P, Z), device=device)
        print(f"Iter {it}: loss={stats['loss']:.4f} policy={stats['policy_loss']:.4f} value={stats['value_loss']:.4f} samples={X.size(0)}")

    return model


if __name__ == "__main__":
    # Quick smoke-run on CPU with tiny settings
    az_train(iterations=1, games_per_iter=1, mcts_simulations=16, lr=1e-3, device="cpu")
