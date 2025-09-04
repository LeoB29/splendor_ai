import math
import os
import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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
    def __init__(self, model: PolicyValueNet, device: str = "cpu", c_puct: float = 1.5, n_simulations: int = 100,
                 dir_alpha: float = 0.3, dir_eps: float = 0.25):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.n_sim = n_simulations
        self.dir_alpha = dir_alpha
        self.dir_eps = dir_eps

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

    def run(self, root_state: GameState, temperature: float = 1.0, add_dirichlet: bool = False) -> Tuple[np.ndarray, int]:
        root = AZNode(root_state)
        # Expand root
        self._expand(root)
        # Optional Dirichlet noise for exploration at root (self-play)
        if add_dirichlet and root.edges:
            actions = list(root.edges.keys())
            priors = np.array([root.edges[a].P for a in actions], dtype=np.float64)
            noise = np.random.dirichlet([self.dir_alpha] * len(actions))
            mixed = (1 - self.dir_eps) * priors + self.dir_eps * noise
            s = mixed.sum()
            if s > 0:
                mixed = mixed / s
            for a, p in zip(actions, mixed):
                root.edges[a].P = float(p)

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

        # Build visit counts and a policy target from them (normalized counts)
        counts = np.zeros(43, dtype=np.float32)
        total = 0
        for a, e in root.edges.items():
            counts[a] = e.N
            total += e.N
        pi = counts / total if total > 0 else counts

        # Choose an action index, respecting temperature and legality
        legal_mask = np.array(legal_actions_mask(root_state), dtype=np.float32)
        legal_idxs = [i for i, v in enumerate(legal_mask) if v]

        if not legal_idxs:
            return pi, -1

        temp = float(temperature)
        if temp <= 1e-3:
            # Argmax over visit counts
            a_idx = int(np.argmax(counts)) if counts.sum() > 0 else int(random.choice(legal_idxs))
        else:
            # Soft sampling from counts ** (1/temp), masked to legal actions
            try:
                probs = np.power(counts.astype(np.float64), 1.0 / temp)
            except Exception:
                probs = counts.astype(np.float64)
            probs *= legal_mask
            s = probs.sum()
            if s <= 0:
                a_idx = int(random.choice(legal_idxs))
            else:
                probs = (probs / s).astype(np.float64)
                a_idx = int(np.random.choice(len(probs), p=probs))
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


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.pis: List[np.ndarray] = []
        self.zs: List[float] = []

    def add(self, X: torch.Tensor, P: torch.Tensor, Z: torch.Tensor):
        for i in range(X.size(0)):
            if len(self.states) >= self.capacity:
                # FIFO eviction
                self.states.pop(0)
                self.pis.pop(0)
                self.zs.pop(0)
            self.states.append(X[i].cpu().numpy())
            self.pis.append(P[i].cpu().numpy())
            self.zs.append(float(Z[i].cpu().item()))

    def size(self) -> int:
        return len(self.states)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = min(batch_size, len(self.states))
        idxs = np.random.choice(len(self.states), size=n, replace=False)
        X = torch.tensor(np.stack([self.states[i] for i in idxs]), dtype=torch.float32)
        P = torch.tensor(np.stack([self.pis[i] for i in idxs]), dtype=torch.float32)
        Z = torch.tensor([self.zs[i] for i in idxs], dtype=torch.float32)
        return X, P, Z


def _find_latest_checkpoint(ckpt_dir: str) -> Optional[Tuple[str, int]]:
    try:
        files = os.listdir(ckpt_dir)
    except FileNotFoundError:
        return None
    best_path = None
    best_iter = -1
    for fname in files:
        if fname.startswith("az_iter_") and fname.endswith(".pt"):
            num_str = fname[len("az_iter_"):-3]
            try:
                it = int(num_str)
            except Exception:
                continue
            if it > best_iter:
                best_iter = it
                best_path = os.path.join(ckpt_dir, fname)
    return (best_path, best_iter) if best_path is not None else None


def evaluate_vs_random(model: PolicyValueNet, games: int = 4, mcts_simulations: int = 32, device: str = "cpu") -> float:
    wins = 0
    for g in range(games):
        state = setup_game(num_players=2)
        mcts = AlphaZeroMCTS(model, device=device, n_simulations=mcts_simulations)
        # Alternate who starts
        my_index = g % 2
        while not state.is_terminal:
            legal = state.get_legal_actions()
            if not legal:
                state.current_player = (state.current_player + 1) % len(state.players)
                continue
            if state.current_player == my_index:
                pi, a_idx = mcts.run(state, temperature=0.0)  # argmax over visits
                if a_idx == -1:
                    # fallback to random legal
                    a = random.choice(legal)
                else:
                    a = index_to_action(a_idx, state)
            else:
                a = random.choice(legal)
            state = state.apply_action(a)
        if state.winner == my_index:
            wins += 1
    return wins / games if games > 0 else 0.0


def az_train(
    iterations: int = 10,
    games_per_iter: int = 8,
    mcts_simulations: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    replay_capacity: int = 20000,
    batch_size: int = 256,
    train_batches_per_iter: int = 20,
    eval_games: int = 6,
    log_dir: str = "logs",
    ckpt_dir: str = "checkpoints",
    resume: bool = True,
    resume_path: Optional[str] = None,
):
    # Setup
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train_log.csv")
    write_header = not os.path.exists(log_path)

    # Infer input size
    dummy_state = setup_game(num_players=2)
    input_size = len(flatten_game_state(dummy_state))
    action_size = 43

    model = PolicyValueNet(input_size=input_size, action_size=action_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=replay_capacity)
    start_iter_global = 0

    # Resume from checkpoint if available/requested
    if resume or resume_path:
        loaded = False
        if resume_path is not None and os.path.exists(resume_path):
            try:
                ck = torch.load(resume_path, map_location=device)
                model.load_state_dict(ck["model"])
                optimizer.load_state_dict(ck["optimizer"])  # type: ignore[arg-type]
                start_iter_global = int(ck.get("iter", 0))
                print(f"[Resume] Loaded checkpoint {resume_path} @ iter {start_iter_global}")
                loaded = True
            except Exception as e:
                print(f"[Resume] Failed to load {resume_path}: {e}")
        if not loaded and resume and os.path.isdir(ckpt_dir):
            latest = _find_latest_checkpoint(ckpt_dir)
            if latest is not None:
                path, itnum = latest
                try:
                    ck = torch.load(path, map_location=device)
                    model.load_state_dict(ck["model"])
                    optimizer.load_state_dict(ck["optimizer"])  # type: ignore[arg-type]
                    start_iter_global = int(ck.get("iter", itnum))
                    print(f"[Resume] Loaded latest checkpoint {path} @ iter {start_iter_global}")
                except Exception as e:
                    print(f"[Resume] Failed to load {path}: {e}")

    # Continue global iteration numbering
    for step in range(1, iterations + 1):
        it = start_iter_global + step
        step_counts: List[int] = []
        for g in range(games_per_iter):
            traj, winner = self_play_episode(model, mcts_simulations=mcts_simulations, device=device, temperature=1.0)
            X, P, Z = compute_targets(traj, winner)
            buffer.add(X, P, Z)
            step_counts.append(len(traj))

        # Train from replay buffer
        losses = []
        pol_losses = []
        val_losses = []
        for _ in range(train_batches_per_iter):
            if buffer.size() == 0:
                break
            batch = buffer.sample(batch_size)
            stats = train_on_batch(model, optimizer, batch, device=device)
            losses.append(stats["loss"])
            pol_losses.append(stats["policy_loss"])
            val_losses.append(stats["value_loss"])

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        avg_pl = float(np.mean(pol_losses)) if pol_losses else float("nan")
        avg_vl = float(np.mean(val_losses)) if val_losses else float("nan")
        avg_steps = float(np.mean(step_counts)) if step_counts else 0.0

        # Evaluate vs random
        win_rate = evaluate_vs_random(model, games=eval_games, mcts_simulations=max(16, mcts_simulations // 2), device=device)

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"az_iter_{it}.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": it,
            "buffer_size": buffer.size(),
        }, ckpt_path)

        # Log CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["iter", "buffer", "avg_steps", "loss", "policy_loss", "value_loss", "win_rate"])
                write_header = False
            writer.writerow([it, buffer.size(), f"{avg_steps:.2f}", f"{avg_loss:.4f}", f"{avg_pl:.4f}", f"{avg_vl:.4f}", f"{win_rate:.3f}"])

        print(
            f"Iter {it:02d} | buffer={buffer.size()} steps={avg_steps:.1f} "
            f"loss={avg_loss:.4f} pol={avg_pl:.4f} val={avg_vl:.4f} win%={win_rate:.1%}"
        )

    return model


if __name__ == "__main__":
    # Example longer run (tweak as desired)
    az_train(
        iterations=5,
        games_per_iter=6,
        mcts_simulations=48,
        lr=1e-3,
        device="cpu",
        replay_capacity=20000,
        batch_size=256,
        train_batches_per_iter=20,
        eval_games=6,
        resume=True,
    )
