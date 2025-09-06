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
from typing import Any

# Optional DirectML (AMD on Windows)
try:
    import torch_directml as _dml  # type: ignore
    DML_DEVICE = _dml.device()
except Exception:
    DML_DEVICE = None

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
        # Children and edges are keyed by (action_index, variant_index)
        # variant_index == 0 for actions without return variants
        self.children: Dict[Tuple[int, int], "AZNode"] = {}
        self.edges: Dict[Tuple[int, int], EdgeStats] = {}
        # Metadata for edges: stores tokens_returned for variant branches
        self.edge_meta: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.priors = priors  # numpy array over actions (optional)

    def is_expanded(self) -> bool:
        return len(self.edges) > 0 or self.state.is_terminal

    def total_N(self) -> int:
        return sum(e.N for e in self.edges.values())


class AlphaZeroMCTS:
    def __init__(self, model: PolicyValueNet, device: str = "cpu", c_puct: float = 1.5, n_simulations: int = 100,
                 dir_alpha: float = 0.3, dir_eps: float = 0.25, returns_top_k: int = 3, mcts_batch: int = 16):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.n_sim = n_simulations
        self.dir_alpha = dir_alpha
        self.dir_eps = dir_eps
        self.returns_top_k = max(1, int(returns_top_k))
        self.mcts_batch = max(1, int(mcts_batch))
        # Stores best return variant per base action from the last run
        self._last_best_returns: Dict[int, Dict[str, int]] = {}
        # Root for tree reuse across moves
        self._root: Optional[AZNode] = None

    def get_best_tokens_returned(self, a_idx: int) -> Optional[Dict[str, int]]:
        return self._last_best_returns.get(a_idx)

    @torch.no_grad()
    def _evaluate_batch(self, states: List[GameState]) -> Tuple[List[np.ndarray], List[float]]:
        if not states:
            return [], []
        xs = [flatten_game_state(s) for s in states]
        x = torch.tensor(np.stack(xs), dtype=torch.float32, device=self.device)
        # Build masks batch
        masks = torch.tensor(np.stack([np.array(legal_actions_mask(s), dtype=np.float32) for s in states]),
                             dtype=torch.float32, device=self.device)
        # AMP only on CUDA
        use_cuda = (isinstance(self.device, torch.device) and self.device.type == "cuda") or (isinstance(self.device, str) and str(self.device).startswith("cuda"))
        if use_cuda:
            try:
                with torch.amp.autocast('cuda'):  # type: ignore[attr-defined]
                    logits, values = self.model(x)
            except Exception:
                with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                    logits, values = self.model(x)
        else:
            logits, values = self.model(x)
        priors_t = masked_softmax(logits, masks)  # (B, A)
        priors = [p.detach().cpu().numpy() for p in priors_t]
        vals = [float(v.item()) for v in values]
        return priors, vals

    def _select(self, node: AZNode) -> Tuple[int, int]:
        # Pick action index maximizing PUCT
        sqrt_sum = math.sqrt(node.total_N() + 1)
        best_score = -1e30
        best_a = None
        for a, e in node.edges.items():  # a is (a_idx, r_idx)
            u = self.c_puct * e.P * (sqrt_sum / (1 + e.N))
            score = e.Q + u
            if score > best_score:
                best_score = score
                best_a = a
        assert best_a is not None
        return best_a

    def _enumerate_return_variants(self, state: GameState, a_idx: int) -> Tuple[List[Dict[str, int]], List[float]]:
        # Build base action to get tokens taken and reserve target if any
        from nn_input_output import TAKE_3_DIFF_COMBOS, IDX_TO_GEM
        player = state.players[state.current_player]
        tokens_taken: Dict[str, int] = {}
        action_type = None
        target = None
        tier = None

        if 0 <= a_idx <= 9:
            gems_idx = TAKE_3_DIFF_COMBOS[a_idx]
            tokens_taken = {IDX_TO_GEM[g]: 1 for g in gems_idx}
            action_type = "take_tokens"
        elif 10 <= a_idx <= 14:
            gem_idx = a_idx - 10
            gem = IDX_TO_GEM[gem_idx]
            tokens_taken = {gem: 2}
            action_type = "take_tokens"
        elif 30 <= a_idx <= 41:
            # Reserve visible card; index_to_action will resolve target/tier, here only tokens_taken matters
            if state.tokens.get("gold", 0) > 0:
                tokens_taken = {"gold": 1}
            else:
                tokens_taken = {}
            action_type = "reserve"
        else:
            # Other actions don't need explicit return variants
            return [], []

        after = player.tokens.copy()
        for c, k in tokens_taken.items():
            after[c] = after.get(c, 0) + k
        total = sum(after.values())
        excess = max(0, total - 10)
        if excess <= 0:
            return [], []

        # Build token list for combinations
        token_list: List[str] = []
        for c, cnt in after.items():
            token_list.extend([c] * max(0, int(cnt)))

        from itertools import combinations
        unique_sets = set()
        for combo in combinations(token_list, excess):
            # Count occurrences
            counts: Dict[str, int] = {}
            for c in combo:
                counts[c] = counts.get(c, 0) + 1
            # Validate within availability
            if all(after.get(c, 0) >= n for c, n in counts.items()):
                unique_sets.add(frozenset(counts.items()))

        candidates: List[Dict[str, int]] = [dict(fs) for fs in unique_sets]

        # Heuristic scoring: avoid returning needed colors and gold
        need: Dict[str, float] = {c: 0.0 for c in ["diamond", "sapphire", "obsidian", "ruby", "emerald"]}
        # Demand from visible cards and reserved
        visible_cards = []
        for t in sorted(state.board.keys()):
            # Skip None placeholders defensively
            visible_cards.extend([c for c in state.board[t] if c is not None])
        reserved_cards = [c for c in player.reserved if c is not None]
        all_cards = visible_cards + reserved_cards
        for card in all_cards:
            try:
                for c, cost in card.cost.items():
                    short = max(0, cost - player.bonuses.get(c, 0) - player.tokens.get(c, 0))
                    need[c] += short
            except Exception:
                # Ignore malformed card objects
                continue
        gold_penalty = (max(need.values()) if need else 0) + 3.0

        def score(ret: Dict[str, int]) -> float:
            s = 0.0
            for c, n in ret.items():
                if c == "gold":
                    s += gold_penalty * n
                else:
                    s += need.get(c, 0.0) * n
            # small preference to return from largest piles
            for c, n in ret.items():
                s -= 0.01 * min(player.tokens.get(c, 0), n)
            return s

        candidates.sort(key=score)
        top = candidates[: self.returns_top_k]
        if not top:
            return [], []
        # Uniform weights across variants
        weights = [1.0 / len(top)] * len(top)
        return top, weights

    def _expand_with_priors(self, node: AZNode, priors: np.ndarray) -> None:
        node.priors = priors
        legal_mask = legal_actions_mask(node.state)
        for a_idx, legal in enumerate(legal_mask):
            if not legal:
                continue
            base_p = float(priors[a_idx])
            variants, weights = self._enumerate_return_variants(node.state, a_idx)
            if variants:
                for r_idx, (ret, w) in enumerate(zip(variants, weights), start=0):
                    key = (a_idx, r_idx)
                    node.edges[key] = EdgeStats(P=base_p * float(w))
                    node.edge_meta[key] = {"tokens_returned": ret}
            else:
                key = (a_idx, 0)
                node.edges[key] = EdgeStats(P=base_p)
                node.edge_meta[key] = {"tokens_returned": {}}

    def _step(self, node: AZNode, key: Tuple[int, int]) -> GameState:
        a_idx, r_idx = key
        action = index_to_action(a_idx, node.state)
        # If this edge has a specific return variant, apply it
        meta = node.edge_meta.get(key)
        if meta and hasattr(action, "tokens_returned"):
            action.tokens_returned = meta.get("tokens_returned", {})
        return node.state.apply_action(action)

    def run(self, root_state: GameState, temperature: float = 1.0, add_dirichlet: bool = False) -> Tuple[np.ndarray, int]:
        # Use reused root if it matches the external state; otherwise create a new root
        root = self._root if self._root is not None else AZNode(root_state)
        # If not expanded yet, expand via batch path
        if not root.is_expanded():
            priors, vals = self._evaluate_batch([root.state])
            v0 = vals[0] if vals else 0.0
            self._expand_with_priors(root, priors[0])
        # Optional Dirichlet noise for exploration at root (self-play)
        if add_dirichlet and root.edges:
            actions = list(root.edges.keys())
            # Aggregate priors per base action to sample Dirichlet in base space
            base_actions = sorted(set(a for a, _ in actions))
            base_priors_list = []
            for a in base_actions:
                vars_for_a = [k for k in actions if k[0] == a]
                base_priors_list.append(sum(root.edges[k].P for k in vars_for_a))
            base_priors = np.array(base_priors_list, dtype=np.float64)
            noise = np.random.dirichlet([self.dir_alpha] * len(base_actions))
            mixed_base = (1 - self.dir_eps) * base_priors + self.dir_eps * noise
            s = mixed_base.sum()
            if s > 0:
                mixed_base = mixed_base / s
            # Redistribute mixed base priors uniformly across variants of each base action
            for a, mb in zip(base_actions, mixed_base):
                variants = [k for k in actions if k[0] == a]
                if not variants:
                    continue
                per = float(mb) / len(variants)
                for key in variants:
                    root.edges[key].P = per

        # Simulations with batched leaf evaluation
        sims_done = 0
        while sims_done < self.n_sim:
            pending_nodes: List[AZNode] = []
            pending_paths: List[List[Tuple[AZNode, Tuple[int, int]]]] = []
            # Collect up to batch leaves
            for _ in range(min(self.mcts_batch, self.n_sim - sims_done)):
                node = root
                path: List[Tuple[AZNode, Tuple[int, int]]] = []
                # Selection
                while node.is_expanded() and not node.state.is_terminal and node.edges:
                    key = self._select(node)
                    path.append((node, key))
                    if key not in node.children:
                        next_state = self._step(node, key)
                        node.children[key] = AZNode(next_state)
                    node = node.children[key]

                # Terminal: immediate backup
                if node.state.is_terminal:
                    v = -1.0
                    for parent, key in reversed(path):
                        e = parent.edges[key]
                        e.N += 1
                        e.W += v
                        v = -v
                    sims_done += 1
                    continue

                # Leaf requires evaluation
                if not node.is_expanded():
                    pending_nodes.append(node)
                    pending_paths.append(path)
                else:
                    # Rare: expanded but no edges; skip
                    sims_done += 1

            # Batch evaluate collected leaves
            if pending_nodes:
                priors_list, vals_list = self._evaluate_batch([n.state for n in pending_nodes])
                for node, path, priors, v in zip(pending_nodes, pending_paths, priors_list, vals_list):
                    self._expand_with_priors(node, priors)
                    # Backup
                    val = float(v)
                    for parent, key in reversed(path):
                        e = parent.edges[key]
                        e.N += 1
                        e.W += val
                        val = -val
                sims_done += len(pending_nodes)

        # Build visit counts and a policy target from them (normalized counts)
        counts = np.zeros(43, dtype=np.float32)
        total = 0
        # Aggregate visit counts per base action
        for (a_idx, r_idx), e in root.edges.items():
            counts[a_idx] += e.N
            total += e.N
        pi = counts / total if total > 0 else counts

        # Choose an action index, respecting temperature and legality
        legal_mask = np.array(legal_actions_mask(root_state), dtype=np.float32)
        legal_idxs = [i for i, v in enumerate(legal_mask) if v]

        if not legal_idxs:
            return pi, -1

        temp = float(temperature)
        if temp <= 1e-3:
            # Greedy: argmax over legal actions only
            masked_counts = counts * legal_mask
            if masked_counts.sum() > 0:
                a_idx = int(np.argmax(masked_counts))
            else:
                a_idx = int(random.choice(legal_idxs))
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

        # Record the most-visited return variant per base action at root for later execution
        best_returns: Dict[int, Dict[str, int]] = {}
        variant_counts: Dict[int, Tuple[int, Tuple[int, int]]] = {}
        for key, e in root.edges.items():
            base, var = key
            cur = variant_counts.get(base)
            if cur is None or e.N > cur[0]:
                variant_counts[base] = (e.N, key)
        for base, (_n, key) in variant_counts.items():
            meta = root.edge_meta.get(key, {})
            best_returns[base] = meta.get("tokens_returned", {})
        self._last_best_returns = best_returns
        # Store root for reuse next move
        self._root = root

        return pi, a_idx

    def reuse_after_play(self, a_idx: int) -> None:
        # Move root to the child corresponding to the most visited variant of the chosen base action
        root = self._root
        if root is None or not root.edges:
            self._root = None
            return
        # Pick the most visited variant for base action
        best_key = None
        best_visits = -1
        for key, e in root.edges.items():
            base, var = key
            if base != a_idx:
                continue
            if e.N > best_visits:
                best_visits = e.N
                best_key = key
        if best_key is None:
            self._root = None
            return
        child = root.children.get(best_key)
        self._root = child if child is not None else None


# -----------------------------
# Self-play and Training
# -----------------------------
@dataclass
class Sample:
    state: np.ndarray  # flattened state (float32)
    pi: np.ndarray     # policy target over actions (float32, sum=1)
    player: int        # player to move at this state


def self_play_episode(model: PolicyValueNet, mcts_simulations: int = 100, device: str = "cpu", temperature: float = 1.0,
                      temp_init: float = 1.0, temp_final: float = 0.0, temp_moves: int = 20,
                      add_dirichlet: bool = True, mcts_batch: int = 16) -> Tuple[List[Sample], int]:
    state = setup_game(num_players=2)
    mcts = AlphaZeroMCTS(model, device=device, n_simulations=mcts_simulations, mcts_batch=mcts_batch)

    trajectory: List[Sample] = []
    consecutive_passes = 0
    move_idx = 0

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

        # Temperature schedule across the game
        temp = temp_init if move_idx < temp_moves else temp_final
        pi, _ = mcts.run(state, temperature=temp, add_dirichlet=add_dirichlet)
        s = flatten_game_state(state)
        trajectory.append(Sample(state=s, pi=pi.astype(np.float32), player=state.current_player))

        # Sample an action: restrict to legal actions and renormalize
        legal_mask = legal_actions_mask(state)
        legal_idxs = [i for i, v in enumerate(legal_mask) if v]
        probs = pi.copy()
        probs *= np.array(legal_mask, dtype=np.float32)
        if probs.sum() <= 0 or not legal_idxs:
            # Fallback: choose a random legal Action directly (covers actions not in NN space, e.g., reserve-from-deck)
            legals = state.get_legal_actions()
            if legals:
                a = random.choice(legals)
                a_idx_for_reuse = None
                try:
                    from nn_input_output import action_to_index
                    a_idx_for_reuse = action_to_index(a, state)
                except Exception:
                    a_idx_for_reuse = None
                state = state.apply_action(a)
                try:
                    if a_idx_for_reuse is not None:
                        mcts.reuse_after_play(a_idx_for_reuse)
                except Exception:
                    pass
                move_idx += 1
                continue
            else:
                # Should not hit because we handled no-legal case earlier; skip turn defensively
                state.current_player = (state.current_player + 1) % len(state.players)
                move_idx += 1
                continue
        else:
            probs = probs / probs.sum()
            a_idx = int(np.random.choice(len(probs), p=probs))

            action = index_to_action(a_idx, state)
            # Use most-visited return variant for the chosen base action
            best_ret = mcts.get_best_tokens_returned(a_idx)
            if best_ret is not None and hasattr(action, "tokens_returned"):
                action.tokens_returned = best_ret
            state = state.apply_action(action)
            # Reuse MCTS tree for next move
            try:
                mcts.reuse_after_play(a_idx)
            except Exception:
                pass
        move_idx += 1

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


def train_on_batch(model: PolicyValueNet, optimizer: optim.Optimizer, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: Any = "cpu",
                   policy_weight: float = 1.0, value_weight: float = 1.0, grad_clip: Optional[float] = None,
                   scaler: Optional[Any] = None) -> Dict[str, float]:
    X, P, Z = batch
    X, P, Z = X.to(device), P.to(device), Z.to(device)
    use_cuda = (isinstance(device, torch.device) and device.type == "cuda") or (isinstance(device, str) and str(device).startswith("cuda"))
    use_amp = scaler is not None and use_cuda

    optimizer.zero_grad(set_to_none=True)

    if use_amp:
        # Prefer new torch.amp API, fallback to torch.cuda.amp
        try:
            autocast_ctx = torch.amp.autocast  # type: ignore[attr-defined]
            with autocast_ctx('cuda'):
                logits, values = model(X)
                log_probs = torch.log_softmax(logits, dim=-1)
                policy_loss = -(P * log_probs).sum(dim=-1).mean()
                value_loss = nn.functional.mse_loss(values, Z)
                loss = policy_weight * policy_loss + value_weight * value_loss
        except Exception:
            # Backward compatibility
            with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                logits, values = model(X)
                log_probs = torch.log_softmax(logits, dim=-1)
                policy_loss = -(P * log_probs).sum(dim=-1).mean()
                value_loss = nn.functional.mse_loss(values, Z)
                loss = policy_weight * policy_loss + value_weight * value_loss
        scaler.scale(loss).backward()
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits, values = model(X)
        # Policy loss: cross-entropy between target pi and predicted log-probs (masked by pi)
        log_probs = torch.log_softmax(logits, dim=-1)
        policy_loss = -(P * log_probs).sum(dim=-1).mean()
        # Value loss: MSE
        value_loss = nn.functional.mse_loss(values, Z)
        loss = policy_weight * policy_loss + value_weight * value_loss
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return {
        "loss": float((policy_weight * policy_loss + value_weight * value_loss).item()),
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


def evaluate_vs_random(model: PolicyValueNet, games: int = 4, mcts_simulations: int = 32, device: str = "cpu", mcts_batch: int = 16, max_moves: int = 250) -> float:
    wins = 0
    for g in range(games):
        state = setup_game(num_players=2)
        mcts = AlphaZeroMCTS(model, device=device, n_simulations=mcts_simulations, mcts_batch=mcts_batch)
        # Alternate who starts
        my_index = g % 2
        move_count = 0
        consecutive_passes = 0
        while not state.is_terminal:
            legal = state.get_legal_actions()
            if not legal:
                state.current_player = (state.current_player + 1) % len(state.players)
                consecutive_passes += 1
                # Break stalemates where neither player has legal moves
                if consecutive_passes >= len(state.players) or move_count >= max_moves:
                    break
                continue
            played_a_idx: Optional[int] = None
            if state.current_player == my_index:
                pi, a_idx = mcts.run(state, temperature=0.0)  # argmax over visits
                if a_idx == -1:
                    # fallback to random legal
                    a = random.choice(legal)
                    played_a_idx = None
                else:
                    a = index_to_action(a_idx, state)
                    # Apply most-visited return variant for this base action if any
                    best_ret = mcts.get_best_tokens_returned(a_idx)
                    if best_ret is not None and hasattr(a, "tokens_returned"):
                        a.tokens_returned = best_ret
                    played_a_idx = a_idx
            else:
                a = random.choice(legal)
                played_a_idx = None
            state = state.apply_action(a)
            move_count += 1
            consecutive_passes = 0
            if played_a_idx is not None:
                try:
                    mcts.reuse_after_play(played_a_idx)
                except Exception:
                    pass
        if state.winner == my_index:
            wins += 1
        # Lightweight eval progress
        interval = max(1, games // 10)
        if ((g + 1) % interval == 0) or (g + 1 == games):
            print(f"[Eval] {g+1}/{games} games done")
    return wins / games if games > 0 else 0.0


def az_train(
    iterations: int = 10,
    games_per_iter: int = 8,
    mcts_simulations: int = 64,
    mcts_batch: int = 16,
    lr: float = 1e-3,
    device: Any = "cpu",
    replay_capacity: int = 20000,
    batch_size: int = 256,
    train_batches_per_iter: int = 20,
    eval_games: int = 6,
    log_dir: str = "logs",
    ckpt_dir: str = "checkpoints",
    resume: bool = True,
    resume_path: Optional[str] = None,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    temp_init: float = 1.0,
    temp_final: float = 0.0,
    temp_moves: int = 20,
    compile_model: bool = False,
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
    # Optional: compile model (PyTorch 2.x) for speed
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("[Info] torch.compile enabled")
        except Exception as e:
            print(f"[Info] torch.compile unavailable: {e}")
    # Optimizer: avoid DirectML CPU-fallback for foreach ops by disabling foreach
    def _configure_optimizer_for_device(opt: optim.Optimizer, dev: Any) -> None:
        try:
            is_cuda = (isinstance(dev, torch.device) and dev.type == "cuda") or (isinstance(dev, str) and str(dev).startswith("cuda"))
            for pg in opt.param_groups:
                # On DirectML/CPU, disable foreach/fused/capturable to avoid CPU fallbacks
                if not is_cuda:
                    try:
                        pg["foreach"] = False  # type: ignore[index]
                    except Exception:
                        pass
                    try:
                        pg["fused"] = False  # type: ignore[index]
                    except Exception:
                        pass
                    try:
                        pg["capturable"] = False  # type: ignore[index]
                    except Exception:
                        pass
        except Exception:
            pass

    try:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, foreach=False)  # type: ignore[call-arg]
    except TypeError:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    _configure_optimizer_for_device(optimizer, device)
    # Enable TF32 on CUDA (Ampere+)
    try:
        if (isinstance(device, torch.device) and device.type == "cuda") or (isinstance(device, str) and str(device).startswith("cuda")):
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    # GradScaler for AMP on CUDA (new API with fallback)
    scaler = None
    try:
        is_cuda = (isinstance(device, torch.device) and device.type == "cuda") or (isinstance(device, str) and str(device).startswith("cuda"))
        if is_cuda:
            try:
                scaler = torch.amp.GradScaler('cuda')  # type: ignore[attr-defined]
            except Exception:
                scaler = torch.cuda.amp.GradScaler(enabled=True)  # type: ignore[attr-defined]
        else:
            scaler = None
    except Exception:
        scaler = None
    buffer = ReplayBuffer(capacity=replay_capacity)
    start_iter_global = 0

    # Resume from checkpoint if available/requested
    def _load_checkpoint_cpu_safe(path: str):
        """Load a checkpoint safely on CPU with best-effort fallbacks.
        Tries weights_only=True first (PyTorch 2.4+ safe loader). If that fails due to
        missing allowlisted globals, attempts to add them. Falls back to a normal
        torch.load(map_location='cpu') as last resort (only use with trusted files).
        """
        import torch as _torch
        # First attempt: safe weights-only if available
        try:
            try:
                return _torch.load(path, map_location='cpu', weights_only=True)  # type: ignore[call-arg]
            except TypeError:
                # Older PyTorch without weights_only
                return _torch.load(path, map_location='cpu')
        except Exception:
            # Try to allowlist needed globals for safe loader, then retry
            try:
                try:
                    from torch.serialization import add_safe_globals  # type: ignore
                    try:
                        from torch._utils import _rebuild_device_tensor_from_numpy  # type: ignore
                        add_safe_globals([_rebuild_device_tensor_from_numpy])  # type: ignore
                    except Exception:
                        pass
                except Exception:
                    pass
                return _torch.load(path, map_location='cpu', weights_only=True)  # type: ignore[call-arg]
            except Exception:
                # Final fallback: unsafe loader on CPU (only acceptable for own checkpoints)
                return _torch.load(path, map_location='cpu')
    if resume or resume_path:
        loaded = False
        if resume_path is not None and os.path.exists(resume_path):
            try:
                # Load safely on CPU, then move to target device
                ck = _load_checkpoint_cpu_safe(resume_path)
                model.load_state_dict(ck["model"])
                optimizer.load_state_dict(ck["optimizer"])  # type: ignore[arg-type]
                _configure_optimizer_for_device(optimizer, device)
                # Move optimizer state tensors to target device
                try:
                    for state in optimizer.state.values():  # type: ignore[attr-defined]
                        for k, v in list(state.items()):
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                except Exception:
                    pass
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
                    ck = _load_checkpoint_cpu_safe(path)
                    model.load_state_dict(ck["model"])
                    optimizer.load_state_dict(ck["optimizer"])  # type: ignore[arg-type]
                    _configure_optimizer_for_device(optimizer, device)
                    # Move optimizer state tensors to target device
                    try:
                        for state in optimizer.state.values():  # type: ignore[attr-defined]
                            for k, v in list(state.items()):
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(device)
                    except Exception:
                        pass
                    start_iter_global = int(ck.get("iter", itnum))
                    print(f"[Resume] Loaded latest checkpoint {path} @ iter {start_iter_global}")
                except Exception as e:
                    print(f"[Resume] Failed to load {path}: {e}")

    # Continue global iteration numbering
    for step in range(1, iterations + 1):
        it = start_iter_global + step
        step_counts: List[int] = []
        print(f"[Iter {it}] Self-play: {games_per_iter} games")
        for g in range(games_per_iter):
            traj, winner = self_play_episode(
                model,
                mcts_simulations=mcts_simulations,
                device=device,
                temperature=1.0,
                temp_init=temp_init,
                temp_final=temp_final,
                temp_moves=temp_moves,
                add_dirichlet=True,
                mcts_batch=mcts_batch,
            )
            X, P, Z = compute_targets(traj, winner)
            buffer.add(X, P, Z)
            step_counts.append(len(traj))
            print(f"[Iter {it}] Self-play {g+1}/{games_per_iter} steps={len(traj)}")

        # Train from replay buffer
        losses = []
        pol_losses = []
        val_losses = []
        print(f"[Iter {it}] Train: {train_batches_per_iter} batches (buffer={buffer.size()})")
        for bi in range(train_batches_per_iter):
            if buffer.size() == 0:
                break
            batch = buffer.sample(batch_size)
            stats = train_on_batch(
                model,
                optimizer,
                batch,
                device=device,
                policy_weight=policy_weight,
                value_weight=value_weight,
                grad_clip=grad_clip,
                scaler=scaler,
            )
            losses.append(stats["loss"])
            pol_losses.append(stats["policy_loss"])
            val_losses.append(stats["value_loss"])
            interval = max(1, train_batches_per_iter // 5)
            if ((bi + 1) % interval == 0) or (bi + 1 == train_batches_per_iter):
                print(f"[Iter {it}] Train {bi+1}/{train_batches_per_iter} loss={stats['loss']:.4f}")

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        avg_pl = float(np.mean(pol_losses)) if pol_losses else float("nan")
        avg_vl = float(np.mean(val_losses)) if val_losses else float("nan")
        avg_steps = float(np.mean(step_counts)) if step_counts else 0.0

        # Evaluate vs random
        print(f"[Iter {it}] Eval: {eval_games} games")
        win_rate = evaluate_vs_random(
            model,
            games=eval_games,
            mcts_simulations=max(16, mcts_simulations // 2),
            device=device,
            mcts_batch=mcts_batch,
        )

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
    # Auto-select best available device: CUDA > DirectML > CPU
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("[Device] Using CUDA GPU")
    elif DML_DEVICE is not None:
        dev = DML_DEVICE
        print("[Device] Using DirectML (AMD/Intel GPU)")
    else:
        dev = "cpu"
        print("[Device] Using CPU")

    # Example longer run (tweak as desired)
    az_train(
        iterations=10,
        games_per_iter=8,
        mcts_simulations=96,
        mcts_batch=32,
        lr=1e-3,
        device=dev,
        replay_capacity=30000,
        batch_size=512,
        train_batches_per_iter=50,
        eval_games=50,
        resume=True,
        weight_decay=1e-4,
        grad_clip=1.0,
        policy_weight=1.0,
        value_weight=1.0,
        temp_init=1.0,
        temp_final=0.0,
        temp_moves=20,
        compile_model=False,
    )
