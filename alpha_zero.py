import math
import hashlib
import os
import csv
import random
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing as mp
import io
import copy

#DEBUG
#import game_state as GS
#GS.DEBUG_GUARDS = True

# Optional DirectML (AMD on Windows)
try:
    import torch_directml as _dml  # type: ignore
    DML_DEVICE = _dml.device()
except Exception:
    DML_DEVICE = None

from game_state import GameState, Action, Card, PlayerState
from nn_input_output import (
    flatten_game_state,
    legal_actions_mask,
    index_to_action,
    flatten_visible_cards,
    permute_colors_in_flat_state,
    permute_policy_colors,
    permute_action_index,
    permute_token_vector,
    CARD_VEC_LEN,
    NUM_TIERS,
    CARDS_PER_TIER,
    RESERVED_PER_PLAYER,
    NOBLES_MAX,
)
from cards_init import setup_game


def _allowlist_checkpoint_globals() -> None:
    """Allowlist legacy tensor/numpy rebuild globals for safe checkpoint loading."""
    try:
        from torch.serialization import add_safe_globals  # type: ignore
    except Exception:
        return
    globs: list[Any] = []
    # Torch legacy tensor rebuild helpers
    try:
        from torch._utils import _rebuild_device_tensor_from_numpy  # type: ignore
        globs.append(_rebuild_device_tensor_from_numpy)
    except Exception:
        pass
    # NumPy legacy pickle globals seen in some checkpoints
    try:
        from numpy.core.multiarray import _reconstruct  # type: ignore
        globs.append(_reconstruct)
    except Exception:
        pass
    try:
        globs.append(np.ndarray)
    except Exception:
        pass
    try:
        globs.append(np.dtype)
    except Exception:
        pass
    # Newer NumPy exposes concrete dtype classes (e.g., Float32DType) in pickles.
    try:
        globs.append(type(np.dtype(np.float32)))
    except Exception:
        pass
    # Python codec helper seen in some serialized storages.
    try:
        from _codecs import encode as _codecs_encode  # type: ignore
        globs.append(_codecs_encode)
    except Exception:
        pass
    if globs:
        try:
            add_safe_globals(globs)  # type: ignore[arg-type]
        except Exception:
            pass


def _model_state_dict_cpu(model: nn.Module) -> Dict[str, Any]:
    """Return a CPU-only copy of model state_dict for portable safe loading."""
    sd = model.state_dict()
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


# -----------------------------
# Policy + Value Network (Attention)
# -----------------------------
class AttentionBlock(nn.Module):
    """Pre-LN transformer encoder block for set-like game entities."""

    def __init__(self, width: int, n_heads: int, dropout: float = 0.05):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class PolicyValueNet(nn.Module):
    """Token-attention encoder for Splendor entities.

    We keep the same fixed 43-action head interface while replacing pooled MLP
    context with cross-entity attention over cards, nobles, players, and bank.
    """

    def __init__(self, input_size: int, action_size: int = 43, width: int = 512, n_blocks: int = 4):
        super().__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.num_players = self._infer_num_players(input_size)
        self.width = int(width)

        # Sequence layout (fixed for action alignment and stable decoding)
        self.n_board = NUM_TIERS * CARDS_PER_TIER
        self.n_res_self = RESERVED_PER_PLAYER
        self.n_nobles = NOBLES_MAX
        self.seq_len = 1 + self.n_board + self.n_res_self + self.n_nobles + 4
        # CLS + board + self reserved + nobles + (self player, opp player, bank, opp reserved pool)

        n_heads = self._pick_num_heads(self.width)

        # Entity encoders
        self.card_encoder = nn.Sequential(
            nn.Linear(CARD_VEC_LEN, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
        )
        self.noble_encoder = nn.Sequential(
            nn.Linear(5, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
        )
        self.player_encoder = nn.Sequential(
            nn.Linear(12, self.width),  # 6 tokens + 5 bonuses + 1 points
            nn.GELU(),
            nn.Linear(self.width, self.width),
        )
        self.bank_encoder = nn.Sequential(
            nn.Linear(6, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
        )

        # Learned token metadata
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.width))
        self.board_pos = nn.Parameter(torch.zeros(1, self.n_board, self.width))
        self.res_pos = nn.Parameter(torch.zeros(1, self.n_res_self, self.width))
        self.noble_pos = nn.Parameter(torch.zeros(1, self.n_nobles, self.width))
        self.type_emb = nn.Embedding(8, self.width)
        # 0 cls, 1 board, 2 self-res, 3 nobles, 4 self-player, 5 opp-player, 6 bank, 7 opp-res-pool

        # Stronger token identity embeddings.
        self.board_tier_emb = nn.Embedding(NUM_TIERS, self.width)
        self.board_slot_emb = nn.Embedding(CARDS_PER_TIER, self.width)
        self.res_slot_emb = nn.Embedding(self.n_res_self, self.width)
        self.noble_slot_emb = nn.Embedding(self.n_nobles, self.width)

        board_tier_ids: list[int] = []
        board_slot_ids: list[int] = []
        for t in range(NUM_TIERS):
            for s in range(CARDS_PER_TIER):
                board_tier_ids.append(t)
                board_slot_ids.append(s)
        self.register_buffer("board_tier_ids", torch.tensor(board_tier_ids, dtype=torch.long), persistent=False)
        self.register_buffer("board_slot_ids", torch.tensor(board_slot_ids, dtype=torch.long), persistent=False)
        self.register_buffer("res_slot_ids", torch.arange(self.n_res_self, dtype=torch.long), persistent=False)
        self.register_buffer("noble_slot_ids", torch.arange(self.n_nobles, dtype=torch.long), persistent=False)

        self.blocks = nn.ModuleList([AttentionBlock(self.width, n_heads=n_heads) for _ in range(max(1, int(n_blocks)))])
        self.final_ln = nn.LayerNorm(self.width)

        # Policy heads
        self.policy_global = nn.Linear(self.width, 16)  # 15 token actions + 1 take-gold
        head_in = self.width * 2
        self.policy_buy = nn.Sequential(
            nn.Linear(head_in, self.width),
            nn.GELU(),
            nn.Linear(self.width, 1),
        )
        self.policy_reserve = nn.Sequential(
            nn.Linear(head_in, self.width),
            nn.GELU(),
            nn.Linear(self.width, 1),
        )
        self.policy_buy_reserved = nn.Sequential(
            nn.Linear(head_in, self.width),
            nn.GELU(),
            nn.Linear(self.width, 1),
        )

        # Return-choice head: scores candidate return vectors for a given base action
        action_emb_dim = max(16, self.width // 16)
        self.return_action_embed = nn.Embedding(action_size, action_emb_dim)
        self.return_head = nn.Sequential(
            nn.Linear(self.width + action_emb_dim + 6, self.width // 2),
            nn.GELU(),
            nn.Linear(self.width // 2, 1),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.width, self.width // 2),
            nn.GELU(),
            nn.Linear(max(1, self.width // 2), 1),
            nn.Tanh(),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.board_pos, mean=0.0, std=0.02)
        nn.init.normal_(self.res_pos, mean=0.0, std=0.02)
        nn.init.normal_(self.noble_pos, mean=0.0, std=0.02)
        nn.init.normal_(self.type_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.board_tier_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.board_slot_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.res_slot_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.noble_slot_emb.weight, mean=0.0, std=0.02)

    @staticmethod
    def _pick_num_heads(width: int) -> int:
        # Pick the largest standard head count that divides width.
        for h in (16, 12, 8, 6, 4, 3, 2, 1):
            if width % h == 0:
                return h
        return 1

    @staticmethod
    def _infer_num_players(input_size: int) -> int:
        board_len = NUM_TIERS * CARDS_PER_TIER * CARD_VEC_LEN
        nobles_len = NOBLES_MAX * 5
        bank_len = 6
        per_player = RESERVED_PER_PLAYER * CARD_VEC_LEN + 12
        base = board_len + bank_len + nobles_len
        if input_size >= base and (input_size - base) % per_player == 0:
            return int((input_size - base) // per_player)
        return 2

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D), mask: (B, N) with True for valid tokens
        m = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        return (x * m).sum(dim=1) / denom

    def _type(self, bsz: int, type_id: int, count: int, device: torch.device) -> torch.Tensor:
        ids = torch.full((bsz, count), int(type_id), device=device, dtype=torch.long)
        return self.type_emb(ids)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        bsz = x.shape[0]
        n_players = self.num_players
        device = x.device

        idx = 0
        board_len = self.n_board * CARD_VEC_LEN
        board = x[:, idx: idx + board_len].view(bsz, self.n_board, CARD_VEC_LEN)
        idx += board_len

        reserved_len = n_players * RESERVED_PER_PLAYER * CARD_VEC_LEN
        reserved = x[:, idx: idx + reserved_len].view(bsz, n_players, RESERVED_PER_PLAYER, CARD_VEC_LEN)
        idx += reserved_len

        player_len = n_players * 12
        players = x[:, idx: idx + player_len].view(bsz, n_players, 12)
        idx += player_len

        bank = x[:, idx: idx + 6]
        idx += 6

        nobles_len = NOBLES_MAX * 5
        nobles = x[:, idx: idx + nobles_len].view(bsz, NOBLES_MAX, 5)

        # Valid-token masks (True = valid)
        board_valid = (board.abs().sum(dim=-1) > 0)
        res_valid = (reserved.abs().sum(dim=-1) > 0)
        noble_valid = (nobles.abs().sum(dim=-1) > 0)

        # Base encodings
        board_tok = self.card_encoder(board)
        self_res = reserved[:, 0]
        self_res_tok = self.card_encoder(self_res)
        noble_tok = self.noble_encoder(nobles)
        self_player_tok = self.player_encoder(players[:, 0]).unsqueeze(1)
        bank_tok = self.bank_encoder(bank).unsqueeze(1)

        # Opponent aggregates (robust to >2 players)
        if n_players > 1:
            opp_players = players[:, 1:]
            opp_player_tok = self.player_encoder(opp_players.reshape(-1, 12)).view(bsz, n_players - 1, self.width).mean(dim=1, keepdim=True)
            opp_res = reserved[:, 1:].reshape(bsz, (n_players - 1) * RESERVED_PER_PLAYER, CARD_VEC_LEN)
            opp_res_valid = res_valid[:, 1:].reshape(bsz, (n_players - 1) * RESERVED_PER_PLAYER)
            opp_res_tok_raw = self.card_encoder(opp_res)
            opp_res_pool = self._masked_mean(opp_res_tok_raw, opp_res_valid).unsqueeze(1)
            opp_player_valid = torch.ones((bsz, 1), dtype=torch.bool, device=device)
            opp_res_pool_valid = opp_res_valid.any(dim=1, keepdim=True)
        else:
            opp_player_tok = torch.zeros_like(self_player_tok)
            opp_res_pool = torch.zeros_like(self_player_tok)
            opp_player_valid = torch.zeros((bsz, 1), dtype=torch.bool, device=device)
            opp_res_pool_valid = torch.zeros((bsz, 1), dtype=torch.bool, device=device)

        board_tier_e = self.board_tier_emb(self.board_tier_ids).unsqueeze(0)
        board_slot_e = self.board_slot_emb(self.board_slot_ids).unsqueeze(0)
        res_slot_e = self.res_slot_emb(self.res_slot_ids).unsqueeze(0)
        noble_slot_e = self.noble_slot_emb(self.noble_slot_ids).unsqueeze(0)

        # Add position + type metadata
        board_tok = board_tok + self.board_pos + board_tier_e + board_slot_e + self._type(bsz, 1, self.n_board, device)
        self_res_tok = self_res_tok + self.res_pos + res_slot_e + self._type(bsz, 2, self.n_res_self, device)
        noble_tok = noble_tok + self.noble_pos + noble_slot_e + self._type(bsz, 3, self.n_nobles, device)
        self_player_tok = self_player_tok + self._type(bsz, 4, 1, device)
        opp_player_tok = opp_player_tok + self._type(bsz, 5, 1, device)
        bank_tok = bank_tok + self._type(bsz, 6, 1, device)
        opp_res_pool = opp_res_pool + self._type(bsz, 7, 1, device)
        cls_tok = self.cls_token.expand(bsz, -1, -1) + self._type(bsz, 0, 1, device)

        # Build token sequence
        seq = torch.cat(
            [
                cls_tok,             # 0
                board_tok,           # 1..12
                self_res_tok,        # 13..15
                noble_tok,           # 16..25
                self_player_tok,     # 26
                opp_player_tok,      # 27
                bank_tok,            # 28
                opp_res_pool,        # 29
            ],
            dim=1,
        )

        # key_padding_mask: True means "ignore this token in attention"
        pad_mask = torch.cat(
            [
                torch.zeros(bsz, 1, dtype=torch.bool, device=device),      # cls
                ~board_valid,                                              # board
                ~res_valid[:, 0],                                          # self reserved
                ~noble_valid,                                              # nobles
                torch.zeros((bsz, 1), dtype=torch.bool, device=device),    # self-player
                ~opp_player_valid,                                         # opp-player (absent in 1p)
                torch.zeros((bsz, 1), dtype=torch.bool, device=device),    # bank
                ~opp_res_pool_valid,                                       # opp-reserved pool
            ],
            dim=1,
        )

        for blk in self.blocks:
            seq = blk(seq, key_padding_mask=pad_mask)
        seq = self.final_ln(seq)

        cls = seq[:, 0]
        board_ctx = seq[:, 1: 1 + self.n_board]
        self_res_ctx = seq[:, 1 + self.n_board: 1 + self.n_board + self.n_res_self]

        # Empty slots should not contribute non-informative policy features.
        board_ctx = board_ctx * board_valid.unsqueeze(-1).to(dtype=board_ctx.dtype)
        self_res_ctx = self_res_ctx * res_valid[:, 0].unsqueeze(-1).to(dtype=self_res_ctx.dtype)

        # Policy heads
        global_logits = self.policy_global(cls)

        cls_board = cls.unsqueeze(1).expand(-1, self.n_board, -1)
        buy_vis = self.policy_buy(torch.cat([board_ctx, cls_board], dim=-1)).squeeze(-1)
        reserve_vis = self.policy_reserve(torch.cat([board_ctx, cls_board], dim=-1)).squeeze(-1)

        cls_res = cls.unsqueeze(1).expand(-1, self.n_res_self, -1)
        buy_res = self.policy_buy_reserved(torch.cat([self_res_ctx, cls_res], dim=-1)).squeeze(-1)

        # Assemble logits in fixed action order
        logits = torch.zeros(bsz, self.action_size, device=device, dtype=x.dtype)
        logits[:, 0:15] = global_logits[:, 0:15]
        neg_large = torch.full_like(buy_vis, -1.0e4)
        logits[:, 15:27] = torch.where(board_valid, buy_vis, neg_large)
        logits[:, 27:30] = torch.where(res_valid[:, 0], buy_res, torch.full_like(buy_res, -1.0e4))
        logits[:, 30:42] = torch.where(board_valid, reserve_vis, neg_large)
        logits[:, 42] = global_logits[:, 15]

        value = self.value_head(cls).squeeze(-1)
        if return_features:
            return logits, value, cls
        return logits, value

    def score_return_candidates(self, h: torch.Tensor, action_idx: int, return_vecs: torch.Tensor) -> torch.Tensor:
        """Score return candidates given a global state embedding and base action index."""
        if action_idx is None or action_idx < 0 or action_idx >= self.action_size:
            return torch.zeros(return_vecs.shape[0], device=return_vecs.device, dtype=return_vecs.dtype)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if return_vecs.dim() == 1:
            return_vecs = return_vecs.unsqueeze(0)
        k = return_vecs.shape[0]
        h_exp = h.expand(k, -1)
        a_idx = torch.tensor([int(action_idx)], device=h.device, dtype=torch.long)
        a_emb = self.return_action_embed(a_idx).expand(k, -1)
        feats = torch.cat([h_exp, a_emb, return_vecs], dim=-1)
        return self.return_head(feats).squeeze(-1)

def _card_short(card: Optional[Card]) -> str:
    if card is None:
        return "None"
    try:
        return f"t{card.tier} pts={card.points} bonus={card.bonus_color} cost={dict(card.cost)}"
    except Exception:
        return "Card(?)"


def log_policy_alignment(model: PolicyValueNet, state: GameState, device: Any = "cpu", top_k: int = 3) -> None:
    """Print per-card logits for a single state to verify slot/action alignment."""
    was_training = model.training
    model.eval()
    try:
        x = torch.tensor(flatten_game_state(state), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(x)
        l = logits.squeeze(0).detach().cpu().numpy()
    finally:
        if was_training:
            model.train()

    vis = flatten_visible_cards(state)
    reserved = list(getattr(state.players[state.current_player], "reserved", []))

    print("[PolicyDebug] top token actions")
    token_idxs = list(range(0, 15)) + [42]
    toks = sorted([(i, l[i]) for i in token_idxs], key=lambda t: t[1], reverse=True)[: max(1, int(top_k))]
    for idx, val in toks:
        print(f"  a={idx:02d} logit={val:+.3f}")

    print("[PolicyDebug] top buy-visible")
    buy_vis = sorted([(15 + i, l[15 + i], vis[i]) for i in range(len(vis))], key=lambda t: t[1], reverse=True)
    for idx, val, card in buy_vis[: max(1, int(top_k))]:
        print(f"  a={idx:02d} logit={val:+.3f} card={_card_short(card)}")

    print("[PolicyDebug] top reserve-visible")
    res_vis = sorted([(30 + i, l[30 + i], vis[i]) for i in range(len(vis))], key=lambda t: t[1], reverse=True)
    for idx, val, card in res_vis[: max(1, int(top_k))]:
        print(f"  a={idx:02d} logit={val:+.3f} card={_card_short(card)}")

    print("[PolicyDebug] buy-reserved slots")
    for i in range(min(3, len(reserved))):
        card = reserved[i]
        print(f"  a={27 + i:02d} logit={l[27 + i]:+.3f} card={_card_short(card)}")


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
                 dir_alpha: float = 0.3, dir_eps: float = 0.25, returns_top_k: int = 3, mcts_batch: int = 16,
                 return_prior_mix: float = 0.8):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.n_sim = n_simulations
        self.dir_alpha = dir_alpha
        self.dir_eps = dir_eps
        self.returns_top_k = max(1, int(returns_top_k))
        self.mcts_batch = max(1, int(mcts_batch))
        # Blend learned return-head priors with heuristic priors for stability.
        self.return_prior_mix = max(0.0, min(1.0, float(return_prior_mix)))
        # Stores best return variant per base action from the last run
        self._last_best_returns: Dict[int, Dict[str, int]] = {}
        # Root for tree reuse across moves
        self._root: Optional[AZNode] = None

    def get_best_tokens_returned(self, a_idx: int) -> Optional[Dict[str, int]]:
        return self._last_best_returns.get(a_idx)

    def get_return_distribution(self, a_idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return candidate return vectors and MCTS visit-based probs for a base action."""
        root = self._root
        if root is None or not root.edges:
            return None
        counts: list[int] = []
        cands: list[list[int]] = []
        total = 0
        token_order = ["diamond", "sapphire", "obsidian", "ruby", "emerald", "gold"]
        for key, e in root.edges.items():
            base, _var = key
            if base != a_idx:
                continue
            meta = root.edge_meta.get(key, {})
            ret = meta.get("tokens_returned", {}) if meta else {}
            vec = [int(ret.get(c, 0)) for c in token_order]
            cands.append(vec)
            counts.append(int(e.N))
            total += int(e.N)
        if total <= 0 or len(cands) <= 1:
            return None
        probs = [c / total for c in counts]
        return np.array(cands, dtype=np.float32), np.array(probs, dtype=np.float32)

    def _state_signature(self, state: GameState) -> bytes:
        """Compact signature for detecting stale tree reuse.

        Note: This intentionally omits hidden deck order. Reuse is still safe
        when the tree advances via its own chosen actions on the same state.
        """
        flat = flatten_game_state(state)
        h = hashlib.blake2b(digest_size=16)
        h.update(flat.tobytes())
        # Include terminal bookkeeping that affects end-of-round logic.
        h.update(bytes([1 if state.is_terminal else 0]))
        winner = 255 if getattr(state, "winner", None) is None else int(state.winner) & 0xFF
        h.update(bytes([winner]))
        h.update(bytes([1 if getattr(state, "pending_round_end", False) else 0]))
        h.update(bytes([int(getattr(state, "start_player", 0)) & 0xFF]))
        return h.digest()

    def reset_root(self) -> None:
        """Drop the cached tree when external moves make it stale."""
        self._root = None

    def _terminal_value(self, state: GameState) -> float:
        """Return terminal value from the perspective of state.current_player.

        This ensures MCTS backs up correct outcomes instead of a fixed loss
        at terminal leaves, which would bias search in the wrong direction.
        """
        if not state.is_terminal:
            return 0.0
        winner = getattr(state, "winner", None)
        if winner is None:
            return 0.0
        return 1.0 if int(winner) == int(state.current_player) else -1.0

    @torch.no_grad()
    def _evaluate_batch(self, states: List[GameState]) -> Tuple[List[np.ndarray], List[float], List[torch.Tensor]]:
        if not states:
            return [], [], []
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
                    logits, values, h = self.model(x, return_features=True)
            except Exception:
                with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                    logits, values, h = self.model(x, return_features=True)
        else:
            logits, values, h = self.model(x, return_features=True)
        priors_t = masked_softmax(logits, masks)  # (B, A)
        priors = [p.detach().cpu().numpy() for p in priors_t]
        vals = [float(v.item()) for v in values]
        hs = [row.detach() for row in h]
        return priors, vals, hs

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
            # Effective take limited by bank availability
            effective_taken: Dict[str, int] = {c: 1 for c in tokens_taken if state.tokens.get(c, 0) > 0}
            action_type = "take_tokens"
        elif 10 <= a_idx <= 14:
            gem_idx = a_idx - 10
            gem = IDX_TO_GEM[gem_idx]
            want = 2
            have = int(state.tokens.get(gem, 0))
            take_n = min(want, have)
            tokens_taken = {gem: 2}
            effective_taken = {gem: take_n} if take_n > 0 else {}
            action_type = "take_tokens"
        elif 30 <= a_idx <= 41:
            # Reserve visible card; index_to_action will resolve target/tier, here only tokens_taken matters
            g = int(state.tokens.get("gold", 0))
            tokens_taken = {"gold": 1} if g > 0 else {}
            effective_taken = tokens_taken.copy()
            action_type = "reserve"
        else:
            # Other actions don't need explicit return variants
            return [], []

        # Build after-state using effective taken amounts
        after = player.tokens.copy()
        for c, k in (effective_taken.items() if 'effective_taken' in locals() else tokens_taken.items()):
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

    @staticmethod
    def _ret_dict_to_vec(ret: Dict[str, int]) -> List[float]:
        order = ["diamond", "sapphire", "obsidian", "ruby", "emerald", "gold"]
        return [float(ret.get(c, 0)) for c in order]

    def _return_variant_weights(
        self,
        a_idx: int,
        variants: List[Dict[str, int]],
        heuristic_weights: List[float],
        h: Optional[torch.Tensor],
    ) -> List[float]:
        if not variants:
            return []
        if len(variants) == 1:
            return [1.0]

        # Normalize heuristic fallback.
        hw = np.array(heuristic_weights, dtype=np.float64) if heuristic_weights else np.ones(len(variants), dtype=np.float64)
        if hw.sum() <= 0:
            hw = np.ones(len(variants), dtype=np.float64)
        hw = hw / hw.sum()

        if h is None:
            return hw.astype(np.float32).tolist()

        try:
            vecs = torch.tensor(
                np.array([self._ret_dict_to_vec(v) for v in variants], dtype=np.float32),
                device=h.device,
                dtype=h.dtype,
            )
            scores = self.model.score_return_candidates(h, a_idx, vecs)
            probs_t = torch.softmax(scores, dim=-1)
            probs = probs_t.detach().float().cpu().numpy().astype(np.float64)
            if not np.isfinite(probs).all() or probs.sum() <= 0:
                return hw.astype(np.float32).tolist()
            probs = probs / probs.sum()
            # Blend model priors with heuristic priors for early-run robustness.
            mix = self.return_prior_mix
            blended = mix * probs + (1.0 - mix) * hw
            blended = blended / max(1e-12, blended.sum())
            return blended.astype(np.float32).tolist()
        except Exception:
            return hw.astype(np.float32).tolist()

    def _expand_with_priors(self, node: AZNode, priors: np.ndarray, h: Optional[torch.Tensor] = None) -> None:
        node.priors = priors
        legal_mask = legal_actions_mask(node.state)
        for a_idx, legal in enumerate(legal_mask):
            if not legal:
                continue
            base_p = float(priors[a_idx])
            variants, weights = self._enumerate_return_variants(node.state, a_idx)
            if variants:
                learned_weights = self._return_variant_weights(a_idx, variants, weights, h)
                for r_idx, (ret, w) in enumerate(zip(variants, learned_weights), start=0):
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
        # Use reused root only if it matches the external state signature
        root_sig = self._state_signature(root_state)
        root = self._root
        if root is None or getattr(root, "signature", None) != root_sig:
            root = AZNode(root_state)
            root.signature = root_sig  # type: ignore[attr-defined]
            self._root = root
        # If not expanded yet, expand via batch path
        if not root.is_expanded():
            priors, vals, hs = self._evaluate_batch([root.state])
            v0 = vals[0] if vals else 0.0
            self._expand_with_priors(root, priors[0], hs[0] if hs else None)
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
            # Redistribute mixed base priors while preserving variant proportions.
            for a, mb in zip(base_actions, mixed_base):
                variants = [k for k in actions if k[0] == a]
                if not variants:
                    continue
                old = np.array([max(0.0, float(root.edges[key].P)) for key in variants], dtype=np.float64)
                if old.sum() <= 0:
                    old = np.ones(len(variants), dtype=np.float64) / float(len(variants))
                else:
                    old = old / old.sum()
                for key, w in zip(variants, old):
                    root.edges[key].P = float(mb) * float(w)

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
                        child = AZNode(next_state)
                        child.signature = self._state_signature(next_state)  # type: ignore[attr-defined]
                        node.children[key] = child
                    node = node.children[key]

                # Terminal: immediate backup
                if node.state.is_terminal:
                    v = self._terminal_value(node.state)
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
                priors_list, vals_list, hs_list = self._evaluate_batch([n.state for n in pending_nodes])
                for node, path, priors, v, h in zip(pending_nodes, pending_paths, priors_list, vals_list, hs_list):
                    self._expand_with_priors(node, priors, h)
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


def _load_ckpt_model(path: str, device: Any, input_size: int, action_size: int, width: int = 512, res_blocks: int = 6) -> Optional[PolicyValueNet]:
    candidates: List[str] = [path]
    # Fallback from legacy full checkpoint path to model-only path.
    if isinstance(path, str) and path.endswith(".pt") and not path.endswith("_model.pt"):
        model_only = path[:-3] + "_model.pt"
        if os.path.exists(model_only):
            candidates.append(model_only)

    last_err: Optional[Exception] = None
    for cand in candidates:
        try:
            model = PolicyValueNet(input_size=input_size, action_size=action_size, width=width, n_blocks=res_blocks).to(device)
            # Safe-only loader for mixed checkpoint formats (CPU/CUDA/DirectML).
            try:
                ck = torch.load(cand, map_location='cpu', weights_only=True)  # type: ignore[call-arg]
            except Exception:
                # Some checkpoints may require allowing legacy tensor/NumPy rebuild globals.
                _allowlist_checkpoint_globals()
                ck = torch.load(cand, map_location='cpu', weights_only=True)  # type: ignore[call-arg]
            sd = ck.get("model", ck)
            incompat = model.load_state_dict(sd, strict=False)
            if getattr(incompat, "missing_keys", None) or getattr(incompat, "unexpected_keys", None):
                print(
                    f"[Arena] Non-strict checkpoint load for {os.path.basename(cand)} "
                    f"(missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)})"
                )
            if cand != path:
                print(f"[Arena] Loaded fallback model-only checkpoint: {cand}")
            return model.to(device)
        except Exception as e:
            last_err = e

    print(f"[Arena] Failed loading model from {path}: {last_err}")
    return None


def arena_vs_model(model: PolicyValueNet, opp: PolicyValueNet, games: int = 100, sims: int = 64, device: Any = "cpu", mcts_batch: int = 16, max_moves: int = 250) -> tuple[float, float]:
    wins = 0
    margins: list[float] = []
    for g in range(games):
        state = setup_game(num_players=2)
        my_index = g % 2
        mcts_me = AlphaZeroMCTS(model, device=device, n_simulations=sims, mcts_batch=mcts_batch)
        mcts_opp = AlphaZeroMCTS(opp, device=device, n_simulations=sims, mcts_batch=mcts_batch)
        move_count = 0
        while not state.is_terminal and move_count < max_moves:
            legal = state.get_legal_actions()
            if not legal:
                state.current_player = (state.current_player + 1) % len(state.players)
                # External state change without an encoded action: tree is stale.
                mcts_me.reset_root()
                mcts_opp.reset_root()
                continue
            if state.current_player == my_index:
                _, a_idx = mcts_me.run(state, temperature=0.0)
                a = index_to_action(a_idx, state) if a_idx != -1 else legal[0]
                best_ret = mcts_me.get_best_tokens_returned(a_idx)
                if best_ret is not None and hasattr(a, "tokens_returned"):
                    a.tokens_returned = best_ret
                # Reuse only for the MCTS that selected the action.
                if a_idx != -1:
                    try:
                        mcts_me.reuse_after_play(a_idx)
                    except Exception:
                        mcts_me.reset_root()
                else:
                    mcts_me.reset_root()
                # Opponent tree is stale (its variant choice may differ).
                mcts_opp.reset_root()
            else:
                _, a_idx = mcts_opp.run(state, temperature=0.0)
                a = index_to_action(a_idx, state) if a_idx != -1 else legal[0]
                best_ret = mcts_opp.get_best_tokens_returned(a_idx)
                if best_ret is not None and hasattr(a, "tokens_returned"):
                    a.tokens_returned = best_ret
                if a_idx != -1:
                    try:
                        mcts_opp.reuse_after_play(a_idx)
                    except Exception:
                        mcts_opp.reset_root()
                else:
                    mcts_opp.reset_root()
                mcts_me.reset_root()
            state = state.apply_action(a)
            move_count += 1
        my_pts = state.players[my_index].points
        opp_pts = state.players[1 - my_index].points
        margins.append(my_pts - opp_pts)
        if state.winner == my_index:
            wins += 1
    return (wins / games if games > 0 else 0.0), (float(np.mean(margins)) if margins else 0.0)


# -----------------------------
# Self-play and Training
# -----------------------------
@dataclass
class Sample:
    state: np.ndarray  # flattened state (float32)
    pi: np.ndarray     # policy target over actions (float32, sum=1)
    player: int        # player to move at this state
    action_idx: int    # base action index chosen (or -1 if unavailable)
    ret_cands: Optional[np.ndarray] = None  # (K, 6) token-return candidates
    ret_probs: Optional[np.ndarray] = None  # (K,) visit-based probs


def self_play_episode(model: PolicyValueNet, mcts_simulations: int = 100, device: str = "cpu", temperature: float = 1.0,
                      temp_init: float = 1.0, temp_final: float = 0.0, temp_moves: int = 20,
                      add_dirichlet: bool = True, mcts_batch: int = 16,
                      policy_log_moves: int = 0, policy_log_topk: int = 3) -> Tuple[List[Sample], int]:
    state = setup_game(num_players=2)
    mcts = AlphaZeroMCTS(model, device=device, n_simulations=mcts_simulations, mcts_batch=mcts_batch)

    trajectory: List[Sample] = []
    consecutive_passes = 0
    move_idx = 0
    ended_by_pass = False

    while not state.is_terminal:
        # If no legal actions, pass the turn (as in game_sim)
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            state.current_player = (state.current_player + 1) % len(state.players)
            consecutive_passes += 1
            # Passing changes the player to move without an encoded action.
            mcts.reset_root()
            # End if both players consecutively have no moves
            if consecutive_passes >= len(state.players):
                ended_by_pass = True
                break
            continue
        consecutive_passes = 0

        # Temperature schedule across the game
        temp = temp_init if move_idx < temp_moves else temp_final
        if int(policy_log_moves) > 0 and move_idx < int(policy_log_moves):
            try:
                log_policy_alignment(model, state, device=device, top_k=policy_log_topk)
            except Exception:
                pass
        pi, _ = mcts.run(state, temperature=temp, add_dirichlet=add_dirichlet)

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
                # Record sample (no return-targets if we didn't follow MCTS)
                s = flatten_game_state(state)
                trajectory.append(Sample(
                    state=s,
                    pi=pi.astype(np.float32),
                    player=state.current_player,
                    action_idx=int(a_idx_for_reuse) if a_idx_for_reuse is not None else -1,
                    ret_cands=None,
                    ret_probs=None,
                ))
                state = state.apply_action(a)
                try:
                    if a_idx_for_reuse is not None:
                        mcts.reuse_after_play(a_idx_for_reuse)
                    else:
                        mcts.reset_root()
                except Exception:
                    mcts.reset_root()
                move_idx += 1
                continue
            else:
                # Should not hit because we handled no-legal case earlier; skip turn defensively
                state.current_player = (state.current_player + 1) % len(state.players)
                mcts.reset_root()
                move_idx += 1
                continue
        else:
            probs = probs / probs.sum()
            a_idx = int(np.random.choice(len(probs), p=probs))

            action = index_to_action(a_idx, state)
            # Return-targets from MCTS visit counts (if any)
            ret = mcts.get_return_distribution(a_idx)
            ret_cands, ret_probs = (ret if ret is not None else (None, None))
            # Record sample before applying action
            s = flatten_game_state(state)
            trajectory.append(Sample(
                state=s,
                pi=pi.astype(np.float32),
                player=state.current_player,
                action_idx=int(a_idx),
                ret_cands=ret_cands,
                ret_probs=ret_probs,
            ))
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

    if state.winner is not None:
        winner = state.winner
    elif ended_by_pass:
        # Provisional winner on pass-break only: highest points, then fewest purchased cards
        max_pts = max(p.points for p in state.players)
        candidates = [i for i, p in enumerate(state.players) if p.points == max_pts]
        if len(candidates) == 1:
            winner = candidates[0]
        else:
            fewest_cards = min(len(state.players[i].cards) for i in candidates)
            tied = [i for i in candidates if len(state.players[i].cards) == fewest_cards]
            winner = tied[0] if len(tied) == 1 else -1
    else:
        winner = -1
    return trajectory, winner


def compute_targets(
    trajectory: List[Sample],
    winner: int,
    color_augments: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """Build training targets, with optional color-permutation augmentation."""
    X_np = np.stack([t.state for t in trajectory]).astype(np.float32)
    P_np = np.stack([t.pi for t in trajectory]).astype(np.float32)
    if winner < 0:
        Z_np = np.zeros(len(trajectory), dtype=np.float32)
    else:
        z = [1.0 if t.player == winner else -1.0 for t in trajectory]
        Z_np = np.array(z, dtype=np.float32)

    ret_actions = [int(t.action_idx) for t in trajectory]
    ret_cands = [t.ret_cands for t in trajectory]
    ret_probs = [t.ret_probs for t in trajectory]

    aug = max(0, int(color_augments))
    if aug > 0 and len(trajectory) > 0:
        xs = [X_np]
        ps = [P_np]
        zs = [Z_np]
        ra_all = [ret_actions]
        rc_all = [ret_cands]
        rp_all = [ret_probs]
        for _ in range(aug):
            X_aug = np.empty_like(X_np)
            P_aug = np.empty_like(P_np)
            ra_aug: list[int] = []
            rc_aug: list[Optional[np.ndarray]] = []
            rp_aug: list[Optional[np.ndarray]] = []
            for i in range(X_np.shape[0]):
                perm = np.random.permutation(5)
                X_aug[i] = permute_colors_in_flat_state(X_np[i], perm)
                P_aug[i] = permute_policy_colors(P_np[i], perm)
                # Action index under color permutation
                ra_aug.append(permute_action_index(ret_actions[i], perm))
                # Permute return candidates if present
                if ret_cands[i] is not None:
                    rc_aug.append(permute_token_vector(ret_cands[i], perm))
                    rp_aug.append(ret_probs[i])
                else:
                    rc_aug.append(None)
                    rp_aug.append(None)
            xs.append(X_aug)
            ps.append(P_aug)
            zs.append(Z_np)
            ra_all.append(ra_aug)
            rc_all.append(rc_aug)
            rp_all.append(rp_aug)
        X_np = np.concatenate(xs, axis=0)
        P_np = np.concatenate(ps, axis=0)
        Z_np = np.concatenate(zs, axis=0)
        ret_actions = [a for block in ra_all for a in block]
        ret_cands = [c for block in rc_all for c in block]
        ret_probs = [p for block in rp_all for p in block]

    X = torch.tensor(X_np, dtype=torch.float32)
    P = torch.tensor(P_np, dtype=torch.float32)
    Z = torch.tensor(Z_np, dtype=torch.float32)
    return X, P, Z, ret_actions, ret_cands, ret_probs


# -----------------------------
# Parallel self-play helpers
# -----------------------------
_SP_MODEL: Optional[PolicyValueNet] = None
_SP_DEVICE: Any = "cpu"


def _sp_init(model_bytes: bytes, input_size: int, action_size: int, width: int, res_blocks: int, device_str: str) -> None:
    """Initializer for worker processes: reconstruct model once per process."""
    global _SP_MODEL, _SP_DEVICE
    # Resolve device from string; fallback to CPU if unavailable
    dev: Any = "cpu"
    try:
        if device_str.startswith("cuda") and torch.cuda.is_available():
            dev = torch.device("cuda")
        elif device_str.startswith("cpu"):
            dev = "cpu"
        else:
            dev = "cpu"
    except Exception:
        dev = "cpu"
    _SP_DEVICE = dev
    # Rebuild model and load weights
    m = PolicyValueNet(input_size=input_size, action_size=action_size, width=width, n_blocks=res_blocks)
    buf = io.BytesIO(model_bytes)
    try:
        state = torch.load(buf, map_location='cpu', weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        # Older PyTorch without weights_only support.
        state = torch.load(buf, map_location='cpu')
    try:
        m.load_state_dict(state)
    except Exception:
        # Best-effort non-strict for potential arch drift
        m.load_state_dict(state, strict=False)
    _SP_MODEL = m.to(dev).eval()


def _sp_run(args: Tuple[int, float, float, float, int, bool, int, Optional[int]]) -> Tuple[List[Sample], int]:
    """Run one self-play episode using the global model.

    Args: (mcts_simulations, temperature, temp_init, temp_final, temp_moves, add_dirichlet, mcts_batch, seed)
    """
    (mcts_simulations, temperature, temp_init, temp_final, temp_moves, add_dirichlet, mcts_batch, seed) = args
    if seed is not None:
        try:
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            torch.manual_seed(seed)
        except Exception:
            pass
    assert _SP_MODEL is not None
    return self_play_episode(
        _SP_MODEL,
        mcts_simulations=mcts_simulations,
        device=_SP_DEVICE,
        temperature=temperature,
        temp_init=temp_init,
        temp_final=temp_final,
        temp_moves=temp_moves,
        add_dirichlet=add_dirichlet,
        mcts_batch=mcts_batch,
    )


def train_on_batch(
    model: PolicyValueNet,
    optimizer: optim.Optimizer,
    batch: Tuple[Any, ...],
    device: Any = "cpu",
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    return_weight: float = 0.5,
    # Ablation: disable return-head loss without changing architecture
    use_return_loss: bool = True,
    grad_clip: Optional[float] = None,
    scaler: Optional[Any] = None,
    entropy_coef: float = 0.0,
    max_policy_loss: Optional[float] = None,
    max_total_loss: Optional[float] = None,
) -> Dict[str, float]:
    # Batch may include return-target info
    if len(batch) >= 6:
        X, P, Z, ret_actions, ret_cands, ret_probs = batch[:6]
    else:
        X, P, Z = batch  # type: ignore[misc]
        ret_actions, ret_cands, ret_probs = [], [], []
    X, P, Z = X.to(device), P.to(device), Z.to(device)
    use_cuda = (isinstance(device, torch.device) and device.type == "cuda") or (isinstance(device, str) and str(device).startswith("cuda"))
    use_amp = scaler is not None and use_cuda

    optimizer.zero_grad(set_to_none=True)

    def _is_divergent(policy_loss_t: torch.Tensor, total_loss_t: torch.Tensor) -> bool:
        try:
            pl = float(policy_loss_t.detach().item())
            tl = float(total_loss_t.detach().item())
        except Exception:
            return True
        if (not np.isfinite(pl)) or (not np.isfinite(tl)):
            return True
        if max_policy_loss is not None and pl > float(max_policy_loss):
            return True
        if max_total_loss is not None and tl > float(max_total_loss):
            return True
        return False

    if use_amp:
        # Prefer new torch.amp API, fallback to torch.cuda.amp
        try:
            autocast_ctx = torch.amp.autocast  # type: ignore[attr-defined]
            with autocast_ctx('cuda'):
                logits, values, h = model(X, return_features=True)
                log_probs = torch.log_softmax(logits, dim=-1)
                policy_loss = -(P * log_probs).sum(dim=-1).mean()
                # Entropy bonus (maximize entropy => subtract from loss)
                probs = torch.softmax(logits, dim=-1)
                entropy = (-(probs * torch.log_softmax(logits, dim=-1))).sum(dim=-1).mean()
                value_loss = nn.functional.mse_loss(values, Z)
                # Return-choice loss (optional)
                ret_loss = torch.zeros((), device=X.device, dtype=h.dtype)
                ret_count = 0
                if ret_cands:
                    for i, cands in enumerate(ret_cands):
                        if cands is None or ret_probs[i] is None:
                            continue
                        if len(cands) <= 1:
                            continue
                        cand_t = torch.tensor(cands, device=X.device, dtype=h.dtype)
                        probs_t = torch.tensor(ret_probs[i], device=X.device, dtype=h.dtype)
                        scores = model.score_return_candidates(h[i], ret_actions[i], cand_t)
                        logp = torch.log_softmax(scores, dim=-1)
                        ret_loss = ret_loss + (-(probs_t * logp).sum())
                        ret_count += 1
                if ret_count > 0:
                    ret_loss = ret_loss / float(ret_count)
                loss = (policy_weight * policy_loss +
                        value_weight * value_loss +
                        float(return_weight) * ret_loss -
                        float(entropy_coef) * entropy)
                if _is_divergent(policy_loss, loss):
                    total_loss = (policy_weight * policy_loss +
                                  value_weight * value_loss +
                                  float(return_weight) * ret_loss)
                    return {
                        "loss": float(total_loss.item()),
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "return_loss": float(ret_loss.item()),
                        "entropy": float(entropy.item()),
                        "skipped": 1.0,
                        "bad_batch": 1.0,
                    }
        except Exception:
            # Backward compatibility
            with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                logits, values, h = model(X, return_features=True)
                log_probs = torch.log_softmax(logits, dim=-1)
                policy_loss = -(P * log_probs).sum(dim=-1).mean()
                probs = torch.softmax(logits, dim=-1)
                entropy = (-(probs * torch.log_softmax(logits, dim=-1))).sum(dim=-1).mean()
                value_loss = nn.functional.mse_loss(values, Z)
                ret_loss = torch.zeros((), device=X.device, dtype=h.dtype)
                ret_count = 0
                if ret_cands:
                    for i, cands in enumerate(ret_cands):
                        if cands is None or ret_probs[i] is None:
                            continue
                        if len(cands) <= 1:
                            continue
                        cand_t = torch.tensor(cands, device=X.device, dtype=h.dtype)
                        probs_t = torch.tensor(ret_probs[i], device=X.device, dtype=h.dtype)
                        scores = model.score_return_candidates(h[i], ret_actions[i], cand_t)
                        logp = torch.log_softmax(scores, dim=-1)
                        ret_loss = ret_loss + (-(probs_t * logp).sum())
                        ret_count += 1
                if ret_count > 0:
                    ret_loss = ret_loss / float(ret_count)
                loss = (policy_weight * policy_loss +
                        value_weight * value_loss +
                        float(return_weight) * ret_loss -
                        float(entropy_coef) * entropy)
                if _is_divergent(policy_loss, loss):
                    total_loss = (policy_weight * policy_loss +
                                  value_weight * value_loss +
                                  float(return_weight) * ret_loss)
                    return {
                        "loss": float(total_loss.item()),
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "return_loss": float(ret_loss.item()),
                        "entropy": float(entropy.item()),
                        "skipped": 1.0,
                        "bad_batch": 1.0,
                    }
        scaler.scale(loss).backward()
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits, values, h = model(X, return_features=True)
        # Policy loss: cross-entropy between target pi and predicted log-probs (masked by pi)
        log_probs = torch.log_softmax(logits, dim=-1)
        policy_loss = -(P * log_probs).sum(dim=-1).mean()
        probs = torch.softmax(logits, dim=-1)
        entropy = (-(probs * torch.log_softmax(logits, dim=-1))).sum(dim=-1).mean()
        # Value loss: MSE
        value_loss = nn.functional.mse_loss(values, Z)
        ret_loss = torch.zeros((), device=X.device, dtype=h.dtype)
        ret_count = 0
        if ret_cands:
            for i, cands in enumerate(ret_cands):
                if cands is None or ret_probs[i] is None:
                    continue
                if len(cands) <= 1:
                    continue
                cand_t = torch.tensor(cands, device=X.device, dtype=h.dtype)
                probs_t = torch.tensor(ret_probs[i], device=X.device, dtype=h.dtype)
                scores = model.score_return_candidates(h[i], ret_actions[i], cand_t)
                logp = torch.log_softmax(scores, dim=-1)
                ret_loss = ret_loss + (-(probs_t * logp).sum())
                ret_count += 1
        if ret_count > 0:
            ret_loss = ret_loss / float(ret_count)
        loss = (policy_weight * policy_loss +
                value_weight * value_loss +
                float(return_weight) * ret_loss -
                float(entropy_coef) * entropy)
        if _is_divergent(policy_loss, loss):
            total_loss = (policy_weight * policy_loss +
                          value_weight * value_loss +
                          float(return_weight) * ret_loss)
            return {
                "loss": float(total_loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "return_loss": float(ret_loss.item()) if "ret_loss" in locals() else 0.0,
                "entropy": float(entropy.item()),
                "skipped": 1.0,
                "bad_batch": 1.0,
            }
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    total_loss = (policy_weight * policy_loss +
                  value_weight * value_loss +
                  float(return_weight) * ret_loss)
    return {
        "loss": float(total_loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "return_loss": float(ret_loss.item()) if "ret_loss" in locals() else 0.0,
        "entropy": float(entropy.item()),
        "skipped": 0.0,
        "bad_batch": 0.0,
    }


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.pis: List[np.ndarray] = []
        self.zs: List[float] = []
        self.ret_actions: List[int] = []
        self.ret_cands: List[Optional[np.ndarray]] = []
        self.ret_probs: List[Optional[np.ndarray]] = []

    def add(
        self,
        X: torch.Tensor,
        P: torch.Tensor,
        Z: torch.Tensor,
        ret_actions: Optional[List[int]] = None,
        ret_cands: Optional[List[Optional[np.ndarray]]] = None,
        ret_probs: Optional[List[Optional[np.ndarray]]] = None,
    ):
        n = X.size(0)
        if ret_actions is None:
            ret_actions = [-1] * n
        if ret_cands is None:
            ret_cands = [None] * n
        if ret_probs is None:
            ret_probs = [None] * n
        for i in range(X.size(0)):
            if len(self.states) >= self.capacity:
                # FIFO eviction
                self.states.pop(0)
                self.pis.pop(0)
                self.zs.pop(0)
                self.ret_actions.pop(0)
                self.ret_cands.pop(0)
                self.ret_probs.pop(0)
            self.states.append(X[i].cpu().numpy())
            self.pis.append(P[i].cpu().numpy())
            self.zs.append(float(Z[i].cpu().item()))
            self.ret_actions.append(int(ret_actions[i]))
            self.ret_cands.append(ret_cands[i])
            self.ret_probs.append(ret_probs[i])

    def size(self) -> int:
        return len(self.states)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        n = min(batch_size, len(self.states))
        idxs = np.random.choice(len(self.states), size=n, replace=False)
        X = torch.tensor(np.stack([self.states[i] for i in idxs]), dtype=torch.float32)
        P = torch.tensor(np.stack([self.pis[i] for i in idxs]), dtype=torch.float32)
        Z = torch.tensor([self.zs[i] for i in idxs], dtype=torch.float32)
        ret_actions = [self.ret_actions[i] for i in idxs]
        ret_cands = [self.ret_cands[i] for i in idxs]
        ret_probs = [self.ret_probs[i] for i in idxs]
        return X, P, Z, ret_actions, ret_cands, ret_probs


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


def _find_latest_model_checkpoint(ckpt_dir: str) -> Optional[Tuple[str, int]]:
    try:
        files = os.listdir(ckpt_dir)
    except FileNotFoundError:
        return None
    best_path = None
    best_iter = -1
    suffix = "_model.pt"
    for fname in files:
        if fname.startswith("az_iter_") and fname.endswith(suffix):
            num_str = fname[len("az_iter_"):-len(suffix)]
            try:
                it = int(num_str)
            except Exception:
                continue
            if it > best_iter:
                best_iter = it
                best_path = os.path.join(ckpt_dir, fname)
    return (best_path, best_iter) if best_path is not None else None


def _load_elo_pool(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            # Migrate older full-checkpoint paths to model-only paths when available.
            migrated: List[Dict[str, Any]] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                ent = dict(item)
                p = ent.get("path")
                if isinstance(p, str) and p.endswith(".pt") and not p.endswith("_model.pt"):
                    cand = p[:-3] + "_model.pt"
                    if os.path.exists(cand):
                        ent["path"] = cand
                migrated.append(ent)
            return migrated
    except Exception:
        pass
    return []


def _save_elo_pool(path: str, pool: List[Dict[str, Any]]) -> None:
    try:
        with open(path, "w") as f:
            json.dump(pool, f, indent=2)
    except Exception:
        pass


_FAMILY_SLICE = {
    "take": list(range(0, 15)) + [42],
    "buy_vis": list(range(15, 27)),
    "buy_res": list(range(27, 30)),
    "reserve": list(range(30, 42)),
}


def _new_eval_policy_diag() -> Dict[str, float]:
    return {
        "moves": 0.0,
        "legal_count_sum": 0.0,
        "top1_legal_sum": 0.0,
        "mass_take_sum": 0.0,
        "mass_buy_vis_sum": 0.0,
        "mass_buy_res_sum": 0.0,
        "mass_reserve_sum": 0.0,
    }


def _accum_eval_policy_diag(diag: Dict[str, float], pi: np.ndarray, legal_mask: np.ndarray) -> None:
    p = np.asarray(pi, dtype=np.float64)
    lm = np.asarray(legal_mask, dtype=np.float64)
    legal_n = float(np.clip(lm.sum(), 0.0, 43.0))
    legal_p = p * lm
    s = float(legal_p.sum())
    if s > 0.0:
        legal_p = legal_p / s
    top1_legal = float(legal_p.max()) if legal_n > 0 else 0.0
    diag["moves"] += 1.0
    diag["legal_count_sum"] += legal_n
    diag["top1_legal_sum"] += top1_legal
    diag["mass_take_sum"] += float(p[_FAMILY_SLICE["take"]].sum())
    diag["mass_buy_vis_sum"] += float(p[_FAMILY_SLICE["buy_vis"]].sum())
    diag["mass_buy_res_sum"] += float(p[_FAMILY_SLICE["buy_res"]].sum())
    diag["mass_reserve_sum"] += float(p[_FAMILY_SLICE["reserve"]].sum())


def _finalize_eval_policy_diag(diag: Dict[str, float]) -> Dict[str, float]:
    moves = max(1.0, float(diag.get("moves", 0.0)))
    return {
        "moves": float(diag.get("moves", 0.0)),
        "legal_count_mean": float(diag.get("legal_count_sum", 0.0) / moves),
        "top1_legal_mean": float(diag.get("top1_legal_sum", 0.0) / moves),
        "mass_take_mean": float(diag.get("mass_take_sum", 0.0) / moves),
        "mass_buy_vis_mean": float(diag.get("mass_buy_vis_sum", 0.0) / moves),
        "mass_buy_res_mean": float(diag.get("mass_buy_res_sum", 0.0) / moves),
        "mass_reserve_mean": float(diag.get("mass_reserve_sum", 0.0) / moves),
    }


def evaluate_vs_random(
    model: PolicyValueNet,
    games: int = 4,
    mcts_simulations: int = 32,
    device: str = "cpu",
    mcts_batch: int = 16,
    max_moves: int = 250,
    return_diagnostics: bool = False,
):
    wins = 0
    diag = _new_eval_policy_diag()
    for g in range(games):
        state = setup_game(num_players=2)
        mcts = AlphaZeroMCTS(model, device=device, n_simulations=mcts_simulations, mcts_batch=mcts_batch)
        # Alternate who starts
        my_index = g % 2
        move_count = 0
        consecutive_passes = 0
        ended_by_pass = False
        while not state.is_terminal:
            legal = state.get_legal_actions()
            if not legal:
                state.current_player = (state.current_player + 1) % len(state.players)
                consecutive_passes += 1
                mcts.reset_root()
                # Break stalemates where neither player has legal moves
                if consecutive_passes >= len(state.players):
                    ended_by_pass = True
                    break
                if move_count >= max_moves:
                    break
                continue
            played_a_idx: Optional[int] = None
            if state.current_player == my_index:
                pi, a_idx = mcts.run(state, temperature=0.0)  # argmax over visits
                _accum_eval_policy_diag(diag, pi, np.array(legal_actions_mask(state), dtype=np.float32))
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
                    mcts.reset_root()
            else:
                # Opponent/random move or unencodable action -> tree is stale.
                mcts.reset_root()
        winner_idx = state.winner
        if winner_idx is None and ended_by_pass:
            # Provisional winner on pass-break only
            max_pts = max(p.points for p in state.players)
            candidates = [i for i, p in enumerate(state.players) if p.points == max_pts]
            if len(candidates) == 1:
                winner_idx = candidates[0]
            else:
                fewest_cards = min(len(state.players[i].cards) for i in candidates)
                tied = [i for i in candidates if len(state.players[i].cards) == fewest_cards]
                winner_idx = tied[0] if len(tied) == 1 else None
        if winner_idx == my_index:
            wins += 1
        # Lightweight eval progress
        interval = max(1, games // 10)
        if ((g + 1) % interval == 0) or (g + 1 == games):
            print(f"[Eval] {g+1}/{games} games done")
    wr = wins / games if games > 0 else 0.0
    if return_diagnostics:
        return wr, _finalize_eval_policy_diag(diag)
    return wr


def _greedy_action(state: GameState) -> Action:
    # Simple heuristic: prefer immediate points, then cheapest next, then take tokens aiding deficits.
    legal = state.get_legal_actions()
    if not legal:
        # Return a no-op style by advancing player handled by caller; choose random
        import random
        return random.choice(legal) if legal else Action("take_tokens", tokens_taken={})

    def afford_deficit(card: Card, player: PlayerState) -> int:
        d = 0
        for c, cost in getattr(card, 'cost', {}).items():
            have = player.tokens.get(c, 0) + player.bonuses.get(c, 0)
            if have < cost:
                d += (cost - have)
        # gold can cover some later; we ignore to keep simple
        return d

    player = state.players[state.current_player]
    best = None
    best_key = (-1_000_000, )
    for a in legal:
        key = (0, 0, 0)
        if a.action_type in ("buy_card", "buy_reserved") and a.target is not None:
            pts = getattr(a.target, 'points', 0)
            # Favor points heavily; small tie-breaker by bonus rarity
            key = (1000 + pts * 100, 10 - player.bonuses.get(getattr(a.target, 'bonus_color', ''), 0), -sum(getattr(a.target, 'cost', {}).values()))
        elif a.action_type == "reserve":
            # Prefer visible reserves with gold when low on tokens
            gain_gold = int(a.tokens_taken.get('gold', 0) > 0)
            key = (100 if gain_gold else 10, 0, 0)
        elif a.action_type == "take_tokens":
            # Score token take by how much it reduces average deficit to cheapest few cards
            # Construct after-take token counts
            after = player.tokens.copy()
            for c, k in a.tokens_taken.items():
                after[c] = after.get(c, 0) + k
            for c, k in a.tokens_returned.items():
                after[c] = max(0, after.get(c, 0) - k)
            tmp_player = PlayerState()
            tmp_player.tokens = after
            tmp_player.bonuses = player.bonuses.copy()
            # Collect visible cards
            vis: list[Card] = []
            for t in sorted(state.board.keys()):
                vis.extend([c for c in state.board[t] if c is not None])
            vis = vis[:12]
            if vis:
                deficits = sorted(afford_deficit(c, tmp_player) for c in vis)
                score = -sum(deficits[:3])  # reduce smallest deficits
            else:
                score = 0
            key = (score, 0, 0)
        if key > best_key:
            best_key = key
            best = a
    return best if best is not None else legal[0]


def evaluate_vs_greedy(
    model: PolicyValueNet,
    games: int = 50,
    mcts_simulations: int = 64,
    device: str = "cpu",
    mcts_batch: int = 16,
    max_moves: int = 250,
    return_diagnostics: bool = False,
):
    wins = 0
    margins = []
    lengths = []
    diag = _new_eval_policy_diag()
    for g in range(games):
        state = setup_game(num_players=2)
        mcts = AlphaZeroMCTS(model, device=device, n_simulations=mcts_simulations, mcts_batch=mcts_batch)
        my_index = g % 2
        move_count = 0
        consecutive_passes = 0
        ended_by_pass = False
        while not state.is_terminal and move_count < max_moves:
            legal = state.get_legal_actions()
            if not legal:
                state.current_player = (state.current_player + 1) % len(state.players)
                consecutive_passes += 1
                mcts.reset_root()
                if consecutive_passes >= len(state.players):
                    ended_by_pass = True
                    break
                continue
            played_a_idx: Optional[int] = None
            if state.current_player == my_index:
                pi, a_idx = mcts.run(state, temperature=0.0)
                _accum_eval_policy_diag(diag, pi, np.array(legal_actions_mask(state), dtype=np.float32))
                if a_idx == -1:
                    a = legal[0]
                    played_a_idx = None
                else:
                    a = index_to_action(a_idx, state)
                    best_ret = mcts.get_best_tokens_returned(a_idx)
                    if best_ret is not None and hasattr(a, "tokens_returned"):
                        a.tokens_returned = best_ret
                    played_a_idx = a_idx
            else:
                a = _greedy_action(state)
            state = state.apply_action(a)
            move_count += 1
            consecutive_passes = 0
            if played_a_idx is not None:
                try:
                    mcts.reuse_after_play(played_a_idx)
                except Exception:
                    mcts.reset_root()
            else:
                mcts.reset_root()
        lengths.append(move_count)
        my_pts = state.players[my_index].points
        opp_pts = state.players[1 - my_index].points
        margins.append(my_pts - opp_pts)
        winner_idx = state.winner
        if winner_idx is None and ended_by_pass:
            # Provisional winner on pass-break only
            max_pts = max(p.points for p in state.players)
            candidates = [i for i, p in enumerate(state.players) if p.points == max_pts]
            if len(candidates) == 1:
                winner_idx = candidates[0]
            else:
                fewest_cards = min(len(state.players[i].cards) for i in candidates)
                tied = [i for i in candidates if len(state.players[i].cards) == fewest_cards]
                winner_idx = tied[0] if len(tied) == 1 else None
        if winner_idx == my_index:
            wins += 1
        # progress
        interval = max(1, games // 10)
        if ((g + 1) % interval == 0) or (g + 1 == games):
            print(f"[EvalGreedy] {g+1}/{games} games done")
    win_rate = wins / games if games > 0 else 0.0
    avg_margin = float(np.mean(margins)) if margins else 0.0
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    if return_diagnostics:
        return win_rate, avg_margin, avg_len, _finalize_eval_policy_diag(diag)
    return win_rate, avg_margin, avg_len


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
    # If False, resume model/iter only and reset optimizer state.
    resume_optimizer_state: bool = True,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    return_weight: float = 0.5,
    # Ablation: disable return-head loss without changing architecture
    use_return_loss: bool = True,
    # Value warmup (optional): if >0, use value_warmup_weight for first N global iterations
    value_warmup_iters: int = 0,
    value_warmup_weight: float = 1.5,
    temp_init: float = 1.0,
    temp_final: float = 0.0,
    temp_moves: int = 20,
    compile_model: bool = False,
    gate_pool: bool = False,
    gate_games: int = 200,
    gate_threshold: float = 0.57,
    # Delay champion gating for the first N global iterations.
    gate_start_iter: int = 1,
    # If True, rejected challengers are replaced by champion weights for next iteration.
    gate_revert_on_reject: bool = False,
    # Elo ladder (training signal strength)
    use_elo_ladder: bool = False,
    ladder_pool_size: int = 4,
    ladder_opponents: int = 2,
    ladder_games: int = 16,
    ladder_sims: int = 48,
    ladder_k: float = 16.0,
    ladder_promote_threshold: float = 0.52,
    ladder_base_elo: float = 1000.0,
    # LR + entropy knobs
    use_cosine_lr: bool = True,
    lr_min: float = 1e-6,
    warmup_iters: int = 2,
    entropy_init: float = 0.01,
    entropy_anneal_iters: int = 8,
    # Anti-divergence guards for unstable backends/runs.
    divergence_policy_loss: float = 50.0,
    divergence_total_loss: float = 200.0,
    divergence_lr_backoff: float = 0.5,
    divergence_min_lr: float = 1e-6,
    divergence_max_skips_per_iter: int = 3,
    divergence_rollback: bool = True,
    # Model architecture
    width: int = 512,
    res_blocks: int = 6,
    # Logging verbosity: print last N CSV lines at iter end (0 disables)
    log_print_tail: int = 5,
    # Periodic big evaluation
    big_eval_every: int = 5,
    big_eval_games: int = 200,
    # Data augmentation: number of random color permutations per game
    color_augments: int = 1,
    # Debug: log per-card policy logits for the first N moves of a game (0 disables)
    policy_log_moves: int = 0,
    policy_log_topk: int = 3,
    # Parallel self-play
    selfplay_workers: int = 0,
    selfplay_device: Optional[str] = None,
    # Eval diagnostics: summarize policy behavior by action family and legality.
    log_policy_diagnostics: bool = True,
):
    # Setup
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    expected_train_header = (
        "iter,buffer,avg_steps,loss,policy_loss,value_loss,return_loss,"
        "win_rand,win_greedy,margin_g,len_g,"
        "diag_g_legal_n,diag_g_top1_legal,diag_g_take,diag_g_buy_vis,diag_g_buy_res,diag_g_reserve"
    )
    log_path = os.path.join(log_dir, "train_log.csv")
    write_header = not os.path.exists(log_path)
    # Keep a single canonical log file by migrating legacy schemas in place.
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", newline="") as rf:
                first = (rf.readline() or "").strip()
            if first and first != expected_train_header:
                legacy_fields = [c.strip() for c in first.split(",")]
                # Create a non-destructive backup before migration.
                backup_base = os.path.join(log_dir, "train_log_legacy")
                backup_path = backup_base + ".csv"
                bi = 2
                while os.path.exists(backup_path):
                    backup_path = f"{backup_base}_{bi}.csv"
                    bi += 1
                try:
                    import shutil as _shutil
                    _shutil.copy2(log_path, backup_path)
                except Exception:
                    backup_path = ""

                tmp_path = log_path + ".tmp"
                new_cols = expected_train_header.split(",")
                with open(log_path, "r", newline="") as src, open(tmp_path, "w", newline="") as dst:
                    r = csv.reader(src)
                    w = csv.writer(dst)
                    _ = next(r, None)  # consume old header
                    w.writerow(new_cols)
                    old_idx = {name: i for i, name in enumerate(legacy_fields)}
                    for row in r:
                        rec = {k: (row[v] if v < len(row) else "") for k, v in old_idx.items()}
                        mapped: list[str] = []
                        for c in new_cols:
                            if c in rec:
                                mapped.append(rec[c])
                            elif c == "win_rand" and "win_rate" in rec:
                                mapped.append(rec["win_rate"])
                            else:
                                mapped.append("")
                        w.writerow(mapped)
                os.replace(tmp_path, log_path)
                write_header = False
                if backup_path:
                    print(f"[Log] Migrated train_log.csv to current schema (backup: {backup_path})")
                else:
                    print("[Log] Migrated train_log.csv to current schema")
        except Exception:
            # Fallback behavior if migration fails for any reason.
            alt = os.path.join(log_dir, "train_log_v2.csv")
            print(f"[Log] Existing train_log schema mismatch; writing to {alt}")
            log_path = alt
            write_header = not os.path.exists(log_path)
    big_eval_path = os.path.join(log_dir, "eval_big.csv")
    big_write_header = not os.path.exists(big_eval_path)

    # Infer input size
    dummy_state = setup_game(num_players=2)
    input_size = len(flatten_game_state(dummy_state))
    action_size = 43

    model = PolicyValueNet(input_size=input_size, action_size=action_size, width=width, n_blocks=res_blocks).to(device)
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
    # Scheduler (cosine with optional warmup)
    scheduler: Optional[CosineAnnealingLR] = None
    if use_cosine_lr:
        try:
            # Guardrail: eta_min must stay below base lr, otherwise cosine schedule
            # ramps LR up (destabilizing low-LR recovery runs).
            eff_lr_min = float(lr_min)
            if eff_lr_min >= float(lr):
                eff_lr_min = max(1e-8, float(lr) * 0.1)
                print(
                    f"[LR] Adjusted lr_min from {float(lr_min):.2e} to {eff_lr_min:.2e} "
                    f"because lr={float(lr):.2e}"
                )
            tmax = max(1, iterations - max(0, int(warmup_iters)))
            scheduler = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=eff_lr_min)
        except Exception:
            scheduler = None
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
        missing allowlisted globals, attempts to add them and retries.
        """
        import torch as _torch
        # First attempt: safe weights-only.
        try:
            return _torch.load(path, map_location='cpu', weights_only=True)  # type: ignore[call-arg]
        except Exception:
            # Try to allowlist needed globals for safe loader, then retry.
            _allowlist_checkpoint_globals()
            return _torch.load(path, map_location='cpu', weights_only=True)  # type: ignore[call-arg]

    def _resume_from_model_only(path: str, iter_hint: int = 0) -> bool:
        nonlocal start_iter_global
        try:
            ck = _load_checkpoint_cpu_safe(path)
            sd = ck.get("model", ck)
            incompat = model.load_state_dict(sd, strict=False)
            if getattr(incompat, "missing_keys", None) or getattr(incompat, "unexpected_keys", None):
                warnings.warn(
                    f"[Resume] Non-strict load for model-only checkpoint {path} "
                    f"(missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)})"
                )
            start_iter_global = int(ck.get("iter", iter_hint))
            print(f"[Resume] Loaded model-only checkpoint {path} @ iter {start_iter_global} (optimizer reset)")
            return True
        except Exception as e:
            print(f"[Resume] Failed to load model-only checkpoint {path}: {e}")
            return False

    if resume or resume_path:
        loaded = False
        if resume_path is not None and os.path.exists(resume_path):
            try:
                # Load safely on CPU, then move to target device
                ck = _load_checkpoint_cpu_safe(resume_path)
                incompat = model.load_state_dict(ck["model"], strict=False)
                if getattr(incompat, "missing_keys", None) or getattr(incompat, "unexpected_keys", None):
                    warnings.warn(
                        f"[Resume] Non-strict full-checkpoint load {resume_path} "
                        f"(missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)})"
                    )
                if resume_optimizer_state and ("optimizer" in ck):
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
                else:
                    try:
                        optimizer.state.clear()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                start_iter_global = int(ck.get("iter", 0))
                if resume_optimizer_state and ("optimizer" in ck):
                    print(f"[Resume] Loaded checkpoint {resume_path} @ iter {start_iter_global} (optimizer restored)")
                else:
                    print(f"[Resume] Loaded checkpoint {resume_path} @ iter {start_iter_global} (optimizer reset)")
                loaded = True
            except Exception as e:
                print(f"[Resume] Full checkpoint load failed for {resume_path}: {e}")
                # Fallback to model-only path when available.
                if isinstance(resume_path, str) and resume_path.endswith(".pt") and not resume_path.endswith("_model.pt"):
                    model_path = resume_path[:-3] + "_model.pt"
                    if os.path.exists(model_path):
                        loaded = _resume_from_model_only(model_path)
        if not loaded and resume and os.path.isdir(ckpt_dir):
            latest = _find_latest_checkpoint(ckpt_dir)
            if latest is not None:
                path, itnum = latest
                try:
                    ck = _load_checkpoint_cpu_safe(path)
                    incompat = model.load_state_dict(ck["model"], strict=False)
                    if getattr(incompat, "missing_keys", None) or getattr(incompat, "unexpected_keys", None):
                        warnings.warn(
                            f"[Resume] Non-strict latest-checkpoint load {path} "
                            f"(missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)})"
                        )
                    if resume_optimizer_state and ("optimizer" in ck):
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
                    else:
                        try:
                            optimizer.state.clear()  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    start_iter_global = int(ck.get("iter", itnum))
                    if resume_optimizer_state and ("optimizer" in ck):
                        print(f"[Resume] Loaded latest checkpoint {path} @ iter {start_iter_global} (optimizer restored)")
                    else:
                        print(f"[Resume] Loaded latest checkpoint {path} @ iter {start_iter_global} (optimizer reset)")
                    loaded = True
                except Exception as e:
                    print(f"[Resume] Latest full checkpoint not loadable ({path}): {e}")
            if not loaded:
                latest_model = _find_latest_model_checkpoint(ckpt_dir)
                if latest_model is not None:
                    model_path, itnum = latest_model
                    loaded = _resume_from_model_only(model_path, iter_hint=itnum)

    # Auto-tune mcts_batch by device (CUDA:64, else:32)
    try:
        is_cuda = (isinstance(device, torch.device) and device.type == "cuda") or (isinstance(device, str) and str(device).startswith("cuda"))
        if is_cuda:
            mcts_batch = max(mcts_batch, 64)
        else:
            mcts_batch = max(mcts_batch, 32)
    except Exception:
        pass

    # Continue global iteration numbering
    for step in range(1, iterations + 1):
        it = start_iter_global + step
        step_counts: List[int] = []
        print(f"[Iter {it}] Self-play: {games_per_iter} games")
        if int(selfplay_workers) and int(selfplay_workers) > 1:
            # Prepare model bytes for workers (state_dict only)
            try:
                sd = model.state_dict()
                buf = io.BytesIO()
                torch.save(sd, buf)
                model_bytes = buf.getvalue()
            except Exception:
                # Fallback: no parallel if serialization fails
                model_bytes = None
            if model_bytes is None:
                workers = 1
            else:
                workers = min(int(selfplay_workers), int(games_per_iter))
            if workers > 1 and model_bytes is not None:
                dev_str = "cpu"
                try:
                    if selfplay_device is not None:
                        dev_str = str(selfplay_device)
                    else:
                        if (isinstance(device, torch.device) and device.type == "cuda") or (isinstance(device, str) and str(device).startswith("cuda")):
                            dev_str = "cuda"
                        elif isinstance(device, str) and device == "cpu":
                            dev_str = "cpu"
                        else:
                            dev_str = "cpu"  # default to CPU for portability
                except Exception:
                    dev_str = "cpu"
                # Spawn pool with safe 'spawn' context for Windows
                try:
                    ctx = mp.get_context("spawn")
                except Exception:
                    ctx = mp
                with ctx.Pool(processes=workers, initializer=_sp_init,
                              initargs=(model_bytes, input_size, action_size, width, res_blocks, dev_str)) as pool:
                    # Prepare per-game args
                    base_seed = random.randint(1, 10_000_000)
                    args_list = [
                        (mcts_simulations, 1.0, float(temp_init), float(temp_final), int(temp_moves), True, int(mcts_batch), base_seed + i)
                        for i in range(games_per_iter)
                    ]
                    for idx, (traj, winner) in enumerate(pool.imap_unordered(_sp_run, args_list), start=1):
                        X, P, Z, ret_actions, ret_cands, ret_probs = compute_targets(traj, winner, color_augments=color_augments)
                        buffer.add(X, P, Z, ret_actions, ret_cands, ret_probs)
                        step_counts.append(len(traj))
                        print(f"[Iter {it}] Self-play {idx}/{games_per_iter} steps={len(traj)}")
            else:
                # Fallback to sequential if workers <=1 or serialization failed
                for g in range(games_per_iter):
                    log_moves = int(policy_log_moves) if (g == 0 and step == 1) else 0
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
                        policy_log_moves=log_moves,
                        policy_log_topk=policy_log_topk,
                    )
                    X, P, Z, ret_actions, ret_cands, ret_probs = compute_targets(traj, winner, color_augments=color_augments)
                    buffer.add(X, P, Z, ret_actions, ret_cands, ret_probs)
                    step_counts.append(len(traj))
                    print(f"[Iter {it}] Self-play {g+1}/{games_per_iter} steps={len(traj)}")
        else:
            # Sequential self-play
            for g in range(games_per_iter):
                log_moves = int(policy_log_moves) if (g == 0 and step == 1) else 0
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
                    policy_log_moves=log_moves,
                    policy_log_topk=policy_log_topk,
                )
                X, P, Z, ret_actions, ret_cands, ret_probs = compute_targets(traj, winner, color_augments=color_augments)
                buffer.add(X, P, Z, ret_actions, ret_cands, ret_probs)
                step_counts.append(len(traj))
                print(f"[Iter {it}] Self-play {g+1}/{games_per_iter} steps={len(traj)}")

        # Train from replay buffer
        losses = []
        pol_losses = []
        val_losses = []
        ret_losses = []
        skipped_batches = 0
        # Rollback snapshot in case repeated divergence is detected within this iteration.
        pre_train_model_sd = _model_state_dict_cpu(model)
        try:
            pre_train_opt_sd = copy.deepcopy(optimizer.state_dict())
        except Exception:
            pre_train_opt_sd = None
        print(f"[Iter {it}] Train: {train_batches_per_iter} batches (buffer={buffer.size()})")
        # Entropy coefficient (linear anneal)
        ent_coef = 0.0
        try:
            ent_coef = float(entropy_init) * max(0.0, 1.0 - max(0, (step - 1)) / max(1, int(entropy_anneal_iters)))
        except Exception:
            ent_coef = 0.0
        # Warmup LR for first warmup_iters iterations
        if scheduler is not None:
            if step <= max(0, int(warmup_iters)):
                for pg in optimizer.param_groups:
                    pg["lr"] = float(lr) * float(step) / float(max(1, int(warmup_iters)))
        # Determine effective value weight (warmup for early global iterations)
        eff_value_weight = value_weight
        try:
            if int(value_warmup_iters) > 0 and it <= int(value_warmup_iters):
                eff_value_weight = float(value_warmup_weight)
        except Exception:
            eff_value_weight = value_weight

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
                value_weight=eff_value_weight,
                return_weight=return_weight if use_return_loss else 0.0,
                grad_clip=grad_clip,
                scaler=scaler,
                entropy_coef=ent_coef,
                max_policy_loss=divergence_policy_loss,
                max_total_loss=divergence_total_loss,
            )
            if float(stats.get("skipped", 0.0)) > 0.5:
                skipped_batches += 1
                # Back off LR when a divergent batch is detected.
                old_lr = None
                new_lr = None
                try:
                    for pg in optimizer.param_groups:
                        cur = float(pg.get("lr", lr))
                        old_lr = cur if old_lr is None else old_lr
                        nxt = max(float(divergence_min_lr), cur * float(divergence_lr_backoff))
                        pg["lr"] = nxt
                        new_lr = nxt
                except Exception:
                    pass
                print(
                    f"[Guard] Skipped divergent batch {bi+1}/{train_batches_per_iter} "
                    f"(policy={stats.get('policy_loss', float('nan')):.4f}, "
                    f"loss={stats.get('loss', float('nan')):.4f}); "
                    f"lr {old_lr if old_lr is not None else float('nan'):.2e} -> "
                    f"{new_lr if new_lr is not None else float('nan'):.2e}"
                )
                if skipped_batches >= int(divergence_max_skips_per_iter):
                    print(f"[Guard] Reached {skipped_batches} skipped batches in iter {it}")
                    if divergence_rollback:
                        try:
                            model.load_state_dict(pre_train_model_sd)
                            if pre_train_opt_sd is not None:
                                optimizer.load_state_dict(pre_train_opt_sd)  # type: ignore[arg-type]
                                _configure_optimizer_for_device(optimizer, device)
                                try:
                                    for state in optimizer.state.values():  # type: ignore[attr-defined]
                                        for k, v in list(state.items()):
                                            if isinstance(v, torch.Tensor):
                                                state[k] = v.to(device)
                                except Exception:
                                    pass
                            else:
                                try:
                                    optimizer.state.clear()  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                            print(f"[Guard] Rolled back to pre-train snapshot for iter {it}")
                        except Exception as e:
                            print(f"[Guard] Rollback failed at iter {it}: {e}")
                    break
                continue
            losses.append(stats["loss"])
            pol_losses.append(stats["policy_loss"])
            val_losses.append(stats["value_loss"])
            ret_losses.append(stats.get("return_loss", 0.0))
            interval = max(1, train_batches_per_iter // 5)
            if ((bi + 1) % interval == 0) or (bi + 1 == train_batches_per_iter):
                # Also report LR and entropy coef occasionally
                try:
                    cur_lr = optimizer.param_groups[0]["lr"]
                except Exception:
                    cur_lr = lr
                print(f"[Iter {it}] Train {bi+1}/{train_batches_per_iter} loss={stats['loss']:.4f} lr={cur_lr:.2e} ent={ent_coef:.4f}")

        if losses:
            avg_loss = float(np.mean(losses))
            avg_pl = float(np.mean(pol_losses))
            avg_vl = float(np.mean(val_losses))
            avg_rl = float(np.mean(ret_losses))
        else:
            avg_loss = 0.0
            avg_pl = 0.0
            avg_vl = 0.0
            avg_rl = 0.0
            if skipped_batches > 0:
                print(f"[Guard] No optimizer steps applied in iter {it} (all guarded/skipped)")
        avg_steps = float(np.mean(step_counts)) if step_counts else 0.0

        # Evaluate vs random (and greedy for extra signal)
        print(f"[Iter {it}] Eval: {eval_games} games vs random")
        if log_policy_diagnostics:
            win_rate, _diag_rand = evaluate_vs_random(
                model,
                games=eval_games,
                mcts_simulations=mcts_simulations,  # full sims for eval
                device=device,
                mcts_batch=mcts_batch,
                return_diagnostics=True,
            )
        else:
            win_rate = evaluate_vs_random(
                model,
                games=eval_games,
                mcts_simulations=mcts_simulations,  # full sims for eval
                device=device,
                mcts_batch=mcts_batch,
            )
        print(f"[Iter {it}] Eval: {eval_games} games vs greedy")
        if log_policy_diagnostics:
            win_g, margin_g, len_g, diag_g = evaluate_vs_greedy(
                model,
                games=eval_games,
                mcts_simulations=mcts_simulations,  # full sims for eval
                device=device,
                mcts_batch=mcts_batch,
                return_diagnostics=True,
            )
        else:
            win_g, margin_g, len_g = evaluate_vs_greedy(
                model,
                games=eval_games,
                mcts_simulations=mcts_simulations,  # full sims for eval
                device=device,
                mcts_batch=mcts_batch,
            )
            diag_g = _finalize_eval_policy_diag(_new_eval_policy_diag())

        # Save checkpoint (full + model-only for safe arena/ladder loading)
        ckpt_path = os.path.join(ckpt_dir, f"az_iter_{it}.pt")
        ckpt_model_path = os.path.join(ckpt_dir, f"az_iter_{it}_model.pt")
        model_sd_cpu = _model_state_dict_cpu(model)
        torch.save({
            "model": model_sd_cpu,
            "optimizer": optimizer.state_dict(),
            "iter": it,
            "buffer_size": buffer.size(),
        }, ckpt_path)
        torch.save({
            "model": model_sd_cpu,
            "iter": it,
        }, ckpt_model_path)

        # Elo ladder evaluation (optional)
        if use_elo_ladder:
            try:
                elo_path = os.path.join(log_dir, "elo_pool.json")
                elo_log = os.path.join(log_dir, "elo_log.csv")
                pool = _load_elo_pool(elo_path)
                # Index by path for quick lookup
                pool_by_path = {p.get("path"): p for p in pool if isinstance(p, dict)}
                cand_entry = pool_by_path.get(ckpt_model_path, {"path": ckpt_model_path, "elo": float(ladder_base_elo), "iter": it})
                cand_elo = float(cand_entry.get("elo", ladder_base_elo))
                # If empty pool, seed it and skip eval
                if not pool:
                    pool.append(cand_entry)
                    _save_elo_pool(elo_path, pool)
                    print(f"[Elo] Initialized pool with {ckpt_model_path}")
                else:
                    # Select opponents
                    opp_pool = [p for p in pool if p.get("path") != ckpt_model_path]
                    if opp_pool:
                        k = min(int(ladder_opponents), len(opp_pool))
                        opponents = random.sample(opp_pool, k=k)
                        avg_wr = 0.0
                        # Prepare CSV log
                        newfile = not os.path.exists(elo_log)
                        with open(elo_log, "a", newline="") as ef:
                            w = csv.writer(ef)
                            if newfile:
                                w.writerow(["iter", "cand_path", "opp_path", "games", "win_rate", "cand_elo_before", "opp_elo_before", "cand_elo_after", "opp_elo_after"])
                            for opp in opponents:
                                opp_path = opp.get("path")
                                opp_elo = float(opp.get("elo", ladder_base_elo))
                                opp_model = _load_ckpt_model(opp_path, device, input_size, action_size, width=width, res_blocks=res_blocks)
                                if opp_model is None:
                                    continue
                                wr, _margin = arena_vs_model(model, opp_model, games=int(ladder_games), sims=int(ladder_sims), device=device, mcts_batch=mcts_batch)
                                expected = 1.0 / (1.0 + 10 ** ((opp_elo - cand_elo) / 400.0))
                                cand_elo_before = cand_elo
                                opp_elo_before = opp_elo
                                cand_elo = cand_elo + float(ladder_k) * (wr - expected)
                                opp_elo = opp_elo + float(ladder_k) * ((1.0 - wr) - (1.0 - expected))
                                opp["elo"] = float(opp_elo)
                                avg_wr += float(wr)
                                w.writerow([it, ckpt_model_path, opp_path, int(ladder_games), f"{wr:.4f}", f"{cand_elo_before:.2f}", f"{opp_elo_before:.2f}", f"{cand_elo:.2f}", f"{opp_elo:.2f}"])
                        avg_wr = avg_wr / max(1, len(opponents))
                        # Promote into pool if strong enough (or update existing)
                        cand_entry["elo"] = float(cand_elo)
                        cand_entry["iter"] = int(it)
                        if ckpt_model_path in pool_by_path:
                            pool_by_path[ckpt_model_path].update(cand_entry)
                        elif avg_wr >= float(ladder_promote_threshold):
                            pool.append(cand_entry)
                            print(f"[Elo] Promoted to pool (avg win={avg_wr:.2%}, elo={cand_elo:.1f})")
                        else:
                            print(f"[Elo] Not promoted (avg win={avg_wr:.2%}, elo={cand_elo:.1f})")
                        # Enforce pool size
                        pool = sorted(pool, key=lambda p: float(p.get("elo", ladder_base_elo)), reverse=True)
                        if len(pool) > int(ladder_pool_size):
                            pool = pool[: int(ladder_pool_size)]
                        _save_elo_pool(elo_path, pool)
            except Exception as e:
                print(f"[Elo] ladder failed: {e}")

        # Optional: gate vs champion pool
        if gate_pool and int(it) >= int(gate_start_iter):
            try:
                champ_path = os.path.join(ckpt_dir, "champion.pt")
                # If no champion yet, set current as champion
                if not os.path.exists(champ_path):
                    torch.save({"model": _model_state_dict_cpu(model), "iter": it}, champ_path)
                    print(f"[Gate] Set initial champion at iter {it}")
                else:
                    dummy_state = setup_game(num_players=2)
                    input_size2 = len(flatten_game_state(dummy_state))
                    opp_model = _load_ckpt_model(champ_path, device, input_size2, action_size, width=width, res_blocks=res_blocks)
                    if opp_model is not None:
                        wr, margin = arena_vs_model(model, opp_model, games=gate_games, sims=max(16, mcts_simulations // 2), device=device, mcts_batch=mcts_batch)
                        print(f"[Gate] vs champion: win%={wr:.1%} margin={margin:.2f}")
                        # Log gating result
                        try:
                            os.makedirs(log_dir, exist_ok=True)
                            gpath = os.path.join(log_dir, "gating_log.csv")
                            newfile = not os.path.exists(gpath)
                            with open(gpath, "a", newline="") as gf:
                                import csv as _csv
                                w = _csv.writer(gf)
                                if newfile:
                                    w.writerow(["iter", "games", "win_rate", "margin"])
                                w.writerow([it, gate_games, f"{wr:.4f}", f"{margin:.3f}"])
                        except Exception:
                            pass
                        if wr >= gate_threshold:
                            torch.save({"model": _model_state_dict_cpu(model), "iter": it}, champ_path)
                            print(f"[Gate] Promoted iter {it} to champion (>= {gate_threshold:.0%})")
                        elif gate_revert_on_reject:
                            try:
                                champ_ck = _load_checkpoint_cpu_safe(champ_path)
                                model.load_state_dict(champ_ck.get("model", champ_ck))
                                # Reset optimizer state after hard revert to avoid stale momentum.
                                try:
                                    optimizer.state.clear()  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                                print(f"[Gate] Rejected iter {it}; reverted to champion weights")
                            except Exception as ge:
                                print(f"[Gate] Failed reverting to champion: {ge}")
            except Exception as e:
                print(f"[Gate] gating failed: {e}")
        elif gate_pool and int(it) < int(gate_start_iter):
            print(f"[Gate] Warmup skip at iter {it} (gate starts at iter {int(gate_start_iter)})")

        # Log CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "iter", "buffer", "avg_steps", "loss", "policy_loss", "value_loss", "return_loss",
                    "win_rand", "win_greedy", "margin_g", "len_g",
                    "diag_g_legal_n", "diag_g_top1_legal", "diag_g_take", "diag_g_buy_vis", "diag_g_buy_res", "diag_g_reserve",
                ])
                write_header = False
            writer.writerow([
                it, buffer.size(), f"{avg_steps:.2f}", f"{avg_loss:.4f}", f"{avg_pl:.4f}", f"{avg_vl:.4f}", f"{avg_rl:.4f}",
                f"{win_rate:.3f}", f"{win_g:.3f}", f"{margin_g:.3f}", f"{len_g:.2f}",
                f"{diag_g.get('legal_count_mean', 0.0):.2f}",
                f"{diag_g.get('top1_legal_mean', 0.0):.3f}",
                f"{diag_g.get('mass_take_mean', 0.0):.3f}",
                f"{diag_g.get('mass_buy_vis_mean', 0.0):.3f}",
                f"{diag_g.get('mass_buy_res_mean', 0.0):.3f}",
                f"{diag_g.get('mass_reserve_mean', 0.0):.3f}",
            ])

        print(
            f"Iter {it:02d} | buffer={buffer.size()} steps={avg_steps:.1f} "
            f"loss={avg_loss:.4f} pol={avg_pl:.4f} val={avg_vl:.4f} ret={avg_rl:.4f} "
            f"win%_rand={win_rate:.1%} win%_greedy={win_g:.1%} margin_g={margin_g:.2f} len_g={len_g:.1f} "
            f"diag_g(legal={diag_g.get('legal_count_mean', 0.0):.1f}, top1={diag_g.get('top1_legal_mean', 0.0):.2f}, "
            f"mass t/bv/br/rv={diag_g.get('mass_take_mean', 0.0):.2f}/"
            f"{diag_g.get('mass_buy_vis_mean', 0.0):.2f}/"
            f"{diag_g.get('mass_buy_res_mean', 0.0):.2f}/"
            f"{diag_g.get('mass_reserve_mean', 0.0):.2f})"
        )

        # Print only a short tail of the CSV log for readability
        try:
            if int(log_print_tail) > 0:
                with open(log_path, "r") as f:
                    lines = [ln.rstrip("\n") for ln in f]
                if lines:
                    header = lines[0]
                    tail_n = max(1, int(log_print_tail))
                    tail = lines[-tail_n:]
                    print("[TrainLogTail]")
                    print(header)
                    for ln in tail:
                        print(ln)
        except Exception:
            pass

        # Periodic big evaluation (low-variance)
        try:
            if int(big_eval_every) > 0 and (it % int(big_eval_every) == 0):
                print(f"[BigEval] Iter {it}: running {big_eval_games} games (random + greedy)")
                big_wr = evaluate_vs_random(
                    model,
                    games=int(big_eval_games),
                    mcts_simulations=mcts_simulations,
                    device=device,
                    mcts_batch=mcts_batch,
                )
                big_wg, big_margin, big_len = evaluate_vs_greedy(
                    model,
                    games=int(big_eval_games),
                    mcts_simulations=mcts_simulations,
                    device=device,
                    mcts_batch=mcts_batch,
                )
                with open(big_eval_path, "a", newline="") as bf:
                    bw = csv.writer(bf)
                    if big_write_header:
                        bw.writerow(["iter", "games", "win_rand", "win_greedy", "margin_g", "len_g"])
                        big_write_header = False
                    bw.writerow([it, int(big_eval_games), f"{big_wr:.3f}", f"{big_wg:.3f}", f"{big_margin:.3f}", f"{big_len:.2f}"])
                print(
                    f"[BigEval] iter={it} games={int(big_eval_games)} "
                    f"rand={big_wr:.1%} greedy={big_wg:.1%} margin_g={big_margin:.2f} len_g={big_len:.1f}"
                )
                # Confirm save location and echo the CSV row for easy copying
                try:
                    print(f"[BigEvalSaved] {big_eval_path}")
                    print(f"iter,games,win_rand,win_greedy,margin_g,len_g")
                    print(f"{it},{int(big_eval_games)},{big_wr:.3f},{big_wg:.3f},{big_margin:.3f},{big_len:.2f}")
                except Exception:
                    pass
        except Exception as e:
            print(f"[BigEval] failed: {e}")

        # Step cosine scheduler after optimizer steps (avoids PyTorch order warning).
        if scheduler is not None and step > max(0, int(warmup_iters)) and len(losses) > 0:
            try:
                scheduler.step()
            except Exception:
                pass

    return model


if __name__ == "__main__":
    # Auto-select best available device: CUDA > DirectML > CPU
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("[Device] Using CUDA GPU")
        # CUDA-friendly default (e.g., T4 16GB): single GPU worker, larger mcts_batch
        # Rationale: fewer train batches (16) reduce early overfit; eval_games=50 stabilizes the eval signal
        az_train(
            iterations=10,
            games_per_iter=48,
            mcts_simulations=256,
            mcts_batch=96,
            lr=5e-4,
            device=dev,
            replay_capacity=50000,
            batch_size=512,
            train_batches_per_iter=16,
            eval_games=50,
            resume=False,
            weight_decay=1e-4,
            grad_clip=1.0,
            policy_weight=1.0,
            value_weight=1.0,
            value_warmup_iters=3,
            value_warmup_weight=1.5,
            temp_init=1.0,
            temp_final=0.0,
            temp_moves=25,
            compile_model=False,
            res_blocks=6,
            width=512,
            selfplay_workers=1,
            selfplay_device='cuda',
            warmup_iters=4,
            entropy_init=0.02,
            entropy_anneal_iters=12,
        )
    elif DML_DEVICE is not None:
        dev = DML_DEVICE
        print("[Device] Using DirectML (AMD/Intel GPU)")
        # DML-friendly: sequential self-play on CPU-like path, moderate batch
        az_train(
            iterations=10,
            games_per_iter=60,
            mcts_simulations=256,
            mcts_batch=32,
            lr=5e-4,
            device=dev,
            replay_capacity=50000,
            batch_size=512,
            train_batches_per_iter=8,
            eval_games=12,
            resume=True,
            weight_decay=1e-4,
            grad_clip=1.0,
            policy_weight=1.0,
            value_weight=1.0,
            value_warmup_iters=3,
            value_warmup_weight=1.5,
            temp_init=1.0,
            temp_final=0.0,
            temp_moves=30,
            compile_model=False,
            res_blocks=4,
            width=512,
            selfplay_workers=0,
            warmup_iters=4,
            entropy_init=0.04,
            entropy_anneal_iters=20,
        )
    else:
        dev = "cpu"
        print("[Device] Using CPU")
        # CPU-friendly: small mcts_batch and fewer train batches
        az_train(
            iterations=10,
            games_per_iter=16,
            mcts_simulations=128,
            mcts_batch=32,
            lr=5e-4,
            device=dev,
            replay_capacity=20000,
            batch_size=256,
            train_batches_per_iter=12,
            eval_games=10,
            resume=False,
            weight_decay=1e-4,
            grad_clip=1.0,
            policy_weight=1.0,
            value_weight=1.0,
            temp_init=1.0,
            temp_final=0.0,
            temp_moves=20,
            compile_model=False,
            res_blocks=6,
            width=512,
            selfplay_workers=0,
            warmup_iters=4,
            entropy_init=0.02,
            entropy_anneal_iters=12,
        )

