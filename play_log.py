import os
import re
import argparse
import random
from typing import Optional, Tuple

import numpy as np
import torch

import alpha_zero as az


def auto_device() -> torch.device | str:
    try:
        if torch.cuda.is_available():
            print("[Device] Using CUDA GPU")
            return torch.device("cuda")
    except Exception:
        pass
    # Optional DirectML
    try:
        import torch_directml as _dml  # type: ignore
        print("[Device] Using DirectML")
        return _dml.device()
    except Exception:
        pass
    print("[Device] Using CPU")
    return "cpu"


def _find_latest_checkpoint(ckpt_dir: str = "checkpoints") -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    best_it = -1
    best_path = None
    for fname in os.listdir(ckpt_dir):
        m = re.match(r"az_iter_(\d+)\.pt$", fname)
        if not m:
            continue
        it = int(m.group(1))
        if it > best_it:
            best_it = it
            best_path = os.path.join(ckpt_dir, fname)
    return best_path


def load_latest_model(device: torch.device | str,
                      width: int = 512,
                      res_blocks_try: Tuple[int, ...] = (6, 4)) -> Tuple[az.PolicyValueNet, Optional[str]]:
    # Infer IO sizes from a dummy state
    dummy = az.setup_game(num_players=2)
    input_size = len(az.flatten_game_state(dummy))
    action_size = 43

    path = _find_latest_checkpoint()
    if path is None:
        print("[Warn] No checkpoints found; using randomly initialized model")
        model = az.PolicyValueNet(input_size=input_size, action_size=action_size, width=width, n_blocks=res_blocks_try[0])
        return model.to(device), None

    # Load state dict safely on CPU, handling legacy pickled tensors
    def _load_ckpt(p: str):
        # 1) Try strict safe load
        try:
            return torch.load(p, map_location='cpu', weights_only=True)  # type: ignore[arg-type]
        except Exception:
            pass
        # 2) Allowlist legacy rebuild helper then retry safe load
        try:
            from torch.serialization import add_safe_globals  # type: ignore
            try:
                from torch._utils import _rebuild_device_tensor_from_numpy  # type: ignore
                add_safe_globals([_rebuild_device_tensor_from_numpy])  # type: ignore
                return torch.load(p, map_location='cpu', weights_only=True)  # type: ignore[arg-type]
            except Exception:
                pass
        except Exception:
            pass
        # 3) Fallback to unsafe load (trusted local files only)
        print("[Warn] Falling back to torch.load(weights_only=False). Ensure checkpoint is trusted.")
        return torch.load(p, map_location='cpu')

    ck = _load_ckpt(path)
    # Support multiple common keys: 'model', 'state_dict', 'model_state_dict'
    if isinstance(ck, dict):
        sd = ck.get('model') or ck.get('state_dict') or ck.get('model_state_dict') or ck
    else:
        sd = ck

    last_err = None
    for rb in res_blocks_try:
        try:
            model = az.PolicyValueNet(input_size=input_size, action_size=action_size, width=width, n_blocks=rb)
            model.load_state_dict(sd)
            print(f"[Load] Loaded {path} with res_blocks={rb}")
            return model.to(device).eval(), path
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load checkpoint {path}: {last_err}")


def action_to_str(a) -> str:
    at = getattr(a, 'action_type', 'unknown')
    parts = [at]
    tk = getattr(a, 'tokens_taken', None)
    if tk:
        parts.append(f"take={dict(tk)}")
    tr = getattr(a, 'tokens_returned', None)
    if tr:
        parts.append(f"ret={dict(tr)}")
    tgt = getattr(a, 'target', None)
    if tgt is not None:
        pts = getattr(tgt, 'points', 0)
        bonus = getattr(tgt, 'bonus_color', '')
        parts.append(f"card(pts={pts}, bonus={bonus})")
    tier = getattr(a, 'tier', None)
    if tier is not None:
        parts.append(f"tier={tier}")
    return ' '.join(parts)


def print_scores(state) -> str:
    ps = []
    for i, p in enumerate(state.players):
        ps.append(f"P{i}: pts={p.points} tokens={sum(p.tokens.values())} gold={p.tokens.get('gold',0)}")
    return ' | '.join(ps)


def play_logged(model: az.PolicyValueNet, opponent: str = 'random', games: int = 1,
                sims: int = 256, mcts_batch: int = 32, device: torch.device | str = "cpu",
                seed: Optional[int] = None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)

    for g in range(games):
        state = az.setup_game(num_players=2)
        mcts = az.AlphaZeroMCTS(model, device=device, n_simulations=sims, mcts_batch=mcts_batch)
        my_index = g % 2
        move = 0
        print(f"\n=== Game {g+1}/{games} vs {opponent} (you are P{my_index}) ===")
        print(f"Start: {print_scores(state)}")
        while not state.is_terminal:
            legal = state.get_legal_actions()
            if not legal:
                state.current_player = (state.current_player + 1) % len(state.players)
                print(f"[Move {move}] P{state.current_player} had no legal moves → pass")
                continue
            if state.current_player == my_index:
                _, a_idx = mcts.run(state, temperature=0.0)
                if a_idx == -1:
                    a = legal[0]
                else:
                    a = az.index_to_action(a_idx, state)
                    best_ret = mcts.get_best_tokens_returned(a_idx)
                    if best_ret is not None and hasattr(a, "tokens_returned"):
                        a.tokens_returned = best_ret
                who = f"Model(P{my_index})"
            else:
                if opponent == 'greedy':
                    a = az._greedy_action(state)
                else:
                    a = random.choice(legal)
                who = f"{opponent.capitalize()}(P{state.current_player})"
            desc = action_to_str(a)
            state = state.apply_action(a)
            print(f"[Move {move}] {who}: {desc} -> {print_scores(state)}")
            try:
                if who.startswith("Model") and 'a_idx' in locals() and a_idx != -1:
                    mcts.reuse_after_play(a_idx)
            except Exception:
                pass
            move += 1
        print(f"End: winner=P{state.winner} | final scores: {print_scores(state)}\n")


def main():
    ap = argparse.ArgumentParser(description="Play latest AZ model vs random/greedy with verbose move logs")
    ap.add_argument('--opponent', choices=['random', 'greedy'], default='random')
    ap.add_argument('--games', type=int, default=1)
    ap.add_argument('--sims', type=int, default=256)
    ap.add_argument('--mcts-batch', type=int, default=32)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'dml'], default='auto')
    args = ap.parse_args()

    if args.device == 'auto':
        dev = auto_device()
    elif args.device == 'cuda':
        dev = torch.device('cuda')
    elif args.device == 'dml':
        try:
            import torch_directml as _dml  # type: ignore
            dev = _dml.device()
        except Exception:
            print('[Warn] DirectML not available; using CPU')
            dev = 'cpu'
    else:
        dev = 'cpu'

    model, ckpt = load_latest_model(dev)
    if ckpt is None:
        print('[Info] Running with a fresh random model (no checkpoints found)')
    else:
        print(f"[Using] {ckpt}")

    play_logged(model, opponent=args.opponent, games=args.games, sims=args.sims, mcts_batch=args.mcts_batch, device=dev, seed=args.seed)


if __name__ == '__main__':
    main()
