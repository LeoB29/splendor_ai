import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    model_pat = re.compile(r"az_iter_(\d+)_model\.pt$")
    full_pat = re.compile(r"az_iter_(\d+)\.pt$")

    model_hits: List[tuple[int, Path]] = []
    full_hits: List[tuple[int, Path]] = []
    for p in ckpt_dir.glob("*.pt"):
        m = model_pat.search(p.name)
        if m:
            model_hits.append((int(m.group(1)), p))
            continue
        m = full_pat.search(p.name)
        if m:
            full_hits.append((int(m.group(1)), p))

    if model_hits:
        return sorted(model_hits, key=lambda t: t[0])[-1][1]
    if full_hits:
        return sorted(full_hits, key=lambda t: t[0])[-1][1]
    return None


def _safe_elo_from_winrate(w: float) -> Optional[float]:
    # Elo difference vs a fixed reference implied by win rate.
    if w <= 0.0 or w >= 1.0:
        return None
    return float(400.0 * math.log10(w / (1.0 - w)))


def _extract_json(stdout: str) -> Dict[str, Any]:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            return json.loads(ln)
    raise RuntimeError("Could not find JSON payload in subprocess output.")


def _run_repo_eval(
    repo_path: Path,
    ckpt_path: Path,
    seeds: List[int],
    games_random: int,
    games_greedy: int,
    mcts_simulations: int,
    mcts_batch: int,
    max_moves: int,
    device: str,
) -> Dict[str, Any]:
    payload = {
        "ckpt": str(ckpt_path),
        "seeds": seeds,
        "games_random": int(games_random),
        "games_greedy": int(games_greedy),
        "mcts_simulations": int(mcts_simulations),
        "mcts_batch": int(mcts_batch),
        "max_moves": int(max_moves),
        "device": str(device),
    }

    code = r'''
import json
import os
import random
import numpy as np
import torch

from cards_init import setup_game
from nn_input_output import flatten_game_state
from alpha_zero import PolicyValueNet, evaluate_vs_random, evaluate_vs_greedy, _allowlist_checkpoint_globals

cfg = json.loads(os.environ["BENCH_CFG"])


def _load_checkpoint(path: str):
    p = path
    ck = None
    try:
        ck = torch.load(p, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        ck = torch.load(p, map_location="cpu")
    except Exception:
        try:
            _allowlist_checkpoint_globals()
            ck = torch.load(p, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
        except Exception:
            ck = torch.load(p, map_location="cpu", weights_only=False)  # type: ignore[call-arg]

    if isinstance(ck, dict) and "model" in ck:
        sd = ck["model"]
    else:
        sd = ck
    if not isinstance(sd, dict):
        raise RuntimeError("Unsupported checkpoint payload type")
    return sd


def _infer_model_dims(sd: dict):
    width = 512
    w = sd.get("policy_global.weight", None)
    if isinstance(w, torch.Tensor) and w.dim() == 2:
        width = int(w.shape[1])

    block_ids = set()
    for k in sd.keys():
        if not isinstance(k, str):
            continue
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_ids.add(int(parts[1]))
    n_blocks = (max(block_ids) + 1) if block_ids else 4
    return width, n_blocks


sd = _load_checkpoint(cfg["ckpt"])
width, n_blocks = _infer_model_dims(sd)

s0 = setup_game(num_players=2)
in_size = len(flatten_game_state(s0))
model = PolicyValueNet(input_size=in_size, action_size=43, width=width, n_blocks=n_blocks)
try:
    model.load_state_dict(sd)
except Exception:
    model.load_state_dict(sd, strict=False)
model.eval()

seeds = [int(x) for x in cfg["seeds"]]
metrics = []
for s in seeds:
    random.seed(s)
    np.random.seed(s % (2**32 - 1))
    torch.manual_seed(s)
    wr_rand = float(evaluate_vs_random(
        model,
        games=int(cfg["games_random"]),
        mcts_simulations=int(cfg["mcts_simulations"]),
        device=cfg["device"],
        mcts_batch=int(cfg["mcts_batch"]),
        max_moves=int(cfg["max_moves"]),
    ))

    s2 = s + 10_000
    random.seed(s2)
    np.random.seed(s2 % (2**32 - 1))
    torch.manual_seed(s2)
    wr_greedy, margin_g, len_g = evaluate_vs_greedy(
        model,
        games=int(cfg["games_greedy"]),
        mcts_simulations=int(cfg["mcts_simulations"]),
        device=cfg["device"],
        mcts_batch=int(cfg["mcts_batch"]),
        max_moves=int(cfg["max_moves"]),
    )
    metrics.append({
        "seed": int(s),
        "win_rand": float(wr_rand),
        "win_greedy": float(wr_greedy),
        "margin_g": float(margin_g),
        "len_g": float(len_g),
    })

arr_wr_r = np.array([m["win_rand"] for m in metrics], dtype=np.float64)
arr_wr_g = np.array([m["win_greedy"] for m in metrics], dtype=np.float64)
arr_mg = np.array([m["margin_g"] for m in metrics], dtype=np.float64)
arr_lg = np.array([m["len_g"] for m in metrics], dtype=np.float64)

result = {
    "checkpoint": cfg["ckpt"],
    "width": int(width),
    "res_blocks": int(n_blocks),
    "seeds": seeds,
    "games_random": int(cfg["games_random"]),
    "games_greedy": int(cfg["games_greedy"]),
    "mcts_simulations": int(cfg["mcts_simulations"]),
    "mcts_batch": int(cfg["mcts_batch"]),
    "device": cfg["device"],
    "win_rand_mean": float(arr_wr_r.mean()) if len(arr_wr_r) else 0.0,
    "win_rand_std": float(arr_wr_r.std()) if len(arr_wr_r) else 0.0,
    "win_greedy_mean": float(arr_wr_g.mean()) if len(arr_wr_g) else 0.0,
    "win_greedy_std": float(arr_wr_g.std()) if len(arr_wr_g) else 0.0,
    "margin_g_mean": float(arr_mg.mean()) if len(arr_mg) else 0.0,
    "len_g_mean": float(arr_lg.mean()) if len(arr_lg) else 0.0,
}
print(json.dumps(result))
'''

    env = os.environ.copy()
    env["BENCH_CFG"] = json.dumps(payload)

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_path),
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark subprocess failed in {repo_path}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return _extract_json(proc.stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare attention and baseline Splendor agents on fixed-seed evals.")
    here = Path(__file__).resolve().parent
    ap.add_argument("--attention-repo", type=Path, default=here)
    ap.add_argument("--baseline-repo", type=Path, default=here.parent / "splendor_ai_baseline")
    ap.add_argument("--attention-ckpt", type=Path, default=None)
    ap.add_argument("--baseline-ckpt", type=Path, default=None)
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--games-random", type=int, default=20)
    ap.add_argument("--games-greedy", type=int, default=20)
    ap.add_argument("--mcts-simulations", type=int, default=128)
    ap.add_argument("--mcts-batch", type=int, default=16)
    ap.add_argument("--max-moves", type=int, default=250)
    ap.add_argument("--device", type=str, default="cpu", help="cpu | cuda | dml_device_string")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("No valid seeds provided.")

    att_repo = args.attention_repo.resolve()
    base_repo = args.baseline_repo.resolve()

    att_ckpt = args.attention_ckpt.resolve() if args.attention_ckpt else _find_latest_checkpoint(att_repo / "checkpoints")
    base_ckpt = args.baseline_ckpt.resolve() if args.baseline_ckpt else _find_latest_checkpoint(base_repo / "checkpoints")

    if att_ckpt is None or not att_ckpt.exists():
        raise SystemExit(f"Could not find attention checkpoint in {att_repo / 'checkpoints'}")
    if base_ckpt is None or not base_ckpt.exists():
        raise SystemExit(f"Could not find baseline checkpoint in {base_repo / 'checkpoints'}")

    print(f"[Benchmark] Attention checkpoint: {att_ckpt}")
    print(f"[Benchmark] Baseline  checkpoint: {base_ckpt}")
    print(f"[Benchmark] Seeds={seeds} games_random={args.games_random} games_greedy={args.games_greedy}")

    att = _run_repo_eval(
        repo_path=att_repo,
        ckpt_path=att_ckpt,
        seeds=seeds,
        games_random=args.games_random,
        games_greedy=args.games_greedy,
        mcts_simulations=args.mcts_simulations,
        mcts_batch=args.mcts_batch,
        max_moves=args.max_moves,
        device=args.device,
    )

    base = _run_repo_eval(
        repo_path=base_repo,
        ckpt_path=base_ckpt,
        seeds=seeds,
        games_random=args.games_random,
        games_greedy=args.games_greedy,
        mcts_simulations=args.mcts_simulations,
        mcts_batch=args.mcts_batch,
        max_moves=args.max_moves,
        device=args.device,
    )

    delta_wr_r = att["win_rand_mean"] - base["win_rand_mean"]
    delta_wr_g = att["win_greedy_mean"] - base["win_greedy_mean"]
    delta_margin_g = att["margin_g_mean"] - base["margin_g_mean"]

    att_elo_rand = _safe_elo_from_winrate(float(att["win_rand_mean"]))
    base_elo_rand = _safe_elo_from_winrate(float(base["win_rand_mean"]))
    att_elo_greedy = _safe_elo_from_winrate(float(att["win_greedy_mean"]))
    base_elo_greedy = _safe_elo_from_winrate(float(base["win_greedy_mean"]))

    delta_elo_rand = None if (att_elo_rand is None or base_elo_rand is None) else (att_elo_rand - base_elo_rand)
    delta_elo_greedy = None if (att_elo_greedy is None or base_elo_greedy is None) else (att_elo_greedy - base_elo_greedy)

    report = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "seeds": seeds,
        "config": {
            "games_random": args.games_random,
            "games_greedy": args.games_greedy,
            "mcts_simulations": args.mcts_simulations,
            "mcts_batch": args.mcts_batch,
            "max_moves": args.max_moves,
            "device": args.device,
        },
        "attention": att,
        "baseline": base,
        "delta": {
            "win_rand_mean": delta_wr_r,
            "win_greedy_mean": delta_wr_g,
            "margin_g_mean": delta_margin_g,
            "elo_rand_delta": delta_elo_rand,
            "elo_greedy_delta": delta_elo_greedy,
        },
    }

    print("\n=== Benchmark Summary ===")
    print(f"Attention win% vs random: {att['win_rand_mean']:.3f} | vs greedy: {att['win_greedy_mean']:.3f} | margin_g: {att['margin_g_mean']:.3f}")
    print(f"Baseline  win% vs random: {base['win_rand_mean']:.3f} | vs greedy: {base['win_greedy_mean']:.3f} | margin_g: {base['margin_g_mean']:.3f}")
    print(f"Delta     win% vs random: {delta_wr_r:+.3f} | vs greedy: {delta_wr_g:+.3f} | margin_g: {delta_margin_g:+.3f}")
    if delta_elo_rand is not None:
        print(f"Delta Elo proxy (random): {delta_elo_rand:+.1f}")
    else:
        print("Delta Elo proxy (random): N/A (winrate at boundary 0 or 1)")
    if delta_elo_greedy is not None:
        print(f"Delta Elo proxy (greedy): {delta_elo_greedy:+.1f}")
    else:
        print("Delta Elo proxy (greedy): N/A (winrate at boundary 0 or 1)")

    out_path = args.out
    if out_path is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = here / "logs" / f"benchmark_attention_vs_baseline_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
