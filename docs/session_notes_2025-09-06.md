# Splendor AI — Session Notes (2025-09-06)

## Summary
- Stabilized rules/engine and added strong training/eval features.
- Implemented residual Policy+Value network (width=512, 4 blocks).
- Added greedy baseline eval and optional champion gating.
- Fixed token-related edge cases: conservation, cap, and return variants.

## Key Changes
- Engine (`game_state.py`)
  - Guards on transfers: never take more than bank; never return more than player; cap gold spend.
  - Deduped “take 3” actions when <3 colors remain; allow reserve-from-deck (not in NN head).
  - Invariants: clamp negatives; enforce ≤4 visible per tier; remove None cards; reconcile per-color conservation to initial supply; auto-return down to 10 tokens (gold first) as a safety net; optional debug asserts.
  - Debug UX: `[NoLegal]` now includes opponent tokens; debug flag prints integrity issues.
- NN I/O (`nn_input_output.py`)
  - Encoder robust to None and tier overfill; `flatten_visible_cards` limits to 4 per tier.
  - legal mask skips actions outside 43-head (e.g., reserve-from-deck) to avoid index errors.
- MCTS / Training (`alpha_zero.py`)
  - Residual `PolicyValueNet` (512×4). Full-sim evals; auto `mcts_batch` by device.
  - Return-variant enumeration uses effective taken amounts (capped by bank) to compute excess.
  - Greedy baseline: `evaluate_vs_greedy()` returns win%, margin, game length.
  - Optional champion gating: arena vs `checkpoints/champion.pt`; promote on threshold; logs CSV.
  - Cosine LR with warmup; entropy bonus early with anneal; CSV logs extended (rand/greedy/margin/len).
  - Iter-end prints full training CSV for quick visibility.
- GUI (`gui.py`) quick human vs AI (UCT) for rule sanity; action list UI.

## How To Run
- Stronger training defaults in `__main__`:
  - `mcts_simulations=256`, full-sim evals, residual 512×4.
- Fast smoke: `python run_fast.py` (small iter)
- GUI vs AI: `python gui.py`

## Current Signals (examples)
- Early iters: policy loss trending down; value ~0.9–1.0; win% vs random > 80–95% on modest configs.
- Greedy baseline win% and positive margin improving as training proceeds.

## Debugging Notes
- To catch root causes immediately: enable assertions
  - Module-wide: `import game_state as GS; GS.DEBUG_GUARDS = True`
  - Or per game: `state.debug = True` after `setup_game`
- Non-fatal `[Warn] Token cap breach …]` is now checked after invariants; with auto-return it should not appear.

## Next TODOs / Recommendations
- Search: increase sims (512–800 on CUDA), `mcts_batch=64`; parallelize self-play across games.
- Model: consider 6–8 residual blocks after plateau.
- Training: raise `train_batches_per_iter` (80–150) and batch size (512–1024 CUDA); keep cosine LR; entropy anneal.
- Opponents: enable gating (`gate_pool=True`, `gate_games=200`, `gate_threshold≈0.57`) after 5–10 warmup iters; optionally use a small pool of champions.
- Evaluation: add big eval every 5 iters (200–500 games) for stable win%/margin tracking.
- Encoding (later): card-level embeddings/attention for stronger generalization.

## Artifacts
- Checkpoints per iter: `checkpoints/az_iter_#.pt`
- Champion (if gating): `checkpoints/champion.pt`
- Logs: `logs/train_log.csv` (iter, buffer, avg_steps, loss, policy_loss, value_loss, win_rand, win_greedy, margin_g, len_g)

