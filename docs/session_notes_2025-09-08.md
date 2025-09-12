# Splendor AI — Session Notes (2025-09-08)

## Summary
- Added stalemate/pass tie-break logic to scoring in self-play and eval.
- Introduced optional value-loss warmup during early iterations.
- Tuned training defaults for CUDA and DirectML profiles.
- Added a small CLI tool to play and log games vs random/greedy.

## Key Changes
- Self-play / Eval (`alpha_zero.py`)
  - Detect games that end by consecutive passes (no legal moves) and assign a provisional winner by: higher points, then fewest purchased cards; otherwise draw.
  - Applied the same pass-based tie-break in `evaluate_vs_random` and `evaluate_vs_greedy` so eval win rates reflect stalemate outcomes consistently.
- Training (`alpha_zero.py`)
  - New knobs: `value_warmup_iters` and `value_warmup_weight`; when enabled, increases value loss weight for the first N iterations to stabilize early learning signals.
  - CUDA default profile: increased `games_per_iter` (48), larger `replay_capacity` (50k), slightly longer temperature phase (`temp_moves=25`).
  - DirectML profile: increased self-play (`games_per_iter=60`), raised `mcts_simulations=256`, set `resume=True`, reduced `train_batches_per_iter=8`, and `replay_capacity=50k`.
- Tooling
  - New `play_log.py`: loads the latest checkpoint and plays verbose games vs `random` or `greedy`, printing moves, token deltas, and scores. Auto-selects device (`cpu`/`cuda`/DirectML) when possible.

## Notes
- Checkpoints and logs were updated from recent runs; some older checkpoints were pruned.
- No engine/rules changes observed since 2025-09-06; changes are confined to training/eval flow and utilities.

## How To Use
- Quick logged playthroughs:
  - `python play_log.py --opponent greedy --games 1 --sims 256 --mcts-batch 32`
- Run training with updated defaults from `alpha_zero.py` `__main__` (auto-tunes per device):
  - `python alpha_zero.py`

