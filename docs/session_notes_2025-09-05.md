# Splendor AI — Session Notes (2025-09-05)

## Summary
- Implemented batched MCTS leaf evaluation and tree reuse to improve search efficiency.
- Hardened checkpoint resume (safe CPU load, device move, weights_only when supported).
- Updated AMP usage to new `torch.amp` API with CUDA fallback.
- Improved DirectML (AMD) stability/perf: disabled foreach/fused where needed; reduced CPU fallbacks.
- Added lightweight progress logs for self-play, training, and eval.
- Added eval safeguards to prevent rare stalls (max move cap, consecutive-pass break).
- Added a quick runner `run_fast.py` for fast sanity checks (CPU/GPU).
- Added debug snapshot when a position has no legal actions to help diagnose rules/engine gaps.

## Key Changes
- AlphaZero MCTS
  - Batched NN evals: queue leaves and evaluate in batches (`mcts_batch`, default 16; set 32–64 on CUDA).
  - Tree reuse: re-root after chosen action; keeps visit stats between moves.
- Training/Resume
  - Checkpoint loading via CPU + device remap; move optimizer state tensors to target device.
  - Use `torch.load(..., weights_only=True)` when available to avoid pickle warning.
  - Optimizer (Adam): disable foreach/fused/capturable on non-CUDA to avoid DirectML CPU fallbacks.
- Mixed Precision
  - Prefer `torch.amp.GradScaler('cuda')` and `torch.amp.autocast('cuda')`, fall back to `torch.cuda.amp` if needed.
- Logging & UX
  - Progress prints during self-play, training (every ~20%), and eval (every ~10%).
  - Eval guard: `max_moves=250` and consecutive-pass handling to break rare stalemates.
- Utilities
  - `run_fast.py`: minimal 1-iter config that runs quickly on CPU/GPU.
  - Debug: one-time snapshot print when `get_legal_actions()` returns empty (bank, player tokens, board/deck sizes, etc.).

## How To Run
- Normal training (longer): `python alpha_zero.py`
  - Defaults (example in `__main__`): `iterations=10, games_per_iter=8, mcts_simulations=96, mcts_batch=32, eval_games=50`.
- Quick sanity (CPU/GPU): `python run_fast.py`
  - Uses `iterations=1, games_per_iter=2, mcts_simulations=64, eval_games=5`, etc.

## Colab (CUDA) Tips
- Clone repo into Drive; set runtime GPU; run `!python alpha_zero.py` from Drive folder so checkpoints/logs persist.
- Recommended CUDA settings: `mcts_batch=32–64`, `batch_size=512–1024`, `compile_model=True`.
- Reduce `eval_games` per iter (e.g., 10–20) and run a large eval periodically to cut iteration time.

## DirectML (AMD) Notes
- Expect small VRAM usage (model is small; workload is MCTS/CPU-bound). This is normal.
- We disable foreach/fused paths in Adam to avoid CPU fallbacks; warnings should be reduced.

## Files Touched Today
- `alpha_zero.py`: batched MCTS, tree reuse, AMP/optimizer/resume fixes, progress logs, eval guards.
- `run_fast.py`: new quick runner.
- `game_state.py`: debug snapshot on empty legal actions.
- `docs/session_notes.md`: updated Latest link.

## Open TODOs / Next Steps
- Reserve-from-deck rule: allow reserving top card from decks (even without gold) to reduce no-legal-action cases.
- Periodic big eval: small eval per iter + larger eval every N iters (gating ready).
- Batched inference across multiple concurrent games to grow batch sizes further.
- Optional: switch to residual network trunk for strength; add optimizer auto-switch (fused AdamW on CUDA).
- Optional: opponent pool (vs previous checkpoints) and promotion gating.

## Quick Reference
- Checkpoints: `checkpoints/az_iter_#.pt`
- Logs (CSV): `logs/train_log.csv`
- Key params: `mcts_simulations`, `mcts_batch`, `games_per_iter`, `batch_size`, `train_batches_per_iter`, `eval_games`.

