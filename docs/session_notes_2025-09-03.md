# Splendor AI — Session Notes (2025-09-03)

## Summary
- Added GPU support and safe speed-ups (CUDA + DirectML on Windows).
- Fixed lossy action decoding by attaching valid `tokens_returned` for take/reserve when exceeding the 10-token cap.
- Implemented MCTS branching over top-K token-return variants (approach 2) without enlarging the policy head.
- Applied most-visited return variant during self-play and evaluation to match search behavior.
- Implemented official end-of-round finish with tie-break (highest points, then fewest purchased cards).

## Key Changes
- `alpha_zero.py`
  - Device auto-detect: CUDA > DirectML > CPU.
  - Optional AMP on CUDA; TF32 enabled where supported.
  - New MCTS variant branching keyed by `(action_index, variant_index)` with return enumeration + heuristic.
  - Dirichlet noise now applied in base-action space then redistributed to variants.
  - Records most-visited return for each base action at root; used in self-play and evaluation.
- `nn_input_output.py`
  - `index_to_action(...)` now computes a valid `tokens_returned` when needed (keeps non-MCTS paths legal).
- `game_state.py`
  - End condition updated to finish current round once any player reaches 15+ points.
  - Tie-break on fewest purchased cards when points tie.

## How To Run
- Quick smoke (CPU/GPU auto):
  - `python test_alpha_zero_smoke.py`
- Short training sanity run:
  - `python -c "from alpha_zero import az_train; az_train(iterations=1, games_per_iter=2, mcts_simulations=32, eval_games=10, resume=True)"`
- Full training (default params in `__main__`):
  - `python alpha_zero.py`

## AMD GPU (Windows) via DirectML
- Install backend: `pip install torch-directml`
- Script auto-selects DirectML and prints `[Device] Using DirectML (AMD/Intel GPU)`.
- AMP is CUDA-only; DirectML runs in full precision.

## Notes on Rules & Modeling
- Token cap (10) now consistently respected across legal action generation, decoding, and MCTS execution.
- End-of-round rule aligns closer to official Splendor (winner decided after equal turns).
- MCTS explores multiple token-return variants (top-K, default 3) and aggregates visits per base action for the policy target.
- Execution (self-play/eval) uses the most-visited return for the chosen base action.

## Tunables
- `AlphaZeroMCTS(..., returns_top_k=3)`: change number of return variants explored.
- `az_train(..., compile_model=False)`: enable `torch.compile` (PyTorch 2.x) when stable.
- Evaluation size (`eval_games`) for stability; MCTS sims for training/eval.

## Recent Console Example
```
Iter 11 | buffer=572 steps=71.5 loss=3.2692 pol=2.2721 val=0.9971 win%=82.0%
...
```
(Expect variance to change after rule/return fixes; increase eval_games for steadier signal.)

## Next TODOs
- Refine return scoring (prioritize nearly purchasable cards; focus top-N cheapest).
- Option to disable variant branching (debug comparisons).
- Optional tertiary tie-break rule (e.g., total tokens) if desired.
