# Splendor AI �?" Session Notes
Latest session notes: [2025-09-05](session_notes_2025-09-05.md)

Date: keep updated per session

## Summary
- Implemented a minimal AlphaZero-style training loop with self-play and MCTS.
- Added policy+value network, MCTS guided by network priors/values, replay buffer, evaluation vs random, CSV logging, and checkpoints.

## Key Files
- `alpha_zero.py`: PolicyValueNet, AlphaZeroMCTS, self-play, replay buffer, `az_train(...)`, logging, checkpoints.
- `test_alpha_zero_smoke.py`: quick end-to-end smoke test (MCTS �+' self-play �+' one train step).
- Existing helpers reused: `game_state.py`, `nn_input_output.py`, `cards_init.py`.

## How To Run
- From IDE or terminal (with Python):
  - `python alpha_zero.py`
- What it does each iteration:
  - Self-play games �+' push samples to replay buffer
  - Train on mini-batches from buffer
  - Evaluate vs a random agent
  - Save checkpoint and append to CSV log

## Current Training Settings (in `alpha_zero.py` `__main__`)
```
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
)
```

## Latest Results (console output)
```
Iter 01 | buffer=441 steps=73.5 loss=4.2769 pol=3.2762 val=1.0007 win%=66.7%
Iter 02 | buffer=877 steps=72.7 loss=3.8423 pol=2.8359 val=1.0064 win%=83.3%
Iter 03 | buffer=1287 steps=68.3 loss=3.6347 pol=2.6312 val=1.0035 win%=83.3%
Iter 04 | buffer=1711 steps=70.7 loss=3.5237 pol=2.5227 val=1.0010 win%=66.7%
Iter 05 | buffer=2149 steps=73.0 loss=3.4514 pol=2.4467 val=1.0047 win%=83.3%
```
- Interpretation:
  - Policy loss trending down (good early learning signal).
  - Value loss ~1 initially (common with A�1 targets and near-zero predictions); should improve with more training/stronger targets.
  - Win% vs random already > 50% (small eval size; expect noise).

## Artifacts
- Checkpoints: `checkpoints/az_iter_#.pt`
- Logs (CSV): `logs/train_log.csv` with columns: `iter,buffer,avg_steps,loss,policy_loss,value_loss,win_rate`

## Next TODOs
- Add root Dirichlet noise and temperature schedule for better exploration.
- Increase evaluation size (e.g., 50�?"100 games) and add self-play vs previous checkpoint.
- Implement resume-from-checkpoint in `az_train` (load latest `.pt` automatically).
- Add weight decay and optional entropy bonus early in training.
- Expand action encoding to fully cover token-return actions (edge cases) if needed.
- Optional: GPU toggle and batched MCTS evals for speed.

## Quick Resume Instructions
- Open this file at session start to re-prime context.
- Re-run training with: `python alpha_zero.py` (or adjust `az_train(...)` params).
- To compare progress, open `logs/train_log.csv` and plot or inspect changes across runs.

---

## Changelog �?" Recent Updates

- Added automatic resume in `az_train` (loads latest checkpoint and continues iteration numbering and logging).
- Added replay buffer training with configurable `batch_size` and `train_batches_per_iter`.
- Added evaluation vs. random agent; bumped default `eval_games` in `__main__` to 50 to reduce variance of win-rate estimates.
- Introduced Dirichlet root noise in MCTS for self-play exploration (`dir_alpha=0.3`, `dir_eps=0.25`).
- Added temperature scheduling in self-play: higher temperature early moves (`temp_init=1.0`) then greedy later (`temp_final=0.0`) after `temp_moves=20` moves.
- Training stability/knobs: weight decay (`1e-4`), optional gradient clipping (`1.0`), adjustable policy/value loss weights.
- Updated `__main__` defaults for a stronger run: `iterations=10`, `games_per_iter=8`, `mcts_simulations=96`, `batch_size=512`, `train_batches_per_iter=50`, `eval_games=50`.

## Why bump eval_games to 50?

With only a handful of evaluation games, win-rate is very noisy (high variance). Evaluating over 50 games averages out randomness, giving a more reliable signal of progress. The tradeoff is extra time spent on evaluation each iteration.

## What is Dirichlet noise?

We add a small amount of random noise sampled from a Dirichlet distribution to the root policy priors during self-play. This encourages exploration of different opening lines instead of always following early strong priors, which helps the network learn a broader policy. Parameters:
- `dir_alpha`: concentration of the Dirichlet; smaller spreads mass more unevenly.
- `dir_eps`: how much noise to mix into the original priors.

## �?oValue head starts at 0�?? �?" simple explanation

The value head predicts who is likely to win from a position, mapped to the range [-1, 1]. At initialization, the network�?Ts weights are random, so its value outputs are typically near 0 (i.e., �?ounsure/tie�??). Your training targets are A�1 at game end, so the initial mean squared error is roughly (A�1 - 0)^2 = 1. As training progresses and targets become consistent, the value head learns to output numbers closer to A�1 for clearly winning/losing positions, and the value loss drops.
