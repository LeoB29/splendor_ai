# AGENTS.md

## Project
- Splendor game engine and AI (MCTS + AlphaZero-style training).
- Rules/state live in `game_state.py`; training pipeline in `alpha_zero.py`.
- Fixed 43-action space is defined in `nn_input_output.py`.
- Refer to `master_notes.md` for change history and rationale.

## Goals
- Improve playing strength and training stability without breaking Splendor rules.
- Keep changes minimal and consistent with the existing action encoding.
- Ultimate objective: beat strong human play and push toward superhuman strength.

## Current Architecture (Attention Branch)
- `PolicyValueNet` in `alpha_zero.py` now uses a token-attention encoder (transformer-style blocks).
- Token set includes:
  - `CLS` token for global decision/value context
  - board card tokens (12)
  - current-player reserved card tokens (3)
  - noble tokens (10)
  - current-player token
  - aggregated-opponent player token
  - bank token
  - aggregated-opponent reserved token
- Policy/value interfaces remain unchanged:
  - same fixed 43-action logits
  - same value scalar
  - same return-choice head API for conditional token-return scoring

## Checkpoint Compatibility
- MLP-era checkpoints from other branches are generally **not shape-compatible** with the attention model.
- For clean attention runs, prefer `resume=False` or resume only from checkpoints created by this attention architecture.

## Common Commands (PowerShell)
- `python -m venv .venv`
- `.venv\Scripts\python -m pip install -r requirements.txt`
- `.venv\Scripts\python run_fast.py`
- `.venv\Scripts\python gui.py`
- `.venv\Scripts\python -m pytest -q`

## Quick Validation Commands
- `python -m py_compile alpha_zero.py train_dml.py run_fast.py run_fast_cuda.py train_cuda.py`
- `python -m pytest -q test_nn_io.py`
- Tiny end-to-end sanity run:
  - `python -c "from alpha_zero import az_train; az_train(iterations=1,games_per_iter=2,mcts_simulations=8,mcts_batch=4,lr=1e-4,device='cpu',replay_capacity=512,batch_size=64,train_batches_per_iter=1,eval_games=1,resume=False,compile_model=False,use_elo_ladder=False,big_eval_every=0,res_blocks=2,width=128,use_return_loss=True,gate_pool=False)"`

## Device
- Prefer DirectML when available (`torch_directml`) on AMD/Intel.
- Use CUDA profiles on NVIDIA.

## Repo Hygiene
- Checkpoints live in `checkpoints/` and logs in `logs/` (do not commit).
- Avoid touching `backup/` and `docs/` unless asked.

## Attention Branch Scripts
- `train_attention_dml.py`
  - Dedicated DirectML training launcher tuned for the attention architecture.
  - Profiles: `stable` (default) and `progress`.
- `benchmark_attention_vs_baseline.py`
  - Cross-repo benchmark orchestrator (attention vs baseline) using fixed seeds.
  - Compares win rates vs random/greedy and reports deltas plus Elo-proxy deltas.
  - Auto-discovers latest checkpoints in both repos unless explicit paths are provided.

### Benchmark Example
- `python benchmark_attention_vs_baseline.py --seeds 0,1 --games-random 20 --games-greedy 20 --mcts-simulations 128 --mcts-batch 16 --device cpu`

