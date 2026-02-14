# Master Notes
Log of changes and rationale (most recent first).

## 2026-02-14
### Stability ramp profile for attention training (A1/A2)
- Implemented a new early-iteration stability ramp in `az_train` (`alpha_zero.py`):
  - New knobs:
    - `stability_ramp`
    - `ramp_min_buffer`
    - `ramp_iters`
    - `ramp_batch_frac_start`
    - `ramp_divergence_policy_start`
    - `ramp_divergence_total_start`
    - `ramp_max_skips_start`
  - Behavior when enabled:
    - skips training while replay buffer is below `ramp_min_buffer`
    - ramps train batches from a low fraction to full batches over `ramp_iters`
    - starts with looser divergence thresholds and anneals down to configured strict thresholds
    - starts with a higher skip budget and anneals to the configured budget
  - Added explicit ramp logging each iteration:
    - current ramp progress `t`
    - effective train batch count
    - effective divergence thresholds
    - effective max skip count
- Added a new default launcher profile in `train_attention_dml.py`:
  - `PROFILE = "ramp"` by default
  - New profile name: `ATTENTION-RAMP`
  - Tuned for early stability:
    - lower base LR (`2.0e-5`)
    - `stability_ramp=True`
    - `ramp_min_buffer=12000`
    - `ramp_iters=8`
    - `ramp_batch_frac_start=0.25`
    - `ramp_divergence_policy_start=14.0`
    - `ramp_divergence_total_start=18.0`
    - `ramp_max_skips_start=2`
- Applied the same code changes to all active ablation worktrees to keep comparisons fair:
  - `splendor_ai_attn_a1`
  - `splendor_ai_attn_a2`
  - `splendor_ai_attention` (main attention worktree)

### Live run metadata snapshot (A2 ramp run)
- Captured during active training process:
  - `PID=22080`
  - `start=2026-02-14 12:35:29` (local time)
  - command: `python .\train_attention_dml.py`
- Run launcher/profile signals (A2 worktree):
  - `PROFILE="ramp"` in `train_attention_dml.py`
  - prints `"[Profile] Using ATTENTION-RAMP profile"`
  - includes `stability_ramp=True`, `ramp_min_buffer=12000`, `ramp_iters=8`
- File fingerprints for reproducibility (A2 worktree):
  - `alpha_zero.py` SHA256: `2C2603065603C8D47948E5227961ED45C842C8E6E67B18530D22A5A87AF4BE2F`
  - `train_attention_dml.py` SHA256: `D5D32EC3883C0744ACC00846F98D75C4926726FEFB2C0B7FAC617A4233212E3F`

## 2026-02-12
### Attention-v2: masking, identity embeddings, policy diagnostics
- Implemented three targeted upgrades in `alpha_zero.py`:
  1. **Padding/masking improvements**
     - Extended token padding behavior to cover opponent-aggregate tokens when absent.
     - Dampened empty board/self-reserved token features before policy projection.
     - Applied strong negative bias to structurally invalid token-dependent action logits (empty board/reserved slots).
  2. **Stronger token identity embeddings**
     - Added board-tier and board-slot embeddings (factorized board identity).
     - Added reserved-slot and noble-slot embeddings.
     - Kept action interface fixed at 43 actions.
  3. **Policy diagnostics for eval loops**
     - Added per-move policy diagnostics collection in `evaluate_vs_random` / `evaluate_vs_greedy` (opt-in).
     - `az_train` now logs greedy-eval diagnostics each iteration:
       - `diag_g_legal_n` (mean legal action count)
       - `diag_g_top1_legal` (mean top-1 legal policy probability)
       - family mass means: `diag_g_take`, `diag_g_buy_vis`, `diag_g_buy_res`, `diag_g_reserve`
     - Iter print now includes `diag_g(...)` summary for fast diagnosis of policy collapse/mode bias.
- Resume/load robustness:
  - Switched multiple checkpoint load paths to non-strict loading with warning output on missing/unexpected keys to tolerate incremental architecture evolution.

### Validation run for attention-v2 changes
- `python -m py_compile alpha_zero.py` passed.
- `python test_alpha_zero_smoke.py` passed.
- Added diagnostics-path smoke:
  - `evaluate_vs_random(..., return_diagnostics=True)` and `evaluate_vs_greedy(..., return_diagnostics=True)` execute and return expected metric dicts.
- Tiny end-to-end `az_train` run (1 iter, tiny settings) succeeded and wrote new diagnostic columns to `logs/_tmp_diag_check/train_log.csv`.

### Attention architecture migration (attention branch)
- Replaced the pooled/residual MLP encoder in `alpha_zero.py::PolicyValueNet` with a transformer-style token-attention encoder.
- Added `AttentionBlock` (pre-LN multi-head self-attention + FFN) and stacked it in `PolicyValueNet`.
- New entity-token design:
  - `CLS` token (global context)
  - board cards (12 tokens)
  - current-player reserved cards (3 tokens)
  - nobles (10 tokens)
  - current-player token
  - opponent aggregate player token
  - bank token
  - opponent reserved aggregate token
- Added learned type embeddings and position embeddings for board/reserved/noble slots.
- Preserved all external training/search interfaces:
  - fixed 43-action policy head layout unchanged
  - value head output unchanged
  - return-choice head (`score_return_candidates`) unchanged at API level
  - `return_features=True` still returns `(logits, value, h)` where `h` is now the CLS embedding.

### Why this change
- Splendor state is naturally set-structured with many interacting visible entities.
- Attention allows richer cross-entity interactions than pooled MLP context while keeping the existing action encoding stable.

### Quick validation performed
- `python -m py_compile alpha_zero.py train_dml.py run_fast.py run_fast_cuda.py train_cuda.py` passed.
- `python -m pytest -q test_nn_io.py` passed (`4 passed`).
- Direct forward sanity check passed: policy/value/feature shapes are correct.
- Tiny end-to-end training sanity run passed (`az_train` with 1 iteration, 2 games, tiny MCTS/config), confirming self-play + train + eval loop compatibility with the new attention model.
- Tiny DirectML sanity run also passed on this branch (`az_train` with 1 game / 1 batch), confirming attention forward+train compatibility on your AMD path (with expected Adam `aten::lerp` CPU fallback warning from DirectML).

### Notes
- Updated `test_alpha_zero_smoke.py` for current `compute_targets(...)` return arity; smoke script now passes end-to-end.
- Checkpoints from prior non-attention architecture are generally not shape-compatible; attention runs should start fresh or resume from attention-generated checkpoints.
### Training + Benchmark tooling
- Added `train_attention_dml.py`:
  - Dedicated DirectML launcher for attention architecture.
  - Includes `stable` and `progress` profiles with attention-safe defaults and existing gate/divergence controls.
- Added `benchmark_attention_vs_baseline.py`:
  - Runs reproducible, fixed-seed comparisons between attention and baseline repos.
  - Auto-discovers latest checkpoints (prefers `*_model.pt` when available).
  - Reports win-rate deltas vs random/greedy, margin deltas, and Elo-proxy deltas.
  - Saves full JSON reports in `logs/benchmark_attention_vs_baseline_*.json`.
- Smoke benchmark executed successfully (tiny config):
  - Command: `python benchmark_attention_vs_baseline.py --seeds 0 --games-random 2 --games-greedy 2 --mcts-simulations 16 --mcts-batch 8 --device cpu`
  - Result snapshot: attention and baseline tied on win rate in this tiny test; attention had better greedy margin (`+4.0`).
## 2026-02-10
### Recovery profile hardening + run clarity
- **Stricter RECOVERY defaults** (`train_dml.py`):
  - `lr=3e-5` (down from `5e-5`)
  - `train_batches_per_iter=8` (down from `10`)
  - `entropy_init=0.015` (down from `0.02`)
  - `resume_path=checkpoints\\az_iter_10.pt` (start from cleaner pre-drift point)
  - tighter divergence guards:
    - `divergence_policy_loss=4.3`
    - `divergence_total_loss=5.8`
    - `divergence_max_skips_per_iter=1`
- **Run-time config echo added** (`train_dml.py`):
  - Startup now prints a single `[Config]` line with:
    - `resume/resume_path/resume_optimizer_state`
    - `lr`, `train_batches_per_iter`, `eval_games`
    - divergence thresholds and `gate_start_iter`
  - Purpose: quickly verify the active profile and avoid confusion from stale runs.
- **Scheduler floor bug fixed** (`alpha_zero.py`, `train_dml.py`):
  - Root cause observed in logs: `lr_min` default (`1e-4`) was above recovery `lr` (`3e-5`), which makes cosine scheduling increase LR over time.
  - This caused recurring guard skips/rollbacks around mid-iterations even in recovery mode.
  - Fixes:
    - `az_train` default `lr_min` changed to `1e-6`.
    - Added runtime guard: if `lr_min >= lr`, auto-adjust to `0.1 * lr` and print an `[LR]` notice.
    - `train_dml.py` profiles now set `lr_min=1e-6` explicitly and print it in `[Config]`.

## 2026-02-09
### Plateau mitigation: phased gating + anchor reset
- **Phased champion gating** (`alpha_zero.py`):
  - Added `gate_start_iter` to delay champion/challenger gating until a chosen global iteration.
  - During warmup iterations, gate is skipped with a clear log line.
- **DML training profile updated** (`train_dml.py`):
  - Added startup cleanup of weak gate anchors/logs:
    - `checkpoints/champion.pt`
    - `logs/elo_pool.json`
    - `logs/elo_log.csv`
    - `logs/gating_log.csv`
  - Relaxed and stabilized gating after warmup:
    - `gate_games=128`
    - `gate_threshold=0.52`
    - `gate_start_iter=6`
  - Kept previously added anti-divergence guards and conservative optimizer settings.
  - Updated run policy: `gate_revert_on_reject=False` in `train_dml.py` so learner weights keep improving between iterations even when not promoted to champion.
  - Champion role clarified:
    - Champion remains the quality/reference model for gate evaluation and promotion decisions.
    - Learner training no longer rolls back to champion on rejection; champion is now selection/deployment control, not a hard rollback target.
  - Added explicit profile switch in `train_dml.py`:
    - `PROFILE` switch supports `recovery`, `progress`, `default`.
    - Default currently set to `progress`.
    - Recovery profile uses:
      - `lr=5e-5`, `warmup_iters=0`, `grad_clip=0.4`
      - `eval_games=40`
      - `resume_path=checkpoints\\az_iter_11.pt`
      - tighter guards (`divergence_policy_loss=6.0`, `divergence_total_loss=10.0`, `divergence_max_skips_per_iter=2`)
    - Progress profile uses:
      - `lr=7.5e-5`, `warmup_iters=1`, `grad_clip=0.45`, `train_batches_per_iter=12`
      - `eval_games=40`, same gating as recovery (`gate_threshold=0.52`, `gate_start_iter=6`)
      - moderate guards (`divergence_policy_loss=8.0`, `divergence_total_loss=12.0`)
    - `DEFAULT` profile remains available for reference.
  - Resume behavior hardened:
    - Added `resume_optimizer_state` to `az_train` (default `True`).
    - `train_dml.py` now uses `resume_optimizer_state=False` across profiles to avoid importing stale optimizer momentum when starting a new run from an old checkpoint.
  - LR scheduler warning fix:
    - Moved cosine scheduler stepping to end-of-iteration (after optimizer steps), removing the `lr_scheduler.step()` before `optimizer.step()` warning path.

## 2026-02-08
### Anti-collapse training hardening (implemented)
- **Batch-level divergence guardrails** (`alpha_zero.py`):
  - `train_on_batch(...)` now supports `max_policy_loss` and `max_total_loss` thresholds.
  - If a batch is non-finite or exceeds thresholds, optimizer step is skipped (`skipped=1`) before backprop/step.
- **Automatic LR backoff on bad batches** (`alpha_zero.py`):
  - On each skipped/divergent batch, LR is reduced by `divergence_lr_backoff` with a floor `divergence_min_lr`.
- **Iteration rollback on repeated divergence** (`alpha_zero.py`):
  - Training snapshots model+optimizer state at iteration start.
  - If skipped batches reach `divergence_max_skips_per_iter`, iteration training rolls back to the pre-train snapshot.
- **Champion/challenger safety gate strengthened** (`alpha_zero.py`):
  - Added `gate_revert_on_reject`. When enabled and challenger fails gate threshold, model weights revert to champion for next iteration.
  - Optimizer state is cleared on champion revert to avoid stale momentum from rejected weights.
- **Logging behavior improved** (`alpha_zero.py`):
  - If all train batches are guarded/skipped in an iteration, training losses log as `0.0` (not `nan`) with a guard message.

### DML long-run profile update
- **More conservative long-run defaults** (`train_dml.py`):
  - `lr=1e-4`, `grad_clip=0.5`
  - `entropy_init=0.02`, `entropy_anneal_iters=20`
  - Enabled strict champion/challenger gate:
    - `gate_pool=True`, `gate_games=64`, `gate_threshold=0.55`, `gate_revert_on_reject=True`
  - Enabled divergence settings in profile:
    - `divergence_policy_loss=50.0`, `divergence_total_loss=200.0`
    - `divergence_lr_backoff=0.5`, `divergence_min_lr=1e-6`
    - `divergence_max_skips_per_iter=3`, `divergence_rollback=True`

### Logging hygiene
- **Train log schema migration** (`alpha_zero.py`):
  - Replaced repeated `train_log_v2.csv` fallback with one-time in-place migration of `logs/train_log.csv` to the current schema.
  - Legacy file is automatically backed up as `logs/train_log_legacy*.csv`.

## 2026-02-07
### Step 1 implemented: learned return-choice priors inside MCTS
- **MCTS now uses return-head scores during expansion** (`alpha_zero.py`):
  - `_evaluate_batch()` now requests model features (`return_features=True`) and returns `(priors, values, h_features)`.
  - Added `_return_variant_weights()` to score token-return variants with `score_return_candidates(...)`.
  - Added `return_prior_mix` (default `0.8`) in `AlphaZeroMCTS` to blend learned return priors with heuristic priors for stability.
  - `_expand_with_priors()` now assigns per-variant edge priors from learned/blended weights instead of uniform heuristic split.
  - Current operating choice: keep `return_prior_mix=0.8` for now; treat as a tunable hyperparameter to revisit with targeted ablation later.
- **Root Dirichlet handling improved for variant actions** (`alpha_zero.py`):
  - Noise is still applied at the base-action level.
  - Mixed prior is redistributed across variants while preserving each variant's current proportion.
- **Decision retained**:
  - Reserve-from-deck remains excluded from the fixed 43-action space (no action-space expansion done here).

### Validation
- **Runtime checks passed**:
  - `python -c "import alpha_zero"` succeeded.
  - `python -m py_compile alpha_zero.py run_fast.py run_fast_cuda.py train_cuda.py train_dml.py` succeeded.
- **DirectML smoke run passed**:
  - `python run_fast.py` completed both iterations successfully on DirectML.
  - No `az_train(... use_return_loss=...)` signature error; run finished end-to-end.
  - Noted expected DirectML CPU fallback warning for `aten::lerp.Scalar_out` in Adam (performance warning, not correctness).

### Checkpoint/resume hardening + DML long-run stability
- **Resume fallback fixed** (`alpha_zero.py`):
  - If latest full checkpoint (`az_iter_*.pt`) is not safe-load compatible, training now falls back to latest model-only checkpoint (`az_iter_*_model.pt`).
  - Model-only resume restores weights and iteration number, and intentionally resets optimizer state (`optimizer reset` message).
- **Arena/Elo checkpoint loading hardened** (`alpha_zero.py`):
  - `_load_ckpt_model(...)` now falls back from `*.pt` to `*_model.pt` automatically when available.
  - This reduces Elo ladder failures caused by legacy full checkpoints.
- **Safe-loader allowlist expanded** (`alpha_zero.py`):
  - Added NumPy concrete dtype class allowlisting (e.g., `Float32DType`) for better compatibility with legacy artifacts.
- **DirectML long-run profile stabilized** (`train_dml.py`):
  - Set `lr=2.5e-4`, `grad_clip=0.7`, `return_weight=0.25` to match validated stable behavior seen in smoke runs.

## 2026-02-05
### Model + MCTS correctness and stability
- **Canonical player perspective in state encoding** (`nn_input_output.py`): always encode players from the current player’s viewpoint to remove label ambiguity (same input requiring different outputs). This improves learning stability and sample efficiency.
- **Terminal value fix in MCTS** (`alpha_zero.py`): backup value now reflects actual winner from the current player’s perspective instead of a constant loss. Prevents search from preferring losing terminal lines.
- **MCTS tree reuse hygiene** (`alpha_zero.py`):
  - Added `reset_root()` and used it whenever the state changes without a corresponding encoded MCTS action (passes, opponent/random moves, unencodable actions). This prevents stale-tree reuse.
  - Added **state signature** (hash of flattened state + terminal bookkeeping) to validate reuse before each search. If mismatch, the tree is rebuilt.

### Representation (card-set bottleneck)
- **Permutation‑equivariant policy/value encoder** (`alpha_zero.py`):
  - Replaced the flat MLP with a shared card encoder + pooled global context.
  - Policy now scores visible/reserved cards via shared per‑card heads, making it equivariant to card order while preserving action index alignment.
- **Deterministic board encoding** (`nn_input_output.py`): board tiers are now flattened in sorted tier order to align with action indexing.
- **Policy alignment debug** (`alpha_zero.py`): added `log_policy_alignment()` and optional `policy_log_moves`/`policy_log_topk` to print top per‑card logits for the first move(s) of training. This helps confirm slot/action alignment.

### Data augmentation
- **Color permutation augmentation** (`nn_input_output.py`, `alpha_zero.py`):
  - Added color-permutation helpers for flattened states and policy vectors.
  - Added `color_augments` parameter to `az_train` (default `1`) to duplicate self-play data with random color relabeling. Enforces game symmetry and improves sample efficiency.

### Action fidelity (return-choice head)
- **Two‑stage return decision** (`alpha_zero.py`, `nn_input_output.py`):
  - Added a return‑choice head that scores candidate token‑return vectors conditioned on the base action and global state embedding.
  - Collected MCTS visit‑based return distributions per move (when multiple return options exist).
  - Trained the return head with cross‑entropy loss (`return_weight`, default `0.5`).
  - Color‑augmentation now permutes action indices and return vectors consistently.
  - Added `use_return_loss` ablation flag to disable the return head loss without changing the architecture.

### Improvement backlog (prioritized)
1. **Action fidelity gaps**
   - Reserve‑from‑deck is excluded from the fixed 43‑action space.
   - Return‑token variants are heuristic/top‑k only; policy never learns that choice.
2. **Representation bottlenecks** (partially addressed)
   - Flat ordering of cards was a limitation; now using permutation‑equivariant card encoder.
3. **Training signal strength**
   - Evaluation vs random/greedy is weak; add Elo‑style ladder with strict promotion vs a pool of past checkpoints.
4. **Search depth / quality**
   - Sim count modest; quality limited by value accuracy and truncated return‑token variants.
   - Consider adaptive sims or deeper search once value improves.
5. **Data symmetry / augmentation**
   - Player‑swap canonicalization and color permutation done; consider additional invariances if added later.
6. **Scale**
   - Larger replay buffer, more self‑play games, longer runs, better hardware utilization.

### Environment
- **Fixed DirectML dependency** (`environment.yml`):
  - `torch-directml==0.2.5` does not exist on PyPI; updated to `torch-directml==0.2.5.dev240914`.
  - DirectML build requires `torch==2.4.1`, so updated Torch pin accordingly.
  - Environment update verified via `conda env update -f environment.yml --prune`; import checks passed.

### Decisions
- **Reserve‑from‑deck**: keep excluded from the fixed 43‑action space for now (considered a low‑value move; avoid cluttering action space). If needed later, add a minimal heuristic or a small action extension.

### Training signal
- **Elo ladder** (`alpha_zero.py`): added optional pool-based evaluation to strengthen training signal vs random/greedy. Uses a JSON pool in `logs/`, evaluates vs sampled opponents, updates Elo, and promotes models that meet a win‑rate threshold.

### Utilities
- **run_fast tuning** (`run_fast.py`): adjusted smoke‑run settings for DirectML (AMD) and enabled `use_elo_ladder` + `use_return_loss` by default for quick validation runs.
- **CUDA profile**: updated `requirements-cuda.txt` and `environment-cuda.yml` to Torch `2.4.1+cu121` and added `run_fast_cuda.py` for easy CUDA smoke runs.
- **Long‑run profiles**: added `train_dml.py` (DirectML) and `train_cuda.py` (CUDA) with longer, more intensive defaults for extended training runs.
- **Stability/ops fixes**:
- Made `_load_ckpt_model` robust to safe-loader failures seen on DirectML checkpoints.
- Added train-log schema detection; if old CSV header is incompatible, training now writes to `logs/train_log_v2.csv`.
- Lowered smoke-run LR on GPU backends in `run_fast.py` for better stability.
- **DirectML smoke stabilization validated**:
- DML smoke now uses `lr=2.5e-4`, `grad_clip=0.7`, `return_weight=0.25`.
- Verified with a full 2-iteration run: no loss explosion (loss stayed ~4.8-5.0 vs prior blow-up).
- Safe checkpoint loading now uses `weights_only=True` paths only; removed unsafe fallback loads.
- Ladder/arena now use model-only checkpoints saved with CPU tensors (`az_iter_*_model.pt`) to keep safe loading compatible on DirectML.

### GPU backend switching (DirectML vs CUDA)
- **DirectML (AMD/Intel)**:
- Create/update env: `conda env create -f environment.yml` or `conda env update -f environment.yml --prune`
- Activate: `conda activate splendor_ai`
- Run smoke: `python run_fast.py`
- Run long: `python train_dml.py`
- **CUDA (NVIDIA)**:
- Create env: `conda env create -f environment-cuda.yml`
- Activate: `conda activate splendor_ai_cuda`
- Run smoke: `python run_fast_cuda.py`
- Run long: `python train_cuda.py`

### Temporary tooling notes
- A one-off sanity check script was created to validate the new encoding + terminal backup, then removed to keep the repo clean.

### Commands run (high-level)
- `conda env update -f environment.yml --prune`
- `conda run -n splendor_ai python -c "import alpha_zero, nn_input_output"`

