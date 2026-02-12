# AGENTS.md

## Project
- Splendor game engine and AI (MCTS + AlphaZero-style training).
- Rules/state live in `game_state.py`; training pipeline in `alpha_zero.py`.
- Fixed 43-action space is defined in `nn_input_output.py`.
- You can refer to master_notes.md for the decisions we previously made together on the project.

## Goals
- Improve playing strength and training stability without breaking Splendor rules.
- Keep changes minimal and consistent with the existing action encoding.
- The ultimate goal is to be able to beat a very strong human player, or even to achieve superhuman performance if possible

## Common Commands (PowerShell)
- `python -m venv .venv`
- `.venv\Scripts\python -m pip install -r requirements.txt`
- `.venv\Scripts\python run_fast.py`
- `.venv\Scripts\python gui.py`
- `.venv\Scripts\python -m pytest -q`

## Device
- Prefer DirectML when available (`torch_directml`).

## Repo Hygiene
- Checkpoints live in `checkpoints/` and logs in `logs/` (do not commit).
- Avoid touching `backup/` and `docs/` unless asked.
