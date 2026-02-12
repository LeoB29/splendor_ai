# Splendor AI

AlphaZero-style self-play training for Splendor with a rules-accurate engine, a baseline MCTS player, and a simple Tkinter GUI.
This is a work in progress, with the main goal of improving the AI winrate until it reaches human or superhuman performance.
The GUI part is secondary at this stage and should not be touched further for now. 

## Quickstart (Windows PowerShell)
1) Create a virtual environment:
   `python -m venv .venv`
2) Install dependencies:
   `.venv\Scripts\python -m pip install -U pip`
   `.venv\Scripts\python -m pip install -r requirements.txt`

## Conda (native)
1) Create and activate the environment:
   `conda env create -f environment.yml`
   `conda activate splendor_ai`
2) Update deps later:
   `conda env update -f environment.yml --prune`

## Conda (CUDA)
1) Create and activate the CUDA environment:
   `conda env create -f environment-cuda.yml`
   `conda activate splendor_ai_cuda`

## DirectML
This repo is set up to use DirectML (`torch-directml`) for AMD/Intel GPUs on Windows. If install fails for your Python version, update the pinned versions in `requirements.txt` to the compatible set from the official torch-directml release notes.

## CUDA (optional)
If you want to switch to CUDA, install from `requirements-cuda.txt` instead:
`.venv\Scripts\python -m pip install -r requirements-cuda.txt`

## Run
- Quick training smoke run: `.venv\Scripts\python run_fast.py`
- GUI (Human vs AI): `.venv\Scripts\python gui.py`
- Verbose model play log: `.venv\Scripts\python play_log.py --opponent random --games 1`
- Simulation (random/MCTS): `.venv\Scripts\python game_sim.py`
- Tests: `.venv\Scripts\python -m pytest -q`

## Project Layout
- `game_state.py`: core rules and state transitions
- `nn_input_output.py`: state flattening + action encoding/masking
- `alpha_zero.py`: MCTS + self-play + training loop
- `run_fast.py`: small training config for quick iteration
- `gui.py`: Tkinter interface
