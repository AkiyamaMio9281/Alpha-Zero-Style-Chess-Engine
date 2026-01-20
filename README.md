# az_chess

AlphaZero-style chess training loop in Python:
- ResNet policy/value network (4672-action policy head)
- PUCT MCTS guided by the network
- Self-play data generation
- Optional expert (UCI / Stockfish) imitation + curriculum
- Training + evaluation utilities

> Training artifacts (checkpoints / data shards) are ignored via `.gitignore`.

---

## Features

- **Action space:** 64 × 73 = 4672 actions (from-square × move-plane)
- **State encoding:** 102 planes by default (8 history frames × 12 piece planes + aux planes)
- **MCTS:** PUCT selection + Dirichlet noise at root
- **Training:** policy KL/CE to target `π` + value MSE to target `z`
- **Expert mode:** UCI engine guidance (e.g., Stockfish) with optional mixed targets

---

## Project Structure

- `engine.py`  
  Board encoding, action mapping (4672), legal-move indexing, outcome-to-`z`.

- `model.py`  
  ResNet backbone + policy/value heads. Checkpoint load/save.

- `predict.py`  
  Loads a checkpoint and exposes a predictor interface for MCTS/self-play.

- `mcts.py`  
  PUCT MCTS implementation (AlphaZero-style).

- `selfplay.py`  
  Pure MCTS self-play. Produces `.npz` training shards containing `(s, π, z)`.

- `selfplay_uci.py`  
  Expert-guided self-play using a UCI engine (Stockfish). Supports pure imitation and mixed targets.

- `trainer.py`  
  Loads shards and trains the network.

- `eval/arena.py`  
  Head-to-head evaluation between two checkpoints using MCTS.

- `autoloop.py` / `autoloopexpert.py`  
  Automation scripts for running repeated self-play → training → (optional) evaluation.

- `play_cli.py`  
  Play against a checkpoint from the terminal.

---

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`
- (Optional) A UCI engine executable (e.g., Stockfish) for expert-guided runs

---

## Setup

Create a virtual environment and install dependencies.

### Windows (CMD/PowerShell)
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check PyTorch / CUDA (optional):
```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

---

## Quick Start

### 1) Play against a model
```bash
python play_cli.py --checkpoint ckpt/model_ep3_step9000.pt --sims 100 --human white
```

### 2) Generate a small self-play dataset (smoke test)
```bash
python selfplay.py --games 2 --threads 1 --sims 50 --out data/smoke --checkpoint ckpt/model_ep3_step9000.pt
```

### 3) Train on that dataset
```bash
python trainer.py --data data/smoke --epochs 1 --steps-per-epoch 200 --batch-size 128 --out ckpt_smoke --resume ckpt/model_ep3_step9000.pt
```

### 4) Evaluate two checkpoints
Run from the repo root:
```bash
python -m eval.arena --new ckpt_smoke/model_ep1_step200.pt --old ckpt/model_ep3_step9000.pt --games 10 --sims 100 --temperature-moves 10
```

---

## Expert / Stockfish Imitation (Recommended for Cold Start)

### 1) Download Stockfish
Download a Stockfish binary for your OS and note the path to the executable.

### 2) Smoke test expert self-play
#### Windows (CMD)
```bat
python selfplay_uci.py --stockfish "C:\path\to\stockfish.exe" --games 2 --threads 1 --sf-threads 1 --out data\_sf_smoke --checkpoint ckpt\model_ep3_step9000.pt --movetime 50 --multipv 1 --expert-alpha 1.0 --sims 0
```

#### macOS/Linux
```bash
python selfplay_uci.py --stockfish "/path/to/stockfish" --games 2 --threads 1 --sf-threads 1 --out data/_sf_smoke --checkpoint ckpt/model_ep3_step9000.pt --movetime 50 --multipv 1 --expert-alpha 1.0 --sims 0
```

Verify shards exist:
- Windows:
  ```bat
  dir data\_sf_smoke\*.npz
  ```
- macOS/Linux:
  ```bash
  ls data/_sf_smoke/*.npz
  ```

---

## Automated Curriculum Loop (Expert → Mixed)

This runs multiple iterations of:
1) expert-guided data generation
2) training
3) (optional) evaluation

### Windows (CMD) example
```bat
python autoloopexpert.py --stockfish "C:\path\to\stockfish.exe" --start-ckpt ckpt\model_ep3_step9000.pt --ckpt-dir ckpt_expert --data-root data\vs_expert --curriculum --iters 10 --pure-iters 3 --device cuda --games 60 --threads 2 --sf-threads 1 --sf-hash 256 --epochs 2 --steps-per-epoch 1500 --batch-size 256 --temperature-moves 12 --max-plies 300
```

### Notes
- `--pure-iters 3` runs a few iterations of **pure imitation** before mixing in MCTS.
- Each worker spawns one UCI engine process. Start with small `--threads` and `--sf-threads 1`.
- If output looks “stuck” on Windows, you can unbuffer Python output:
  - CMD:
    ```bat
    set PYTHONUNBUFFERED=1
    ```
  - PowerShell:
    ```powershell
    $env:PYTHONUNBUFFERED="1"
    ```

---

## Outputs

- Self-play shards: `data/.../*.npz`
- Checkpoints: `ckpt*/model_ep*_step*.pt`

---

## Troubleshooting

- **`No .npz shards in ...`**  
  Usually means expert self-play did not generate games (often an incorrect `--stockfish` path).

- **Slow expert runs / high CPU usage**  
  Reduce `--threads` and keep `--sf-threads 1` initially.

- **CUDA not detected**  
  Ensure you installed a CUDA-enabled PyTorch build and that your NVIDIA driver is installed.
