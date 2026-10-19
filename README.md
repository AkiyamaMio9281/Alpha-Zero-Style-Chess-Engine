# az_chess

[![tests](https://github.com/AkiyamaMio9281/Alpha-Zero-Style-Chess-Engine/actions/workflows/tests.yml/badge.svg)](https://github.com/AkiyamaMio9281/Alpha-Zero-Style-Chess-Engine/actions/workflows/tests.yml)

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

- `kernels/`  
  The policy-prior softmax over legal moves: NumPy reference and gather-first versions (`reference.py`), and Triton kernels for the GPU (`masked_softmax.py`).

- `bench/`  
  `profile_search.py` breaks a search's time down by component; `bench_masked_softmax.py` benchmarks the softmax implementations.

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

## Testing

Unit tests cover the action encoding (`engine.py`) and MCTS (`mcts.py`) — the
two places where a silent bug would otherwise just produce a model that
quietly trains on garbage.

```bash
pip install -r requirements-dev.txt
pytest
```

---

## GPU policy softmax (Triton)

Every node expansion turns 4672 policy logits into priors over the ~30 legal
moves. `kernels/masked_softmax.py` does this on the GPU in Triton, two ways:

- **v1, dense:** the row softmax with a legal-move mask, one program per
  row. The bandwidth-bound reference point.
- **v2, gather:** reads only the legal logits and writes them packed. Legal
  moves come as a fixed-shape `(B, 256)` index buffer rather than a
  variable-length list, so the kernel can be captured into the same CUDA
  graph as the forward pass (tested: replays match eager bit for bit).

Measured before writing either (`bench/profile_search.py`): the softmax took
2–3% of a search and the logits' device-to-host copy 0.2%. Gathering the
legal logits before the softmax, still in NumPy, cut its cost 2–3x and is
what MCTS uses now.

Kernel time on the GPU (fp16 logits, `do_bench` median with L2 flushed,
RTX 5090 Laptop under WSL2). Both tables are from one run:

| batch | dense working set | torch eager | torch.compile | v1 dense | v2 gather | v1 bandwidth |
|---|---|---|---|---|---|---|
| 1 | 0.03 MB* | 22.3 µs | 7.9 µs | 5.9 µs | 5.9 µs | 6 GB/s |
| 8 | 0.2 MB* | 24.3 µs | 7.9 µs | 6.0 µs | 5.9 µs | 44 GB/s |
| 64 | 2.0 MB* | 24.4 µs | 7.9 µs | 7.9 µs | 5.9 µs | 265 GB/s |
| 256 | 8.0 MB* | 40.7 µs | 16.1 µs | 10.0 µs | 6.0 µs | 839 GB/s* |
| 1024 | 32 MB* | 110.2 µs | 46.8 µs | 26.4 µs | 7.9 µs | 1270 GB/s* |
| 4096 | 128 MB | 800.4 µs | 183.0 µs | 157.4 µs | 14.1 µs | 851 GB/s |

\* The working set fits in the 64 MB L2. `do_bench` flushes L2 before each
run, but part of the output is still in L2 when the kernel finishes, so
these rows overstate DRAM bandwidth; 1270 GB/s at batch 1024 is not
physically possible from DRAM. Only batch 4096, at twice the size of L2, is
a DRAM figure: 851 GB/s is about 95% of the GPU's 896 GB/s spec bandwidth,
and above the 748 GB/s a 1 GiB device-to-device copy reaches, because v1
reads more than it writes. Below batch 64 the time is a flat ~6–8 µs of
launch latency whatever the size, the same latency bound the forward pass
shows.

v1 is 4–5x faster than PyTorch eager from batch 256 up, and on par with
`torch.compile`: which of the two is ahead at large batch changes between
runs (1.2–1.8x ahead at batch 128–4096 in this one, 3–8% behind at batch
256 in an earlier one), so the gap between them is not a result. v2's
time barely moves with batch because it touches only the ~31 legal entries
per row.

End to end, from logits on the GPU to priors on the CPU (wall time):

| batch | NumPy, gather-first | v2 gather, one upload | v2 vs NumPy |
|---|---|---|---|
| 1 | 89 µs | 192 µs | 0.46x |
| 8 | 142 µs | 198 µs | 0.72x |
| 32 | 281 µs | 195 µs | 1.44x |
| 256 | 1779 µs | 314 µs | 5.66x |
| 4096 | 45770 µs | 3151 µs | 14.5x |

"One upload" packs the indices and counts into one pinned buffer and copies
it in a single non-blocking transfer. The kernel wins from batch 32 up. MCTS
here evaluates 1 to 8 positions per call, and there the path's fixed cost --
each CUDA call costs 35–55 µs of latency under WSL2, whatever its size --
outweighs the NumPy softmax it replaces, so the search keeps the NumPy path.
Reproduce with `python bench/bench_masked_softmax.py` (needs Linux, CUDA and
Triton).

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
python selfplay_uci.py --uci-path "C:\path\to\stockfish.exe" --games 2 --threads 1 --sf-threads 1 --out data\_sf_smoke --checkpoint ckpt\model_ep3_step9000.pt --uci-movetime 50 --expert-multipv 1 --expert-alpha 1.0 --sims 0
```

#### macOS/Linux
```bash
python selfplay_uci.py --uci-path "/path/to/stockfish" --games 2 --threads 1 --sf-threads 1 --out data/_sf_smoke --checkpoint ckpt/model_ep3_step9000.pt --uci-movetime 50 --expert-multipv 1 --expert-alpha 1.0 --sims 0
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
