# autoloop.py (streaming)
from __future__ import annotations
from shutil import move
import os, sys, re, argparse, json, subprocess
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

SELFPLAY = str(HERE / "selfplay.py")
TRAINER  = str(HERE / "trainer.py")
ARENA    = str(HERE / "eval" / "arena.py")

def run_stream(cmd: list[str], cwd: Path = HERE, env: dict | None = None) -> str:
    """Run a subprocess and stream its stdout/stderr live.
    Returns the full captured stdout for post-parsing.
    """
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    # Force unbuffered output: -u plus the environment variable.
    proc_env = os.environ.copy()
    proc_env["PYTHONUNBUFFERED"] = "1"
    if env:
        proc_env.update(env)

    # Read and echo line by line (works on Windows and Unix).
    p = subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
        env=proc_env,
    )
    lines: list[str] = []
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)
    return "".join(lines)

def parse_ckpt(trainer_stdout: str) -> str | None:
    m = None
    for m in re.finditer(r"\[ckpt\]\s+saved:\s+(.+?\.pt)\b", trainer_stdout):
        pass
    return m.group(1) if m else None

def parse_win(arena_stdout: str) -> float | None:
    m = re.search(r"Win%=?\s*([0-9]+(?:\.[0-9]+)?)\s*%", arena_stdout)
    return float(m.group(1)) if m else None

def save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--temperature-moves", type=int, default=20)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--ckpt-root", type=str, default="ckpt")
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--resblocks", type=int, default=12)
    ap.add_argument("--in-planes", type=int, default=102)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--eval-sims", type=int, default=200)
    ap.add_argument("--promote-threshold", type=float, default=55.0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--base-ckpt", type=str, default="")
    ap.add_argument("--keep-gens", type=int, default=5)
    # These were never forwarded to selfplay.py, so it always ran on its own
    # defaults no matter what this script was told.
    ap.add_argument("--batch-max", type=int, default=128)
    ap.add_argument("--batch-wait-ms", type=int, default=5)
    ap.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True,
                    help="capture the forward pass as a CUDA graph and replay it. Verified bit-identical to eager and falls back automatically, so it is on by default here; --no-cuda-graph disables it.")
    args = ap.parse_args()

    data_root = HERE / args.data_root
    ckpt_root = HERE / args.ckpt_root
    state_path = HERE / "autoloop_state.json"

    current_ckpt = args.base_ckpt if args.base_ckpt.strip() else None
    history: list[dict] = []

    for r in range(1, args.rounds + 1):
        print(f"\n========== ROUND {r}/{args.rounds}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==========", flush=True)

        # 1) Self-play
        gen_dir = data_root / f"gen{r}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        sp_cmd = [
            PY, "-u", SELFPLAY,  # -u keeps the child unbuffered
            "--games", str(args.games),
            "--threads", str(args.threads),
            "--sims", str(args.sims),
            "--temperature-moves", str(args.temperature_moves),
            "--max-plies", str(args.max_plies),
            "--out", str(gen_dir),
            "--channels", str(args.channels),
            "--resblocks", str(args.resblocks),
            "--batch-max", str(args.batch_max),
            "--batch-wait-ms", str(args.batch_wait_ms),
        ]
        if args.cuda_graph:
            sp_cmd += ["--cuda-graph"]
        if args.device:
            sp_cmd += ["--device", args.device]
        if current_ckpt:
            sp_cmd += ["--checkpoint", current_ckpt]
        out_sp = run_stream(sp_cmd)

        # 2) Train on the whole data_root -- a replay window across generations --
        # not just this round's gen_dir. The window size is set by the --keep-gens
        # pruning below.
        tr_cmd = [
            PY, "-u", TRAINER,
            "--data", str(data_root),
            "--epochs", "1",
            "--steps-per-epoch", str(args.steps_per_epoch),
            "--batch-size", str(args.batch_size),
            "--channels", str(args.channels),
            "--resblocks", str(args.resblocks),
            "--in-planes", str(args.in_planes),
            "--lr", str(args.lr),
            "--out", str(ckpt_root),
        ]
        # Resume from the active checkpoint instead of starting every round from
        # random weights. Without this each round trains a brand-new random model,
        # the arena cannot beat the previous generation, every round is REJECTED,
        # current_ckpt never advances, and the loop spins in place. Worse, trainer
        # names checkpoints from epoch/step, so without resuming every round writes
        # model_ep1_step<spe>.pt and overwrites the previous generation's weights,
        # leaving the arena to play one file against itself.
        if current_ckpt and Path(current_ckpt).exists():
            tr_cmd += ["--resume", current_ckpt]
        out_tr = run_stream(tr_cmd)
        new_ckpt = parse_ckpt(out_tr)
        if not new_ckpt:
            raise RuntimeError("Could not parse a checkpoint path from the trainer output.")

        # 3) Evaluate. The first round has no previous model, so it is promoted directly.
        promoted = False
        winp = None
        if current_ckpt:
            arena_cmd = [
                PY, "-u", ARENA,
                "--new", new_ckpt,
                "--old", current_ckpt,
                "--games", str(args.eval_games),
                "--sims", str(args.eval_sims),
                "--temperature-moves", str(args.temperature_moves),
                # Must match what selfplay and trainer were given, or arena
                # builds a differently-shaped network and dies loading weights.
                "--channels", str(args.channels),
                "--resblocks", str(args.resblocks),
                "--in-planes", str(args.in_planes),
            ]
            # Arena plays whole games at search depth and is the most
            # expensive stage of a round. Measured at 100 sims: 405 ms/move
            # eager against 135 ms/move with the graph.
            if args.cuda_graph:
                arena_cmd += ["--cuda-graph"]
            out_ar = run_stream(arena_cmd)
            winp = parse_win(out_ar)
            if winp is None:
                raise RuntimeError("Could not parse Win% from the arena output.")
            promoted = (winp >= args.promote_threshold)
            if promoted:
                current_ckpt = new_ckpt
        else:
            promoted = True
            current_ckpt = new_ckpt

        # 4) Record and prune
        rec = {
            "round": r,
            "gen_dir": str(gen_dir),
            "new_ckpt": new_ckpt,
            "active_ckpt": current_ckpt,
            "promoted": promoted,
            "win_percent": winp,
        }
        history.append(rec)
        save_state(state_path, {"rounds_done": r, "history": history})
        print(f"[autoloop] round {r} {'PROMOTED' if promoted else 'REJECTED'}; active_ckpt={current_ckpt}", flush=True)

        if args.keep_gens > 0:
            gens = sorted([p for p in data_root.glob("gen*") if p.is_dir()], key=lambda p: int(p.name[3:]))
            if len(gens) > args.keep_gens:
                for p in gens[:-args.keep_gens]:
                    try:
                        import shutil; shutil.rmtree(p)
                        print(f"[autoloop] removed old data dir: {p}", flush=True)
                    except Exception as e:
                        print(f"[autoloop] failed to remove {p}: {e}", flush=True)

    print("\n[autoloop] ALL ROUNDS FINISHED.", flush=True)
    print(f"[autoloop] active checkpoint: {current_ckpt}", flush=True)

if __name__ == "__main__":
    main()
