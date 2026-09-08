# autoloopexpert.py (with curriculum scheduling)
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

def run_cmd(args: List[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    print("\n==> RUN:", " ".join([f'"{a}"' if " " in a else a for a in args]), flush=True)
    proc = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    lines: List[str] = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    print(f"<== EXIT {proc.returncode}\n", flush=True)
    return proc.returncode, "".join(lines)

def latest_ckpt(ckpt_dir: str) -> Optional[str]:
    p = Path(ckpt_dir)
    if not p.exists(): return None
    cands = sorted(p.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    return str(cands[0]) if cands else None

def parse_win(arena_stdout: str) -> Optional[float]:
    m = re.search(r"Win%=?\s*([0-9]+(?:\.[0-9]+)?)\s*%", arena_stdout)
    return float(m.group(1)) if m else None

def save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False))

def prune_old_data_dirs(data_root: str, keep: int) -> None:
    if keep <= 0:
        return
    root = Path(data_root)
    if not root.exists():
        return
    dirs = sorted([p for p in root.glob("it*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if len(dirs) <= keep:
        return
    for p in dirs[:-keep]:
        try:
            shutil.rmtree(p)
            print(f"[autoloop] removed old data dir: {p}", flush=True)
        except Exception as e:
            print(f"[autoloop] failed to remove {p}: {e}", flush=True)

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def main():
    ap = argparse.ArgumentParser()
    # Required
    ap.add_argument("--stockfish", required=True, help="Path to Stockfish/other UCI engine executable")
    # Optional
    ap.add_argument("--start-ckpt", default="", help="Initial checkpoint to start from (resume). If empty, use latest in --ckpt-dir; if none found, use dummy.")
    ap.add_argument("--ckpt-dir", default="ckpt", help="Directory to save and search for checkpoints")
    ap.add_argument("--data-root", default="data/vs_expert", help="Root directory to save self-play shards")
    ap.add_argument("--keep-iters", type=int, default=5, help="Keep only the most recent N iterations' data dirs on disk (0 = keep all)")
    ap.add_argument("--iters", type=int, default=3, help="Number of outer loops (self-play+train)")
    ap.add_argument("--device", default="", help="cuda or cpu for model inference during self-play")
    # Self-play base options (used when not using curriculum or as bounds in curriculum)
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--temperature-moves", type=int, default=12)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--batch-max", type=int, default=128)
    ap.add_argument("--batch-wait-ms", type=int, default=5)
    ap.add_argument("--opponent-color", choices=["white","black"], default="black")
    # Training options
    ap.add_argument("--epochs", type=int, default=2, help="epochs per iteration")
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--resume", action="store_true", help="Force resume from the start-ckpt on the first train step (default: auto)")
    ap.add_argument("--no-resume", action="store_true", help="Do not resume (override)")
    # Optional evaluation (off by default)
    ap.add_argument("--do-arena", action="store_true", help="If set, run eval/arena.py after each iteration and gate promotion on the result")
    ap.add_argument("--arena-games", type=int, default=40)
    ap.add_argument("--arena-sims", type=int, default=400)
    ap.add_argument("--promote-threshold", type=float, default=55.0, help="Win%% vs previous checkpoint required to promote (only used with --do-arena)")

    # Curriculum switches and bounds
    ap.add_argument("--curriculum", action="store_true", help="Enable easy->hard schedule for self-play parameters")
    ap.add_argument("--pure-iters", type=int, default=2, help="Number of initial pure-imitation iterations (sims=0, alpha=1.0, multipv=1)")
    ap.add_argument("--sims-start", type=int, default=0)
    ap.add_argument("--sims-end", type=int, default=400)
    ap.add_argument("--alpha-start", type=float, default=1.0)
    ap.add_argument("--alpha-end", type=float, default=0.5)
    ap.add_argument("--multipv-start", type=int, default=1)
    ap.add_argument("--multipv-end", type=int, default=6)
    ap.add_argument("--movetime-start", type=int, default=100)
    ap.add_argument("--movetime-end", type=int, default=300)
    ap.add_argument("--skill-start", type=int, default=5)
    ap.add_argument("--skill-end", type=int, default=20)
    ap.add_argument("--expert-cp-scale", type=float, default=174.0,
                    help="centipawn scale of the expert softmax (see selfplay_uci.py)")
    ap.add_argument("--sf-threads", type=int, default=1)
    ap.add_argument("--sf-hash", type=int, default=64)
    ap.add_argument("--resign-cp", type=int, default=0)
    ap.add_argument("--resign-plies", type=int, default=0)

    args = ap.parse_args()

    py = sys.executable
    proj = Path(__file__).resolve().parent
    state_path = proj / "autoloopexpert_state.json"
    history: List[dict] = []

    # Determine initial checkpoint
    cur_ckpt = args.start_ckpt.strip()
    if not cur_ckpt:
        cur_ckpt = latest_ckpt(args.ckpt_dir) or ""
        if cur_ckpt:
            print(f"[autoloop] Found latest ckpt: {cur_ckpt}")
    if cur_ckpt and not Path(cur_ckpt).exists():
        print(f"[autoloop] WARNING: start-ckpt not found: {cur_ckpt}. Will fallback to dummy for selfplay.")
        cur_ckpt = ""

    for it in range(1, args.iters + 1):
        print(f"\n====================== ITERATION {it}/{args.iters} ======================\n")

        # Decide parameters for this iteration
        if not args.curriculum:
            sims = args.sims_end
            alpha = args.alpha_end
            multipv = args.multipv_end
            movetime = args.movetime_end
            skill = args.skill_end
        else:
            # Pure imitation for the first pure-iters iterations
            if it <= args.pure_iters:
                sims = 0
                alpha = 1.0
                multipv = 1
                movetime = args.movetime_start
                skill = args.skill_start
            else:
                # linearly progress from start to end across remaining iters
                t = (it - args.pure_iters) / max(1, (args.iters - args.pure_iters))
                t = min(max(t, 0.0), 1.0)
                sims = int(lerp(args.sims_start, args.sims_end, t))
                alpha = float(lerp(args.alpha_start, args.alpha_end, t))
                multipv = int(lerp(args.multipv_start, args.multipv_end, t))
                movetime = int(lerp(args.movetime_start, args.movetime_end, t))
                skill = int(lerp(args.skill_start, args.skill_end, t))

        print(f"[autoloop] iter {it}: sims={sims}, alpha={alpha:.2f}, multipv={multipv}, movetime={movetime}ms, skill={skill}")

        # 1) Self-play vs expert -> shards
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = Path(args.data_root) / f"it{it}_{stamp}"
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        ckpt_flag = cur_ckpt if cur_ckpt else "none"

        sp_cmd = [
            py, str(proj / "selfplay_uci.py"),
            "--games", str(args.games),
            "--threads", str(args.threads),
            "--sims", str(sims),
            "--temperature-moves", str(args.temperature_moves),
            "--max-plies", str(args.max_plies),
            "--out", str(data_dir),
            "--checkpoint", ckpt_flag,
            "--uci-path", str(args.stockfish),
            "--opponent-color", str(args.opponent_color),
            "--uci-movetime", str(movetime),
            "--expert-multipv", str(multipv),
            "--expert-alpha", str(alpha),
            "--expert-cp-scale", str(args.expert_cp_scale),
            "--history", str(args.history),
            "--batch-max", str(args.batch_max),
            "--batch-wait-ms", str(args.batch_wait_ms),
            "--sf-threads", str(args.sf_threads),
            "--sf-hash", str(args.sf_hash),
            "--skill-level", str(skill),
            "--resign-cp", str(args.resign_cp),
            "--resign-plies", str(args.resign_plies),
        ]
        if args.device:
            sp_cmd += ["--device", args.device]

        rc, _ = run_cmd(sp_cmd, cwd=str(proj))
        if rc != 0:
            sys.exit(rc)

        # 2) Train on the newly generated data
        # Train on the whole data-root (replay window across recent iterations),
        # not just this iteration's fresh data_dir. Window size is controlled
        # by the --keep-iters pruning below.
        train_cmd = [
            py, str(proj / "trainer.py"),
            "--data", str(args.data_root),
            "--epochs", str(args.epochs),
            "--steps-per-epoch", str(args.steps_per_epoch),
            "--batch-size", str(args.batch_size),
            "--out", str(args.ckpt_dir),
        ]
        if not args.no_resume:
            if args.resume or (cur_ckpt and Path(cur_ckpt).exists()):
                train_cmd += ["--resume", str(cur_ckpt)]
        rc, _ = run_cmd(train_cmd, cwd=str(proj))
        if rc != 0:
            sys.exit(rc)

        # 3) Find the checkpoint just produced by training
        prev_ckpt = cur_ckpt
        new_ckpt = latest_ckpt(args.ckpt_dir)
        if not new_ckpt:
            print("[autoloop] WARNING: no checkpoint found after training; keeping previous checkpoint.")
            continue
        print(f"[autoloop] New checkpoint: {new_ckpt}")

        # 4) Optional evaluation: only promote the new checkpoint if it beats the
        # previous one by --promote-threshold win%. Otherwise keep training from
        # the last known-good checkpoint instead of silently drifting.
        promoted = True
        winp = None
        if args.do_arena and prev_ckpt and Path(prev_ckpt).exists() and new_ckpt != prev_ckpt:
            arena_cmd = [
                py, str(proj / "eval" / "arena.py"),
                "--new", new_ckpt,
                "--old", prev_ckpt,
                "--games", str(args.arena_games),
                "--sims", str(args.arena_sims),
                "--temperature-moves", str(args.temperature_moves),
            ]
            rc, out_ar = run_cmd(arena_cmd, cwd=str(proj))
            if rc != 0:
                print("[autoloop] arena.py returned non-zero; promoting by default (continuing).")
            else:
                winp = parse_win(out_ar)
                if winp is None:
                    print("[autoloop] WARNING: could not parse Win% from arena output; promoting by default.")
                else:
                    promoted = (winp >= args.promote_threshold)

        cur_ckpt = new_ckpt if promoted else prev_ckpt
        status = "PROMOTED" if promoted else "REJECTED"
        winp_str = f" win%={winp:.1f}" if winp is not None else ""
        print(f"[autoloop] iter {it} {status}{winp_str}; active_ckpt={cur_ckpt}")

        # 5) Record + persist this iteration's history, and prune old data dirs
        history.append({
            "iteration": it,
            "data_dir": str(data_dir),
            "sims": sims, "alpha": alpha, "multipv": multipv, "movetime": movetime, "skill": skill,
            "new_ckpt": new_ckpt,
            "active_ckpt": cur_ckpt,
            "promoted": promoted,
            "win_percent": winp,
        })
        save_state(state_path, {"iters_done": it, "history": history})
        prune_old_data_dirs(args.data_root, args.keep_iters)

    print("\n[autoloop] All iterations finished.")
    if cur_ckpt:
        print(f"[autoloop] Latest checkpoint: {cur_ckpt}")

if __name__ == "__main__":
    main()
