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
    # 确保非缓冲输出：-u + 环境变量
    proc_env = os.environ.copy()
    proc_env["PYTHONUNBUFFERED"] = "1"
    if env:
        proc_env.update(env)

    # 逐行读取并原样打印（Windows/Unix 均可）
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
    args = ap.parse_args()

    data_root = HERE / args.data_root
    ckpt_root = HERE / args.ckpt_root
    state_path = HERE / "autoloop_state.json"

    current_ckpt = args.base_ckpt if args.base_ckpt.strip() else None
    history: list[dict] = []

    for r in range(1, args.rounds + 1):
        print(f"\n========== ROUND {r}/{args.rounds}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==========", flush=True)

        # 1) 自博弈
        gen_dir = data_root / f"gen{r}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        sp_cmd = [
            PY, "-u", SELFPLAY,  # -u 确保子进程不缓冲
            "--games", str(args.games),
            "--threads", str(args.threads),
            "--sims", str(args.sims),
            "--temperature-moves", str(args.temperature_moves),
            "--max-plies", str(args.max_plies),
            "--out", str(gen_dir),
            "--channels", str(args.channels),
            "--resblocks", str(args.resblocks),
        ]
        if args.device:
            sp_cmd += ["--device", args.device]
        if current_ckpt:
            sp_cmd += ["--checkpoint", current_ckpt]
        out_sp = run_stream(sp_cmd)

        # 2) 训练：读整个 data_root（跨代回放窗口），而不是只看这一轮的 gen_dir。
        # 窗口大小由下面的 --keep-gens 磁盘清理控制。
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
        # 从当前活跃的 checkpoint 续训，而不是每轮从随机权重重来。缺了这一步，
        # 每轮训出来的都是全新的随机模型，arena 自然打不过上一代，于是
        # 永远 REJECTED、current_ckpt 永远不更新，整个循环空转。而且 trainer 的
        # 文件名由 epoch/step 决定，不续训的话每轮都叫 model_ep1_step<spe>.pt，
        # 会把上一代权重直接覆盖，arena 于是在拿同一个文件自己跟自己下。
        if current_ckpt and Path(current_ckpt).exists():
            tr_cmd += ["--resume", current_ckpt]
        out_tr = run_stream(tr_cmd)
        new_ckpt = parse_ckpt(out_tr)
        if not new_ckpt:
            raise RuntimeError("无法在 trainer 输出中解析到 checkpoint 路径。")

        # 3) 评测（第一轮没有旧模型则直接晋级）
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
            out_ar = run_stream(arena_cmd)
            winp = parse_win(out_ar)
            if winp is None:
                raise RuntimeError("无法在 arena 输出中解析到 Win%。")
            promoted = (winp >= args.promote_threshold)
            if promoted:
                current_ckpt = new_ckpt
        else:
            promoted = True
            current_ckpt = new_ckpt

        # 4) 记录与清理
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
