# bench/profile_search.py
"""Where does an MCTS search spend its time? Step 1 of TRITON_PLAN.md.

Answers two questions before any GPU kernel gets written:

1. What share of a search's wall time goes to the policy softmax in
   MCTS._finish_expand, and how much of that a gather-first NumPy softmax
   (kernels.reference.masked_softmax_compact) already removes.
2. How long the device-to-host copy of the policy logits actually takes on
   the GPU. This is read from the profiler's Memcpy DtoH entries, not from the
   wall time of .cpu(): .cpu() is a synchronisation point, so its wall time
   also contains the wait for the forward pass (or graph replay) to finish.

Timings use the path self-play actually runs: CUDA graph on, AMP on, and the
request batcher in front of the predictor with --batch-wait-ms 0, matching the
DEVLOG's 46.6 ms/search measurement.

Usage:
  python bench/profile_search.py --checkpoint ckpt/model_ep5_step10000.pt
  python bench/profile_search.py --checkpoint ... --sims 64 400 --eval-batch 1 8 --reps 15

Run on a revision without mcts._legal_softmax it times the code as it stands
("as-is") and skips the per-softmax breakdown.

On a hybrid CPU, pin the run to the performance cores or Windows moves the
thread onto efficiency cores mid-run and the spread doubles. On the Core
Ultra 9 275HX this was measured on, the P-cores are logical CPUs 0, 1, 10-13,
22 and 23:
  start /b /wait /high /affinity C03C03 python bench\\profile_search.py --checkpoint ...

Only compare numbers from the same run. The same code measured 53.9 and 63.0
ms/search in two separate runs, so differences of a few percent between runs
are noise; that is why the softmax implementations are interleaved within one.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import cProfile
import io
import pstats
import statistics
import time
from typing import Callable, Dict, List, Optional

import chess
import numpy as np
import torch

import mcts
from mcts import MCTS, self_play_config
from net.batch_predict import make_batched_predictor
from predict import load_predictor

# Opening, middlegame and endgame positions, so the legal-move counts the
# softmax sees resemble a real game's rather than the start position's 20.
FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 8",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "2rq1rk1/pb2bppp/1pn1pn2/2pp4/2PP4/1PN1PN2/PB2BPPP/2RQ1RK1 b - - 3 12",
    "r4rk1/1pp1qppp/p1np1n2/4p3/2B1P1b1/2NP1N2/PPP1QPPP/R4RK1 w - - 0 10",
    "8/5pk1/6p1/3R4/5P2/r5P1/5K2/8 w - - 0 40",
    "8/8/4k3/3p4/3P1K2/8/8/8 w - - 0 50",
]


class Timer:
    """Accumulates wall time spent inside a wrapped callable."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.seconds = 0.0
        self.calls = 0

    def wrap(self, fn: Callable) -> Callable:
        def timed(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.seconds += time.perf_counter() - t0
                self.calls += 1
        return timed


def run_sweep(predict_fn, sims: int, eval_batch: int, rng: np.random.Generator) -> None:
    """One full search from each position, the way selfplay.py runs one per move."""
    for fen in FENS:
        m = MCTS(predict_fn=predict_fn, mcts_cfg=self_play_config(sims, eval_batch), rng=rng)
        m.set_root(chess.Board(fen))
        m.run_simulations()


def time_config(predict_fn, sims: int, eval_batch: int, impls: Dict[str, Optional[Callable]],
                reps: int, warmup: int) -> Dict[str, dict]:
    t_predict, t_expand, t_softmax = Timer(), Timer(), Timer()
    timed_predict = t_predict.wrap(predict_fn)
    original_expand = MCTS._finish_expand
    MCTS._finish_expand = t_expand.wrap(original_expand)
    rng = np.random.default_rng(0)

    def use(impl):
        if impl is not None:
            mcts._legal_softmax = t_softmax.wrap(impl)

    try:
        for impl in impls.values():
            use(impl)
            for _ in range(warmup):       # also captures the CUDA graph for each batch size
                run_sweep(timed_predict, sims, eval_batch, rng)

        samples: Dict[str, List[dict]] = {name: [] for name in impls}
        names = list(impls)
        for r in range(reps):
            # Alternate the order so drift in GPU clocks does not favour one side.
            for name in (names if r % 2 == 0 else names[::-1]):
                use(impls[name])
                for t in (t_predict, t_expand, t_softmax):
                    t.reset()
                t0 = time.perf_counter()
                run_sweep(timed_predict, sims, eval_batch, rng)
                wall = time.perf_counter() - t0
                n = len(FENS)
                samples[name].append({
                    "wall": wall / n,
                    "predict": t_predict.seconds / n,
                    "expand": t_expand.seconds / n,
                    "softmax": t_softmax.seconds / n,
                    "leaves": t_expand.calls / n,
                })
    finally:
        MCTS._finish_expand = original_expand

    out = {}
    for name, rows in samples.items():
        out[name] = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
        walls = sorted(r["wall"] for r in rows)
        out[name]["wall_lo"], out[name]["wall_hi"] = walls[len(walls) // 4], walls[(3 * len(walls)) // 4]
    return out


def _device_time(evt) -> float:
    """Self device time in microseconds; the attribute was renamed across torch versions."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(evt, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def profile_device(predict_fn, sims: int, eval_batch: int, sweeps: int) -> dict:
    from torch.profiler import ProfilerActivity, profile

    rng = np.random.default_rng(1)
    run_sweep(predict_fn, sims, eval_batch, rng)  # warm up outside the trace
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(sweeps):
            run_sweep(predict_fn, sims, eval_batch, rng)
        torch.cuda.synchronize()

    searches = sweeps * len(FENS)
    res = {"dtoh_us": 0.0, "dtoh_n": 0, "htod_us": 0.0, "htod_n": 0, "kernel_us": 0.0, "top": []}
    for e in prof.key_averages():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        us = _device_time(e)
        if us <= 0:
            continue
        if "Memcpy DtoH" in e.key:
            res["dtoh_us"] += us
            res["dtoh_n"] += e.count
        elif "Memcpy HtoD" in e.key:
            res["htod_us"] += us
            res["htod_n"] += e.count
        elif "Memcpy" not in e.key and "Memset" not in e.key:
            res["kernel_us"] += us
        res["top"].append((us, e.count, e.key))
    for k in ("dtoh_us", "htod_us", "kernel_us"):
        res[k] /= searches
    for k in ("dtoh_n", "htod_n"):
        res[k] /= searches
    res["top"].sort(reverse=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sims", type=int, nargs="+", default=[64, 400])
    ap.add_argument("--eval-batch", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--profile-sweeps", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--no-batcher", action="store_true")
    ap.add_argument("--batch-wait-ms", type=int, default=0)
    ap.add_argument("--cprofile", action="store_true",
                    help="also print the top functions by self time for the first config")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--resblocks", type=int, default=12)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs a CUDA device")

    base = load_predictor(args.checkpoint, channels=args.channels, resblocks=args.resblocks,
                          amp=not args.no_amp, cuda_graph=not args.no_cuda_graph)
    predict_fn = base if args.no_batcher else make_batched_predictor(
        base, max_batch=128, max_wait_ms=args.batch_wait_ms)

    if hasattr(mcts, "_legal_softmax"):
        from kernels.reference import masked_softmax_compact, masked_softmax_ref
        impls: Dict[str, Optional[Callable]] = {"ref": masked_softmax_ref,
                                                "compact": masked_softmax_compact}
    else:
        impls = {"as-is": None}

    print(f"# {torch.cuda.get_device_name()}  torch {torch.__version__}  "
          f"amp={not args.no_amp} cuda_graph={not args.no_cuda_graph} "
          f"batcher={'off' if args.no_batcher else f'wait={args.batch_wait_ms}ms'}  "
          f"positions={len(FENS)} reps={args.reps}")
    print()
    print("| sims | eval_batch | softmax | ms/search (IQR) | leaves | predict_fn ms | "
          "_finish_expand ms | of which softmax ms | softmax % of wall |")
    print("|---|---|---|---|---|---|---|---|---|")
    for sims in args.sims:
        for eb in args.eval_batch:
            res = time_config(predict_fn, sims, eb, impls, args.reps, args.warmup)
            for name, r in res.items():
                sm = f"{r['softmax'] * 1e3:.2f}" if impls[name] is not None else "-"
                pct = f"{100 * r['softmax'] / r['wall']:.1f}%" if impls[name] is not None else "-"
                print(f"| {sims} | {eb} | {name} | {r['wall'] * 1e3:.1f} "
                      f"({r['wall_lo'] * 1e3:.1f}-{r['wall_hi'] * 1e3:.1f}) | {r['leaves']:.0f} | "
                      f"{r['predict'] * 1e3:.1f} | {r['expand'] * 1e3:.2f} | {sm} | {pct} |",
                      flush=True)

    print()
    print("Device time per search from torch.profiler (GPU-side durations, not wall time):")
    print()
    print("| sims | eval_batch | DtoH us (copies) | us per DtoH | HtoD us (copies) | kernels us |")
    print("|---|---|---|---|---|---|")
    last = None
    for sims in args.sims:
        for eb in args.eval_batch:
            d = profile_device(predict_fn, sims, eb, args.profile_sweeps)
            per = d["dtoh_us"] / d["dtoh_n"] if d["dtoh_n"] else 0.0
            print(f"| {sims} | {eb} | {d['dtoh_us']:.0f} ({d['dtoh_n']:.0f}) | {per:.1f} | "
                  f"{d['htod_us']:.0f} ({d['htod_n']:.0f}) | {d['kernel_us']:.0f} |", flush=True)
            last = d
    if last is not None:
        print()
        print("Top device events, last config (total us over the trace, count, name):")
        for us, n, key in last["top"][:8]:
            print(f"  {us:10.0f}  {n:6d}  {key[:90]}")

    if args.cprofile:
        sims, eb = args.sims[0], args.eval_batch[0]
        pr = cProfile.Profile()
        pr.enable()
        run_sweep(predict_fn, sims, eb, np.random.default_rng(2))
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(20)
        print(f"\ncProfile, sims={sims} eval_batch={eb}, one sweep of {len(FENS)} searches, by self time:")
        print(s.getvalue())


if __name__ == "__main__":
    main()
