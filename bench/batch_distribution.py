# bench/batch_distribution.py
"""What batch sizes does the predict batcher actually form during self-play?

bench/bench_masked_softmax.py found the GPU softmax only beats the NumPy
path from batch 32 up. MCTS submits 1 to eval_batch_size positions at a
time, but with several self-play threads sharing one PredictBatcher their
requests can be merged into larger forward passes. This measures how large,
by recording the size of every base_predict call while N threads run
searches concurrently, the way selfplay.py runs them.

Reports, per configuration, the mean forward-pass batch, the leaf-weighted
mean (the batch a typical position is evaluated in -- what the softmax
would see), the share of positions evaluated in batches of 32 or more, and
throughput.

Usage (pin to the P-cores on a hybrid CPU, see profile_search.py):
  python bench/batch_distribution.py --checkpoint ckpt/model_ep5_step10000.pt
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import threading
import time
from collections import Counter
from typing import List

import chess
import numpy as np

from mcts import MCTS, self_play_config
from net.batch_predict import make_batched_predictor
from predict import load_predictor

from profile_search import FENS


def run_config(base_predict, threads: int, eval_batch: int, wait_ms: int, sims: int,
               searches_per_thread: int) -> dict:
    sizes: List[int] = []
    lock = threading.Lock()

    def recording_predict(xs):
        with lock:
            sizes.append(len(xs))
        return base_predict(xs)

    predict_fn = make_batched_predictor(recording_predict, max_batch=128, max_wait_ms=wait_ms)

    def worker(idx: int) -> None:
        rng = np.random.default_rng(idx)
        for k in range(searches_per_thread):
            m = MCTS(predict_fn=predict_fn, mcts_cfg=self_play_config(sims, eval_batch), rng=rng)
            m.set_root(chess.Board(FENS[(idx + k) % len(FENS)]))
            m.run_simulations()

    ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(threads)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0

    s = np.array(sizes)
    leaves = s.sum()
    return {
        "calls": len(s),
        "mean": s.mean(),
        "leaf_mean": (s * s).sum() / leaves,
        "p50": float(np.median(s)),
        "max": int(s.max()),
        "ge32": (s[s >= 32].sum() / leaves),
        "distinct": len(Counter(sizes)),
        "searches_s": threads * searches_per_thread / wall,
        "leaves_s": leaves / wall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--eval-batch", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--wait-ms", type=int, nargs="+", default=[5, 0])
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--searches-per-thread", type=int, default=4)
    ap.add_argument("--no-cuda-graph", action="store_true")
    args = ap.parse_args()

    base = load_predictor(args.checkpoint, amp=True, cuda_graph=not args.no_cuda_graph)
    # Capture graphs / warm up outside the measurements.
    run_config(base, 1, 8, 0, 16, 1)

    print(f"# sims={args.sims}, {args.searches_per_thread} searches per thread, max_batch=128, "
          f"cuda_graph={not args.no_cuda_graph}")
    print()
    print("| threads | eval_batch | wait ms | forward calls | mean batch | leaf-weighted mean | median | max | "
          "positions in batch >= 32 | distinct sizes | searches/s | positions/s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for eb in args.eval_batch:
        for wait in args.wait_ms:
            for th in args.threads:
                r = run_config(base, th, eb, wait, args.sims, args.searches_per_thread)
                print(f"| {th} | {eb} | {wait} | {r['calls']} | {r['mean']:.1f} | {r['leaf_mean']:.1f} | "
                      f"{r['p50']:.0f} | {r['max']} | {100 * r['ge32']:.0f}% | {r['distinct']} | "
                      f"{r['searches_s']:.1f} | {r['leaves_s']:.0f} |", flush=True)


if __name__ == "__main__":
    main()
