# bench/bench_masked_softmax.py
"""Benchmark the masked policy softmax: Triton kernels against the baselines.

Step 3 of TRITON_PLAN.md. Produces three tables:

1. Measured peak DRAM bandwidth -- a large device-to-device copy -- used as
   the roofline ceiling instead of the spec-sheet figure.

2. Kernel only, GPU time (triton.testing.do_bench, median, L2 flushed
   between runs): torch eager, torch.compile, Triton v1 (dense) and Triton
   v2 (gather). Inputs and outputs stay on the GPU. Effective bandwidth is
   the minimum bytes the operation has to move divided by the time:
     dense:  read fp16 logits + read u8 mask + write fp32 probs = 7 B/entry
     gather: read the legal fp16 logits + idx + count, write the (B, 256) out
   Eager and compiled torch are charged the dense minimum, so their
   bandwidth is comparable with v1's.

3. End to end, wall time: from logits on the GPU to one prior array per
   position on the CPU, which is what MCTS needs. Compares the gather-first
   NumPy softmax MCTS uses now (copy the whole row back, softmax on the CPU)
   with torch eager (mask built on the CPU and uploaded, softmax on the GPU,
   whole row back) and with v2 (indices packed and uploaded, kernel, packed
   rows back), the latter both with idx and count uploaded separately and
   through one pinned buffer in a single non-blocking copy. Paths are
   interleaved; only compare numbers from one run.

4. v2's end-to-end path split into its CUDA calls at small batch, to show
   where its fixed floor goes.

Logits are fp16, as the network emits them under AMP. Needs CUDA and Triton
(run it in the WSL environment from TRITON_PLAN.md).
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import statistics
import time
from typing import Callable, Dict, List

import chess
import numpy as np
import torch
import triton
from triton.testing import do_bench

from engine import ACTION_SIZE, legal_moves_index_map
from kernels.masked_softmax import GATHER_BLOCK, masked_softmax_dense, masked_softmax_gather, pack_legal
from kernels.reference import masked_softmax_compact

DEV = torch.device("cuda")


def game_positions(games: int = 8, max_plies: int = 120, seed: int = 0) -> List[np.ndarray]:
    """Legal-move index sets from random games, for a realistic spread of counts."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(games):
        board = chess.Board()
        while not board.is_game_over() and board.ply() < max_plies:
            out.append(np.array(sorted(legal_moves_index_map(board)), dtype=np.int32))
            moves = list(board.legal_moves)
            board.push(moves[rng.integers(len(moves))])
    return out


def torch_masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """What you would write in PyTorch: the reference's -1e9 mask, fp32 math."""
    return torch.softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)


torch_masked_softmax_compiled = torch.compile(torch_masked_softmax)


def bench_gpu(fn: Callable[[], object]) -> float:
    """Median GPU time in microseconds."""
    return do_bench(fn, warmup=50, rep=300, return_mode="median") * 1e3


def peak_bandwidth_gbs(mib: int = 1024) -> float:
    src = torch.empty(mib * 2**20 // 4, device=DEV, dtype=torch.float32).normal_()
    dst = torch.empty_like(src)
    us = bench_gpu(lambda: dst.copy_(src))
    gbs = 2 * src.numel() * 4 / (us * 1e-6) / 1e9
    del src, dst
    torch.cuda.empty_cache()
    return gbs


def make_batch(pool: List[np.ndarray], B: int, rng: np.random.Generator):
    legal = [pool[i] for i in rng.integers(len(pool), size=B)]
    logits = rng.normal(0, 3, (B, ACTION_SIZE)).astype(np.float16)
    return logits, legal


def kernel_table(pool, batches, peak, rng) -> None:
    print("| B | legal/row | eager us | compile us | v1 dense us | v2 gather us | "
          "eager GB/s | compile GB/s | v1 GB/s (% peak) | v2 GB/s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for B in batches:
        logits_np, legal = make_batch(pool, B, rng)
        logits = torch.from_numpy(logits_np).to(DEV)
        mask_np = np.zeros((B, ACTION_SIZE), dtype=bool)
        for i, l in enumerate(legal):
            mask_np[i, l] = True
        mask = torch.from_numpy(mask_np).to(DEV)
        idx_np, count_np = pack_legal(legal)
        idx, count = torch.from_numpy(idx_np).to(DEV), torch.from_numpy(count_np).to(DEV)

        # Correctness guard: a fast wrong answer is not a result.
        ref = torch_masked_softmax(logits, mask)
        torch.testing.assert_close(masked_softmax_dense(logits, mask), ref, rtol=0, atol=1e-6)
        torch.testing.assert_close(torch_masked_softmax_compiled(logits, mask), ref, rtol=0, atol=1e-6)
        g = masked_softmax_gather(logits, idx, count)
        for i, l in enumerate(legal[:8]):
            torch.testing.assert_close(g[i, :len(l)], ref[i, torch.from_numpy(l).long().to(DEV)],
                                       rtol=0, atol=1e-6)

        t_eager = bench_gpu(lambda: torch_masked_softmax(logits, mask))
        t_comp = bench_gpu(lambda: torch_masked_softmax_compiled(logits, mask))
        t_v1 = bench_gpu(lambda: masked_softmax_dense(logits, mask))
        t_v2 = bench_gpu(lambda: masked_softmax_gather(logits, idx, count))

        dense_bytes = B * ACTION_SIZE * (2 + 1 + 4)
        gather_bytes = int(count_np.sum()) * 2 + B * GATHER_BLOCK * 2 + B * 4 + B * GATHER_BLOCK * 4
        gbs = lambda nbytes, us: nbytes / (us * 1e-6) / 1e9
        v1_gbs = gbs(dense_bytes, t_v1)
        print(f"| {B} | {count_np.mean():.0f} | {t_eager:.1f} | {t_comp:.1f} | {t_v1:.1f} | {t_v2:.1f} | "
              f"{gbs(dense_bytes, t_eager):.0f} | {gbs(dense_bytes, t_comp):.0f} | "
              f"{v1_gbs:.0f} ({100 * v1_gbs / peak:.0f}%) | {gbs(gather_bytes, t_v2):.1f} |", flush=True)


def end_to_end_table(pool, batches, iters, rng) -> None:
    print("| B | numpy compact us | torch eager us | v2 gather us | v2 one upload us | packing us | "
          "numpy us/row | v2 one upload us/row | v2 vs numpy | v2 one upload vs numpy |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for B in batches:
        logits_np, legal = make_batch(pool, B, rng)
        logits = torch.from_numpy(logits_np).to(DEV)

        def numpy_compact():
            l = logits.cpu().numpy().astype(np.float32)
            return [masked_softmax_compact(l[i], legal[i]) for i in range(B)]

        def torch_eager():
            mask_np = np.zeros((B, ACTION_SIZE), dtype=bool)
            for i, l in enumerate(legal):
                mask_np[i, l] = True
            p = torch_masked_softmax(logits, torch.from_numpy(mask_np).to(DEV)).cpu().numpy()
            return [p[i, legal[i]] for i in range(B)]

        def v2_gather():
            idx_np, count_np = pack_legal(legal)
            out = masked_softmax_gather(logits, torch.from_numpy(idx_np).to(DEV),
                                        torch.from_numpy(count_np).to(DEV)).cpu().numpy()
            return [out[i, :count_np[i]] for i in range(B)]

        # Same kernel, one upload instead of two: idx and count share one
        # pinned int32 staging buffer, copied with a single non-blocking H2D,
        # so the path pays one synchronous round trip (the final .cpu())
        # instead of three or four.
        n_idx = B * GATHER_BLOCK
        host = torch.empty(n_idx + B, dtype=torch.int32, pin_memory=True)
        host_np = host.numpy()
        dev_buf = torch.empty(n_idx + B, dtype=torch.int32, device=DEV)

        def v2_one_upload():
            idx_np, count_np = pack_legal(legal)
            host_np[:n_idx] = idx_np.ravel()
            host_np[n_idx:] = count_np
            dev_buf.copy_(host, non_blocking=True)
            out = masked_softmax_gather(logits, dev_buf[:n_idx].view(B, GATHER_BLOCK),
                                        dev_buf[n_idx:]).cpu().numpy()
            return [out[i, :count_np[i]] for i in range(B)]

        def packing():
            return pack_legal(legal)

        a, b, c, d = numpy_compact(), torch_eager(), v2_gather(), v2_one_upload()
        for i in range(B):
            np.testing.assert_allclose(b[i], a[i], rtol=0, atol=1e-6)
            np.testing.assert_allclose(c[i], a[i], rtol=0, atol=1e-6)
            np.testing.assert_allclose(d[i], a[i], rtol=0, atol=1e-6)

        paths: Dict[str, Callable] = {"numpy": numpy_compact, "eager": torch_eager,
                                      "v2": v2_gather, "v2_one": v2_one_upload, "pack": packing}
        samples: Dict[str, List[float]] = {k: [] for k in paths}
        names = list(paths)
        for _ in range(10):                      # warm up every path
            for f in paths.values():
                f()
        for r in range(iters):
            for name in (names if r % 2 == 0 else names[::-1]):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                paths[name]()
                samples[name].append(time.perf_counter() - t0)
        med = {k: statistics.median(v) * 1e6 for k, v in samples.items()}
        print(f"| {B} | {med['numpy']:.0f} | {med['eager']:.0f} | {med['v2']:.0f} | {med['v2_one']:.0f} | "
              f"{med['pack']:.0f} | {med['numpy'] / B:.1f} | {med['v2_one'] / B:.1f} | "
              f"{med['numpy'] / med['v2']:.2f}x | {med['numpy'] / med['v2_one']:.2f}x |", flush=True)


def v2_breakdown_table(pool, batches, iters, rng) -> None:
    """Where v2's end-to-end floor goes at small batch: each CUDA call it makes,
    timed on its own with a sync after it."""
    print("| B | pack us | H2D idx us | H2D count us | kernel launch+sync us | D2H out us | sum us |")
    print("|---|---|---|---|---|---|---|")
    for B in batches:
        logits_np, legal = make_batch(pool, B, rng)
        logits = torch.from_numpy(logits_np).to(DEV)
        idx_np, count_np = pack_legal(legal)
        idx, count = torch.from_numpy(idx_np).to(DEV), torch.from_numpy(count_np).to(DEV)
        out = masked_softmax_gather(logits, idx, count)

        steps: Dict[str, Callable] = {
            "pack": lambda: pack_legal(legal),
            "h2d_idx": lambda: torch.from_numpy(idx_np).to(DEV),
            "h2d_count": lambda: torch.from_numpy(count_np).to(DEV),
            "kernel": lambda: masked_softmax_gather(logits, idx, count, out=out),
            "d2h": lambda: out.cpu(),
        }
        samples: Dict[str, List[float]] = {k: [] for k in steps}
        for _ in range(10):
            for f in steps.values():
                f()
            torch.cuda.synchronize()
        for _ in range(iters):
            for name, f in steps.items():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                f()
                torch.cuda.synchronize()
                samples[name].append(time.perf_counter() - t0)
        med = {k: statistics.median(v) * 1e6 for k, v in samples.items()}
        print(f"| {B} | {med['pack']:.0f} | {med['h2d_idx']:.0f} | {med['h2d_count']:.0f} | "
              f"{med['kernel']:.0f} | {med['d2h']:.0f} | {sum(med.values()):.0f} |", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256])
    ap.add_argument("--iters", type=int, default=300, help="end-to-end samples per path and batch size")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs a CUDA device")
    pool = game_positions()
    print(f"# {torch.cuda.get_device_name()}  torch {torch.__version__}  triton {triton.__version__}  "
          f"positions={len(pool)} (legal moves per position: mean "
          f"{np.mean([len(l) for l in pool]):.0f}, max {max(len(l) for l in pool)})")
    print()
    peak = peak_bandwidth_gbs()
    print(f"Measured peak DRAM bandwidth (1 GiB device-to-device copy): {peak:.0f} GB/s")
    print()
    print("Kernel only, GPU time (do_bench median, L2 flushed):")
    print()
    kernel_table(pool, args.batches, peak, np.random.default_rng(1))
    print()
    print("End to end, wall time: logits on the GPU -> one prior array per position on the CPU:")
    print()
    end_to_end_table(pool, args.batches, args.iters, np.random.default_rng(2))
    print()
    print("v2's end-to-end path step by step, each followed by a sync:")
    print()
    v2_breakdown_table(pool, [b for b in args.batches if b <= 8] or args.batches[:1],
                       args.iters, np.random.default_rng(3))


if __name__ == "__main__":
    main()
