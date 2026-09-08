# eval/arena.py
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from typing import List, Tuple
import numpy as np
import chess

from engine import EncodeConfig
from mcts import MCTS, eval_config
from predict import load_predictor

def play_game(predict_white, predict_black, sims: int, temperature_moves: int, max_plies: int = 300) -> str:
    board = chess.Board()
    history: List[chess.Board] = []
    encode_cfg = EncodeConfig(history=8)
    move_no = 0

    while True:
        if board.is_game_over(claim_draw=True) or move_no >= max_plies:
            oc = board.outcome(claim_draw=True)
            return oc.result() if oc else "1/2-1/2"

        predict = predict_white if board.turn else predict_black
        mcts = MCTS(predict_fn=predict, mcts_cfg=eval_config(sims=sims), encode_cfg=encode_cfg)
        mcts.set_root(board, history)
        mcts.run_simulations()
        tau = 1.0 if move_no < temperature_moves else 0.0
        move, _ = mcts.select_action(tau=tau)

        prev_board = board.copy(stack=False)
        board.push(move)
        history.append(prev_board)
        move_no += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=str, required=True, help="new model checkpoint")
    ap.add_argument("--old", type=str, required=True, help="old model checkpoint")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--temperature-moves", type=int, default=20)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--resblocks", type=int, default=12)
    ap.add_argument("--in-planes", type=int, default=102)
    ap.add_argument("--cuda-graph", action="store_true",
                    help="capture the forward pass as a CUDA graph and replay it; removes per-kernel launch overhead (CUDA only, falls back to eager)")
    args = ap.parse_args()

    # 加载预测器
    # Model size has to be passed through: load_predictor builds the network from
    # these before loading weights, and a mismatch is a hard size error, not a
    # strict=False warning. Without them arena can only ever evaluate
    # default-sized checkpoints.
    size = dict(in_planes=args.in_planes, channels=args.channels, resblocks=args.resblocks)
    p_new = load_predictor(args.new, cuda_graph=args.cuda_graph, **size)
    p_old = load_predictor(args.old, cuda_graph=args.cuda_graph, **size)

    results = []
    for g in range(args.games):
        # 颜色对称：偶数局 new=White, 奇数局 new=Black
        if g % 2 == 0:
            res = play_game(p_new, p_old, args.sims, args.temperature_moves)
        else:
            res = play_game(p_old, p_new, args.sims, args.temperature_moves)
            # 翻转视角统计：把 result 变成“新模型”的结果
            if res == "1-0":
                res = "0-1"
            elif res == "0-1":
                res = "1-0"
        results.append(res)
        print(f"[arena] game {g+1}/{args.games} result={res}", flush=True)

    w = results.count("1-0")
    l = results.count("0-1")
    d = results.count("1/2-1/2")
    print(f"[arena] NEW vs OLD  W:{w}  L:{l}  D:{d}  Win%={(w + 0.5*d) / max(1, len(results)) * 100:.1f}%")

if __name__ == "__main__":
    main()
