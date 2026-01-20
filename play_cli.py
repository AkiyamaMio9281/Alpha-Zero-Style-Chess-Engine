from __future__ import annotations
import argparse, time
from typing import List
import numpy as np
import chess

from engine import EncodeConfig, legal_moves_index_map, ACTION_SIZE, encode_board
from mcts import MCTS, MCTSConfig
# Prefer the user's net.predict if present; else fall back to a minimal local builder.
try:
    from predict import load_predictor
except Exception:
    # Fallback: minimal predictor via model.py
    from model import load_model, ModelConfig
    import torch
    def load_predictor(checkpoint, in_planes=102, channels=128, resblocks=12, amp=False, device=None):
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        m = load_model(checkpoint, cfg=ModelConfig(in_planes, channels, resblocks), device=dev)
        m.eval()
        def pred(feats_batch: List[np.ndarray]):
            x = torch.from_numpy(np.stack(feats_batch).astype(np.float32)).to(dev)
            with torch.no_grad():
                pl, v = m(x)
            return pl.cpu().numpy().astype(np.float32), v.cpu().numpy().astype(np.float32)
        return pred

def build_ai(checkpoint: str, device: str, channels: int, resblocks: int, in_planes: int):
    predict = load_predictor(
        checkpoint=checkpoint,
        in_planes=in_planes, channels=channels, resblocks=resblocks,
        amp=False, device=device if device else None,
    )
    return predict

def ai_choose_move(board: chess.Board, history: List[chess.Board], predict_fn, sims: int, history_T: int) -> chess.Move:
    # For human play we disable root Dirichlet noise.
    mcts = MCTS(
        predict_fn=predict_fn,
        mcts_cfg=MCTSConfig(sims=sims, cpuct=1.25, dirichlet_alpha=0.30, dirichlet_epsilon=0.03),
        encode_cfg=EncodeConfig(history=history_T),
    )
    mcts.set_root(board, history)
    mcts.run_simulations()
    move, _ = mcts.select_action(tau=0.0)  # greedy by visit count
    return move

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="", help="cuda or cpu")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--human", default="white", choices=["white","black"])
    ap.add_argument("--history", type=int, default=8, help="frames T used by encoder (keep 8 unless you changed it)")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--resblocks", type=int, default=12)
    ap.add_argument("--in-planes", type=int, default=102)
    args = ap.parse_args()

    predict_fn = build_ai(args.checkpoint, args.device, args.channels, args.resblocks, args.in_planes)

    board = chess.Board()
    history: List[chess.Board] = []
    move_no = 0

    print("UCI input like 'e2e4', commands: 'undo', 'fen', 'board', 'quit'.")
    print(board, "\n")

    human_white = (args.human == "white")
    if not human_white:
        mv = ai_choose_move(board, history, predict_fn, args.sims, args.history)
        prev_board = board.copy(stack=False)
        board.push(mv)
        history.append(prev_board)
        move_no += 1
        print(f"AI ({'White' if not human_white else 'Black'}): {mv.uci()}\n{board}\n")

    while True:
        try:
            s = input("你走: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n退出。"); break
        if s in ("q", "quit", "exit"): break
        if s in ("b", "board"):
            print(board, "\n"); continue
        if s == "fen":
            print(board.fen()); continue
        if s in ("u", "undo"):
            # pop AI move then human move if present
            if len(board.move_stack) > 0: board.pop()
            if len(board.move_stack) > 0: board.pop()
            if history: history.pop()
            if history: history.pop()
            print(board, "\n"); continue

        try:
            mv = chess.Move.from_uci(s)
        except Exception:
            print("格式错误，用UCI（如 e2e4）。"); continue
        if mv not in board.legal_moves:
            print("非法走子。"); continue

        prev_board = board.copy(stack=False)
        board.push(mv)
        history.append(prev_board)
        move_no += 1

        if board.is_game_over(claim_draw=True):
            oc = board.outcome(claim_draw=True)
            print(f"对局结束: {oc.result() if oc else '*'}  {oc.termination if oc else ''}")
            break

        t0 = time.time()
        mv_ai = ai_choose_move(board, history, predict_fn, args.sims, args.history)
        dt = (time.time() - t0) * 1000
        prev_board = board.copy(stack=False)
        board.push(mv_ai)
        history.append(prev_board)
        move_no += 1
        print(f"AI: {mv_ai.uci()}   ({dt:.0f} ms, sims={args.sims})\n{board}\n")

        if board.is_game_over(claim_draw=True):
            oc = board.outcome(claim_draw=True)
            print(f"对局结束: {oc.result() if oc else '*'}  {oc.termination if oc else ''}")
            break

if __name__ == "__main__":
    main()
