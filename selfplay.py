# selfplay.py
from __future__ import annotations
import engine
import os
import time
import uuid
import argparse
import threading
from typing import List, Tuple, Callable, Optional
import queue
import numpy as np
import chess
from engine import legal_moves_index_map, ACTION_SIZE



from engine import (
    EncodeConfig,
    encode_board,
    board_outcome_to_z,
)
from mcts import MCTS, MCTSConfig

# ===== 即时日志 =====
def log(*a, **k):
    print(*a, **k, flush=True)

# ===== Dummy evaluator（均匀先验 + 价值0）=====
def dummy_predict(feats_batch: List[np.ndarray]):
    B = len(feats_batch)
    logits = np.zeros((B, 4672), dtype=np.float32)
    values = np.zeros((B,), dtype=np.float32)
    return logits, values

# ===== 可选：小型批量推理聚合器（把多个线程的叶子攒成 batch 一起推理）=====
class _Req:
    __slots__ = ("x", "evt", "out_logits", "out_value")
    def __init__(self, x: np.ndarray):
        self.x = x
        self.evt = threading.Event()
        self.out_logits: Optional[np.ndarray] = None
        self.out_value: Optional[float] = None

class PredictBatcher:
    def __init__(self, base_predict: Callable[[List[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                 max_batch: int = 128, max_wait_ms: int = 5):
        self.base_predict = base_predict
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.q: "queue.Queue[_Req]" = queue.Queue()
        self.stop = False
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        while not self.stop:
            try:
                first = self.q.get(timeout=0.01)
            except queue.Empty:
                continue
            batch = [first]
            t0 = time.time()
            while len(batch) < self.max_batch:
                remain = self.max_wait_ms/1000.0 - (time.time() - t0)
                if remain <= 0: break
                try:
                    nxt = self.q.get(timeout=remain)
                    batch.append(nxt)
                except queue.Empty:
                    break
            xs = [r.x for r in batch]
            logits, values = self.base_predict(xs)  # (B,A), (B,)
            for i, r in enumerate(batch):
                r.out_logits = logits[i]
                r.out_value  = float(values[i])
                r.evt.set()

    def predict_one(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        r = _Req(x)
        self.q.put(r)
        r.evt.wait()
        return r.out_logits, r.out_value  # type: ignore

def make_batched_predictor(base_predict: Callable[[List[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                           max_batch: int, max_wait_ms: int):
    """
    返回与 base_predict 相同签名的函数，但内部用队列把每个样本汇聚成大 batch。
    """
    batcher = PredictBatcher(base_predict, max_batch=max_batch, max_wait_ms=max_wait_ms)
    def predict(feats_batch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        outs_logits, outs_values = [], []
        for x in feats_batch:
            lg, v = batcher.predict_one(x)
            outs_logits.append(lg)
            outs_values.append(v)
        return np.stack(outs_logits), np.asarray(outs_values, dtype=np.float32)
    return predict

def _can_force_draw(board: chess.Board) -> bool:
    try:
        if hasattr(board, "can_claim_draw") and board.can_claim_draw():
            return True
    except Exception:
        pass
    for m in ("can_claim_threefold_repetition","is_fivefold_repetition",
              "can_claim_fifty_moves","is_seventyfive_moves"):
        if hasattr(board, m) and getattr(board, m)():
            return True
    return False

def play_one_game(predict_fn, sims: int, temperature_moves: int,
                  rng: np.random.Generator, max_plies: int, quiet: bool,
                  encode_cfg: EncodeConfig):
    board = chess.Board()

    feats_list: List[np.ndarray] = []
    pi_list: List[np.ndarray] = []
    persp_list: List[bool] = []

    move_no = 0
    history: List[chess.Board] = []

    if not quiet:
        log(f"[game] start, stm={'W' if board.turn else 'B'}")
    forced_draw = False

    while True:
        if board.is_game_over(claim_draw=True): break
        if _can_force_draw(board): forced_draw = True; break
        if move_no >= max_plies:   forced_draw = True; break

        if (move_no % 5) == 0 and not quiet:
            try:
                n_legal = board.legal_moves.count()  # python-chess 1.10+
            except Exception:
                n_legal = len(list(board.legal_moves))
            log(f"[game] move_no={move_no}, legal={n_legal}")

        # 如果 sims <= 0：直接在合法步里均匀抽样（更快地攒随机数据）
        if sims <= 0:
            legal = list(board.legal_moves)
            # 记录特征与均匀 π
            # 记录特征与“合法步均匀” π
            feats = encode_board(board, prev_boards=history[-(encode_cfg.history-1):] if history else None, cfg=encode_cfg)

            legal_map = legal_moves_index_map(board)               # {idx -> Move}
            legal_idx = np.fromiter(legal_map.keys(), dtype=np.int32)
            pi = np.zeros((ACTION_SIZE,), dtype=np.float32)
            if legal_idx.size == 0:
                break
            pi[legal_idx] = 1.0 / legal_idx.size

            feats_list.append(feats)
            pi_list.append(pi)
            persp_list.append(board.turn)

# 选招并推进
            mv = legal_map[int(legal_idx[rng.integers(legal_idx.size)])]
            prev_board = board.copy(stack=False)
            board.push(mv)
            history.append(prev_board)
            move_no += 1
            continue


        # 用 MCTS 搜索
        mcts = MCTS(
            predict_fn=predict_fn,
            mcts_cfg=MCTSConfig(sims=sims, cpuct=2.0, dirichlet_alpha=0.30, dirichlet_epsilon=0.25),
            encode_cfg=encode_cfg,
            rng=rng,
        )
        mcts.set_root(board, history)
        mcts.run_simulations()

        tau = 1.0 if move_no < temperature_moves else 0.0
        move, pi = mcts.select_action(tau=tau)

        feats = encode_board(board, prev_boards=history[-(encode_cfg.history-1):] if history else None, cfg=encode_cfg)
        feats_list.append(feats)
        pi_list.append(pi.astype(np.float32))
        persp_list.append(board.turn)

        prev_board = board.copy(stack=False)
        board.push(move)
        history.append(prev_board)
        move_no += 1

    if forced_draw:
        result = "1/2-1/2"; z_list = [0.0 for _ in persp_list]
        if not quiet:
            log(f"[game] forced draw: result={result}, plies={move_no}")
    else:
        oc = board.outcome(claim_draw=True)
        result = oc.result() if oc is not None else "*"
        z_list = [board_outcome_to_z(board, perspective_white=p) for p in persp_list]
        if not quiet:
            log(f"[game] end result={result}, plies={move_no}")

    return feats_list, pi_list, z_list, result

def save_shard(out_dir: str, feats_list, pi_list, z_list, result: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    Cmax = max(f.shape[0] for f in feats_list) if feats_list else 102
    fixed_feats = []
    for f in feats_list:
        if f.shape[0] == Cmax:
            fixed_feats.append(f)
        else:
            pad = np.zeros((Cmax - f.shape[0], 8, 8), dtype=f.dtype)
            fixed_feats.append(np.concatenate([f, pad], axis=0))
    arr_feats = np.stack(fixed_feats) if fixed_feats else np.zeros((0, Cmax, 8, 8), dtype=np.float32)
    arr_pi    = np.stack(pi_list) if pi_list else np.zeros((0, 4672), dtype=np.float32)
    arr_z     = np.asarray(z_list, dtype=np.float32) if z_list else np.zeros((0,), dtype=np.float32)

    shard_id = f"selfplay_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    path = os.path.join(out_dir, shard_id + ".npz")
    np.savez_compressed(path, feats=arr_feats, pi=arr_pi, z=arr_z, result=result)
    return path

def _build_predict_fn(args) -> Callable[[List[np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    # 选择预测器：如提供 checkpoint 就加载真模型；否则用 dummy
    if args.checkpoint and args.checkpoint.strip().lower() not in ("", "none"):
        try:
            # 为了兼容你后来把 net.model 改成 model，这里优先尝试 net.predict，再从 model 中兜底
            try:
                from predict import load_predictor  # 先尝试包路径
            except Exception:
                # 直接从顶层 model 导入
                import importlib
                mod = importlib.import_module("model")
                def load_predictor(checkpoint, in_planes=102, channels=128, resblocks=12, amp=True, device=None):
                    m = mod.load_model(checkpoint, cfg=mod.ModelConfig(in_planes, channels, resblocks),
                                       device=(device if device else None))
                    # 期望 mod 里有 predict(model, feats) 或 model(feats)
                    def pred(feats_list: List[np.ndarray]):
                        import torch, numpy as _np
                        x = _np.stack(feats_list).astype(_np.float32)
                        x = torch.from_numpy(x).to(next(m.parameters()).device)
                        with torch.no_grad():
                            pl, v = m(x)
                        return pl.detach().float().cpu().numpy(), v.detach().float().cpu().numpy()
                    return pred
            base_predict = load_predictor(
                checkpoint=args.checkpoint,
                in_planes=102, channels=args.channels, resblocks=args.resblocks,
                amp=(not args.no_amp),
                device=(args.device if args.device else None),
            )
            # 批量聚合开关
            if args.batch_max > 1:
                predict_fn = make_batched_predictor(base_predict, max_batch=args.batch_max, max_wait_ms=args.batch_wait_ms)
                log(f"[selfplay] using checkpoint (batched x{args.batch_max}): {args.checkpoint}")
            else:
                predict_fn = base_predict
                log(f"[selfplay] using checkpoint: {args.checkpoint}")
            return predict_fn
        except Exception as e:
            log(f"[selfplay] failed to load checkpoint ({args.checkpoint}), fallback to dummy. Error: {e}")
            return dummy_predict
    else:
        log("[selfplay] no checkpoint provided; using dummy evaluator.")
        return dummy_predict

def _worker_loop(idx: int, num_games: int, sims: int, temperature_moves: int,
                 out_dir: str, max_plies: int, predict_fn, quiet: bool, encode_cfg: EncodeConfig):
    rng = np.random.default_rng(seed=(idx + 1) * 20250814)
    for g in range(num_games):
        if not quiet:
            log(f"[thread {idx}] starting game {g+1}/{num_games} (sims={sims}, max_plies={max_plies})")
        feats, pi, z, result = play_one_game(predict_fn, sims, temperature_moves, rng, max_plies, quiet, encode_cfg)
        p = save_shard(out_dir, feats, pi, z, result)
        if not quiet:
            log(f"[thread {idx}] game {g+1}/{num_games} saved: {p}  result={result}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--temperature-moves", type=int, default=20)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--out", type=str, default="data/shards")
    # 可选 checkpoint 与模型规模
    ap.add_argument("--checkpoint", type=str, default="", help="optional model checkpoint for inference")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--resblocks", type=int, default=12)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", type=str, default="", help="e.g., cuda or cpu")
    # 批量推理参数
    ap.add_argument("--batch-max", type=int, default=128, help=">=2 启用批量聚合；1 表示关闭")
    ap.add_argument("--batch-wait-ms", type=int, default=5)
    # 静默
    ap.add_argument("--quiet", action="store_true")
    # 编码历史长度（与 engine.EncodeConfig 对齐）
    ap.add_argument("--history", type=int, default=8)
    args = ap.parse_args()

    encode_cfg = EncodeConfig(history=args.history)

    predict_fn = _build_predict_fn(args)

    per_thread = [args.games // args.threads] * args.threads
    for i in range(args.games % args.threads):
        per_thread[i] += 1

    threads: List[threading.Thread] = []
    for i, n in enumerate(per_thread):
        if n <= 0:
            continue
        th = threading.Thread(
            target=_worker_loop,
            args=(i, n, args.sims, args.temperature_moves, args.out, args.max_plies, predict_fn, args.quiet, encode_cfg),
            daemon=True
        )
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

    log(f"self-play finished. shards saved to: {args.out}")

if __name__ == "__main__":
    main()
