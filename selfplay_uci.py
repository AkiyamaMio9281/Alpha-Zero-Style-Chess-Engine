# selfplay_uci.py (expert-vs-self data generator with curriculum knobs)
from __future__ import annotations
"""
生成“和UCI专家（如Stockfish）对弈”的训练数据（AlphaZero风格 π、z），可选混合自家MCTS分布。
支持：每线程独立引擎、防崩重启、Skill Level、早投降、纯模仿/均匀先验等。

示例（纯模仿启动）：
python selfplay_uci.py ^
  --games 200 --threads 4 ^
  --sims 0 --temperature-moves 0 ^
  --checkpoint none ^
  --out "data/vs_expert_boot" ^
  --uci-path "C:\path\to\stockfish.exe" ^
  --opponent-color black ^
  --uci-movetime 100 ^
  --expert-multipv 1 ^
  --expert-alpha 1.0 ^
  --skill-level 5

示例（混回自博弈）：
python selfplay_uci.py ^
  --games 120 --threads 4 ^
  --sims 300 --temperature-moves 12 ^
  --checkpoint "C:\...\ckpt\model_ep3_step9000.pt" ^
  --out "data/vs_expert_mix" ^
  --uci-path "C:\path\to\stockfish.exe" ^
  --opponent-color black ^
  --uci-movetime 250 ^
  --expert-multipv 6 ^
  --expert-alpha 0.6 ^
  --skill-level 15
"""
import os
import time
import uuid
import argparse
import threading
from typing import List, Tuple, Callable, Optional, Dict
import queue
import numpy as np
import chess
import chess.engine as ce
import asyncio
import concurrent.futures

from engine import (
    ACTION_SIZE,
    EncodeConfig,
    encode_board,
    legal_moves_index_map,
    board_outcome_to_z,
)

allow_resign = False  # 默认关


# ------------------------------
# UCI Expert wrapper (one instance per thread)
# ------------------------------
class UCIExpert:
    def __init__(self, path: str, threads: int = 1, hash_mb: int = 64, skill_level: Optional[int] = None):
        self.path = path
        self.sf_threads = int(threads)
        self.sf_hash = int(hash_mb)
        self.sf_skill = None if skill_level is None else int(skill_level)
        self.engine = ce.SimpleEngine.popen_uci(path)
        self._configure()

    def _configure(self):
        try:
            opts = {"Threads": self.sf_threads, "Hash": self.sf_hash}
            if self.sf_skill is not None:
                # Stockfish supports "Skill Level" 0..20; some engines may ignore.
                opts["Skill Level"] = self.sf_skill
            self.engine.configure(opts)
        except Exception:
            pass  # ignore unsupported options

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass

    def restart(self):
        try:
            self.close()
        finally:
            self.engine = ce.SimpleEngine.popen_uci(self.path)
            self._configure()

    def bestmove(self, board: chess.Board, movetime_ms: int) -> chess.Move:
        info = self.engine.play(board, ce.Limit(time=movetime_ms / 1000.0))
        return info.move

    def analyse_cp(self, board: chess.Board, movetime_ms: int = 30) -> Optional[int]:
        """Return centipawn eval from side-to-move POV (positive=good for side-to-move)."""
        try:
            info = self.engine.analyse(board, ce.Limit(time=movetime_ms / 1000.0))
            if "score" not in info:
                return None
            sc = info["score"].white()
            cp = sc.score(mate_score=100000)
            if not board.turn:
                cp = -cp
            return int(cp)
        except Exception:
            return None

    def distribution(self, board: chess.Board, multipv: int, movetime_ms: int) -> Dict[chess.Move, float]:
        """Return a probability distribution over top MultiPV moves derived from centipawn scores.
        Robust to engines/driver quirks: retries (fallbacks) & restarts on CancelledError/AssertionError.
        """
        limit = ce.Limit(time=movetime_ms / 1000.0)
        for attempt in (0, 1, 2):
            try:
                if attempt == 0:
                    infos = self.engine.analyse(board, limit, multipv=multipv, info=ce.INFO_ALL)
                else:
                    infos = self.engine.analyse(board, limit, multipv=multipv)
                cand = []
                for inf in infos:
                    if "pv" in inf and inf["pv"] and "score" in inf:
                        mv = inf["pv"][0]
                        sc = inf["score"].white()  # unify as white's POV
                        cp = sc.score(mate_score=100000)
                        if not board.turn:  # if side to move is black, flip
                            cp = -cp
                        cand.append((mv, float(cp)))
                if not cand:
                    return {}
                beta = 1.0 / 1200.0  # temperature for softmax over centipawns
                xs = np.array([c for _, c in cand], dtype=np.float64)
                p = np.exp(beta * (xs - xs.max()))
                p = p / p.sum()
                return {mv: float(pi) for (mv, _), pi in zip(cand, p)}
            except (concurrent.futures.CancelledError, asyncio.CancelledError, AssertionError, RuntimeError):
                try:
                    self.restart()
                except Exception:
                    pass
                if attempt >= 2:
                    return {}
            except Exception:
                if attempt >= 1:
                    return {}
        return {}

# ------------------------------
# Batched predictor helper
# ------------------------------
_Req = Tuple[np.ndarray, "queue.Queue[Tuple[np.ndarray, float]]"]

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
                remain = self.max_wait_ms / 1000.0 - (time.time() - t0)
                if remain <= 0:
                    break
                try:
                    nxt = self.q.get(timeout=remain)
                    batch.append(nxt)
                except queue.Empty:
                    break
            feats_batch = [x for x, _ in batch]
            logits, values = self.base_predict(feats_batch)
            for i, (_, out_q) in enumerate(batch):
                out_q.put((logits[i], float(values[i])))

    def predict_one(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        out_q: "queue.Queue[Tuple[np.ndarray, float]]" = queue.Queue(maxsize=1)
        self.q.put((x, out_q))
        logits, v = out_q.get()
        return logits, v

def make_batched_predictor(base_predict: Callable[[List[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                           max_batch: int, max_wait_ms: int):
    batcher = PredictBatcher(base_predict, max_batch=max_batch, max_wait_ms=max_wait_ms)
    def predict(feats_batch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        outs_logits, outs_values = [], []
        for x in feats_batch:
            lg, v = batcher.predict_one(x)
            outs_logits.append(lg)
            outs_values.append(v)
        return np.stack(outs_logits), np.asarray(outs_values, dtype=np.float32)
    return predict

# ------------------------------
# Predictor builder
# ------------------------------
def build_predictor(args) -> Callable[[List[np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    if args.checkpoint and args.checkpoint.strip().lower() not in ("", "none"):
        try:
            from predict import load_predictor
        except Exception:
            import importlib, torch
            mod = importlib.import_module("model")
            def load_predictor(checkpoint, in_planes=102, channels=128, resblocks=12, amp=True, device=None):
                m = mod.load_model(checkpoint, cfg=mod.ModelConfig(in_planes, channels, resblocks),
                                   device=(device if device else None))
                m.eval()
                def pred(feats_list: List[np.ndarray]):
                    x = np.stack(feats_list).astype(np.float32)
                    xt = torch.from_numpy(x).to(next(m.parameters()).device)
                    with torch.no_grad():
                        pl, v = m(xt)
                    return pl.detach().float().cpu().numpy(), v.detach().float().cpu().numpy()
                return pred
        base_predict = load_predictor(
            checkpoint=args.checkpoint,
            in_planes=102, channels=args.channels, resblocks=args.resblocks,
            amp=(not args.no_amp),
            device=(args.device if args.device else None),
        )
    else:
        def base_predict(feats_list: List[np.ndarray]):
            B = len(feats_list)
            logits = np.zeros((B, ACTION_SIZE), dtype=np.float32)
            values = np.zeros((B,), dtype=np.float32)
            return logits, values

    if args.batch_max and args.batch_max > 1:
        return make_batched_predictor(base_predict, max_batch=args.batch_max, max_wait_ms=args.batch_wait_ms)
    return base_predict

# ------------------------------
# Save .npz shard
# ------------------------------
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
    arr_pi    = np.stack(pi_list) if pi_list else np.zeros((0, ACTION_SIZE), dtype=np.float32)
    arr_z     = np.asarray(z_list, dtype=np.float32) if z_list else np.zeros((0,), dtype=np.float32)
    shard_id = f"expert_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    path = os.path.join(out_dir, shard_id + ".npz")
    np.savez_compressed(path, feats=arr_feats, pi=arr_pi, z=arr_z, result=result)
    return path

# ------------------------------
# One game vs expert
# ------------------------------
from mcts import MCTS, MCTSConfig

def play_one_game_vs_expert(predict_fn,
                            sims: int,
                            temperature_moves: int,
                            rng: np.random.Generator,
                            max_plies: int,
                            quiet: bool,
                            encode_cfg: EncodeConfig,
                            expert: UCIExpert,
                            expert_color_white: bool,
                            expert_alpha: float,
                            expert_multipv: int,
                            expert_movetime: int,
                            resign_cp: int = 0,
                            resign_plies: int = 0) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool], str]:
    board = chess.Board()
    feats_list: List[np.ndarray] = []
    pi_list: List[np.ndarray] = []
    persp_list: List[bool] = []
    history: List[chess.Board] = []
    move_no = 0
    bad_counter = 0

    if not quiet:
        print(f"[game] start vs UCI {'W' if expert_color_white else 'B'}  stm={'W' if board.turn else 'B'}")

    forced_draw = False
    while True:
        if board.is_game_over(claim_draw=True):
            break
        if max_plies > 0 and move_no >= max_plies:
            forced_draw = True
            break

        # Optional resignation if position is hopeless for side-to-move (our model's side only)
        if resign_cp and resign_plies and (board.turn != expert_color_white):
            cp = expert.analyse_cp(board, movetime_ms=20)
            if (cp is not None) and (cp < -abs(resign_cp)):
                bad_counter += 1
                if bad_counter >= resign_plies:
                    if not quiet:
                        print(f"[game] resign triggered at ply {move_no} (cp={cp})")
                    if allow_resign:
                        pass  
            else:
                bad_counter = 0

        feats = encode_board(board, prev_boards=history[-(encode_cfg.history-1):] if history else None, cfg=encode_cfg)
        legal_map = legal_moves_index_map(board)  # {idx -> Move}
        legal_idx = np.fromiter(legal_map.keys(), dtype=np.int32)
        if legal_idx.size == 0:
            break

        side_is_expert = (board.turn is True) == expert_color_white

        if side_is_expert:
            # Expert distribution
            dist = expert.distribution(board, multipv=expert_multipv, movetime_ms=expert_movetime)
            pi_exp = np.zeros((ACTION_SIZE,), dtype=np.float32)
            for mv, p in dist.items():
                for a_idx, mv_leg in legal_map.items():
                    if mv_leg == mv:
                        pi_exp[a_idx] = p
                        break
            s = float(pi_exp.sum())
            if s <= 0:
                pi_exp[legal_idx] = 1.0 / float(legal_idx.size)
            else:
                pi_exp /= s

            pi_mcts = np.zeros((ACTION_SIZE,), dtype=np.float32)
            if expert_alpha < 1.0 and sims > 0:
                mcts = MCTS(
                    predict_fn=predict_fn,
                    mcts_cfg=MCTSConfig(sims=sims, cpuct=2.0, dirichlet_alpha=0.30, dirichlet_epsilon=0.03),
                    encode_cfg=encode_cfg,
                    rng=rng,
                )
                mcts.set_root(board, history)
                mcts.run_simulations()
                _, pi_tmp = mcts.select_action(tau=1.0 if move_no < temperature_moves else 0.0)
                pi_mcts = pi_tmp.astype(np.float32)

            alpha = float(np.clip(expert_alpha, 0.0, 1.0))
            pi = (1.0 - alpha) * pi_mcts + alpha * pi_exp

            feats_list.append(feats)
            pi_list.append(pi.astype(np.float32))
            persp_list.append(board.turn)

            mv = expert.bestmove(board, movetime_ms=expert_movetime)
            if (mv is None) or (mv not in list(board.legal_moves)):
                a = int(np.argmax(pi))
                mv = legal_map.get(a, None) or list(board.legal_moves)[0]

            prev_board = board.copy(stack=False)
            board.push(mv)
            history.append(prev_board)
            move_no += 1
            continue

        else:
            # Our move via MCTS (or random/uniform if sims==0)
            if sims > 0:
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
            else:
                # Uniform over legal moves
                pi = np.zeros((ACTION_SIZE,), dtype=np.float32)
                pi[legal_idx] = 1.0 / float(legal_idx.size)
                # Sample a move uniformly
                a = int(np.random.choice(legal_idx))
                move = legal_map[a]

            feats_list.append(feats)
            pi_list.append(pi.astype(np.float32))
            persp_list.append(board.turn)

            prev_board = board.copy(stack=False)
            board.push(move)
            history.append(prev_board)
            move_no += 1
            continue

    if forced_draw:
        result = "1/2-1/2"
        z_list = [0.0 for _ in persp_list]
        if not quiet:
            print(f"[game] forced draw: result={result}, plies={move_no}")
    else:
        oc = board.outcome(claim_draw=True)
        result = oc.result() if oc is not None else "*"
        z_list = [board_outcome_to_z(board, perspective_white=p) for p in persp_list]
        if not quiet:
            print(f"[game] end result={result}, plies={move_no}")

    return feats_list, pi_list, z_list, result

# ------------------------------
# Worker & main
# ------------------------------
def _worker_loop(idx: int, num_games: int, sims: int, temperature_moves: int,
                 out_dir: str, max_plies: int, predict_fn, quiet: bool, encode_cfg: EncodeConfig,
                 uci_path: str, expert_color_white: bool, expert_alpha: float, expert_multipv: int, expert_movetime: int,
                 resign_cp: int, resign_plies: int, sf_threads: int, sf_hash: int, sf_skill: Optional[int]):
    rng = np.random.default_rng(seed=(idx + 1) * 20250901)
    expert = UCIExpert(uci_path, threads=sf_threads, hash_mb=sf_hash, skill_level=sf_skill)  # one engine per worker
    try:
        for g in range(num_games):
            if not quiet:
                print(f"[thread {idx}] starting game {g+1}/{num_games}")
            feats, pi, z, result = play_one_game_vs_expert(
                predict_fn, sims, temperature_moves, rng, max_plies, quiet, encode_cfg,
                expert, expert_color_white, expert_alpha, expert_multipv, expert_movetime,
                resign_cp=resign_cp, resign_plies=resign_plies
            )
            p = save_shard(out_dir, feats, pi, z, result)
            if not quiet:
                print(f"[thread {idx}] game {g+1}/{num_games} saved: {p}  result={result}")
    finally:
        expert.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--temperature-moves", type=int, default=20)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--out", type=str, default="data/shards")

    # Model
    ap.add_argument("--checkpoint", type=str, default="", help="optional model checkpoint for inference; use 'none' for dummy")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--resblocks", type=int, default=12)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", type=str, default="", help="e.g., cuda or cpu")

    # Predictor batching
    ap.add_argument("--batch-max", type=int, default=128, help=">=2 enables batch aggregator")
    ap.add_argument("--batch-wait-ms", type=int, default=5)

    # Encoding
    ap.add_argument("--history", type=int, default=8)

    # Expert options
    ap.add_argument("--uci-path", type=str, required=True, help="path to UCI engine executable (e.g., stockfish)")
    ap.add_argument("--opponent-color", choices=["white", "black"], default="black", help="which color the expert plays")
    ap.add_argument("--uci-movetime", type=int, default=200, help="expert think time per move in ms")
    ap.add_argument("--expert-multipv", type=int, default=6, help="use MultiPV candidates to build a soft target")
    ap.add_argument("--expert-alpha", type=float, default=0.7, help="mixing weight: pi = (1-a)*pi_mcts + a*pi_expert on expert turns")
    ap.add_argument("--sf-threads", type=int, default=1, help="Stockfish Threads per worker")
    ap.add_argument("--sf-hash", type=int, default=64, help="Stockfish Hash (MB) per worker")
    ap.add_argument("--skill-level", type=int, default=-1, help="Stockfish Skill Level (0..20). Negative = don't set.")

    # Early resignation (optional)
    ap.add_argument("--resign-cp", type=int, default=0, help="Trigger resignation if eval < -resign-cp for resign-plies plies; 0=disabled")
    ap.add_argument("--resign-plies", type=int, default=0, help="Number of consecutive plies below threshold to resign; 0=disabled")

    # Misc
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args()

    # Build predictor
    predict_fn = build_predictor(args)
    encode_cfg = EncodeConfig(history=args.history)

    expert_color_white = (args.opponent_color == "white")
    sf_skill = None if args.skill_level < 0 else int(args.skill_level)

    # Threading: split games roughly evenly
    n = max(1, args.games // max(1, args.threads))
    threads = []
    for i in range(args.threads):
        th = threading.Thread(
            target=_worker_loop,
            args=(i, n, args.sims, args.temperature_moves, args.out, args.max_plies,
                  predict_fn, args.quiet, encode_cfg,
                  args.uci_path, expert_color_white, float(args.expert_alpha), int(args.expert_multipv), int(args.uci_movetime),
                  int(args.resign_cp), int(args.resign_plies), int(args.sf_threads), int(args.sf_hash), sf_skill),
            daemon=True
        )
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

    print(f"[selfplay_uci] finished. shards saved to: {args.out}")

if __name__ == "__main__":
    main()
