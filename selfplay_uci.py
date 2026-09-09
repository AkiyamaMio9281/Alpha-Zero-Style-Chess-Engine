# selfplay_uci.py (expert-vs-self data generator with curriculum knobs)
from __future__ import annotations
r"""
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
from net.batch_predict import make_batched_predictor


# ------------------------------
# UCI Expert wrapper (one instance per thread)
# ------------------------------
class UCIExpert:
    def __init__(self, path: str, threads: int = 1, hash_mb: int = 64,
                 skill_level: Optional[int] = None, cp_scale: float = 174.0):
        self.path = path
        self.sf_threads = int(threads)
        self.sf_hash = int(hash_mb)
        self.sf_skill = None if skill_level is None else int(skill_level)
        # Centipawn scale of the softmax that turns engine scores into soft
        # targets. 174 = 400/ln(10), the scale of the standard centipawn to
        # win-probability conversion, so a 174cp edge is worth e:1 in the
        # target. The previous 1200 was so flat that a 95cp spread -- most of a
        # pawn -- came out as a 1.08x ratio between best and worst candidate,
        # i.e. a target carrying almost no information about which move is good.
        self.cp_scale = float(cp_scale)
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
                beta = 1.0 / self.cp_scale
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
# Predictor builder
# ------------------------------
def build_predictor(args) -> Callable[[List[np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    if args.checkpoint and args.checkpoint.strip().lower() not in ("", "none"):
        from predict import load_predictor
        base_predict = load_predictor(
            checkpoint=args.checkpoint,
            in_planes=102, channels=args.channels, resblocks=args.resblocks,
            amp=(not args.no_amp),
            device=(args.device if args.device else None),
            cuda_graph=getattr(args, "cuda_graph", False),
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


def _pick_imitation_move(pi: np.ndarray, legal_map: Dict[int, chess.Move],
                         legal_idx: np.ndarray, policy: str,
                         rng: np.random.Generator) -> chess.Move:
    """Choose our move during the imitation phase (sims=0).

    Playing uniformly at random looks like it buys state diversity, but against
    a real engine it just loses immediately: measured against Stockfish at skill
    15, four games ran 6, 16, 32 and 42 plies and all four were losses, so the
    data was almost entirely openings and already-lost positions with z = -1 on
    every one of our samples. The network never saw a middlegame.

    Sampling from the expert's own distribution -- which has already been
    computed for the label, so it is free -- keeps games a realistic length and
    still varies the line, with --expert-cp-scale controlling how much. Use rng,
    not np.random: the worker's seed is what makes a run reproducible, and the
    global RNG is shared across threads.
    """
    if policy == "random":
        return legal_map[int(rng.choice(legal_idx))]

    probs = pi[legal_idx].astype(np.float64)
    total = probs.sum()
    if total <= 0:                      # expert gave nothing; fall back
        return legal_map[int(rng.choice(legal_idx))]
    probs /= total

    if policy == "expert-best":
        return legal_map[int(legal_idx[int(np.argmax(probs))])]
    return legal_map[int(rng.choice(legal_idx, p=probs))]


# ------------------------------
# One game vs expert
# ------------------------------
from mcts import MCTS, self_play_config, expert_mix_config


def expert_pi(expert: UCIExpert, board: chess.Board, legal_map: Dict[int, chess.Move],
              legal_idx: np.ndarray, multipv: int, movetime: int) -> Tuple[np.ndarray, bool]:
    """The expert's move distribution as a policy target over ACTION_SIZE.

    Returns (pi, from_expert). ``from_expert`` is False when the engine gave
    nothing back and the uniform fallback was used -- worth counting, because a
    uniform target is an inverted training signal, not a weak one.
    """
    dist = expert.distribution(board, multipv=multipv, movetime_ms=movetime)
    move_to_idx = {mv: a_idx for a_idx, mv in legal_map.items()}
    pi = np.zeros((ACTION_SIZE,), dtype=np.float32)
    for mv, p in dist.items():
        a_idx = move_to_idx.get(mv)
        if a_idx is not None:
            pi[a_idx] = p
    total = float(pi.sum())
    if total <= 0:
        pi[legal_idx] = 1.0 / float(legal_idx.size)
        return pi, False
    pi /= total
    return pi, True

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
                            resign_plies: int = 0,
                            eval_batch_size: int = 1,
                            imitation_move: str = "expert-sample") -> Tuple[List[np.ndarray], List[np.ndarray], List[bool], str]:
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
    resigned_side_white: Optional[bool] = None
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
                    resigned_side_white = board.turn
                    break
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
            pi_exp, _ = expert_pi(expert, board, legal_map, legal_idx,
                                  expert_multipv, expert_movetime)

            pi_mcts = np.zeros((ACTION_SIZE,), dtype=np.float32)
            if expert_alpha < 1.0 and sims > 0:
                mcts = MCTS(
                    predict_fn=predict_fn,
                    mcts_cfg=expert_mix_config(sims=sims, eval_batch_size=eval_batch_size),
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
                    mcts_cfg=self_play_config(sims=sims, eval_batch_size=eval_batch_size),
                    encode_cfg=encode_cfg,
                    rng=rng,
                )
                mcts.set_root(board, history)
                mcts.run_simulations()
                tau = 1.0 if move_no < temperature_moves else 0.0
                move, pi = mcts.select_action(tau=tau)
            else:
                # Behaviour cloning on our own turns too: label the position
                # with the expert's distribution rather than a uniform one.
                # Recording uniform here is not a weak imitation target but an
                # inverted one -- measured on a pure-imitation run, 46% of
                # samples were uniform over every legal move, training the
                # network towards "all moves are equally good".
                pi, _ = expert_pi(expert, board, legal_map, legal_idx,
                                  expert_multipv, expert_movetime)
                move = _pick_imitation_move(pi, legal_map, legal_idx, imitation_move, rng)

            feats_list.append(feats)
            pi_list.append(pi.astype(np.float32))
            persp_list.append(board.turn)

            prev_board = board.copy(stack=False)
            board.push(move)
            history.append(prev_board)
            move_no += 1
            continue

    if resigned_side_white is not None:
        winner_white = not resigned_side_white
        result = "1-0" if winner_white else "0-1"
        z_list = [1.0 if (p == winner_white) else -1.0 for p in persp_list]
        if not quiet:
            print(f"[game] resignation: result={result}, plies={move_no}")
    elif forced_draw:
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


def expert_is_white(opponent_color: str, worker_idx: int, game_idx: int) -> bool:
    """Which colour the expert takes for one game.

    "alternate" swaps every game. With a fixed colour the same side loses every
    game -- the engine is stronger than we are -- and although z then comes out
    numerically balanced, because both perspectives are recorded, it is
    perfectly correlated with which side you are. The encoding is side-to-move
    relative but a position still reveals whether you moved first, so the value
    head learns "am I the engine" rather than "who is winning". Trained on 200
    fixed-colour games it returned -1.000 for the starting position and +1.000
    after 1.e4. Offsetting by the worker index keeps the split even when each
    worker plays an odd number of games.
    """
    if opponent_color == "alternate":
        return (worker_idx + game_idx) % 2 == 0
    return opponent_color == "white"


# ------------------------------
# Worker & main
# ------------------------------
def _worker_loop(idx: int, num_games: int, sims: int, temperature_moves: int,
                 out_dir: str, max_plies: int, predict_fn, quiet: bool, encode_cfg: EncodeConfig,
                 uci_path: str, opponent_color: str, expert_alpha: float, expert_multipv: int, expert_movetime: int,
                 resign_cp: int, resign_plies: int, sf_threads: int, sf_hash: int, sf_skill: Optional[int],
                 eval_batch_size: int = 1, cp_scale: float = 174.0,
                 imitation_move: str = "expert-sample"):
    rng = np.random.default_rng(seed=(idx + 1) * 20250901)
    expert = UCIExpert(uci_path, threads=sf_threads, hash_mb=sf_hash, skill_level=sf_skill,
                       cp_scale=cp_scale)  # one engine per worker
    try:
        for g in range(num_games):
            expert_color_white = expert_is_white(opponent_color, idx, g)
            if not quiet:
                side = "W" if expert_color_white else "B"
                print(f"[thread {idx}] starting game {g+1}/{num_games} (expert plays {side})")
            feats, pi, z, result = play_one_game_vs_expert(
                predict_fn, sims, temperature_moves, rng, max_plies, quiet, encode_cfg,
                expert, expert_color_white, expert_alpha, expert_multipv, expert_movetime,
                resign_cp=resign_cp, resign_plies=resign_plies, eval_batch_size=eval_batch_size,
                imitation_move=imitation_move,
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
    ap.add_argument("--cuda-graph", action="store_true",
                    help="capture the forward pass as a CUDA graph and replay it; removes per-kernel launch overhead (CUDA only, falls back to eager)")
    ap.add_argument("--device", type=str, default="", help="e.g., cuda or cpu")

    # Predictor batching
    ap.add_argument("--batch-max", type=int, default=128, help=">=2 enables batch aggregator")
    ap.add_argument("--batch-wait-ms", type=int, default=5)
    ap.add_argument("--eval-batch-size", type=int, default=1,
                     help=">=2 evaluates that many MCTS leaves per network call within a single search "
                          "(virtual loss keeps them from collapsing onto the same path), instead of one "
                          "leaf at a time. 1 (default) is the original unbatched search, unchanged.")

    # Encoding
    ap.add_argument("--history", type=int, default=8)

    # Expert options
    ap.add_argument("--uci-path", type=str, required=True, help="path to UCI engine executable (e.g., stockfish)")
    ap.add_argument("--opponent-color", choices=["white", "black", "alternate"], default="alternate",
                    help="which colour the expert plays. 'alternate' swaps every game, which is "
                         "the default because a fixed colour makes the outcome predictable from "
                         "which side you are and the value head learns that instead of the "
                         "position.")
    ap.add_argument("--uci-movetime", type=int, default=200, help="expert think time per move in ms")
    ap.add_argument("--expert-multipv", type=int, default=6, help="use MultiPV candidates to build a soft target")
    ap.add_argument("--expert-alpha", type=float, default=0.7, help="mixing weight: pi = (1-a)*pi_mcts + a*pi_expert on expert turns")
    ap.add_argument("--imitation-move", choices=["expert-sample", "expert-best", "random"],
                    default="expert-sample",
                    help="how we pick our own move when sims=0. expert-sample draws from the "
                         "expert's distribution (realistic game length, still varied); random "
                         "loses in a handful of plies and yields only openings and lost positions.")
    ap.add_argument("--expert-cp-scale", type=float, default=174.0,
                    help="centipawn scale of the softmax over engine scores. 174 = 400/ln(10), "
                         "matching the standard centipawn-to-win-probability conversion. Larger "
                         "flattens the soft targets, smaller sharpens them.")
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

    sf_skill = None if args.skill_level < 0 else int(args.skill_level)

    # Threading: split games evenly, remainder to the first threads (matches selfplay.py)
    per_thread = [args.games // args.threads] * args.threads
    for i in range(args.games % args.threads):
        per_thread[i] += 1

    threads = []
    for i, n in enumerate(per_thread):
        if n <= 0:
            continue
        th = threading.Thread(
            target=_worker_loop,
            args=(i, n, args.sims, args.temperature_moves, args.out, args.max_plies,
                  predict_fn, args.quiet, encode_cfg,
                  args.uci_path, str(args.opponent_color), float(args.expert_alpha), int(args.expert_multipv), int(args.uci_movetime),
                  int(args.resign_cp), int(args.resign_plies), int(args.sf_threads), int(args.sf_hash), sf_skill,
                  int(args.eval_batch_size), float(args.expert_cp_scale),
                  str(args.imitation_move)),
            daemon=True
        )
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

    print(f"[selfplay_uci] finished. shards saved to: {args.out}")

if __name__ == "__main__":
    main()
