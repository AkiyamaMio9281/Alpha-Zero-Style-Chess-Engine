# mcts.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple, Optional, Any, Dict, List
import numpy as np
import chess

from engine import (
    ACTION_SIZE,
    EncodeConfig,
    encode_board,
    legal_moves_index_map,
    index_to_move,
    board_outcome_to_z,
)

# ========= Interfaces & Config =========
class Evaluator(Protocol):
    def __call__(self, feats_batch: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (policy_logits, value):
        - policy_logits: (B, ACTION_SIZE)
        - value:         (B,)  in [-1, 1]
        """
        ...

@dataclass
class MCTSConfig:
    sims: int = 200
    cpuct: float = 2.0
    dirichlet_alpha: float = 0.30
    dirichlet_epsilon: float = 0.25
    resign_threshold: Optional[float] = None   # e.g., -0.95
    virtual_loss: float = 1.0

# ========= Named presets =========
# selfplay.py, selfplay_uci.py (own-move branch) and eval/arena.py all used to
# hardcode this same (cpuct, dirichlet_alpha, dirichlet_epsilon) triple
# separately, which is exactly the kind of thing that drifts silently when one
# copy gets tuned and the others don't. Centralized here instead.
def self_play_config(sims: int) -> MCTSConfig:
    return MCTSConfig(sims=sims, cpuct=2.0, dirichlet_alpha=0.30, dirichlet_epsilon=0.25)

# selfplay_uci.py's expert-turn MCTS/expert mix: less root noise since the
# resulting pi is blended with the expert's own distribution.
def expert_mix_config(sims: int) -> MCTSConfig:
    return MCTSConfig(sims=sims, cpuct=2.0, dirichlet_alpha=0.30, dirichlet_epsilon=0.03)

# play_cli.py: gentler search for interactive human play (lower cpuct, low noise).
def human_play_config(sims: int) -> MCTSConfig:
    return MCTSConfig(sims=sims, cpuct=1.25, dirichlet_alpha=0.30, dirichlet_epsilon=0.03)

# ========= Node =========
class Node:
    __slots__ = ("P", "N", "W", "children", "is_terminal", "expanded")
    def __init__(self):
        self.P: Dict[int, float] = {}       # prior for legal actions
        self.N: Dict[int, int] = {}         # visit counts
        self.W: Dict[int, float] = {}       # total value
        self.children: Dict[int, Node] = {} # action -> child node
        self.is_terminal: bool = False
        self.expanded: bool = False
    def q(self, a: int) -> float:
        n = self.N.get(a, 0)
        return 0.0 if n == 0 else self.W.get(a, 0.0) / n
    def n(self, a: int) -> int:
        return self.N.get(a, 0)
    @property
    def total_visits(self) -> int:
        return sum(self.N.values()) if self.N else 0

# ========= MCTS =========
class MCTS:
    def __init__(
        self,
        predict_fn: Evaluator,
        mcts_cfg: Optional[MCTSConfig] = None,
        encode_cfg: Optional[EncodeConfig] = None,
        rng: Optional[np.random.Generator] = None,
        **kwargs: Any,
    ) -> None:
        self.predict_fn = predict_fn
        self.cfg = mcts_cfg or MCTSConfig()
        self.encode_cfg = encode_cfg or EncodeConfig()
        self.rng = rng or np.random.default_rng()

        self.root: Optional[Node] = None
        self.root_board: Optional[chess.Board] = None
        self.root_history: List[chess.Board] = []
        self._root_noise_applied: bool = False

    # ----- Root helpers -----
    def set_root(self, board: chess.Board, history: Optional[List[chess.Board]] = None) -> None:
        self.root = Node()
        self.root_board = board.copy(stack=False)
        self.root_history = list(history) if history else []
        self._root_noise_applied = False
        self._expand_root()

    def _get_root_legal_map(self) -> Dict[int, chess.Move]:
        return legal_moves_index_map(self.root_board)

    # ----- Core loop -----
    def run_simulations(self, sims: Optional[int] = None) -> None:
        sims = sims or self.cfg.sims
        for _ in range(sims):
            self._simulate_once()

    def select_action(self, tau: float = 1.0) -> Tuple[chess.Move, np.ndarray]:
        """Return (selected_move, pi[4672]) from root visit counts."""
        assert self.root is not None and self.root_board is not None
        root = self.root
        counts = np.zeros((ACTION_SIZE,), dtype=np.float32)
        for a, n in root.N.items():
            counts[a] = n

        # temperature
        if tau > 0:
            probs = counts ** (1.0 / tau)
            s = probs.sum()
            if s <= 0:
                # all zeros -> use priors
                pri = np.zeros_like(counts)
                for a, p in root.P.items():
                    pri[a] = p
                s = pri.sum()
                probs = pri / s if s > 0 else np.ones_like(counts) / len(counts)
            else:
                probs = probs / s
            # sample from legal only
            legal_map = self._get_root_legal_map()
            legal_idx = np.fromiter(legal_map.keys(), dtype=np.int32)
            legal_probs = probs[legal_idx]
            s2 = legal_probs.sum()
            if s2 <= 0:
                # fallback to uniform on legal
                legal_probs = np.ones_like(legal_probs) / len(legal_probs)
            else:
                legal_probs = legal_probs / s2
            a = int(self.rng.choice(legal_idx, p=legal_probs))
            move = legal_map[a]
            pi = probs
        else:
            # argmax visits (fallback to best prior if all zero)
            if counts.sum() == 0:
                a = max(root.P.items(), key=lambda kv: kv[1])[0]
            else:
                a = int(np.argmax(counts))
            move = self._get_root_legal_map().get(a)
            if move is None:
                # if argmax is illegal (shouldn't happen), pick best legal prior
                a = max(self._get_root_legal_map().keys(), key=lambda k: root.P.get(k, 0.0))
                move = self._get_root_legal_map()[a]
            s = counts.sum()
            pi = (counts / s) if s > 0 else counts

        return move, pi

    def apply_move(self, move: chess.Move) -> None:
        """Advance root to child reached by 'move'."""
        assert self.root is not None and self.root_board is not None
        legal_map = self._get_root_legal_map()
        inv_map = {mv: idx for idx, mv in legal_map.items()}
        if move not in inv_map:
            return
        a_idx = inv_map[move]
        child = self.root.children.get(a_idx)

        new_board = self.root_board.copy(stack=True)
        prev_board = self.root_board.copy(stack=False)
        new_board.push(move)
        new_history = list(self.root_history)
        new_history.append(prev_board)

        if child is None:
            self.root = Node()
            self.root_board = new_board
            self.root_history = new_history
            self._expand_root()
        else:
            self.root = child
            self.root_board = new_board
            self.root_history = new_history
            if not child.expanded:
                self._expand_root_from_existing()

    # ----- Simulation -----
    def _simulate_once(self) -> None:
        path: List[Tuple[Node, int, chess.Board]] = []  # (node, action, board_before_move)
        node = self.root
        board = self.root_board.copy(stack=True)
        history = list(self.root_history)

        while True:
            if board.is_game_over(claim_draw=True):
                z = board_outcome_to_z(board, perspective_white=board.turn)
                self._backup(path, z)
                return

            if not node.expanded:
                v = self._expand(node, board, history)
                self._backup(path, v)
                return

            a = self._select_by_puct(node)
            path.append((node, a, board.copy(stack=False)))
            move = self._move_from_index(node, board, a)
            board.push(move)
            history.append(path[-1][2])

            child = node.children.get(a)
            if child is None:
                child = Node()
                node.children[a] = child
            node = child

    def _expand_root(self) -> None:
        _ = self._expand(self.root, self.root_board.copy(stack=False), list(self.root_history))
        # Dirichlet noise at root
        if not self._root_noise_applied:
            legal = list(self.root.P.keys())
            if len(legal) > 0 and self.cfg.dirichlet_epsilon > 0:
                alpha = self.cfg.dirichlet_alpha
                noise = self.rng.dirichlet([alpha] * len(legal))
                for idx, a in enumerate(legal):
                    self.root.P[a] = (1 - self.cfg.dirichlet_epsilon) * self.root.P[a] + self.cfg.dirichlet_epsilon * noise[idx]
            self._root_noise_applied = True

    def _expand_root_from_existing(self) -> None:
        if not self.root.expanded:
            _ = self._expand(self.root, self.root_board.copy(stack=False), list(self.root_history))

    def _expand(self, node: Node, board: chess.Board, history: List[chess.Board]) -> float:
        if board.is_game_over(claim_draw=True):
            node.is_terminal = True
            v = board_outcome_to_z(board, perspective_white=board.turn)
            node.expanded = True
            return v

        feats = encode_board(
            board,
            prev_boards=history[-(self.encode_cfg.history-1):] if history else None,
            cfg=self.encode_cfg,
        )
        logits, values = self.predict_fn([feats])  # (1, A), (1,)
        logits = None if logits is None else np.asarray(logits)[0].astype(np.float32)
        v = float(np.asarray(values).reshape(-1)[0])

        legal_map = legal_moves_index_map(board)
        legal_idx = list(legal_map.keys())
        if not legal_idx:
            node.is_terminal = True
            node.expanded = True
            return 0.0

        pri: Dict[int, float] = {}
        if logits is None:
            p = 1.0 / len(legal_idx)
            for a in legal_idx:
                pri[a] = p
        else:
            x = np.full((ACTION_SIZE,), -1e9, dtype=np.float32)
            x[np.array(legal_idx, dtype=np.int32)] = 0.0
            x = logits + x
            x = x - np.max(x)
            ex = np.exp(x).astype(np.float32)
            s = ex.sum()
            probs = ex / s if s > 0 else np.ones_like(ex) / len(ex)
            for a in legal_idx:
                pri[a] = float(probs[a])

        node.P = pri
        node.expanded = True
        node.is_terminal = False
        return v

    def _select_by_puct(self, node: Node) -> int:
        total_n = node.total_visits
        cp = self.cfg.cpuct
        best_a, best_score = None, -1e30
        for a, p in node.P.items():
            q = node.q(a)
            n = node.n(a)
            u = cp * p * (np.sqrt(total_n + 1e-8) / (1 + n))
            s = q + u
            if s > best_score:
                best_score = s
                best_a = a
        assert best_a is not None
        return best_a

    def _move_from_index(self, node: Node, board: chess.Board, action_index: int) -> chess.Move:
        mv = index_to_move(board, action_index)
        if mv is None:
            # Fallback: pick legal with max prior
            legal = legal_moves_index_map(board)
            if legal:
                best = max(legal.keys(), key=lambda a: node.P.get(a, 0.0))
                return legal[best]
            return chess.Move.null()
        return mv

    def _backup(self, path: List[Tuple[Node, int, chess.Board]], leaf_value: float) -> None:
        v = leaf_value
        for node, a, _ in reversed(path):
            v = -v  # switch perspective (child->parent)
            node.N[a] = node.N.get(a, 0) + 1
            node.W[a] = node.W.get(a, 0.0) + v
