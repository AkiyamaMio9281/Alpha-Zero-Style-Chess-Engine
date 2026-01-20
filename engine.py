# engine.py
# AlphaZero-style chess encoding & action mapping utilities
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
import chess

# === 8x8x73 action space ===
#  - 56 sliding planes: 8 directions * steps 1..7
#  - 8 knight planes
#  - 9 under-promotion planes: 3 dirs (L,F,R) × (N,B,R)
ACTION_SIZE = 8 * 8 * 73  # 4672

SLIDE_DIRS: List[Tuple[int,int]] = [
    (1, 0),  (-1, 0),  (0, 1),  (0, -1),   # E, W, N, S
    (1, 1),  (-1, 1),  (1, -1), (-1, -1),  # NE, NW, SE, SW
]
KNIGHT_DELTAS: List[Tuple[int,int]] = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2)
]
# 兵升变的三个方向（相对于“行棋方朝上”的坐标系）
PROMO_DIRS: List[Tuple[int,int]] = [(-1, 1), (0, 1), (1, 1)]
PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]  # Q 用滑走平面

@dataclass
class EncodeConfig:
    history: int = 8  # AlphaZero uses T=8
    include_castling: bool = True
    include_side_to_move: bool = True

# ------------------------------
# Helpers (orientation & mapping)
# ------------------------------
def _orient_square(sq: int, stm_white: bool) -> int:
    return sq if stm_white else chess.square_mirror(sq)

def _deorient_square(sq_o: int, stm_white: bool) -> int:
    return sq_o if stm_white else chess.square_mirror(sq_o)

def _sq_to_xy(sq: int) -> tuple[int, int]:
    return chess.square_file(sq), chess.square_rank(sq)

def _xy_to_sq(x: int, y: int) -> int:
    return chess.square(x, y)

def _inside(x: int, y: int) -> bool:
    return 0 <= x < 8 and 0 <= y < 8

def _from_plane_index(from_sq_oriented: int, plane: int) -> int:
    return from_sq_oriented * 73 + plane

# ------------------------------
# Move <-> index (8x8x73)
# ------------------------------
def move_to_index(board: chess.Board, move: chess.Move) -> Optional[int]:
    """Map a legal chess.Move to AlphaZero 8x8x73 index (relative to side-to-move)."""
    stm_white = board.turn
    fr = move.from_square
    to = move.to_square

    fr_o = _orient_square(fr, stm_white)
    to_o = _orient_square(to, stm_white)
    fx, fy = _sq_to_xy(fr_o)
    tx, ty = _sq_to_xy(to_o)
    dx, dy = tx - fx, ty - fy

    # Under-promotions: planes 64..72 (3 dirs × 3 pieces)
    if move.promotion in PROMO_PIECES:
        ndx = 0 if dx == 0 else (1 if dx > 0 else -1)
        if dy == 1 and ndx in (-1, 0, 1):
            dir_idx = [-1, 0, 1].index(ndx)
            piece_idx = PROMO_PIECES.index(move.promotion)
            plane = 64 + dir_idx * 3 + piece_idx
            return _from_plane_index(fr_o, plane)
        return None

    # Knights: planes 56..63
    if (dx, dy) in KNIGHT_DELTAS:
        plane = 56 + KNIGHT_DELTAS.index((dx, dy))
        return _from_plane_index(fr_o, plane)

    # Sliding / straights / diagonals
    if dx == 0 and dy == 0:
        return None
    sdx = 0 if dx == 0 else (1 if dx > 0 else -1)
    sdy = 0 if dy == 0 else (1 if dy > 0 else -1)
    if (sdx, sdy) in SLIDE_DIRS:
        dir_idx = SLIDE_DIRS.index((sdx, sdy))
        steps = max(abs(dx), abs(dy))
        if 1 <= steps <= 7:
            plane = dir_idx * 7 + (steps - 1)  # 0..55
            return _from_plane_index(fr_o, plane)
    return None

def index_to_move(board: chess.Board, action_index: int) -> Optional[chess.Move]:
    """Decode AlphaZero index back to a chess.Move for the CURRENT board."""
    if action_index < 0 or action_index >= ACTION_SIZE:
        return None

    stm_white = board.turn
    from_o = action_index // 73
    plane  = action_index %  73
    fx, fy = _sq_to_xy(from_o)

    tx = ty = None
    promo_piece = None

    if plane <= 55:
        dir_idx = plane // 7
        steps   = (plane % 7) + 1
        dx, dy  = SLIDE_DIRS[dir_idx]
        tx, ty  = fx + dx * steps, fy + dy * steps
    elif 56 <= plane <= 63:
        dx, dy = KNIGHT_DELTAS[plane - 56]
        tx, ty = fx + dx, fy + dy
    else:
        dir_idx   = (plane - 64) // 3
        piece_idx = (plane - 64) %  3
        dx, dy    = PROMO_DIRS[dir_idx]
        tx, ty    = fx + dx, fy + dy
        promo_piece = PROMO_PIECES[piece_idx]

    if tx is None or not _inside(tx, ty):
        return None

    to_o = _xy_to_sq(tx, ty)
    fr   = _deorient_square(from_o, stm_white)
    to   = _deorient_square(to_o,   stm_white)

    if promo_piece is None:
        piece = board.piece_at(fr)
        if piece and piece.piece_type == chess.PAWN:
            last_rank = 7 if stm_white else 0
            if chess.square_rank(to) == last_rank:
                promo_piece = chess.QUEEN

    mv = chess.Move(fr, to, promotion=promo_piece)
    return mv if mv in board.legal_moves else None

def legal_moves_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros((ACTION_SIZE,), dtype=bool)
    for mv in board.legal_moves:
        idx = move_to_index(board, mv)
        if idx is not None:
            mask[idx] = True
    return mask

def legal_moves_index_map(board: chess.Board) -> Dict[int, chess.Move]:
    mapping: Dict[int, chess.Move] = {}
    for mv in board.legal_moves:
        idx = move_to_index(board, mv)
        if idx is not None:
            mapping[idx] = mv
    return mapping

def legal_action_indices(board: chess.Board) -> np.ndarray:
    return np.flatnonzero(legal_moves_mask(board)).astype(np.int32)

def index_to_move_from_legal(board: chess.Board, action_index: int) -> Optional[chess.Move]:
    return legal_moves_index_map(board).get(action_index)

# ------------------------------
# Board encoding (features)
# ------------------------------
PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]

def _planes_for_color(b: chess.Board, color: bool, perspective_white: bool) -> np.ndarray:
    planes = np.zeros((6, 8, 8), dtype=np.float32)
    for i, p in enumerate(PIECE_ORDER):
        for sq in b.pieces(p, color):
            sq_o = _orient_square(sq, perspective_white)
            x, y = _sq_to_xy(sq_o)
            planes[i, y, x] = 1.0
    return planes

def _aux_planes(b: chess.Board, cfg: EncodeConfig, perspective_white: bool) -> np.ndarray:
    planes = []
    if cfg.include_side_to_move:
        planes.append(np.ones((8, 8), dtype=np.float32))
    if cfg.include_castling:
        def castling_plane(color: bool, ks: bool) -> np.ndarray:
            ok = b.has_kingside_castling_rights(color) if ks else b.has_queenside_castling_rights(color)
            return np.ones((8, 8), dtype=np.float32) if ok else np.zeros((8, 8), dtype=np.float32)
        stm = b.turn
        opp = not stm
        planes.append(castling_plane(stm, True))
        planes.append(castling_plane(stm, False))
        planes.append(castling_plane(opp, True))
        planes.append(castling_plane(opp, False))
    hm = min(b.halfmove_clock, 100) / 100.0
    planes.append(np.full((8, 8), hm, dtype=np.float32))
    return np.stack(planes, axis=0)

def _encode_frame(b: chess.Board) -> np.ndarray:
    stm = b.turn
    own = _planes_for_color(b, stm, stm)
    opp = _planes_for_color(b, not stm, stm)
    return np.concatenate([own, opp], axis=0)  # (12,8,8)

def encode_board(
    board: chess.Board,
    prev_boards: Optional[List[chess.Board]] = None,
    cfg: EncodeConfig = EncodeConfig(),
) -> np.ndarray:
    """Encode into (C,8,8) planes, AlphaZero style (always T frames)."""
    T = cfg.history
    frames: List[chess.Board] = [board]

    if prev_boards and len(prev_boards) > 0:
        take = prev_boards[-(T-1):]
        frames.extend(reversed(take))
    else:
        btmp = board.copy(stack=True)
        for _ in range(T - 1):
            if len(btmp.move_stack) > 0:
                btmp.pop()
                frames.append(btmp.copy(stack=False))
            else:
                frames.append(frames[-1])

    # 固定帧数：不足 T 用“最早一帧”补齐；超出则截断
    if len(frames) >= T:
        frames = frames[:T]
    else:
        if len(frames) == 0:
            frames = [board] * T
        else:
            pad = [frames[-1]] * (T - len(frames))
            frames = frames + pad

    plane_list = [_encode_frame(fb) for fb in frames]  # T × (12,8,8)
    planes = np.concatenate(plane_list, axis=0)        # (T*12,8,8)
    aux = _aux_planes(board, cfg, board.turn)          # (P,8,8)
    feats = np.concatenate([planes, aux], axis=0).astype(np.float32)
    return feats

# ------------------------------
# Outcome -> z helpers
# ------------------------------
def outcome_to_z(outcome: chess.Outcome, perspective_white: bool) -> float:
    if outcome is None:
        raise ValueError("outcome_to_z: outcome is None (game not finished).")
    if outcome.winner is None:
        return 0.0
    return 1.0 if (outcome.winner is True) == perspective_white else -1.0

def board_outcome_to_z(board: chess.Board, perspective_white: bool, claim_draw: bool = True) -> float:
    oc = board.outcome(claim_draw=claim_draw)
    return outcome_to_z(oc, perspective_white)

def result_string_to_z(result: str, perspective_white: bool) -> float:
    r = result.strip()
    if r == "1-0":
        winner_white = True
    elif r == "0-1":
        winner_white = False
    elif r in ("1/2-1/2", "½-½", "0.5-0.5"):
        return 0.0
    else:
        raise ValueError(f"Unknown result string: {result}")
    return 1.0 if winner_white == perspective_white else -1.0

# ------------------------------
# Smoke test
# ------------------------------
if __name__ == "__main__":
    b = chess.Board()
    cfg = EncodeConfig()
    feats = encode_board(b, cfg=cfg)
    mask = legal_moves_mask(b)
    print("Features shape:", feats.shape)   # expect (102, 8, 8)
    print("ACTION_SIZE:", ACTION_SIZE)      # 4672
    print("Num legal moves at start:", mask.sum())
