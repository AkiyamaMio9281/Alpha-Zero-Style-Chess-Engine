import chess
import numpy as np
import pytest

from engine import (
    ACTION_SIZE,
    PIECE_ORDER,
    EncodeConfig,
    board_outcome_to_z,
    encode_board,
    index_to_move,
    legal_action_indices,
    legal_moves_index_map,
    legal_moves_mask,
    move_to_index,
    outcome_to_z,
    result_string_to_z,
)

# FENs covering: start position, black-to-move, a straight/under promotion,
# and castling (both sides, both directions available).
ROUNDTRIP_FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "8/P6k/8/8/8/8/7K/8 w - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
]


@pytest.mark.parametrize("fen", ROUNDTRIP_FENS)
def test_move_index_roundtrip(fen):
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    assert legal, f"test FEN has no legal moves: {fen}"
    for mv in legal:
        idx = move_to_index(board, mv)
        assert idx is not None, f"move {mv.uci()} failed to encode"
        assert 0 <= idx < ACTION_SIZE
        decoded = index_to_move(board, idx)
        assert decoded == mv, f"roundtrip mismatch for {mv.uci()}: got {decoded}"


@pytest.mark.parametrize("fen", ROUNDTRIP_FENS)
def test_legal_moves_mask_matches_board(fen):
    board = chess.Board(fen)
    mask = legal_moves_mask(board)
    assert mask.dtype == bool
    assert mask.shape == (ACTION_SIZE,)
    assert int(mask.sum()) == len(list(board.legal_moves))


@pytest.mark.parametrize("fen", ROUNDTRIP_FENS)
def test_legal_moves_index_map_matches_board(fen):
    board = chess.Board(fen)
    idx_map = legal_moves_index_map(board)
    assert set(idx_map.values()) == set(board.legal_moves)
    assert len(idx_map) == len(list(board.legal_moves))
    # keys must agree with legal_action_indices
    assert set(idx_map.keys()) == set(int(i) for i in legal_action_indices(board))


def test_encode_board_shape_default_history():
    board = chess.Board()
    feats = encode_board(board, cfg=EncodeConfig())
    assert feats.shape == (102, 8, 8)
    assert feats.dtype == np.float32


def test_encode_board_shape_stable_across_game_length():
    board = chess.Board()
    cfg = EncodeConfig(history=8)
    history = []
    for san in ["e4", "e5", "Nf3", "Nc6"]:
        prev = board.copy(stack=False)
        board.push_san(san)
        history.append(prev)
        feats = encode_board(board, prev_boards=history[-(cfg.history - 1):], cfg=cfg)
        assert feats.shape == (102, 8, 8)


def test_outcome_to_z_decisive_and_draw():
    white_win = chess.Outcome(chess.Termination.CHECKMATE, True)
    assert outcome_to_z(white_win, perspective_white=True) == 1.0
    assert outcome_to_z(white_win, perspective_white=False) == -1.0

    draw = chess.Outcome(chess.Termination.STALEMATE, None)
    assert outcome_to_z(draw, perspective_white=True) == 0.0
    assert outcome_to_z(draw, perspective_white=False) == 0.0

    with pytest.raises(ValueError):
        outcome_to_z(None, perspective_white=True)


def test_board_outcome_to_z_checkmate():
    board = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        board.push_san(san)
    assert board.is_checkmate()
    # Black delivered mate, so White (side that just got mated) lost.
    assert board_outcome_to_z(board, perspective_white=True) == -1.0
    assert board_outcome_to_z(board, perspective_white=False) == 1.0


def test_board_outcome_to_z_threefold_repetition_draw():
    board = chess.Board()
    shuffle = ["Nf3", "Nf6", "Ng1", "Ng8"] * 2  # returns to start position 3 times total
    for san in shuffle:
        board.push_san(san)
    assert board.can_claim_draw()
    assert board_outcome_to_z(board, perspective_white=True) == 0.0
    assert board_outcome_to_z(board, perspective_white=False) == 0.0


def test_result_string_to_z():
    assert result_string_to_z("1-0", perspective_white=True) == 1.0
    assert result_string_to_z("1-0", perspective_white=False) == -1.0
    assert result_string_to_z("0-1", perspective_white=True) == -1.0
    assert result_string_to_z("0-1", perspective_white=False) == 1.0
    assert result_string_to_z("1/2-1/2", perspective_white=True) == 0.0
    with pytest.raises(ValueError):
        result_string_to_z("bogus", perspective_white=True)


# ----- history-frame perspective -----

# Black is missing the g8 knight, so the two sides have different knight counts.
# Material has to be asymmetric for this class of bug to be visible at all: in a
# symmetric position, mirroring the board exactly compensates for swapping
# own/opp, and both the correct and the broken encoding produce the same planes.
ASYMMETRIC_FEN = "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KNIGHT_PLANE = PIECE_ORDER.index(chess.KNIGHT)


def _frame_knight_counts(feats, n_frames):
    """(own, opp) knight counts for each of the first n_frames history frames.
    Frame k occupies channels [12k, 12k+12): first 6 own, last 6 opponent."""
    return [(int(feats[12 * k + KNIGHT_PLANE].sum()),
             int(feats[12 * k + 6 + KNIGHT_PLANE].sum()))
            for k in range(n_frames)]


@pytest.mark.parametrize("moves, mover", [
    (["e4", "e5", "d4", "d5", "Nf3"], chess.BLACK),
    (["e4", "e5", "d4", "d5"], chess.WHITE),
])
def test_history_frames_all_use_the_current_players_perspective(moves, mover):
    """Every history frame must be encoded from the perspective of the player to
    move on the *current* board. Encoding each frame from its own side-to-move
    -- as this did originally -- makes "own" alternate between the two players
    down the history stack, so the 96 history channels carry an alternating
    transform instead of a temporal signal."""
    board = chess.Board(ASYMMETRIC_FEN)
    history = []
    for san in moves:
        prev = board.copy(stack=False)
        board.push_san(san)
        history.append(prev)
    assert board.turn == mover

    feats = encode_board(board, prev_boards=history, cfg=EncodeConfig(history=8))

    own_expected = len(board.pieces(chess.KNIGHT, mover))
    opp_expected = len(board.pieces(chess.KNIGHT, not mover))
    assert own_expected != opp_expected, "test position must be asymmetric to be meaningful"

    counts = _frame_knight_counts(feats, len(moves))
    assert counts == [(own_expected, opp_expected)] * len(moves), (
        f"own/opp knight counts alternate across frames: {counts}"
    )


def test_encoding_is_invariant_under_color_mirroring():
    """Swapping both colours and flipping the board must leave the features
    untouched, since everything is encoded relative to the side to move. This is
    a general property of the encoding rather than a check on frame
    perspective -- the per-frame version of the bug above satisfies it too."""
    board = chess.Board()
    history = []
    for san in ["e4", "c5", "Nf3", "d6", "d4", "cxd4"]:
        prev = board.copy(stack=False)
        board.push_san(san)
        history.append(prev)

    cfg = EncodeConfig(history=8)
    direct = encode_board(board, prev_boards=history, cfg=cfg)
    mirrored = encode_board(board.mirror(), prev_boards=[h.mirror() for h in history], cfg=cfg)
    assert np.array_equal(direct, mirrored)
