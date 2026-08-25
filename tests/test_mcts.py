import chess
import numpy as np
import pytest

from engine import ACTION_SIZE, legal_moves_index_map
from mcts import MCTS, MCTSConfig


def dummy_predict(feats_batch):
    """Uniform-prior, zero-value evaluator (no model needed)."""
    B = len(feats_batch)
    logits = np.zeros((B, ACTION_SIZE), dtype=np.float32)
    values = np.zeros((B,), dtype=np.float32)
    return logits, values


def test_run_simulations_visits_root_exactly_sims_times():
    board = chess.Board()
    cfg = MCTSConfig(sims=30, cpuct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    m = MCTS(predict_fn=dummy_predict, mcts_cfg=cfg, rng=np.random.default_rng(0))
    m.set_root(board)
    m.run_simulations()
    assert m.root.total_visits == cfg.sims


def test_select_action_returns_legal_move_with_valid_pi():
    board = chess.Board()
    m = MCTS(predict_fn=dummy_predict, mcts_cfg=MCTSConfig(sims=30), rng=np.random.default_rng(1))
    m.set_root(board)
    m.run_simulations()

    move, pi = m.select_action(tau=1.0)
    assert move in board.legal_moves
    assert pi.shape == (ACTION_SIZE,)
    assert (pi >= 0).all()
    assert pi.sum() == pytest.approx(1.0, abs=1e-4)

    move_greedy, _ = m.select_action(tau=0.0)
    assert move_greedy in board.legal_moves


def test_root_priors_normalized_after_dirichlet_noise():
    board = chess.Board()
    m = MCTS(
        predict_fn=dummy_predict,
        mcts_cfg=MCTSConfig(sims=1, dirichlet_alpha=0.3, dirichlet_epsilon=0.25),
        rng=np.random.default_rng(2),
    )
    m.set_root(board)
    priors = list(m.root.P.values())
    assert len(priors) == len(list(board.legal_moves))
    assert all(p > 0 for p in priors)
    assert sum(priors) == pytest.approx(1.0, abs=1e-4)


def test_apply_move_advances_root_board_and_reuses_tree():
    board = chess.Board()
    m = MCTS(predict_fn=dummy_predict, mcts_cfg=MCTSConfig(sims=20), rng=np.random.default_rng(3))
    m.set_root(board)
    m.run_simulations()

    move, _ = m.select_action(tau=0.0)
    move_idx = next(a for a, mv in legal_moves_index_map(board).items() if mv == move)
    child = m.root.children.get(move_idx)
    m.apply_move(move)

    expected = board.copy(stack=True)
    expected.push(move)
    assert m.root_board.fen() == expected.fen()
    if child is not None:
        assert m.root is child  # tree reused instead of rebuilt from scratch


def test_set_root_marks_checkmate_as_terminal():
    board = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        board.push_san(san)
    assert board.is_checkmate()

    m = MCTS(predict_fn=dummy_predict, mcts_cfg=MCTSConfig(sims=5))
    m.set_root(board)
    assert m.root.is_terminal
    assert m.root.expanded
    assert m.root.total_visits == 0  # nothing to simulate from a terminal root
