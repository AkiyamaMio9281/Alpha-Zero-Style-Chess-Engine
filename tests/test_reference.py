import chess
import numpy as np
import pytest

import mcts
from engine import ACTION_SIZE, legal_moves_index_map
from kernels.reference import masked_softmax_compact, masked_softmax_ref
from mcts import MCTS, MCTSConfig, Node

# Positions spanning the range of legal-move counts the softmax sees.
FENS = {
    "start": chess.STARTING_FEN,                                                       # 20
    "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # 48
    "promotions": "8/2P1k3/8/8/8/8/3p1K2/8 w - - 0 1",                                 # under-promotions
    "max_218": "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1",                  # 218, the chess maximum
    "single": "k7/2K5/8/8/8/8/8/1R6 b - - 0 1",                                        # exactly 1
}


def _legal_idx(fen):
    return np.array(list(legal_moves_index_map(chess.Board(fen)).keys()), dtype=np.int32)


def _original_softmax(logits, legal_idx):
    """The inline code _finish_expand ran before it was extracted, verbatim,
    so the extraction is checked against the original rather than against itself."""
    x = np.full((ACTION_SIZE,), -1e9, dtype=np.float32)
    x[np.array(legal_idx, dtype=np.int32)] = 0.0
    x = logits + x
    x = x - np.max(x)
    ex = np.exp(x).astype(np.float32)
    s = ex.sum()
    probs = ex / s if s > 0 else np.ones_like(ex) / len(ex)
    return {int(a): float(probs[a]) for a in legal_idx}


@pytest.mark.parametrize("fen", FENS.values(), ids=FENS.keys())
def test_reference_bit_identical_to_original_code(fen):
    legal_idx = _legal_idx(fen)
    logits = np.random.default_rng(0).normal(0, 3, ACTION_SIZE).astype(np.float32)
    ref = masked_softmax_ref(logits, legal_idx)
    got = {int(a): float(p) for a, p in zip(legal_idx, ref)}
    assert got == _original_softmax(logits, legal_idx)  # exact, not approx


@pytest.mark.parametrize("fen", FENS.values(), ids=FENS.keys())
def test_finish_expand_priors_bit_identical_to_original_code(fen):
    board = chess.Board(fen)
    legal_idx = _legal_idx(fen)
    logits = np.random.default_rng(0).normal(0, 3, ACTION_SIZE).astype(np.float32)

    node = Node()
    MCTS(predict_fn=None, mcts_cfg=MCTSConfig())._finish_expand(node, board, logits)

    assert node.P == _original_softmax(logits, legal_idx)  # exact, not approx


@pytest.mark.parametrize("fen", FENS.values(), ids=FENS.keys())
@pytest.mark.parametrize("scale", [0.1, 3.0, 30.0])
def test_compact_matches_reference(fen, scale):
    legal_idx = _legal_idx(fen)
    rng = np.random.default_rng(1)
    for _ in range(20):
        logits = rng.normal(0, scale, ACTION_SIZE).astype(np.float32)
        ref = masked_softmax_ref(logits, legal_idx)
        out = masked_softmax_compact(logits, legal_idx)
        assert out.dtype == np.float32
        # Same per-entry values; only the float32 summation order differs.
        np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-9)
        assert out.sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("fn", [masked_softmax_ref, masked_softmax_compact])
def test_single_legal_move_gets_probability_exactly_one(fn):
    legal_idx = _legal_idx(FENS["single"])
    logits = np.random.default_rng(2).normal(0, 5, ACTION_SIZE).astype(np.float32)
    out = fn(logits, legal_idx)
    assert out.tolist() == [1.0]


@pytest.mark.parametrize("fn", [masked_softmax_ref, masked_softmax_compact])
def test_huge_logits_do_not_overflow(fn):
    legal_idx = _legal_idx(FENS["kiwipete"])
    logits = np.zeros(ACTION_SIZE, dtype=np.float32)
    logits[legal_idx] = np.linspace(9000.0, 10000.0, legal_idx.size, dtype=np.float32)
    out = fn(logits, legal_idx)
    assert np.isfinite(out).all()
    assert out.sum() == pytest.approx(1.0, abs=1e-6)
    assert int(np.argmax(out)) == legal_idx.size - 1


@pytest.mark.parametrize("fn", [masked_softmax_ref, masked_softmax_compact])
def test_fp16_sourced_logits(fn):
    """Under AMP the network emits fp16 logits that predict.py casts to fp32;
    values on the fp16 grid must go through unchanged."""
    legal_idx = _legal_idx(FENS["start"])
    logits = np.random.default_rng(3).normal(0, 4, ACTION_SIZE).astype(np.float16).astype(np.float32)
    out = fn(logits, legal_idx)
    np.testing.assert_allclose(out, masked_softmax_ref(logits, legal_idx), rtol=1e-6, atol=1e-9)


def test_large_illegal_logits_are_ignored():
    """An illegal logit far above every legal one must not leak into the
    priors. The reference adds -1e9, so this holds up to a gap of ~1e9."""
    legal_idx = _legal_idx(FENS["start"])
    logits = np.zeros(ACTION_SIZE, dtype=np.float32)
    illegal = np.setdiff1d(np.arange(ACTION_SIZE), legal_idx)
    logits[illegal] = 1e8
    expected = np.full(legal_idx.size, 1.0 / legal_idx.size, dtype=np.float32)
    np.testing.assert_allclose(masked_softmax_ref(logits, legal_idx), expected, rtol=1e-6)
    np.testing.assert_allclose(masked_softmax_compact(logits, legal_idx), expected, rtol=1e-6)


def test_where_the_minus_1e9_mask_stops_working():
    """Pins the boundary of the equivalence argument. Once an illegal logit
    beats the best legal one by more than 1e9, the reference's mask no longer
    pushes it below the legal entries: the softmax's mass goes to the illegal
    action and every legal prior collapses to 0. The compact version never
    looks at illegal entries, so it is unaffected. No trained network produces
    a 1e9 logit gap; this documents the edge rather than a case that occurs."""
    legal_idx = _legal_idx(FENS["start"])
    logits = np.zeros(ACTION_SIZE, dtype=np.float32)
    illegal = np.setdiff1d(np.arange(ACTION_SIZE), legal_idx)
    logits[illegal[0]] = 2e9

    assert masked_softmax_ref(logits, legal_idx).sum() == 0.0
    assert masked_softmax_compact(logits, legal_idx).sum() == pytest.approx(1.0, abs=1e-6)


def test_legal_softmax_is_swappable(monkeypatch):
    """bench/profile_search.py measures both implementations by rebinding
    mcts._legal_softmax; the search has to go through that name."""
    calls = []

    def spy(logits, legal_idx):
        calls.append(legal_idx.size)
        return masked_softmax_compact(logits, legal_idx)

    monkeypatch.setattr(mcts, "_legal_softmax", spy)
    node = Node()
    MCTS(predict_fn=None, mcts_cfg=MCTSConfig())._finish_expand(
        node, chess.Board(), np.zeros(ACTION_SIZE, dtype=np.float32))
    assert calls == [20]
    assert sum(node.P.values()) == pytest.approx(1.0, abs=1e-6)
