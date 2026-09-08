import chess
import numpy as np
import pytest

from engine import ACTION_SIZE, legal_moves_index_map
from selfplay_uci import UCIExpert, expert_pi


class _FakeExpert:
    """Stands in for a UCI engine: returns a fixed move distribution."""

    def __init__(self, dist):
        self.dist = dist
        self.calls = 0

    def distribution(self, board, multipv, movetime_ms):
        self.calls += 1
        return self.dist


def _position():
    board = chess.Board()
    for san in "e4 c5 Nf3".split():
        board.push_san(san)
    legal_map = legal_moves_index_map(board)
    legal_idx = np.fromiter(legal_map.keys(), dtype=np.int32)
    return board, legal_map, legal_idx


def test_expert_pi_lands_on_the_right_action_indices():
    board, legal_map, legal_idx = _position()
    moves = list(legal_map.values())[:3]
    fake = _FakeExpert({moves[0]: 0.5, moves[1]: 0.3, moves[2]: 0.2})

    pi, from_expert = expert_pi(fake, board, legal_map, legal_idx, multipv=3, movetime=10)

    assert from_expert
    assert pi.shape == (ACTION_SIZE,)
    assert pi.sum() == pytest.approx(1.0)
    assert np.count_nonzero(pi) == 3
    inverse = {mv: idx for idx, mv in legal_map.items()}
    assert pi[inverse[moves[0]]] == pytest.approx(0.5)
    assert pi[inverse[moves[2]]] == pytest.approx(0.2)


def test_expert_pi_reports_the_uniform_fallback():
    """A silent uniform target is an inverted training signal, not a weak one,
    so the caller has to be able to tell that it happened."""
    board, legal_map, legal_idx = _position()
    pi, from_expert = expert_pi(_FakeExpert({}), board, legal_map, legal_idx,
                                multipv=3, movetime=10)

    assert not from_expert
    assert pi.sum() == pytest.approx(1.0)
    assert np.count_nonzero(pi) == len(legal_idx)


def test_expert_pi_ignores_moves_outside_the_legal_map():
    board, legal_map, legal_idx = _position()
    good = list(legal_map.values())[0]
    bogus = chess.Move.from_uci("a1a8")            # not legal here
    assert bogus not in legal_map.values()

    pi, from_expert = expert_pi(_FakeExpert({good: 0.6, bogus: 0.4}), board,
                                legal_map, legal_idx, multipv=2, movetime=10)

    assert from_expert
    assert np.count_nonzero(pi) == 1               # the bogus move contributed nothing
    assert pi.sum() == pytest.approx(1.0)          # and the rest was renormalised


# ----- centipawn -> soft target sharpness -----

class _StubScore:
    def __init__(self, cp):
        self._cp = cp

    def white(self):
        return self

    def score(self, mate_score=None):
        return self._cp


class _StubEngine:
    """Minimal stand-in for python-chess's SimpleEngine.analyse."""

    def __init__(self, cps, moves):
        self.infos = [{"pv": [mv], "score": _StubScore(cp)} for mv, cp in zip(moves, cps)]

    def analyse(self, board, limit, multipv=None, info=None):
        return self.infos


def _expert_with(cps, moves):
    expert = object.__new__(UCIExpert)          # bypass popen_uci
    expert.engine = _StubEngine(cps, moves)
    return expert


def test_cp_scale_controls_how_sharp_the_soft_targets_are():
    """The default used to be 1200, flat enough that a 95cp spread -- most of a
    pawn -- produced a 1.08x ratio between the best and worst candidate, i.e. a
    target that barely said which move was better."""
    board, legal_map, _ = _position()
    moves = list(legal_map.values())[:4]
    cps = [-26, -32, -96, -121]                 # 95cp spread, as measured

    def ratio(scale):
        expert = _expert_with(cps, moves)
        expert.cp_scale = scale
        probs = np.array(list(expert.distribution(board, multipv=4, movetime_ms=10).values()))
        assert probs.sum() == pytest.approx(1.0)
        return probs.max() / probs.min()

    flat = ratio(1200.0)
    default = ratio(174.0)
    sharp = ratio(80.0)

    assert flat < 1.15                          # the old default: nearly uniform
    assert default > 1.6                        # win-probability scale
    assert sharp > default                      # smaller scale is sharper


# ----- how we pick our own move during the imitation phase -----

from selfplay_uci import _pick_imitation_move


def _pi_over(legal_map, weights):
    """Build a policy vector giving the first len(weights) legal moves those
    weights, normalised."""
    pi = np.zeros((ACTION_SIZE,), dtype=np.float32)
    for idx, w in zip(list(legal_map.keys())[:len(weights)], weights):
        pi[idx] = w
    return pi / pi.sum()


def test_expert_best_takes_the_argmax():
    board, legal_map, legal_idx = _position()
    pi = _pi_over(legal_map, [0.1, 0.7, 0.2])
    expected = legal_map[list(legal_map.keys())[1]]

    rng = np.random.default_rng(0)
    for _ in range(5):
        assert _pick_imitation_move(pi, legal_map, legal_idx, "expert-best", rng) == expected


def test_expert_sample_stays_inside_the_expert_support():
    """Sampling has to vary the line -- that is the whole point -- but never
    wander outside the moves the expert actually proposed."""
    board, legal_map, legal_idx = _position()
    pi = _pi_over(legal_map, [0.5, 0.3, 0.2])
    support = {legal_map[k] for k in list(legal_map.keys())[:3]}

    rng = np.random.default_rng(1)
    drawn = {_pick_imitation_move(pi, legal_map, legal_idx, "expert-sample", rng)
             for _ in range(60)}

    assert drawn <= support
    assert len(drawn) > 1


def test_random_ignores_the_policy():
    board, legal_map, legal_idx = _position()
    pi = _pi_over(legal_map, [1.0])                 # all mass on one move
    only = legal_map[list(legal_map.keys())[0]]

    rng = np.random.default_rng(2)
    drawn = {_pick_imitation_move(pi, legal_map, legal_idx, "random", rng) for _ in range(60)}

    assert len(drawn) > 1 and drawn != {only}       # spread across legal moves
    assert all(mv in legal_map.values() for mv in drawn)


def test_falls_back_to_random_when_the_expert_returned_nothing():
    board, legal_map, legal_idx = _position()
    empty = np.zeros((ACTION_SIZE,), dtype=np.float32)

    rng = np.random.default_rng(3)
    for policy in ("expert-sample", "expert-best"):
        mv = _pick_imitation_move(empty, legal_map, legal_idx, policy, rng)
        assert mv in legal_map.values()
