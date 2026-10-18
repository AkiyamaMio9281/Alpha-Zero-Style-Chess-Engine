import functools

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from engine import ACTION_SIZE, legal_moves_index_map
from kernels.masked_softmax import (
    GATHER_BLOCK,
    masked_softmax_dense,
    masked_softmax_gather,
    pack_legal,
)
from kernels.reference import masked_softmax_ref

DEV = torch.device("cuda")
FEN_218 = "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1"
FEN_SINGLE = "k7/2K5/8/8/8/8/8/1R6 b - - 0 1"


@functools.lru_cache(maxsize=None)
def _game_positions():
    """Legal-move index sets from every position of a few random games, so
    the counts the kernels see are a real game's spread, not one position's."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(8):
        board = chess.Board()
        while not board.is_game_over() and board.ply() < 120:
            out.append(np.array(sorted(legal_moves_index_map(board)), dtype=np.int32))
            moves = list(board.legal_moves)
            board.push(moves[rng.integers(len(moves))])
    return out


def _batch(B, seed=0, scale=3.0):
    rng = np.random.default_rng(seed)
    pool = _game_positions()
    legal = [pool[i] for i in rng.integers(len(pool), size=B)]
    logits = rng.normal(0, scale, (B, ACTION_SIZE)).astype(np.float32)
    return logits, legal


def _mask(legal):
    m = np.zeros((len(legal), ACTION_SIZE), dtype=bool)
    for i, l in enumerate(legal):
        m[i, l] = True
    return m


def _run_dense(logits, legal):
    out = masked_softmax_dense(torch.from_numpy(logits).to(DEV), torch.from_numpy(_mask(legal)).to(DEV))
    return out.cpu().numpy()


def _run_gather(logits, legal):
    idx, count = pack_legal(legal)
    out = masked_softmax_gather(torch.from_numpy(logits).to(DEV),
                                torch.from_numpy(idx).to(DEV), torch.from_numpy(count).to(DEV))
    return out.cpu().numpy()


def _check_dense(out, logits, legal, atol=1e-6):
    for i, l in enumerate(legal):
        np.testing.assert_allclose(out[i, l], masked_softmax_ref(logits[i], l), rtol=0, atol=atol)
        illegal = np.ones(ACTION_SIZE, dtype=bool)
        illegal[l] = False
        assert (out[i, illegal] == 0).all()


def _check_gather(out, logits, legal, atol=1e-6):
    for i, l in enumerate(legal):
        np.testing.assert_allclose(out[i, :len(l)], masked_softmax_ref(logits[i], l), rtol=0, atol=atol)
        assert (out[i, len(l):] == 0).all()


def test_game_positions_have_a_real_spread():
    counts = [len(l) for l in _game_positions()]
    assert len(counts) > 300 and min(counts) < 10 and max(counts) > 40


@pytest.mark.parametrize("B", [1, 7, 64, 256])
def test_dense_matches_reference(B):
    logits, legal = _batch(B, seed=B)
    _check_dense(_run_dense(logits, legal), logits, legal)


@pytest.mark.parametrize("B", [1, 7, 64, 256])
def test_gather_matches_reference(B):
    logits, legal = _batch(B, seed=B)
    _check_gather(_run_gather(logits, legal), logits, legal)


@pytest.mark.parametrize("run,check", [(_run_dense, _check_dense), (_run_gather, _check_gather)],
                         ids=["dense", "gather"])
def test_fp16_logits(run, check):
    """Under AMP the network emits fp16 logits. The kernels widen them to fp32
    on load, which is exact, and predict.py's CPU path widens the same values
    the same way -- so the fp32 reference on the widened values is the right
    target and the fp32 tolerance still applies."""
    logits, legal = _batch(64, seed=5)
    half = logits.astype(np.float16)
    if run is _run_dense:
        out = masked_softmax_dense(torch.from_numpy(half).to(DEV),
                                   torch.from_numpy(_mask(legal)).to(DEV)).cpu().numpy()
    else:
        idx, count = pack_legal(legal)
        out = masked_softmax_gather(torch.from_numpy(half).to(DEV), torch.from_numpy(idx).to(DEV),
                                    torch.from_numpy(count).to(DEV)).cpu().numpy()
    check(out, half.astype(np.float32), legal)


@pytest.mark.parametrize("run,check", [(_run_dense, _check_dense), (_run_gather, _check_gather)],
                         ids=["dense", "gather"])
def test_huge_logits_do_not_overflow(run, check):
    logits, legal = _batch(16, seed=6)
    logits += 1e4
    out = run(logits, legal)
    assert np.isfinite(out).all()
    check(out, logits, legal)


@pytest.mark.parametrize("run", [_run_dense, _run_gather], ids=["dense", "gather"])
def test_single_legal_move_is_exactly_one(run):
    legal = [np.array(sorted(legal_moves_index_map(chess.Board(FEN_SINGLE))), dtype=np.int32)]
    assert len(legal[0]) == 1
    logits = np.random.default_rng(7).normal(0, 5, (1, ACTION_SIZE)).astype(np.float32)
    out = run(logits, legal)
    assert out[0, legal[0][0] if run is _run_dense else 0] == 1.0


@pytest.mark.parametrize("run,check", [(_run_dense, _check_dense), (_run_gather, _check_gather)],
                         ids=["dense", "gather"])
def test_218_legal_moves(run, check):
    legal = [np.array(sorted(legal_moves_index_map(chess.Board(FEN_218))), dtype=np.int32)]
    assert len(legal[0]) == 218
    logits = np.random.default_rng(8).normal(0, 3, (1, ACTION_SIZE)).astype(np.float32)
    check(run(logits, legal), logits, legal)


@pytest.mark.parametrize("run", [_run_dense, _run_gather], ids=["dense", "gather"])
def test_row_without_legal_moves_is_zero_not_nan(run):
    """MCTS never asks for a finished position, but a fully masked row must
    still not produce NaN (-inf - (-inf)); it comes out as zeros, and the
    rows around it are unaffected."""
    logits, legal = _batch(3, seed=9)
    legal[1] = np.zeros(0, dtype=np.int32)
    out = run(logits, legal)
    assert not np.isnan(out).any()
    assert (out[1] == 0).all()
    assert out[0].sum() == pytest.approx(1.0, abs=1e-5)
    assert out[2].sum() == pytest.approx(1.0, abs=1e-5)


def test_gather_never_reads_padding_slots():
    """Fill every padding slot with the index of an illegal action whose
    logit is huge. If a padding slot were dereferenced, that action would
    take all the probability mass."""
    logits, legal = _batch(32, seed=10)
    idx, count = pack_legal(legal)
    for i, l in enumerate(legal):
        trap = int(np.setdiff1d(np.arange(ACTION_SIZE), l)[0])
        logits[i, trap] = 1e4
        idx[i, len(l):] = trap
    out = masked_softmax_gather(torch.from_numpy(logits).to(DEV),
                                torch.from_numpy(idx).to(DEV), torch.from_numpy(count).to(DEV))
    _check_gather(out.cpu().numpy(), logits, legal)


def test_gather_accepts_int32_idx_and_preallocated_out():
    logits, legal = _batch(8, seed=11)
    idx, count = pack_legal(legal)
    out = torch.full((8, GATHER_BLOCK), float("nan"), device=DEV)
    ret = masked_softmax_gather(torch.from_numpy(logits).to(DEV),
                                torch.from_numpy(idx.astype(np.int32)).to(DEV),
                                torch.from_numpy(count).to(DEV), out=out)
    assert ret is out
    _check_gather(out.cpu().numpy(), logits, legal)   # padding overwritten with 0, not left NaN


def test_bad_inputs_are_rejected():
    logits = torch.zeros((2, ACTION_SIZE), device=DEV)
    with pytest.raises(ValueError):
        masked_softmax_dense(logits.cpu(), torch.zeros((2, ACTION_SIZE), dtype=torch.bool))
    with pytest.raises(ValueError):
        masked_softmax_dense(logits, torch.zeros((2, ACTION_SIZE), dtype=torch.float32, device=DEV))
    with pytest.raises(ValueError):
        masked_softmax_gather(logits, torch.zeros((2, 100), dtype=torch.int16, device=DEV),
                              torch.zeros(2, dtype=torch.int32, device=DEV))
    with pytest.raises(ValueError):
        pack_legal([np.arange(GATHER_BLOCK + 1)])


def test_gather_captures_into_the_forward_passs_cuda_graph():
    """The point of the fixed-shape layout: forward pass and kernel recorded
    as one graph, then replayed on new positions and new legal-move sets by
    copying into the static buffers. Replays must match eager runs."""
    from model import AlphaZeroChess, ModelConfig

    torch.manual_seed(0)
    model = AlphaZeroChess(ModelConfig(in_planes=102, channels=16, resblocks=1)).to(DEV).eval()
    B = 8
    static_x = torch.zeros((B, 102, 8, 8), device=DEV)
    static_idx = torch.zeros((B, GATHER_BLOCK), dtype=torch.int16, device=DEV)
    static_count = torch.zeros((B,), dtype=torch.int32, device=DEV)
    static_out = torch.empty((B, GATHER_BLOCK), device=DEV)

    def step():
        logits, _ = model(static_x)
        masked_softmax_gather(logits, static_idx, static_count, out=static_out)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad():
        for _ in range(3):      # compile the Triton kernel before capture starts
            step()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        step()

    rng = np.random.default_rng(12)
    for seed in (1, 2, 3):
        _, legal = _batch(B, seed=seed)
        idx, count = pack_legal(legal)
        x = torch.from_numpy(rng.random((B, 102, 8, 8)).astype(np.float32)).to(DEV)
        static_x.copy_(x)
        static_idx.copy_(torch.from_numpy(idx))
        static_count.copy_(torch.from_numpy(count))
        graph.replay()
        with torch.no_grad():
            logits_eager, _ = model(x)
            eager = masked_softmax_gather(logits_eager, static_idx, static_count)
        torch.testing.assert_close(static_out, eager, rtol=0, atol=0)
        _check_gather(static_out.cpu().numpy(), logits_eager.cpu().numpy(), legal)
