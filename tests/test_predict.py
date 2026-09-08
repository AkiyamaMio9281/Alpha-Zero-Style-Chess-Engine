import numpy as np
import pytest

torch = pytest.importorskip("torch")

from model import AlphaZeroChess, ModelConfig
from predict import ACTION_SIZE, load_predictor

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.fixture
def checkpoint(tmp_path):
    """One saved set of weights, so eager and graph predictors are comparable.
    Loading from a missing path would give each of them fresh random weights."""
    torch.manual_seed(0)
    cfg = ModelConfig(in_planes=102, channels=16, resblocks=1)
    path = tmp_path / "weights.pt"
    torch.save({"model": AlphaZeroChess(cfg).state_dict()}, path)
    return str(path)


def _feats(n):
    rng = np.random.default_rng(0)
    return [rng.random((102, 8, 8)).astype(np.float32) for _ in range(n)]


def _predictor(checkpoint, **kw):
    return load_predictor(checkpoint, in_planes=102, channels=16, resblocks=1,
                          amp=False, **kw)


def test_empty_batch_returns_empty_arrays(checkpoint):
    predict = _predictor(checkpoint, device="cpu")
    logits, value = predict([])
    assert logits.shape == (0, ACTION_SIZE)
    assert value.shape == (0,)


def test_cuda_graph_flag_is_inert_on_cpu(checkpoint):
    """Graphs need CUDA. Asking for them on CPU must still produce results
    rather than raising."""
    plain = _predictor(checkpoint, device="cpu", cuda_graph=False)
    asked = _predictor(checkpoint, device="cpu", cuda_graph=True)
    feats = _feats(4)
    assert np.allclose(plain(feats)[0], asked(feats)[0])


@requires_cuda
def test_cuda_graph_matches_eager_exactly(checkpoint):
    """Replaying a captured graph runs the same kernels on the same weights, so
    the results should be identical, not merely close."""
    eager = _predictor(checkpoint, device="cuda", cuda_graph=False)
    graph = _predictor(checkpoint, device="cuda", cuda_graph=True)

    feats = _feats(8)
    le, ve = eager(feats)
    lg, vg = graph(feats)
    assert np.array_equal(le, lg)
    assert np.array_equal(ve, vg)


@requires_cuda
def test_cuda_graph_handles_several_batch_sizes(checkpoint):
    """A graph is captured per batch size. MCTS uses at least two (1 at the
    root, eval_batch_size for the rest, plus a remainder), so a graph captured
    for one size must not be replayed for another."""
    eager = _predictor(checkpoint, device="cuda", cuda_graph=False)
    graph = _predictor(checkpoint, device="cuda", cuda_graph=True)

    for n in (1, 3, 8, 3, 1):          # repeats exercise the cache
        feats = _feats(n)
        assert np.array_equal(eager(feats)[0], graph(feats)[0]), f"mismatch at batch {n}"


@requires_cuda
def test_graph_results_are_copied_out_before_the_next_replay(checkpoint):
    """Replay writes into the graph's own output buffers. The predictor must
    hand back a copy, or an earlier result would mutate when the next batch is
    evaluated."""
    graph = _predictor(checkpoint, device="cuda", cuda_graph=True)

    first = graph(_feats(4))[0].copy()
    kept = graph(_feats(4))[0]
    rng = np.random.default_rng(99)
    graph([rng.random((102, 8, 8)).astype(np.float32) for _ in range(4)])

    assert np.array_equal(kept, first)   # unchanged by the third call
