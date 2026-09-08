import concurrent.futures as cf
import threading

import numpy as np
import pytest

from net.batch_predict import make_batched_predictor

ACTION_SIZE = 4672


def _recording_base():
    """Evaluator whose output depends only on each position's own content, the
    way a real network behaves. An evaluator that keyed off the batch index
    instead would silently mask ordering bugs."""
    seen = {"calls": 0, "sizes": []}

    def base(feats):
        seen["calls"] += 1
        seen["sizes"].append(len(feats))
        logits = np.stack([np.full(ACTION_SIZE, float(f[0, 0, 0]), np.float32) for f in feats])
        values = np.asarray([float(f[0, 0, 0]) * 0.5 for f in feats], np.float32)
        return logits, values

    return base, seen


def _positions(n, offset=0):
    out = []
    for i in range(n):
        f = np.zeros((102, 8, 8), np.float32)
        f[0, 0, 0] = offset + i + 1
        out.append(f)
    return out


def test_a_batch_becomes_one_call_not_one_call_per_position():
    """The whole point of the batcher. The original implementation submitted and
    waited one position at a time, so a batch of 8 turned into 8 forward passes
    of size 1, each also paying the max_wait_ms coalescing window."""
    base, seen = _recording_base()
    predict = make_batched_predictor(base, max_batch=128, max_wait_ms=5)

    predict(_positions(8))

    assert seen["calls"] == 1
    assert seen["sizes"] == [8]


def test_results_line_up_with_the_positions_that_were_submitted():
    base, _ = _recording_base()
    predict = make_batched_predictor(base, max_batch=128, max_wait_ms=5)

    logits, values = predict(_positions(8))

    assert logits.shape == (8, ACTION_SIZE)
    assert values.shape == (8,)
    assert np.allclose(values, [(i + 1) * 0.5 for i in range(8)])
    for i in range(8):
        assert np.allclose(logits[i], i + 1)


def test_concurrent_callers_do_not_get_each_others_results():
    """Requests from several threads share one queue and get coalesced into the
    same forward pass, so each caller has to be handed back exactly its own
    rows."""
    base, _ = _recording_base()
    predict = make_batched_predictor(base, max_batch=128, max_wait_ms=5)
    errors = []

    def worker(k):
        _, values = predict(_positions(8, offset=k * 100))
        expected = [(k * 100 + i + 1) * 0.5 for i in range(8)]
        if not np.allclose(values, expected):
            errors.append((k, values, expected))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


def test_empty_batch_returns_empty_arrays():
    # np.stack([]) raises, so the empty case needs its own path.
    base, seen = _recording_base()
    predict = make_batched_predictor(base, max_batch=128, max_wait_ms=5)

    logits, values = predict([])

    assert logits.shape == (0, ACTION_SIZE)
    assert values.shape == (0,)
    assert seen["calls"] == 0


def test_evaluator_failure_raises_instead_of_hanging():
    """Waiters block on an Event that only the batcher thread sets. If the
    evaluator raises and nobody sets it, the caller blocks forever -- a hang
    rather than a traceback."""
    def boom(feats):
        raise RuntimeError("evaluator exploded")

    predict = make_batched_predictor(boom, max_batch=128, max_wait_ms=5)

    with cf.ThreadPoolExecutor(1) as pool:
        future = pool.submit(predict, _positions(4))
        with pytest.raises(RuntimeError, match="evaluator exploded"):
            future.result(timeout=10)


def test_zero_wait_still_batches_what_is_already_queued():
    """max_wait_ms=0 means "batch whatever has already arrived, never wait for
    more" -- the right setting for a single-threaded caller submitting a whole
    eval_batch_size group at once. It must not collapse the batch back to 1,
    which is what a wait-window check alone would do."""
    base, seen = _recording_base()
    predict = make_batched_predictor(base, max_batch=128, max_wait_ms=0)

    _, values = predict(_positions(8))

    assert seen["sizes"] == [8]
    assert np.allclose(values, [(i + 1) * 0.5 for i in range(8)])


def test_max_batch_is_respected():
    base, seen = _recording_base()
    predict = make_batched_predictor(base, max_batch=3, max_wait_ms=0)

    _, values = predict(_positions(8))

    assert all(n <= 3 for n in seen["sizes"]), seen["sizes"]
    assert sum(seen["sizes"]) == 8
    assert np.allclose(values, [(i + 1) * 0.5 for i in range(8)])

