# net/batch_predict.py
from __future__ import annotations
import threading, queue, time
from typing import List, Tuple, Callable, Optional
import numpy as np

class _Req:
    __slots__ = ("x", "evt", "out_logits", "out_value", "error")
    def __init__(self, x: np.ndarray):
        self.x = x
        self.evt = threading.Event()
        self.out_logits = None
        self.out_value = None
        self.error: Optional[BaseException] = None

class PredictBatcher:
    def __init__(self, base_predict: Callable[[List[np.ndarray]], Tuple[np.ndarray,np.ndarray]],
                 max_batch: int = 128, max_wait_ms: int = 5):
        self.base_predict = base_predict
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.q: "queue.Queue[_Req]" = queue.Queue()
        self.stop = False
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        while not self.stop:
            try:
                first = self.q.get(timeout=0.01)
            except queue.Empty:
                continue
            batch = [first]
            # Drain everything already queued first -- none of it needs waiting for.
            # When a single caller submits eval_batch_size positions at once they are
            # all queued already, and waiting any longer is waiting for nothing.
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            # Only then the real coalescing window, for requests other threads send a
            # moment later. max_wait_ms=0 therefore means "batch whatever has arrived,
            # never wait", not the old trap where it collapsed every batch to size 1.
            t0 = time.time()
            while len(batch) < self.max_batch:
                remain = self.max_wait_ms/1000.0 - (time.time() - t0)
                if remain <= 0: break
                try:
                    nxt = self.q.get(timeout=remain)
                    batch.append(nxt)
                except queue.Empty:
                    break
            xs = [r.x for r in batch]
            try:
                logits, values = self.base_predict(xs)  # (B,A), (B,)
            except BaseException as e:
                # Hand the failure back to the waiters. Without this every
                # caller in the batch blocks on its Event forever, so a broken
                # evaluator hangs the search instead of raising something
                # readable.
                for r in batch:
                    r.error = e
                    r.evt.set()
                continue
            for i, r in enumerate(batch):
                r.out_logits = logits[i]
                r.out_value  = values[i]
                r.evt.set()

    def submit(self, x: np.ndarray) -> _Req:
        """Queue one position and return immediately, without waiting for it.

        Callers holding several positions must submit them all before waiting on
        any of them. Submitting and waiting one at a time -- which is what
        predict_one does -- puts a single request in the queue at a time, so the
        batching thread never has more than one to coalesce and every position
        pays a full round trip plus the max_wait_ms window.
        """
        r = _Req(x)
        self.q.put(r)
        return r

    def wait_result(self, r: _Req) -> Tuple[np.ndarray, float]:
        r.evt.wait()
        if r.error is not None:
            raise r.error
        return r.out_logits, r.out_value

    def predict_one(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        lg, v = self.wait_result(self.submit(x))
        return lg, float(v)

def make_batched_predictor(base_predict, max_batch: int = 128, max_wait_ms: int = 5):
    """Return a function with the same signature as predict_fn that batches internally.

    The whole group is queued before any of it is waited on, so the batching thread
    sees every request from the caller at once and merges them into one base_predict
    call. The original called predict_one per position -- submit, then block -- which
    split a group of N into N batch-of-1 forward passes, each also paying the full
    max_wait_ms window: exactly the overhead batching exists to remove.
    """
    batcher = PredictBatcher(base_predict, max_batch=max_batch, max_wait_ms=max_wait_ms)
    def predict(feats_batch: List[np.ndarray]):
        if not feats_batch:
            # np.stack([]) raises; mirror predict.py's empty-batch shapes.
            return np.zeros((0, 4672), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        reqs = [batcher.submit(x) for x in feats_batch]      # queue all of them
        outs = [batcher.wait_result(r) for r in reqs]        # then collect
        return (np.stack([lg for lg, _ in outs]),
                np.asarray([v for _, v in outs], dtype=np.float32))
    return predict
