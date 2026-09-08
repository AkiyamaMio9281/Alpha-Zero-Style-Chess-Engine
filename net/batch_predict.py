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
            # 先把已经在队列里的全部排空——这些不需要等。单个调用方一次交
            # eval_batch_size 个位置时，它们已经全在队列里，再等只是白等。
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            # 然后才是真正的凑批窗口，用于等其他线程稍后到达的请求。
            # max_wait_ms=0 因此表示“只批处理已到达的，从不等待”，而不是以前
            # 那种会把 batch 直接退化成 1 的陷阱语义。
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
    """返回与原 predict_fn 相同签名的函数，但内部做批量聚合。

    整批先全部入队，再统一等待：这样批处理线程一次就能看到调用方的全部请求，
    合并成一次 base_predict 调用。原实现逐个 predict_one（提交即阻塞），
    一批 N 个位置会被拆成 N 次 batch=1 的前向，还要各自吃满 max_wait_ms 的
    凑批窗口——正是批处理本该消除的开销。
    """
    batcher = PredictBatcher(base_predict, max_batch=max_batch, max_wait_ms=max_wait_ms)
    def predict(feats_batch: List[np.ndarray]):
        if not feats_batch:
            # np.stack([]) raises; mirror predict.py's empty-batch shapes.
            return np.zeros((0, 4672), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        reqs = [batcher.submit(x) for x in feats_batch]      # 全部入队
        outs = [batcher.wait_result(r) for r in reqs]        # 再统一收
        return (np.stack([lg for lg, _ in outs]),
                np.asarray([v for _, v in outs], dtype=np.float32))
    return predict
