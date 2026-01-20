# net/batch_predict.py
from __future__ import annotations
import threading, queue, time
from typing import List, Tuple, Callable
import numpy as np

class _Req:
    __slots__ = ("x", "evt", "out_logits", "out_value")
    def __init__(self, x: np.ndarray):
        self.x = x
        self.evt = threading.Event()
        self.out_logits = None
        self.out_value = None

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
            t0 = time.time()
            # 抓更多请求直到凑满或超时
            while len(batch) < self.max_batch:
                remain = self.max_wait_ms/1000.0 - (time.time() - t0)
                if remain <= 0: break
                try:
                    nxt = self.q.get(timeout=remain)
                    batch.append(nxt)
                except queue.Empty:
                    break
            xs = [r.x for r in batch]
            logits, values = self.base_predict(xs)  # (B,A), (B,)
            for i, r in enumerate(batch):
                r.out_logits = logits[i]
                r.out_value  = values[i]
                r.evt.set()

    def predict_one(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        r = _Req(x)
        self.q.put(r)
        r.evt.wait()
        return r.out_logits, float(r.out_value)

def make_batched_predictor(base_predict, max_batch: int = 128, max_wait_ms: int = 5):
    """返回与原 predict_fn 相同签名的函数，但内部做批量聚合。"""
    batcher = PredictBatcher(base_predict, max_batch=max_batch, max_wait_ms=max_wait_ms)
    def predict(feats_batch: List[np.ndarray]):
        outs_logits, outs_values = [], []
        for x in feats_batch:
            lg, v = batcher.predict_one(x)
            outs_logits.append(lg); outs_values.append(v)
        return np.stack(outs_logits), np.asarray(outs_values, dtype=np.float32)
    return predict
