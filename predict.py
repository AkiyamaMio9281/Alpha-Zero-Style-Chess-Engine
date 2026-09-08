from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

# We import from the user's top-level model.py
from model import load_model, ModelConfig

ACTION_SIZE = 4672


class _CudaGraphRunner:
    """Replays a captured CUDA graph of the forward pass, one graph per batch size.

    The network is a fixed-shape CNN held in eval mode, so a forward pass issues
    the identical sequence of kernels every time. Profiling a 64-simulation
    search showed 62.8 ms of wall time inside the forward passes but only
    3.62 ms of actual kernel execution on the GPU: the rest was per-kernel
    launch and dispatch cost, paid ~250 times per search on 8x8 inputs that the
    GPU finishes almost instantly. Capturing the sequence once and replaying it
    submits the whole graph as a single operation instead.

    Graphs need static shapes, so one is captured lazily per distinct batch
    size (MCTS uses few: 1 for the root, eval_batch_size for the rest, plus a
    remainder). Capture failures and unseen batch sizes past the cache limit
    fall back to an ordinary eager forward rather than erroring.

    The returned tensors are the graph's own output buffers and are overwritten
    by the next replay, so callers must copy out before calling again. The
    predictor below does that immediately via .cpu().numpy().
    """

    def __init__(self, model, device: torch.device, use_amp: bool, max_cached: int = 8):
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.max_cached = max_cached
        self._graphs: Dict[int, tuple] = {}
        self._disabled = False

    def _capture(self, batch: int, in_planes: int) -> tuple:
        static_in = torch.zeros((batch, in_planes, 8, 8), device=self.device)

        # Warm up on a side stream first: capture records whatever the ops do,
        # so any lazy allocation or autotuning has to happen before it starts.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                with torch.no_grad(), torch.autocast(device_type="cuda", enabled=self.use_amp):
                    self.model(static_in)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=self.use_amp):
            with torch.cuda.graph(graph):
                logits, value = self.model(static_in)
        return static_in, graph, logits, value

    def try_run(self, xt: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Run via a captured graph, or return None to let the caller go eager."""
        if self._disabled:
            return None
        batch = int(xt.shape[0])
        entry = self._graphs.get(batch)
        if entry is None:
            if len(self._graphs) >= self.max_cached:
                return None
            try:
                entry = self._capture(batch, int(xt.shape[1]))
            except Exception as e:
                print(f"[predict] CUDA graph capture failed ({e}); using eager forward.", flush=True)
                self._disabled = True
                return None
            self._graphs[batch] = entry
        static_in, graph, logits, value = entry
        static_in.copy_(xt)
        graph.replay()
        return logits, value


def load_predictor(
    checkpoint: str,
    in_planes: int = 102,
    channels: int = 128,
    resblocks: int = 12,
    amp: bool = False,
    device: Optional[str] = None,
    cuda_graph: bool = False,
):
    """
    Returns a function predict(feats_batch: List[np.ndarray]) -> (logits[B,4672], value[B])
    that runs the AlphaZero model on CPU/CUDA depending on device.

    cuda_graph=True captures the forward pass as a CUDA graph per batch size and
    replays it, which removes the kernel-launch overhead that dominates this
    model's runtime. It requires CUDA and falls back to eager on anything it
    cannot capture.
    """
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = ModelConfig(in_planes=in_planes, channels=channels, resblocks=resblocks)
    model = load_model(checkpoint=checkpoint, cfg=cfg, device=dev)
    model.eval()
    # autocast only pays off on CUDA; CPU fp32 stays the default path.
    use_amp = amp and dev.type == "cuda"

    runner = None
    if cuda_graph:
        if dev.type != "cuda":
            print("[predict] --cuda-graph ignored: device is not CUDA.", flush=True)
        else:
            runner = _CudaGraphRunner(model, dev, use_amp)

    def _predict(feats_batch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if len(feats_batch) == 0:
            return np.zeros((0, ACTION_SIZE), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        x = np.stack(feats_batch).astype(np.float32)
        xt = torch.from_numpy(x).to(dev, non_blocking=True)

        out = runner.try_run(xt) if runner is not None else None
        if out is None:
            with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=use_amp):
                out = model(xt)
        logits, value = out
        return (logits.detach().cpu().numpy().astype(np.float32),
                value.detach().cpu().numpy().astype(np.float32))

    return _predict
