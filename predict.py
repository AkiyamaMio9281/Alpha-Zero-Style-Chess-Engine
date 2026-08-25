from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import torch

# We import from the user's top-level model.py
from model import load_model, ModelConfig

def load_predictor(
    checkpoint: str,
    in_planes: int = 102,
    channels: int = 128,
    resblocks: int = 12,
    amp: bool = False,
    device: Optional[str] = None,
):
    """
    Returns a function predict(feats_batch: List[np.ndarray]) -> (logits[B,4672], value[B])
    that runs the AlphaZero model on CPU/CUDA depending on device.
    """
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = ModelConfig(in_planes=in_planes, channels=channels, resblocks=resblocks)
    model = load_model(checkpoint=checkpoint, cfg=cfg, device=dev)
    model.eval()
    # autocast only pays off on CUDA; CPU fp32 stays the default path.
    use_amp = amp and dev.type == "cuda"

    def _predict(feats_batch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if len(feats_batch) == 0:
            return np.zeros((0, 4672), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        x = np.stack(feats_batch).astype(np.float32)
        xt = torch.from_numpy(x).to(dev, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=use_amp):
            logits, value = model(xt)  # (B, 4672), (B,)
        return logits.detach().cpu().numpy().astype(np.float32), value.detach().cpu().numpy().astype(np.float32)

    return _predict