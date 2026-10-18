# kernels/reference.py
"""CPU implementations of the policy-prior softmax over legal moves.

Both functions take one position's raw policy logits (ACTION_SIZE,) and the
action indices of its legal moves, and return the prior for each legal move,
in the order of legal_idx. They are the baselines any GPU kernel for this step
has to match (masked_softmax_ref) and beat (masked_softmax_compact).
"""
from __future__ import annotations
import numpy as np

from engine import ACTION_SIZE


def masked_softmax_ref(logits: np.ndarray, legal_idx: np.ndarray) -> np.ndarray:
    """The softmax MCTS._finish_expand has always used, extracted unchanged.

    It adds -1e9 to every illegal action and runs the softmax over the whole
    ACTION_SIZE row, then keeps the legal entries. Roughly 43 of the 4672
    entries are legal in a typical position, so about 99% of the exp work is
    on values that are thrown away.
    """
    x = np.full((ACTION_SIZE,), -1e9, dtype=np.float32)
    x[legal_idx] = 0.0
    x = logits + x
    x = x - np.max(x)
    ex = np.exp(x).astype(np.float32)
    s = ex.sum()
    probs = ex / s if s > 0 else np.ones_like(ex) / len(ex)
    return probs[legal_idx]


def masked_softmax_compact(logits: np.ndarray, legal_idx: np.ndarray) -> np.ndarray:
    """Gather the legal logits first, then softmax over just those.

    Equivalent to masked_softmax_ref: there every illegal entry sits at
    logit - 1e9, so exp() underflows it to exactly 0 unless an illegal logit
    exceeds the largest legal one by about 1e9, and the max it subtracts is
    always a legal logit. The per-entry values match; only the order of the
    float32 sum differs, so results agree to a few ulps rather than bit for bit.

    The one input where the two diverge is a non-finite logit row. The
    reference then falls back to 1/ACTION_SIZE per legal move -- priors that do
    not even sum to 1 -- while this propagates the NaN. Neither is a meaningful
    prior; the network does not produce such rows.
    """
    x = logits[legal_idx].astype(np.float32, copy=False)
    x = x - x.max()
    ex = np.exp(x)
    return ex / ex.sum()
