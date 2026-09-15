import pytest

torch = pytest.importorskip("torch")

from model import ACTION_SIZE, AlphaZeroChess, ModelConfig


def _tiny_cfg(**kw) -> ModelConfig:
    # These tests cover the policy head's output properties and flatten order, so a
    # small network is enough -- no need for 12 residual blocks.
    return ModelConfig(in_planes=102, channels=16, resblocks=1, **kw)


def test_policy_logits_can_be_negative():
    """With BN/ReLU on the output layer the logits are clamped non-negative: every
    suppressed move collapses to the same value and becomes indistinguishable after
    softmax. Training runs and the loss falls, it just never learns a policy --
    exactly the kind of bug that never crashes and silently trains on garbage."""
    model = AlphaZeroChess(_tiny_cfg())
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.randn(8, 102, 8, 8))

    assert (logits < 0).any(), "policy logits are non-negative — output layer has an activation"
    assert (logits > 0).any()


def test_forward_shapes_and_value_range():
    model = AlphaZeroChess(_tiny_cfg())
    model.eval()
    with torch.no_grad():
        logits, v = model(torch.randn(4, 102, 8, 8))

    assert logits.shape == (4, ACTION_SIZE)
    assert v.shape == (4,)
    assert bool((v >= -1.0).all()) and bool((v <= 1.0).all())


def test_policy_planes_flatten_order_is_from_sq_times_73_plus_plane():
    """The policy head flattens (B,73,8,8) into the engine's from_sq*73+plane order.
    Getting that order wrong raises nothing; it just lines every policy target up
    with the wrong move."""
    model = AlphaZeroChess(_tiny_cfg())
    model.eval()

    # Replace conv2 with a predictable output: the value at (plane, rank, file) is
    # plane*100 + from_sq, where from_sq = rank*8 + file, as in chess.square(file, rank).
    marker = (torch.arange(73).view(73, 1) * 100 + torch.arange(64).view(1, 64)).float()

    class _Marker(torch.nn.Module):
        def forward(self, h: torch.Tensor) -> torch.Tensor:
            return marker.view(1, 73, 8, 8).expand(h.size(0), -1, -1, -1)

    model.policy.conv2 = _Marker()
    with torch.no_grad():
        logits, _ = model(torch.randn(2, 102, 8, 8))

    for from_sq in (0, 27, 63):
        for plane in (0, 55, 72):
            assert logits[0, from_sq * 73 + plane].item() == pytest.approx(plane * 100 + from_sq)


def test_fc_policy_head_still_builds():
    # policy_type="fc" is the compatibility path for old checkpoints; changing the
    # planes head must not break it.
    model = AlphaZeroChess(_tiny_cfg(policy_type="fc"))
    model.eval()
    with torch.no_grad():
        logits, v = model(torch.randn(2, 102, 8, 8))

    assert logits.shape == (2, ACTION_SIZE)
    assert v.shape == (2,)
