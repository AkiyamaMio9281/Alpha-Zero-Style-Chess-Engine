import pytest

torch = pytest.importorskip("torch")

from model import ACTION_SIZE, AlphaZeroChess, ModelConfig


def _tiny_cfg(**kw) -> ModelConfig:
    # 这里测的是策略头的输出性质与展平顺序，小网络就够，不需要 12 个 resblock。
    return ModelConfig(in_planes=102, channels=16, resblocks=1, **kw)


def test_policy_logits_can_be_negative():
    """输出层一旦带上 BN/ReLU，logits 就被钳成非负：所有被抑制的走法塌到同一个
    值，softmax 之后完全无法区分。训练照跑、loss 照降，只是学不出策略——正是
    那种不会崩、只会静默训练垃圾的 bug。"""
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
    """策略头把 (B,73,8,8) 展平成 engine 的 from_sq*73+plane 顺序。顺序错位不会
    报错，只会让每条策略目标都对到别的走法上。"""
    model = AlphaZeroChess(_tiny_cfg())
    model.eval()

    # 用可预测的输出层替换 conv2：(plane, rank, file) 处的值 = plane*100 + from_sq，
    # 其中 from_sq = rank*8 + file（与 chess.square(file, rank) 一致）。
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
    # policy_type="fc" 是旧 checkpoint 的兼容路径，改 planes 头时别把它弄坏。
    model = AlphaZeroChess(_tiny_cfg(policy_type="fc"))
    model.eval()
    with torch.no_grad():
        logits, v = model(torch.randn(2, 102, 8, 8))

    assert logits.shape == (2, ACTION_SIZE)
    assert v.shape == (2,)
