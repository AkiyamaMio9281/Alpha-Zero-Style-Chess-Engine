# model.py (backward-compatible policy head)
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# AlphaZero policy head 输出 4672 维（8x8x73）
ACTION_SIZE = 4672

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)

class PolicyHeadFC(nn.Module):
    """旧版策略头：1x1 conv -> BN -> flatten -> FC(4672)。"""
    def __init__(self, in_channels: int, hidden: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(hidden)
        self.fc   = nn.Linear(hidden * 8 * 8, ACTION_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn(self.conv(x)))
        h = torch.flatten(h, 1)
        logits = self.fc(h)  # (B, ACTION_SIZE)
        return logits

class PolicyHeadPlanes(nn.Module):
    """新版策略头：直接输出 73 个“动作平面”，保证与引擎的 from_sq*73+plane 顺序一致。

    输出层必须是裸 conv —— 不接 BN，也不接激活。早期版本把 73 通道的输出
    直接套了 BN+ReLU，于是所有 logits >= 0：被抑制的走法全部被钳到 0，
    softmax 之后概率完全相同，策略头根本没法区分它们，而且 ReLU 负半区
    的梯度是死的。训练照跑、loss 照降，只是学不出策略。
    """
    def __init__(self, in_channels: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or in_channels
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, 73, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.conv2(h)  # (B, 73, 8, 8)，无激活，logits 可正可负
        # (B, 73, 8, 8) -> (B, 8, 8, 73) -> (B, 8*8*73)
        # 展平后的下标即 (rank*8+file)*73 + plane = from_sq*73 + plane，
        # 与 engine._from_plane_index 一致。
        B = h.size(0)
        logits = h.permute(0, 2, 3, 1).contiguous().view(B, 8 * 8 * 73)
        return logits

class ValueHead(nn.Module):
    def __init__(self, in_channels: int, hidden_ch: int = 1, hidden_fc: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_ch, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(hidden_ch)
        self.fc1  = nn.Linear(hidden_ch * 8 * 8, hidden_fc)
        self.fc2  = nn.Linear(hidden_fc, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn(self.conv(x)))
        h = torch.flatten(h, 1)
        h = F.relu(self.fc1(h))
        v = torch.tanh(self.fc2(h)).squeeze(-1)  # (B,)
        return v

@dataclass
class ModelConfig:
    in_planes: int = 102      # C=102 (T=8 => 96 + aux 6)
    channels: int = 128
    resblocks: int = 12
    policy_type: str = "planes"  # "planes"（推荐，严格对齐 8x8x73）或 "fc"（兼容旧结构）

class AlphaZeroChess(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_planes, cfg.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(cfg.channels) for _ in range(cfg.resblocks)])
        if cfg.policy_type == "fc":
            self.policy = PolicyHeadFC(cfg.channels)
        else:
            self.policy = PolicyHeadPlanes(cfg.channels)
        self.value  = ValueHead(cfg.channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C, 8, 8) float32
        returns: (policy_logits[B,ACTION_SIZE], value[B])
        """
        h = self.stem(x)
        h = self.tower(h)
        logits = self.policy(h)
        value  = self.value(h)
        return logits, value

# --------- load_model（带旧权重自动适配） ---------
import os

def _guess_policy_type_from_state(sd: Dict[str, Any]) -> str:
    """根据 checkpoint 的 key 猜测使用的是 'fc' 还是 'planes' 策略头。"""
    has_fc = any(k.startswith("policy.fc.") for k in sd.keys())
    if has_fc:
        return "fc"
    # 如果没有 fc 权重，按新版 planes 处理
    return "planes"

def load_model(
    checkpoint: Optional[str] = None,
    cfg: ModelConfig = ModelConfig(),
    device: Optional[torch.device] = None,
) -> AlphaZeroChess:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 如果给了 checkpoint，就先 peek 一下 policy 类型，确保构建匹配的结构；
    # 如果没给，就按 cfg.policy_type 创建（默认 planes）。
    sd = None
    if checkpoint and checkpoint.strip().lower() != "none" and os.path.exists(checkpoint):
        try:
            obj = torch.load(checkpoint, map_location="cpu")
        except Exception:
            try:
                obj = torch.load(checkpoint, map_location="cpu", weights_only=True)  # type: ignore
            except TypeError:
                obj = torch.load(checkpoint, map_location="cpu")
        if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
            sd = obj
        elif hasattr(obj, "state_dict"):
            sd = obj.state_dict()

    cfg_eff = cfg
    if sd is not None:
        # 覆盖 policy_type 以匹配权重
        pt = _guess_policy_type_from_state(sd)
        if pt != cfg.policy_type:
            cfg_eff = ModelConfig(in_planes=cfg.in_planes, channels=cfg.channels,
                                  resblocks=cfg.resblocks, policy_type=pt)

    model = AlphaZeroChess(cfg_eff).to(device)

    # 加载权重（strict=False 以允许不同策略头之间部分加载）
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # 修策略头输出激活之前存的 planes checkpoint 用的是 policy.conv/policy.bn，
        # 新结构是 policy.conv1/bn1/conv2，整个策略头会 missing 掉。这必须显式说
        # 清楚：stem/tower/value 头都照常恢复了，唯独策略头是随机初始化的，需要
        # 重新训练——否则很容易以为在续训，却对棋力突然变差摸不着头脑。
        if any(k.startswith("policy.") for k in missing):
            print("[load_model] WARNING: policy head weights were NOT restored and are "
                  "randomly initialized (this checkpoint predates the policy-head fix "
                  "that removed the output activation). The backbone and value head "
                  "resumed normally; the policy head has to be retrained.", flush=True)
        if missing or unexpected:
            print(f"[load_model] loaded with missing={missing} unexpected={unexpected}", flush=True)

    model.eval()
    return model
