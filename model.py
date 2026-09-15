# model.py (backward-compatible policy head)
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# The AlphaZero policy head outputs 4672 logits (8x8x73).
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
    """Legacy policy head: 1x1 conv -> BN -> flatten -> FC(4672)."""
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
    """Policy head that emits the 73 action planes directly, in the same
    from_sq*73+plane order the engine uses.

    The output layer must be a bare conv -- no BN and no activation. An earlier
    version ran the 73-channel output through BN+ReLU, so every logit was >= 0:
    every suppressed move was clamped to 0, came out with identical probability
    after softmax, and the head had no way to tell them apart, with no gradient
    in ReLU's negative half. Training ran and the loss fell; it just never
    learned a policy.
    """
    def __init__(self, in_channels: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or in_channels
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, 73, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.conv2(h)  # (B, 73, 8, 8), no activation, so logits can be negative
        # (B, 73, 8, 8) -> (B, 8, 8, 73) -> (B, 8*8*73)
        # The flattened index is (rank*8+file)*73 + plane = from_sq*73 + plane,
        # matching engine._from_plane_index.
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
    policy_type: str = "planes"  # "planes" (recommended, aligned with 8x8x73) or "fc" (legacy)

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

# --------- load_model (adapts legacy weights automatically) ---------
import os

def _guess_policy_type_from_state(sd: Dict[str, Any]) -> str:
    """Guess from the checkpoint's keys whether it uses the 'fc' or 'planes' policy head."""
    has_fc = any(k.startswith("policy.fc.") for k in sd.keys())
    if has_fc:
        return "fc"
    # No fc weights, so treat it as the planes head.
    return "planes"

def load_model(
    checkpoint: Optional[str] = None,
    cfg: ModelConfig = ModelConfig(),
    device: Optional[torch.device] = None,
) -> AlphaZeroChess:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Given a checkpoint, peek at its policy type first so the matching structure
    # is built; without one, build from cfg.policy_type (planes by default).
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
        # Override policy_type to match the weights.
        pt = _guess_policy_type_from_state(sd)
        if pt != cfg.policy_type:
            cfg_eff = ModelConfig(in_planes=cfg.in_planes, channels=cfg.channels,
                                  resblocks=cfg.resblocks, policy_type=pt)

    model = AlphaZeroChess(cfg_eff).to(device)

    # Load weights. strict=False allows a partial load across policy-head variants.
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # A planes checkpoint saved before the policy-head fix uses policy.conv and
        # policy.bn, while the current structure is policy.conv1/bn1/conv2, so the
        # whole policy head goes missing. Say so explicitly: the stem, tower and value
        # head resumed normally, but the policy head is randomly initialised and needs
        # retraining. Otherwise it looks like an ordinary resume and the sudden drop in
        # strength is baffling.
        if any(k.startswith("policy.") for k in missing):
            print("[load_model] WARNING: policy head weights were NOT restored and are "
                  "randomly initialized (this checkpoint predates the policy-head fix "
                  "that removed the output activation). The backbone and value head "
                  "resumed normally; the policy head has to be retrained.", flush=True)
        if missing or unexpected:
            print(f"[load_model] loaded with missing={missing} unexpected={unexpected}", flush=True)

    model.eval()
    return model
