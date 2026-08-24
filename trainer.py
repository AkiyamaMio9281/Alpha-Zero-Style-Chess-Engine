# trainer.py
from __future__ import annotations
import os
import glob
import math
import argparse
import random
from collections import OrderedDict
from typing import List, Tuple, Iterator, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import AlphaZeroChess, ModelConfig, ACTION_SIZE, load_model

# ==== 数据迭代器：随机抽样自博弈分片 ====
class SelfPlayDataset:
    """
    从目录下的 .npz 分片中随机采样样本 (feats[C,8,8], pi[4672], z[1])。
    支持动态 padding/裁剪特征通道到 in_planes。
    """
    def __init__(self, shards_dir: str, in_planes: int = 102, max_cached_files: int = 64):
        self.files = sorted(glob.glob(os.path.join(shards_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz shards in {shards_dir}")
        self.in_planes = in_planes
        self.max_cached_files = max_cached_files
        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

    def __len__(self):
        # 不能准确返回全集大小；训练时用 steps_per_epoch 控制
        return 10**9

    def _load_file(self, path: str) -> Dict[str, np.ndarray]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        data = np.load(path)
        self._cache[path] = {"feats": data["feats"], "pi": data["pi"], "z": data["z"]}
        if len(self._cache) > self.max_cached_files:
            self._cache.popitem(last=False)
        return self._cache[path]

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, ps, zs = [], [], []
        for _ in range(batch_size):
            f = random.choice(self.files)
            arr = self._load_file(f)
            n = arr["feats"].shape[0]
            i = random.randrange(n)
            x = arr["feats"][i]  # (C?,8,8)
            # 调整到 in_planes：不足则 0-pad，超出则裁剪
            C = x.shape[0]
            if C < self.in_planes:
                pad = np.zeros((self.in_planes - C, 8, 8), dtype=x.dtype)
                x = np.concatenate([x, pad], axis=0)
            elif C > self.in_planes:
                x = x[:self.in_planes]
            xs.append(x.astype(np.float32))
            ps.append(arr["pi"][i].astype(np.float32))     # (4672,)
            zs.append(np.float32(arr["z"][i]))             # ()
        return np.stack(xs), np.stack(ps), np.asarray(zs)

# ==== 训练 ====
def train_one_epoch(
    model: AlphaZeroChess,
    dataset: SelfPlayDataset,
    optimizer: optim.Optimizer,
    device: torch.device,
    batch_size: int,
    steps: int,
    value_weight: float = 1.0,
    weight_decay: float = 1e-4,
) -> Dict[str, float]:
    model.train()
    ce = nn.KLDivLoss(reduction="batchmean")  # 使用 KLDiv 与 soft targets（等价于 CE with soft labels）
    mse = nn.MSELoss()
    log_softmax = nn.LogSoftmax(dim=1)

    loss_sum = 0.0
    ce_sum = 0.0
    mse_sum = 0.0

    for step in range(steps):
        x_np, pi_np, z_np = dataset.sample(batch_size)
        x = torch.from_numpy(x_np).to(device)
        pi = torch.from_numpy(pi_np).to(device)
        z = torch.from_numpy(z_np).to(device)

        logits, v = model(x)  # (B,4672),(B,)
        logp = log_softmax(logits)  # (B,4672)
        # KLDivLoss 期望 input=log_probs, target=probs
        policy_loss = ce(logp, pi)
        value_loss = mse(v, z)
        loss = policy_loss + value_weight * value_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_sum += float(loss.item())
        ce_sum += float(policy_loss.item())
        mse_sum += float(value_loss.item())

        if (step + 1) % 50 == 0:
            print(f"[train] step {step+1}/{steps}  loss={loss_sum/(step+1):.4f}  CE={ce_sum/(step+1):.4f}  MSE={mse_sum/(step+1):.4f}", flush=True)

    return {
        "loss": loss_sum / steps,
        "policy_ce": ce_sum / steps,
        "value_mse": mse_sum / steps,
    }

def save_ckpt(model: AlphaZeroChess, optimizer: optim.Optimizer, path: str, step: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(), "step": step}, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="dir with .npz shards")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--resblocks", type=int, default=12)
    ap.add_argument("--in-planes", type=int, default=102)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default="ckpt")
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig(in_planes=args.in_planes, channels=args.channels, resblocks=args.resblocks)
    model = load_model(args.resume if args.resume else None, cfg=cfg, device=device)

    dataset = SelfPlayDataset(args.data, in_planes=args.in_planes)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 简单阶梯衰减（参考 AlphaZero 的阶梯式）：1/10 每个 epoch
    def adjust_lr(ep_idx: int):
        base = args.lr
        decay = 10 ** (ep_idx)  # 1, 10, 100 ...
        for g in optimizer.param_groups:
            g["lr"] = base / decay

    global_step = 0
    for ep in range(1, args.epochs + 1):
        adjust_lr(ep - 1)
        print(f"== Epoch {ep}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.5f}", flush=True)
        stats = train_one_epoch(
            model, dataset, optimizer, device,
            batch_size=args.batch_size,
            steps=args.steps_per_epoch,
        )
        global_step += args.steps_per_epoch
        ckpt_path = os.path.join(args.out, f"model_ep{ep}_step{global_step}.pt")
        save_ckpt(model, optimizer, ckpt_path, step=global_step)
        print(f"[ckpt] saved: {ckpt_path}   stats: {stats}", flush=True)

if __name__ == "__main__":
    main()
