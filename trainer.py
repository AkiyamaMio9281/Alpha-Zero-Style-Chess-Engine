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
    递归查找子目录（如 gen1/, gen2/, ...），所以既可以指向单个分片目录，
    也可以指向多轮自对弈数据的根目录，实现跨代的经验回放。
    支持动态 padding/裁剪特征通道到 in_planes。
    """
    def __init__(self, shards_dir: str, in_planes: int = 102, max_cached_files: int = 64):
        self.files = sorted(glob.glob(os.path.join(shards_dir, "**", "*.npz"), recursive=True))
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

    def sample(self, batch_size: int, files_per_batch: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw a batch, reading from *files_per_batch* shards rather than one
        shard per sample.

        Picking a shard independently for every sample means a batch of 256
        touches 256 shards. With more shards than fit in the cache that misses
        almost every time, and each miss decompresses a whole shard to keep one
        position out of it. Measured on 200 shards with the 64-shard cache:
        32% hit rate, 174 misses per batch, 525 ms to build one batch against
        27 ms to train on it -- the GPU sat idle 94.8% of the time and a
        2000-step epoch took 17 minutes instead of one.

        Reading several samples per shard cuts the lookups by the same factor.
        The cost is that a batch is drawn from fewer games, so its samples are
        more correlated; 16 shards per batch keeps that mild while removing
        almost all of the decompression.
        """
        xs, ps, zs = [], [], []
        groups = max(1, min(files_per_batch, batch_size, len(self.files)))
        base, extra = divmod(batch_size, groups)
        for g in range(groups):
            f = random.choice(self.files)
            arr = self._load_file(f)
            n = arr["feats"].shape[0]
            for _ in range(base + (1 if g < extra else 0)):
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
    files_per_batch: int = 16,
) -> Dict[str, float]:
    model.train()
    ce = nn.KLDivLoss(reduction="batchmean")  # 使用 KLDiv 与 soft targets（等价于 CE with soft labels）
    mse = nn.MSELoss()
    log_softmax = nn.LogSoftmax(dim=1)

    loss_sum = 0.0
    ce_sum = 0.0
    mse_sum = 0.0

    for step in range(steps):
        x_np, pi_np, z_np = dataset.sample(batch_size, files_per_batch=files_per_batch)
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

def save_ckpt(model: AlphaZeroChess, optimizer: optim.Optimizer, path: str, epoch: int, step: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(), "epoch": epoch, "step": step}, path)

def _parse_milestones(spec: str) -> List[int]:
    """Parse "20000,40000" (commas and/or whitespace) into a sorted step list."""
    return sorted(int(tok) for tok in spec.replace(",", " ").split() if tok.strip())


def lr_at_step(base_lr: float, global_step: int, milestones: List[int], lr_min: float) -> float:
    """阶梯衰减，以累计 global_step 为准，而不是以“第几次 resume”为准。

    旧实现是 base_lr / 10**epoch_index，而 epoch_index 直接取自断点
    checkpoint 里的 epoch 字段，所以每次 --resume 都会把指数永久推高一截：
    autoloopexpert.py 每轮迭代 resume 一次、跑 2 个 epoch，从 model_ep3 接着练的
    话 lr 依次是 2e-4、2e-5（第 1 轮），2e-6、2e-7（第 2 轮），不到十几轮
    就彻底数值死掉——而循环依旧打印正常的 loss、照常存 checkpoint。

    改用 global_step 后，学习率只取决于实际训了多少步：同一段训练拆成
    几次 resume 都得到相同的 lr。lr_min 再提供一个下界，保证无论里程碑配得
    多激进，学习率都不会跌到 0。
    """
    passed = sum(1 for m in milestones if global_step >= m)
    return max(base_lr / (10.0 ** passed), lr_min)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="dir with .npz shards")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--files-per-batch", type=int, default=16,
                    help="how many shards one batch is drawn from. Lower reads less from disk "
                         "but correlates the batch; equal to --batch-size restores the old "
                         "one-shard-per-sample behaviour.")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--resblocks", type=int, default=12)
    ap.add_argument("--in-planes", type=int, default=102)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--lr-milestones", type=str, default="20000,40000",
                    help="comma-separated global_step milestones; lr drops 10x at each. "
                         "Empty string disables decay.")
    ap.add_argument("--lr-min", type=float, default=1e-5,
                    help="floor for the decayed lr; it can never drop below this")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default="ckpt")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--tag", type=str, default="",
                    help="inserted into the checkpoint filename. Without it the name is derived "
                         "from epoch and step alone, so two runs resuming from the same "
                         "checkpoint write the same file and the earlier one is lost -- which is "
                         "what happens every time an autoloop iteration is rejected and the next "
                         "one restarts from the same weights.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig(in_planes=args.in_planes, channels=args.channels, resblocks=args.resblocks)
    model = load_model(args.resume if args.resume else None, cfg=cfg, device=device)

    dataset = SelfPlayDataset(args.data, in_planes=args.in_planes)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 恢复训练进度：epoch 计数决定学习率阶梯衰减到哪一档，optimizer 状态
    # （SGD momentum 缓冲）也一并恢复，否则每次 --resume 学习率都会从头衰减、
    # momentum 也会被清零重来。
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=device)
        except Exception as e:
            ckpt = None
            print(f"[trainer] could not re-read checkpoint for resume state ({e}); starting fresh optimizer/schedule.", flush=True)
        if isinstance(ckpt, dict):
            if "opt" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["opt"])
                except Exception as e:
                    print(f"[trainer] could not restore optimizer state ({e}); starting fresh optimizer.", flush=True)
            start_epoch = int(ckpt.get("epoch", 0))
            global_step = int(ckpt.get("step", 0))
            if start_epoch or global_step:
                print(f"[trainer] resuming from epoch={start_epoch}, step={global_step}", flush=True)

    # 阶梯衰减：以 global_step 为准，而不是以累计 epoch 为准。后者会让每次
    # --resume 都把学习率再除一个 10，在 autoloop 里几轮之后就降到 0。
    milestones = _parse_milestones(args.lr_milestones)
    if milestones:
        print(f"[trainer] lr schedule: {args.lr:g} / 10^(milestones passed), "
              f"milestones={milestones} on global_step, floor={args.lr_min:g}", flush=True)
    else:
        print(f"[trainer] lr schedule: constant {args.lr:g}", flush=True)

    end_epoch = start_epoch + args.epochs
    for ep in range(start_epoch + 1, end_epoch + 1):
        lr = lr_at_step(args.lr, global_step, milestones, args.lr_min)
        for g in optimizer.param_groups:
            g["lr"] = lr
        print(f"== Epoch {ep}/{end_epoch}  global_step={global_step}  lr={lr:.6g}", flush=True)
        stats = train_one_epoch(
            model, dataset, optimizer, device,
            batch_size=args.batch_size,
            steps=args.steps_per_epoch,
            files_per_batch=args.files_per_batch,
        )
        global_step += args.steps_per_epoch
        tag = f"{args.tag}_" if args.tag else ""
        ckpt_path = os.path.join(args.out, f"model_{tag}ep{ep}_step{global_step}.pt")
        save_ckpt(model, optimizer, ckpt_path, epoch=ep, step=global_step)
        print(f"[ckpt] saved: {ckpt_path}   stats: {stats}", flush=True)

if __name__ == "__main__":
    main()
