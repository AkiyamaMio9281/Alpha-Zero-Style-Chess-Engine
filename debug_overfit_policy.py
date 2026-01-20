# debug_overfit_policy.py
from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import chess

from engine import EncodeConfig, encode_board, legal_moves_index_map, move_to_index
from model import load_model, ModelConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--in-planes", type=int, default=102)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--resblocks", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--target", default="e7e5", help="target move UCI for position after 1.e4 (black to move)")
    args = ap.parse_args()

    dev = torch.device(args.device if (args.device!="auto") else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = ModelConfig(in_planes=args.in_planes, channels=args.channels, resblocks=args.resblocks)
    model = load_model(args.checkpoint, cfg=cfg, device=dev)
    model.train()

    # Build one training sample: position after 1.e4 (black to move), target move = args.target
    b = chess.Board(); b.push_san("e4")
    feats = encode_board(b, cfg=EncodeConfig(history=8))
    x = torch.from_numpy(feats[None].astype(np.float32)).to(dev)  # (1,C,8,8)

    mp = legal_moves_index_map(b)
    # Build pi one-hot
    try:
        tgt_mv = chess.Move.from_uci(args.target)
    except Exception:
        raise SystemExit(f"Bad --target UCI: {args.target}")
    if tgt_mv not in b.legal_moves:
        raise SystemExit(f"Target move {args.target} not legal in this position")
    tgt_idx = None
    for idx, mv in mp.items():
        if mv == tgt_mv:
            tgt_idx = idx
            break
    if tgt_idx is None:
        raise SystemExit("Mapping error: target move not found in legal_moves_index_map")
    pi = np.zeros((len(mp) * 73), dtype=np.float32)  # wrong shape on purpose to force use of ACTION_SIZE?
    # Actually we need ACTION_SIZE
    from engine import ACTION_SIZE
    pi = np.zeros((ACTION_SIZE,), dtype=np.float32)
    pi[tgt_idx] = 1.0
    y = torch.from_numpy(pi[None]).to(dev)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    def eval_topk():
        model.eval()
        with torch.no_grad():
            logits, v = model(x)
            logits = logits[0].detach().float().cpu().numpy()
        # Mask to legal
        mask = np.full_like(logits, -1e9, dtype=np.float32)
        for i in mp.keys(): mask[i] = 0.0
        p = np.exp(logits + mask - (logits + mask).max()); p /= p.sum()
        top = sorted(mp.keys(), key=lambda i: p[i], reverse=True)[:10]
        print("TOP:", [(mp[i].uci(), float(p[i])) for i in top][:10], flush=True)
        model.train()

    print("Before training:")
    eval_topk()

    for step in range(1, args.steps+1):
        logits, v = model(x)
        logp = torch.log_softmax(logits, dim=1)
        loss = -(y * logp).sum(dim=1).mean()  # CE with one-hot
        opt.zero_grad(); loss.backward(); opt.step()
        if step % (args.steps//4) == 0 or step == 1:
            print(f"step {step}  loss={float(loss):.4f}")
            eval_topk()

if __name__ == "__main__":
    main()
