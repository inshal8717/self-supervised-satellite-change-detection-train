from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ContrastivePatchDataset
from .model import SatelliteEncoder, nt_xent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contrastive pretraining on unlabelled satellite imagery")
    p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=50); p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=64); p.add_argument("--samples-per-epoch", type=int, default=4096)
    p.add_argument("--bands", type=int, default=4); p.add_argument("--width", type=int, default=32)
    p.add_argument("--projection-dim", type=int, default=128); p.add_argument("--temperature", type=float, default=.2)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    ds = ContrastivePatchDataset(args.data, args.patch_size, args.samples_per_epoch, args.bands)
    loader = DataLoader(ds, args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=args.device.startswith("cuda"))
    model = SatelliteEncoder(args.bands, args.width, args.projection_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    config = vars(args); best = float("inf"); history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for x1, x2 in loader:
            x1, x2 = x1.to(args.device), x2.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = nt_xent(model(x1), model(x2), args.temperature)
            loss.backward(); optimizer.step(); total += loss.item()
        scheduler.step(); avg = total / max(len(loader), 1); history.append({"epoch": epoch, "loss": avg})
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "loss": avg, "config": config}
        torch.save(state, out / "last.pt")
        if avg < best: best = avg; torch.save(state, out / "best.pt")
        print(f"epoch={epoch:03d} loss={avg:.5f}")
    (out / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__": main()

