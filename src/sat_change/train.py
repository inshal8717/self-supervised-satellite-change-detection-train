"""Contrastive (SimCLR-style) pretraining on unlabelled imagery.

Example:
    python -m sat_change.train --data data/synthetic/images --output outputs/pretrain --epochs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ContrastivePatchDataset
from .model import SatelliteEncoder, nt_xent
from .runtime import default_device

log = logging.getLogger("sat_change.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contrastive pretraining on unlabelled satellite imagery")
    p.add_argument("--data", required=True, help="directory of .npy/.tif imagery (masks are ignored)")
    p.add_argument("--output", required=True, help="directory for checkpoints and history.json")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--samples-per-epoch", type=int, default=4096)
    p.add_argument("--bands", type=int, default=4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=default_device())
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    if args.samples_per_epoch < args.batch_size:
        raise SystemExit("--samples-per-epoch must be at least --batch-size")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dataset = ContrastivePatchDataset(args.data, args.patch_size, args.samples_per_epoch, args.bands)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=args.device.startswith("cuda"),
    )
    model = SatelliteEncoder(args.bands, args.width, args.projection_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    config = vars(args)
    best_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x1, x2 in loader:
            x1, x2 = x1.to(args.device), x2.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = nt_xent(model(x1), model(x2), args.temperature)
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()

        avg = total / len(loader)
        history.append({"epoch": epoch, "loss": avg})
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": avg,
            "config": config,
        }
        torch.save(state, out / "last.pt")
        if avg < best_loss:
            best_loss = avg
            torch.save(state, out / "best.pt")
        log.info("epoch=%03d loss=%.5f", epoch, avg)

    (out / "history.json").write_text(json.dumps(history, indent=2))
    log.info("wrote checkpoints and history to %s", out)


if __name__ == "__main__":
    main()
