from __future__ import annotations

import subprocess
import sys


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True)


run("scripts/make_synthetic.py", "--output", "data/synthetic", "--scenes", "8", "--size", "96")
run("-m", "sat_change.train", "--data", "data/synthetic/images", "--output", "outputs/demo", "--epochs", "2", "--samples-per-epoch", "128", "--batch-size", "16", "--patch-size", "48")
run("-m", "sat_change.evaluate", "--images", "data/synthetic/images", "--masks", "data/synthetic/masks", "--checkpoint", "outputs/demo/best.pt", "--output", "outputs/demo/metrics.json")

