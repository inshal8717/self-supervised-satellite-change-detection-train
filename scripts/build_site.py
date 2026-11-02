"""Build the static project page (docs/index.html) from reproducible results.

Reads docs/data/history.json and docs/data/metrics.json, written by
scripts/reproduce_results.py, and writes a self-contained page with inline SVG
charts. Architecture diagram is rendered client-side by Mermaid from a CDN.

Usage:
    python scripts/build_site.py
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/inshal8717/self-supervised-satellite-change-detection-train"
PAGES_URL = "https://inshal8717.github.io/self-supervised-satellite-change-detection-train/"
ARCHITECTURE = """flowchart LR
    A[Unlabelled Sentinel-2 imagery<br/>B02 B03 B04 B08] --> B[Per-band normalisation]
    B --> C[Random patch + two augmented views]
    C --> E[Shared CNN encoder]
    E --> H[NT-Xent contrastive loss]
    H --> E
    B --> I[Before / after scenes]
    I --> J[Same encoder, dense features at 3 scales]
    J --> K[Cosine distance, upsampled and averaged]
    K --> L[Otsu threshold]
    L --> M[Change mask and score map]"""


def loss_chart(history: list[dict]) -> str:
    width, height, pad = 640, 280, 48
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    x0, x1 = min(epochs), max(epochs)
    y_max = max(losses) * 1.1
    def sx(e: float) -> float:
        return pad + (e - x0) / max(x1 - x0, 1) * (width - 2 * pad)
    def sy(v: float) -> float:
        return height - pad - v / y_max * (height - 2 * pad)
    points = " ".join(f"{sx(h['epoch']):.1f},{sy(h['loss']):.1f}" for h in history)
    ticks = "".join(
        f'<line x1="{pad}" x2="{width - pad}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>'
        f'<text x="{pad - 6}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v:.1f}</text>'
        for v in [y_max * i / 4 for i in range(5)]
    )
    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="Training loss by epoch">
  {ticks}
  <line x1="{pad}" x2="{pad}" y1="{pad}" y2="{height - pad}" class="axis"/>
  <line x1="{pad}" x2="{width - pad}" y1="{height - pad}" y2="{height - pad}" class="axis"/>
  <polyline points="{points}" class="series"/>
  <text x="{width / 2}" y="{height - 10}" class="tick" text-anchor="middle">epoch</text>
  <text x="14" y="{height / 2}" class="tick" text-anchor="middle" transform="rotate(-90 14 {height / 2})">NT-Xent loss</text>
</svg>"""


def bar_chart(items: list[tuple[str, float]], label: str) -> str:
    """Horizontal bars for values in [0, 1]."""
    width, row_h, label_w = 640, 36, 170
    height = row_h * len(items) + 10
    bars = []
    for i, (name, value) in enumerate(items):
        y = 5 + i * row_h
        bar_w = max(0.0, min(value, 1.0)) * (width - label_w - 70)
        bars.append(
            f'<text x="{label_w - 10}" y="{y + 20}" class="tick" text-anchor="end">{html.escape(name)}</text>'
            f'<rect x="{label_w}" y="{y + 6}" width="{bar_w:.1f}" height="20" class="bar"/>'
            f'<text x="{label_w + bar_w + 8:.1f}" y="{y + 21}" class="tick">{value:.3f}</text>'
        )
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)}">'
            + "".join(bars) + "</svg>")


def comparison_section(baselines: dict[str, dict[str, float]]) -> str:
    """Comparison table, AUROC bars, and a verdict written from the numbers."""
    methods = ["trained encoder", "random encoder", "spectral difference"]
    cols = [("auroc", "AUROC"), ("f1", "F1"), ("iou", "IoU"), ("precision", "Precision"), ("recall", "Recall")]
    head = "<th>method</th>" + "".join(f"<th>{label}</th>" for _, label in cols)
    body = "".join(
        f"<tr><td>{m}</td>" + "".join(f"<td>{baselines[m][k]:.3f}</td>" for k, _ in cols) + "</tr>"
        for m in methods
    )
    table = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    bars = bar_chart([(m, baselines[m]["auroc"]) for m in methods], "Held-out AUROC by method")

    trained = baselines["trained encoder"]
    spectral = baselines["spectral difference"]
    random_enc = baselines["random encoder"]
    if trained["auroc"] < spectral["auroc"]:
        extra = ""
        if random_enc["f1"] > trained["f1"] and random_enc["iou"] > trained["iou"]:
            extra = " The random-weight encoder also scores higher than the trained one on F1 and IoU."
        verdict = (
            f"<strong>On this held-out set, the trained encoder does not beat a plain spectral difference.</strong> "
            f"Its AUROC is {trained['auroc']:.2f}, near the 0.50 expected from chance, against "
            f"{spectral['auroc']:.2f} for the spectral difference.{extra} "
            f"Pretraining on this data, with this budget, is not yet adding value."
        )
    else:
        verdict = (
            f"The trained encoder scores AUROC {trained['auroc']:.2f}, against {spectral['auroc']:.2f} for the "
            f"spectral difference baseline."
        )
    return (f'<div class="card">{table}</div>'
            f'<div class="card">{bars}</div>'
            f'<div class="verdict"><p>{verdict}</p></div>')


def scene_table(scenes: list[dict]) -> str:
    cols = ["scene", "iou", "f1", "precision", "recall", "auroc", "threshold"]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    rows = []
    for s in scenes:
        cells = "".join(
            f"<td>{html.escape(str(s[c]))}</td>" if c == "scene" else f"<td>{s[c]:.3f}</td>" for c in cols
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build(history: list[dict], report: dict, baselines: dict[str, dict[str, float]]) -> str:
    final_loss = history[-1]["loss"]
    epochs = history[-1]["epoch"]
    scenes = report["scenes"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Self-supervised satellite change detection</title>
<style>
  :root {{ --ink: #1f2933; --muted: #5f6b7a; --accent: #0f766e; --line: #d9e2ec; --bg: #f7fafc; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; color: var(--ink); background: var(--bg); margin: 0; line-height: 1.55; }}
  main {{ max-width: 880px; margin: 0 auto; padding: 32px 20px 64px; }}
  h1 {{ font-size: 2rem; margin: 0 0 8px; }}
  h2 {{ margin-top: 40px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }}
  .lede {{ font-size: 1.15rem; color: var(--muted); }}
  .links a {{ margin-right: 16px; color: var(--accent); font-weight: 600; }}
  .card {{ background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 18px; margin: 16px 0; }}
  svg {{ width: 100%; height: auto; }}
  .grid {{ stroke: var(--line); }}
  .axis {{ stroke: var(--muted); }}
  .series {{ fill: none; stroke: var(--accent); stroke-width: 2.5; }}
  .bar {{ fill: var(--accent); opacity: 0.85; }}
  .tick {{ fill: var(--muted); font-size: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.95rem; }}
  th, td {{ text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--line); }}
  th:first-child, td:first-child {{ text-align: left; }}
  code {{ background: #edf2f7; padding: 2px 6px; border-radius: 4px; }}
  pre {{ background: #1f2933; color: #f7fafc; padding: 14px; border-radius: 8px; overflow-x: auto; }}
  .note {{ color: var(--muted); font-size: 0.95rem; }}
  .mermaid {{ text-align: center; }}
</style>
</head>
<body>
<main>
<h1>Self-supervised satellite change detection</h1>
<p class="lede">Detects change between two satellite images of the same place, using a convolutional
encoder pretrained with contrastive learning on <strong>unlabelled</strong> imagery. No change labels are
needed for training.</p>
<p class="note"><strong>Status:</strong> on held-out synthetic scenes, the pretrained model does not yet beat a simple
per-pixel spectral difference. The comparison is below. This is a research baseline with an honest negative result.</p>
<p class="links">
  <a href="{REPO_URL}">Source code</a>
  <a href="{REPO_URL}#quick-start">Quick start</a>
  <a href="{REPO_URL}#limitations">Limitations</a>
</p>

<h2>How it works</h2>
<div class="card mermaid">
<pre class="mermaid">
{html.escape(ARCHITECTURE)}
</pre>
</div>
<p>The encoder is a small fully convolutional network. Before and after scenes pass through the same encoder, and the
distance between their feature maps at three scales gives a dense change score. Otsu thresholding turns that score
into a mask without any labels.</p>

<h2>Training</h2>
<div class="card">{loss_chart(history)}</div>
<p class="note">NT-Xent loss over {epochs} epochs, reproduced by <code>scripts/reproduce_results.py</code>.
Final epoch loss: {final_loss:.3f}.</p>

<h2>Held-out results</h2>
<p>The encoder was pretrained on scenes 0 to 7 and evaluated on scenes 8 to 11, which it never saw. Ground-truth masks
were used only for scoring. Two baselines use the same scenes and the same thresholding: a plain per-pixel spectral
difference, and the same network with random, untrained weights.</p>
{comparison_section(baselines)}
<div class="card">{scene_table(scenes)}</div>
<p class="note">These are synthetic scenes: a changed region is a flat, distinct spectral signature with mild noise, so
they are much easier than real change. The threshold is chosen per scene by Otsu, which keeps it label-free.</p>

<h2>Reproduce</h2>
<pre>pip install -e ".[geo,dashboard]"
python scripts/reproduce_results.py   # about a few minutes on CPU
python scripts/build_site.py          # rebuilds this page</pre>
<p>Run the interactive dashboard locally with <code>streamlit run app.py</code>. It is not hosted, because this project
page is static.</p>

<h2>Limitations</h2>
<ul>
  <li>Research baseline. It is not a disaster-monitoring product.</li>
  <li>Outputs a binary change mask. It does not say what kind of change happened.</li>
  <li>Assumes the two images are already aligned at the pixel level.</li>
  <li>Real-world accuracy has not been measured. Public benchmarks with trusted masks are the next step.</li>
</ul>
<p class="note">Source: <a href="{REPO_URL}">{REPO_URL.replace("https://", "")}</a> ·
Page generated by <code>scripts/build_site.py</code>.</p>
</main>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
  mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "docs" / "data"))
    parser.add_argument("--output", default=str(ROOT / "docs" / "index.html"))
    args = parser.parse_args()

    data = Path(args.data)
    history = json.loads((data / "history.json").read_text())
    report = json.loads((data / "metrics.json").read_text())
    baselines = json.loads((data / "baselines.json").read_text())
    Path(args.output).write_text(build(history, report, baselines), encoding="utf-8")
    print(f"wrote {args.output}  (live at {PAGES_URL})")


if __name__ == "__main__":
    main()
