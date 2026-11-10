"""Build the static project page (docs/index.html) from reproducible results.

Reads the JSON files in docs/data/ written by scripts/reproduce_results.py and
scripts/make_example_images.py, and writes a self-contained page. The charts and
the diagram are inline SVG, so the page loads no scripts and has no external
dependencies. The example images are PNGs in docs/img/.

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

# (key in baselines.json, plain-English name shown on the page)
METHODS = [
    ("trained", "Trained network (shared stretch, current)"),
    ("trained_per_photo", "Same trained network, old per-photo stretch"),
    ("untrained", "Same network, untrained (shared stretch)"),
    ("diff_shared", "Colour difference, shared stretch"),
    ("diff_per_photo", "Colour difference, per-photo stretch"),
    ("diff_raw", "Colour difference, raw values"),
]


def esc(text: object) -> str:
    return html.escape(str(text))


# ---------------------------------------------------------------- diagram ---

def _box(x: int, y: int, w: int, h: int, lines: list[str], strong: bool = False) -> str:
    cls = "box strong" if strong else "box"
    rows = "".join(
        f'<text x="{x + w / 2}" y="{y + h / 2 + (i - (len(lines) - 1) / 2) * 17 + 5}" text-anchor="middle" class="label">'
        f"{esc(line)}</text>"
        for i, line in enumerate(lines)
    )
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/>{rows}'


def _arrow(x1: int, y1: int, x2: int, y2: int) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="arrow" marker-end="url(#head)"/>'


def flow_diagram() -> str:
    """Two-row diagram in plain words: learn first, then compare two dates."""
    w, h, gap = 200, 74, 40
    xs = [10 + i * (w + gap) for i in range(3)]
    top, bottom = 38, 190
    parts = [
        '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker></defs>',
        '<text x="10" y="20" class="step">Step 1: learn what satellite images look like (no change labels needed)</text>',
        _box(xs[0], top, w, h, ["Lots of satellite", "photos, unlabelled"]),
        _arrow(xs[0] + w, top + h // 2, xs[1], top + h // 2),
        _box(xs[1], top, w, h, ["Cut out a small patch,", "distort it in two ways"]),
        _arrow(xs[1] + w, top + h // 2, xs[2], top + h // 2),
        _box(xs[2], top, w, h, ["Network learns: both", "versions are the same patch"], strong=True),
        f'<text x="10" y="{bottom - 18}" class="step">Step 2: compare the same place on two dates</text>',
        _box(xs[0], bottom, w, h, ["Photo before", "and photo after"]),
        _arrow(xs[0] + w, bottom + h // 2, xs[1], bottom + h // 2),
        _box(xs[1], bottom, w, h, ["Network describes", "every pixel of both"], strong=True),
        _arrow(xs[1] + w, bottom + h // 2, xs[2], bottom + h // 2),
        _box(xs[2], bottom, w, h, ["Pixels whose description", "differs a lot = changed"]),
    ]
    width = xs[2] + w + 10
    height = bottom + h + 10
    return (f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="Step 1 learns from unlabelled photos. Step 2 compares two dates and marks pixels that differ.">'
            + "".join(parts) + "</svg>")


# ----------------------------------------------------------------- charts ---

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
        f'<line x1="{pad}" x2="{width - pad}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="gridline"/>'
        f'<text x="{pad - 6}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v:.1f}</text>'
        for v in [y_max * i / 4 for i in range(5)]
    )
    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="Training loss by epoch">
  {ticks}
  <line x1="{pad}" x2="{pad}" y1="{pad}" y2="{height - pad}" class="axis"/>
  <line x1="{pad}" x2="{width - pad}" y1="{height - pad}" y2="{height - pad}" class="axis"/>
  <polyline points="{points}" class="series"/>
  <text x="{width / 2}" y="{height - 10}" class="tick" text-anchor="middle">epoch (one pass of training)</text>
  <text x="14" y="{height / 2}" class="tick" text-anchor="middle" transform="rotate(-90 14 {height / 2})">loss (lower is better)</text>
</svg>"""


def bar_chart(items: list[tuple[str, float]], label: str) -> str:
    """Horizontal bars for values in [0, 1], with a dashed line at 0.5 (guessing)."""
    width, row_h, label_w = 720, 40, 330
    height = row_h * len(items) + 24
    scale = width - label_w - 70
    guess_x = label_w + 0.5 * scale
    bars = [
        f'<line x1="{guess_x:.1f}" x2="{guess_x:.1f}" y1="0" y2="{row_h * len(items) + 6}" class="guess"/>',
        f'<text x="{guess_x:.1f}" y="{row_h * len(items) + 20}" class="tick" text-anchor="middle">0.5 = guessing</text>',
    ]
    for i, (name, value) in enumerate(items):
        y = 5 + i * row_h
        bar_w = max(0.0, min(value, 1.0)) * scale
        bars.append(
            f'<text x="{label_w - 10}" y="{y + 21}" class="tick" text-anchor="end">{esc(name)}</text>'
            f'<rect x="{label_w}" y="{y + 6}" width="{bar_w:.1f}" height="22" class="bar"/>'
            f'<text x="{label_w + bar_w + 8:.1f}" y="{y + 22}" class="tick">{value:.3f}</text>'
        )
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}">'
            + "".join(bars) + "</svg>")


# --------------------------------------------------------------- sections ---

def example_section(example: dict | None) -> str:
    """The before/after figure, with captions written from the measured numbers."""
    if not example:
        return ""
    panels = [
        ("before", "Before", "The first photo."),
        ("after", "After", "The second photo of the same place."),
        ("truth", "What really changed",
         f"{example['truth_changed_pct']:.1f}% of the scene. This is the answer key."),
        ("model", "Trained network (current)",
         f"Marks {example['model_changed_pct']:.1f}% as changed. Score (F1): {example['model_f1']:.2f}."),
        ("model_old", "Same network, old scaling",
         f"Marks {example['model_old_changed_pct']:.0f}% as changed. Score (F1): {example['model_old_f1']:.2f}."),
        ("raw", "Colour difference, raw",
         f"Marks {example['raw_changed_pct']:.1f}% as changed. Score (F1): {example['raw_f1']:.2f}."),
    ]
    figures = "".join(
        f'<figure><img src="img/example_{key}.png" alt="{esc(title)}" width="288" height="288">'
        f"<figcaption><strong>{esc(title)}</strong><br>{esc(text)}</figcaption></figure>"
        for key, title, text in panels
    )
    return (
        f'<p>Here is one scene the network never saw during training ({esc(example["scene"])}, a synthetic scene). '
        "Orange means &ldquo;changed&rdquo;. The first two images are the question, the third is the answer key, and "
        "the rest are attempts to find the answer. The changed area is obvious to the eye, and a good method should "
        "find just that patch.</p>"
        f'<div class="panels">{figures}</div>'
    )


def comparison_section(baselines: dict[str, dict[str, float]]) -> str:
    """Comparison table, bar chart of AUROC, and a verdict written from the numbers."""
    methods = [(k, name) for k, name in METHODS if k in baselines]
    cols = [("auroc", "AUROC"), ("f1", "F1"), ("iou", "IoU"), ("precision", "Precision"), ("recall", "Recall")]
    head = "<th>method</th>" + "".join(f"<th>{label}</th>" for _, label in cols)
    body = "".join(
        f"<tr><td>{esc(name)}</td>" + "".join(f"<td>{baselines[k][c]:.3f}</td>" for c, _ in cols) + "</tr>"
        for k, name in methods
    )
    table = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    bars = bar_chart([(name, baselines[k]["auroc"]) for k, name in methods], "Held-out AUROC by method")

    verdict = verdict_text(baselines)
    return (f'<div class="card">{table}</div><div class="card">{bars}</div>'
            f'<div class="verdict"><p>{verdict}</p></div>')


def _get(baselines: dict[str, dict[str, float]], key: str, metric: str = "auroc") -> float | None:
    return baselines.get(key, {}).get(metric)


def verdict_text(baselines: dict[str, dict[str, float]]) -> str:
    """Plain-English reading of the comparison, written from the measured numbers."""
    trained = _get(baselines, "trained")
    old = _get(baselines, "trained_per_photo")
    untrained = _get(baselines, "untrained")
    simple = {k: _get(baselines, k) for k in ("diff_shared", "diff_raw") if _get(baselines, k) is not None}
    best_simple = max(simple.values()) if simple else None
    parts = []
    if old is not None and trained - old > 0.2:
        parts.append(
            f"<strong>One change fixed the network.</strong> The same trained weights score {old:.3f} when each photo "
            f"is brightness-stretched on its own (close to guessing) and {trained:.3f} when both photos share one "
            "stretch. A big change in one photo shifts how that photo is stretched, so with separate stretches the "
            "whole scene looked different."
        )
    elif trained < 0.6:
        parts.append(f"<strong>The trained network does not work yet.</strong> It scores {trained:.3f}, close to guessing.")
    else:
        parts.append(f"The trained network scores {trained:.3f}.")
    if untrained is not None and trained - untrained < 0.01:
        parts.append(
            f"<strong>But the pretraining is not shown to help.</strong> The same network with random, untrained "
            f"weights scores {untrained:.3f}. Self-supervised pretraining is the point of this project, so that is the "
            "result to take seriously."
        )
    if best_simple is not None and trained >= best_simple - 0.01:
        parts.append(
            f"<strong>The network also does not beat a plain colour difference.</strong> Subtracting the photos with "
            f"no network at all scores {best_simple:.3f}, the same. These synthetic scenes are easy: the change is a "
            "flat patch on a smooth background, so they show that the code works, not that the network adds anything. "
            "Real, harder scenes are the real test."
        )
    elif best_simple is not None:
        parts.append(f"The best plain colour difference scores {best_simple:.3f}.")
    f1_net, f1_simple = _get(baselines, "trained", "f1"), _get(baselines, "diff_shared", "f1")
    if f1_simple is not None and f1_net < f1_simple - 0.1:
        parts.append(
            f"Separately, the network's final map is worse than its ranking suggests: the automatic cut-off marks too "
            f"many pixels as changed, so its map scores F1 {f1_net:.2f} against {f1_simple:.2f} for the plain difference."
        )
    return " ".join(parts)


def status_line(baselines: dict[str, dict[str, float]]) -> str:
    trained = _get(baselines, "trained")
    old = _get(baselines, "trained_per_photo")
    if old is not None and trained - old > 0.2:
        return (
            f"After fixing a brightness-scaling flaw, the trained network scores {trained:.3f} on held-out synthetic "
            "scenes. It is not yet shown to beat a plain colour difference or an untrained network, and it has not "
            "been tested on real data."
        )
    return f"On held-out synthetic scenes the trained network scores {trained:.3f}."


def scene_table(scenes: list[dict]) -> str:
    cols = ["scene", "iou", "f1", "precision", "recall", "auroc"]
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    rows = []
    for s in scenes:
        cells = "".join(
            f"<td>{esc(s[c])}</td>" if c == "scene" else f"<td>{s[c]:.3f}</td>" for c in cols
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


GLOSSARY = [
    ("Change detection", "Finding what is different between two photos of the same place taken at different times."),
    ("Self-supervised learning", "Training a network without anyone labelling the answers. Here it learns from "
     "unlabelled photos by recognising that two distorted copies of the same patch are still the same patch."),
    ("Encoder", "The neural network. It turns each small area of a photo into a list of numbers that describes it."),
    ("Held-out", "Photos the network never saw while training. Testing on them shows whether it learned something "
     "general, instead of memorising."),
    ("Baseline", "A simple method to compare against. If a fancy method can't beat a simple one, it isn't helping yet."),
    ("AUROC", "A score from 0 to 1 for how well a method ranks truly changed pixels above unchanged ones. "
     "0.5 is the same as guessing, 1.0 is perfect."),
    ("F1", "A score from 0 to 1 for the final changed/unchanged map. It penalises both missed changes and false alarms."),
    ("Otsu threshold", "An automatic way to pick the cut-off that turns a change score into a yes/no map, without labels."),
    ("Synthetic scenes", "Computer-generated photos with a known changed patch. They are easy and are only a check "
     "that the code works, not a measure of real-world accuracy."),
    ("Sentinel-2", "The European satellites whose 10-metre images this project is designed to use."),
]


def glossary_section() -> str:
    items = "".join(f"<dt>{esc(term)}</dt><dd>{esc(text)}</dd>" for term, text in GLOSSARY)
    return f"<dl>{items}</dl>"


STYLE = """
  :root { --ink: #1f2933; --muted: #5f6b7a; --accent: #0f766e; --warm: #ea7a00; --line: #d9e2ec; --bg: #f7fafc; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; color: var(--ink); background: var(--bg);
         margin: 0; line-height: 1.6; }
  main { max-width: 900px; margin: 0 auto; padding: 32px 20px 64px; }
  h1 { font-size: 2rem; margin: 0 0 8px; line-height: 1.25; }
  h2 { margin-top: 44px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
  .lede { font-size: 1.2rem; color: var(--muted); }
  .links a { margin-right: 16px; color: var(--accent); font-weight: 600; }
  .card { background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 18px; margin: 16px 0;
          overflow-x: auto; }
  .verdict { border-left: 4px solid var(--warm); background: #fff8ef; padding: 4px 18px; margin: 16px 0;
             border-radius: 0 8px 8px 0; }
  svg { width: 100%; height: auto; }
  .panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
  figure { margin: 0; background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
  figure img { width: 100%; height: auto; display: block; border-radius: 6px; image-rendering: pixelated; }
  figcaption { font-size: 0.9rem; color: var(--muted); margin-top: 8px; }
  .box { fill: #fff; stroke: var(--accent); stroke-width: 1.5; }
  .box.strong { fill: #e6f4f1; stroke-width: 2.5; }
  .label { fill: var(--ink); font-size: 14px; }
  .step { fill: var(--accent); font-size: 15px; font-weight: 600; }
  .arrow { stroke: var(--muted); stroke-width: 2; }
  .arrowhead { fill: var(--muted); }
  .gridline { stroke: var(--line); }
  .axis { stroke: var(--muted); }
  .series { fill: none; stroke: var(--accent); stroke-width: 2.5; }
  .bar { fill: var(--accent); opacity: 0.85; }
  .guess { stroke: var(--warm); stroke-dasharray: 5 4; stroke-width: 1.5; }
  .tick { fill: var(--muted); font-size: 12.5px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
  th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--line); }
  th:first-child, td:first-child { text-align: left; }
  code { background: #edf2f7; padding: 2px 6px; border-radius: 4px; }
  pre { background: #1f2933; color: #f7fafc; padding: 14px; border-radius: 8px; overflow-x: auto; }
  dt { font-weight: 700; margin-top: 12px; }
  dd { margin: 2px 0 0; color: var(--muted); }
  .note { color: var(--muted); font-size: 0.95rem; }
"""


def build(history: list[dict], report: dict, baselines: dict[str, dict[str, float]],
          example: dict | None = None) -> str:
    final_loss = history[-1]["loss"]
    epochs = history[-1]["epoch"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Self-supervised satellite change detection</title>
<style>{STYLE}</style>
</head>
<body>
<main>
<h1>Finding what changed between two satellite photos</h1>
<p class="lede">Give it two photos of the same place taken months apart, and it marks the pixels that
changed: cleared forest, new buildings, flooding. It learns what satellite images look like from
<strong>unlabelled</strong> photos, so nobody has to hand-label thousands of examples.</p>
<p class="note"><strong>Status:</strong> a research project. {status_line(baselines)}</p>
<p class="links">
  <a href="{REPO_URL}">Source code</a>
  <a href="{REPO_URL}#quick-start">Quick start</a>
  <a href="#glossary">Glossary</a>
</p>

<h2>What it does, in one picture</h2>
{example_section(example)}

<h2>How it works</h2>
<div class="card">{flow_diagram()}</div>
<p>First it learns, with no answers given, what ordinary satellite imagery looks like. Then, for a before and after
photo, it describes each pixel of both with the same network, and marks the pixels whose descriptions differ
most. Both photos are first scaled with one shared brightness stretch, so ground that did not change looks the
same in both. A cut-off then picks which pixels count as changed.</p>

<h2>How well does it work?</h2>
<p>The network was trained on 8 synthetic scenes and tested on 4 others it had never seen. Every method below was
tested on the same 4 scenes and uses the same cut-off rule. Higher is better. A score of 0.5 on AUROC means no
better than guessing. The first two rows are the same trained weights and differ only in how the photos are
brightness-stretched before the network sees them. See the glossary for what each column means.</p>
{comparison_section(baselines)}

<h2>Training details</h2>
<div class="card">{loss_chart(history)}</div>
<p class="note">Loss over {epochs} epochs, reproduced by <code>scripts/reproduce_results.py</code>.
Final loss: {final_loss:.3f}. The loss falls, so the network is learning its training task.</p>
<h3>Results scene by scene (trained network)</h3>
<div class="card">{scene_table(report["scenes"])}</div>
<p class="note">These are synthetic scenes: the changed patch is a flat, distinct colour with mild noise, so
they are much easier than real change. They check that the pipeline works. They do not measure real-world
accuracy.</p>

<h2>Reproduce it</h2>
<pre>pip install -e ".[geo,dashboard]"
python scripts/reproduce_results.py   # trains and tests, a few minutes on a CPU
python scripts/build_site.py          # rebuilds this page</pre>
<p>The interactive dashboard runs locally with <code>streamlit run app.py</code>. It is not hosted, because this
page is static.</p>

<h2>Limitations</h2>
<ul>
  <li>A research baseline, not a monitoring product.</li>
  <li>It outputs a yes/no change map. It does not say what kind of change happened.</li>
  <li>It assumes the two photos are already lined up pixel for pixel.</li>
  <li>It has not been tested on real labelled change data. That is the necessary next step.</li>
</ul>

<h2 id="glossary">Glossary</h2>
{glossary_section()}
<p class="note">Source: <a href="{REPO_URL}">{REPO_URL.replace("https://", "")}</a> &middot;
Page generated by <code>scripts/build_site.py</code>.</p>
</main>
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
    example_path = data / "example.json"
    example = json.loads(example_path.read_text()) if example_path.exists() else None
    Path(args.output).write_text(build(history, report, baselines, example), encoding="utf-8")
    print(f"wrote {args.output}  (live at {PAGES_URL})")


if __name__ == "__main__":
    main()
