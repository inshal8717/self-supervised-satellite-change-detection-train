"""Tests for the static project page and its example images."""

import json
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_site  # noqa: E402

HISTORY = [{"epoch": 1, "loss": 5.0}, {"epoch": 2, "loss": 4.0}]
SCENES = [{"scene": "scene008", "iou": 0.1, "f1": 0.2, "precision": 0.3, "recall": 0.4, "auroc": 0.5}]


def _metrics(auroc: float) -> dict:
    return {"auroc": auroc, "f1": 0.2, "iou": 0.1, "precision": 0.1, "recall": 0.8}


BASELINES = {
    "trained encoder": _metrics(0.49),
    "random encoder": _metrics(0.70),
    "spectral difference": _metrics(0.74),
    "raw difference": _metrics(1.0),
}
EXAMPLE = {
    "scene": "scene008", "truth_changed_pct": 5.2, "model_changed_pct": 73.0, "baseline_changed_pct": 48.0,
    "raw_changed_pct": 5.3, "model_f1": 0.1, "baseline_f1": 0.15, "raw_f1": 0.99,
}


def _page() -> str:
    return build_site.build(HISTORY, {"scenes": SCENES}, BASELINES, EXAMPLE)


def test_page_has_no_scripts_and_no_mermaid():
    """The page must not load a diagram library: it caused a visible 'Syntax error' once."""
    page = _page().lower()
    assert "mermaid" not in page
    assert "<script" not in page


def test_every_svg_is_well_formed():
    svgs = re.findall(r"<svg.*?</svg>", _page(), re.S)
    assert len(svgs) >= 3  # diagram, loss curve, AUROC bars
    for svg in svgs:
        ET.fromstring(svg)


def test_verdict_is_written_from_the_numbers():
    page = _page()
    assert "does not work yet" in page
    assert "1.00" in page  # raw difference score appears in the explanation

    better = dict(BASELINES, **{"trained encoder": _metrics(0.95)})
    page = build_site.build(HISTORY, {"scenes": SCENES}, better, EXAMPLE)
    assert "does not work yet" not in page


def test_page_builds_without_example_or_raw_baseline():
    older = {k: v for k, v in BASELINES.items() if k != "raw difference"}
    page = build_site.build(HISTORY, {"scenes": SCENES}, older, None)
    assert "<h1>" in page and "cause is visible" not in page


def test_committed_page_matches_committed_data():
    """docs/index.html must be regenerated whenever the results in docs/data change."""
    data = ROOT / "docs" / "data"
    if not (data / "baselines.json").exists():
        pytest.skip("results not committed")
    example = data / "example.json"
    expected = build_site.build(
        json.loads((data / "history.json").read_text()),
        json.loads((data / "metrics.json").read_text()),
        json.loads((data / "baselines.json").read_text()),
        json.loads(example.read_text()) if example.exists() else None,
    )
    assert (ROOT / "docs" / "index.html").read_text(encoding="utf-8") == expected


def test_committed_page_images_exist():
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    for src in re.findall(r'<img src="([^"]+)"', page):
        assert (ROOT / "docs" / src).is_file(), src


def test_png_writer_round_trips(tmp_path: Path):
    pytest.importorskip("torch")  # make_example_images imports the model code
    import make_example_images

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    path = tmp_path / "x.png"
    make_example_images.write_png(path, rgb)

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (7, 5)
    idat_start = data.index(b"IDAT") + 4
    idat_len = struct.unpack(">I", data[idat_start - 8:idat_start - 4])[0]
    raw = zlib.decompress(data[idat_start:idat_start + idat_len])
    rows = [raw[i * (1 + 7 * 3) + 1:(i + 1) * (1 + 7 * 3)] for i in range(5)]
    assert np.array_equal(np.frombuffer(b"".join(rows), np.uint8).reshape(5, 7, 3), rgb)
