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
    "trained": _metrics(0.999),
    "trained_per_photo": _metrics(0.49),
    "untrained": _metrics(0.70),
    "diff_shared": _metrics(1.0),
    "diff_per_photo": _metrics(0.74),
    "diff_raw": _metrics(1.0),
}
EXAMPLE = {
    "scene": "scene008", "truth_changed_pct": 5.2, "model_changed_pct": 5.3, "model_old_changed_pct": 73.0,
    "raw_changed_pct": 5.3, "model_f1": 0.99, "model_old_f1": 0.1, "raw_f1": 0.99,
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


def test_verdict_when_the_fix_works_but_a_plain_difference_ties():
    page = _page()
    assert "One change fixed the network" in page
    assert "does not beat a plain colour difference" in page
    assert "0.490" in page and "1.000" in page
    assert "pretraining is not shown to help" not in page  # untrained network is far behind here


def test_verdict_does_not_claim_a_tie_when_the_network_is_clearly_behind():
    behind = dict(BASELINES, trained=_metrics(0.90))
    page = build_site.build(HISTORY, {"scenes": SCENES}, behind, EXAMPLE)
    assert "One change fixed the network" in page
    assert "does not beat a plain colour difference" not in page
    assert "best plain colour difference scores 1.000" in page


def test_verdict_flags_that_an_untrained_network_does_as_well():
    page = build_site.build(HISTORY, {"scenes": SCENES}, dict(BASELINES, untrained=_metrics(0.998)), EXAMPLE)
    assert "pretraining is not shown to help" in page
    assert "random, untrained weights scores 0.998" in page


def test_verdict_flags_a_weak_final_map_when_the_ranking_is_good():
    weak = dict(BASELINES, trained=dict(_metrics(0.999), f1=0.74), diff_shared=dict(_metrics(1.0), f1=0.999))
    page = build_site.build(HISTORY, {"scenes": SCENES}, weak, EXAMPLE)
    assert "marks too many pixels as changed" in page
    assert "marks too many pixels as changed" not in _page()  # F1s are equal in the default fixture


def test_verdict_when_the_network_does_not_work():
    broken = dict(BASELINES, trained=_metrics(0.45))
    page = build_site.build(HISTORY, {"scenes": SCENES}, broken, EXAMPLE)
    assert "does not work yet" in page
    assert "One change fixed the network" not in page


def test_status_line_follows_the_numbers():
    assert "not yet shown to beat a plain colour difference" in build_site.status_line(BASELINES)
    assert "not yet shown" not in build_site.status_line(dict(BASELINES, trained_per_photo=_metrics(0.95)))


def test_page_builds_without_example_or_optional_rows():
    older = {k: v for k, v in BASELINES.items() if k not in ("diff_raw", "trained_per_photo")}
    page = build_site.build(HISTORY, {"scenes": SCENES}, older, None)
    assert "<h1>" in page and "One change fixed the network" not in page


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
