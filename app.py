"""Streamlit dashboard for change detection on a registered before/after pair.

Run locally from the repo root:
    pip install -e ".[geo,dashboard]"
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import streamlit as st

from sat_change.io import read_image
from sat_change.metrics import rank_auc, segmentation_metrics
from sat_change.postprocess import binary_mask
from sat_change.runtime import load_model, score_pair

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
MAX_MAP_POINTS = 5000
IMAGE_TYPES = ["npy", "tif", "tiff"]

st.set_page_config(page_title="Satellite Change Detection", page_icon=":satellite:", layout="wide")


@st.cache_resource(show_spinner=False)
def cached_model(checkpoint: str):
    return load_model(checkpoint)


@contextmanager
def uploaded_as_file(uploaded) -> Iterator[Path]:
    """Write an upload to a temp file for the duration of the block, then delete it."""
    suffix = Path(uploaded.name).suffix or ".npy"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded.getvalue())
        path = Path(handle.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def display_rgb(arr: np.ndarray) -> np.ndarray:
    """First three bands (or a repeated single band) stretched to [0, 1] for display."""
    image = arr[:3] if arr.shape[0] >= 3 else np.repeat(arr[:1], 3, axis=0)
    image = np.moveaxis(image, 0, -1)
    lo, hi = np.nanpercentile(image, [2, 98])
    return np.clip((image - lo) / max(float(hi - lo), 1e-6), 0, 1)


def npy_bytes(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, arr)
    return buffer.getvalue()


def geotiff_bytes(arr: np.ndarray, profile: dict) -> bytes | None:
    """Single-band GeoTIFF in memory, or None when there is no CRS or rasterio is missing."""
    if not profile:
        return None
    try:
        from rasterio.io import MemoryFile
    except ImportError:
        return None
    output = dict(profile)
    output.update(driver="GTiff", count=1, dtype="float32", height=arr.shape[0], width=arr.shape[1], compress="deflate")
    with MemoryFile() as memory:
        with memory.open(**output) as dataset:
            dataset.write(arr.astype(np.float32), 1)
        return memory.read()


def changed_points(mask: np.ndarray, transform, crs) -> tuple[np.ndarray, np.ndarray]:
    """Lon/lat of changed pixels, evenly subsampled to at most MAX_MAP_POINTS.

    Subsampling picks indices into the flat list of changed pixels, so each
    row/column pair still refers to the same pixel.
    """
    from rasterio.transform import xy
    from rasterio.warp import transform as warp_transform

    flat = np.flatnonzero(mask > 0)
    step = max(1, flat.size // MAX_MAP_POINTS)
    rows, cols = np.unravel_index(flat[::step], mask.shape)
    xs, ys = xy(transform, rows, cols)
    lon, lat = warp_transform(crs, "EPSG:4326", xs, ys)
    return np.asarray(lat), np.asarray(lon)


def render_map(mask: np.ndarray, profile: dict) -> None:
    if not profile or "transform" not in profile or "crs" not in profile:
        st.info("The map layer needs a GeoTIFF input with coordinate reference system (CRS) metadata.")
        return
    try:
        import pandas as pd

        lat, lon = changed_points(mask, profile["transform"], profile["crs"])
        st.map(pd.DataFrame({"lat": lat, "lon": lon}), latitude="lat", longitude="lon", size=5, color="#ef4444")
    except (ImportError, ValueError, TypeError) as exc:
        st.warning(f"Could not render the map layer: {exc}")


def sidebar_inputs(checkpoints: list[Path]):
    with st.sidebar:
        st.header("Inputs")
        checkpoint = st.selectbox("Model checkpoint", checkpoints, format_func=lambda p: str(p.relative_to(ROOT))) if checkpoints else None
        before = st.file_uploader("Before image", type=IMAGE_TYPES)
        after = st.file_uploader("After image", type=IMAGE_TYPES)
        mask = st.file_uploader("Ground-truth mask (optional)", type=IMAGE_TYPES)
        tile_size = st.number_input("Tile size", min_value=16, value=512, step=16)
        overlap = st.number_input("Tile overlap", min_value=0, value=32, step=8)
        threshold = st.number_input("Threshold (0 = Otsu)", min_value=0.0, value=0.0, step=0.01)
        run = st.button("Run detection", type="primary", use_container_width=True)
    return checkpoint, before, after, mask, int(tile_size), int(overlap), (None if threshold == 0 else float(threshold)), run


def show_evaluation(mask: np.ndarray, score: np.ndarray, target_upload) -> None:
    with uploaded_as_file(target_upload) as path:
        target, _ = read_image(path)
    truth = target[0] > 0.5
    if truth.shape != mask.shape:
        st.error(f"Mask shape {truth.shape} does not match the image {mask.shape}.")
        return
    metrics = segmentation_metrics(mask, truth)
    metrics["auroc"] = rank_auc(score, truth)
    st.subheader("Evaluation")
    cols = st.columns(5)
    for column, name in zip(cols, ("iou", "f1", "precision", "recall", "auroc")):
        column.metric(name.upper(), f"{metrics[name]:.3f}")


def main() -> None:
    st.title("Satellite Change Detection")
    st.caption("Self-supervised feature comparison for registered before/after imagery.")

    checkpoints = sorted(OUTPUTS.glob("**/*.pt"))
    checkpoint, before_up, after_up, mask_up, tile_size, overlap, fixed_threshold, run = sidebar_inputs(checkpoints)

    if not run:
        st.info("Choose a checkpoint and upload a registered before/after pair to begin.")
        return
    if checkpoint is None:
        st.error("No checkpoint was found under outputs/. Train a model first (see README).")
        return
    if before_up is None or after_up is None:
        st.error("Upload both a before and an after image.")
        return

    try:
        with uploaded_as_file(before_up) as before_path, uploaded_as_file(after_up) as after_path:
            with st.spinner("Running tiled change detection..."):
                score, profile = score_pair(cached_model(str(checkpoint)), before_path, after_path, tile_size=tile_size, overlap=overlap)
            before, _ = read_image(before_path)
            after, _ = read_image(after_path)
    except ValueError as exc:
        st.error(str(exc))
        return

    mask, threshold = binary_mask(score, fixed_threshold)
    changed_fraction = float(mask.mean())
    st.success("Change detection complete.")
    cards = st.columns(4)
    cards[0].metric("Threshold", f"{threshold:.4f}")
    cards[1].metric("Changed area", f"{changed_fraction:.2%}")
    cards[2].metric("Image size", f"{score.shape[1]} x {score.shape[0]}")
    cards[3].metric("Changed pixels", f"{int(mask.sum()):,}")

    if mask_up is not None:
        show_evaluation(mask, score, mask_up)

    import plotly.graph_objects as go

    st.subheader("Results")
    columns = st.columns(4)
    columns[0].image(display_rgb(before), caption="Before")
    columns[1].image(display_rgb(after), caption="After")
    heatmap = go.Figure(go.Heatmap(z=score, colorscale="Turbo")).update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
    columns[2].plotly_chart(heatmap, use_container_width=True)
    columns[3].image(mask, caption="Binary change mask")

    st.subheader("Geospatial layer")
    render_map(mask, profile)

    st.download_button(
        "Download metadata",
        json.dumps({"threshold": threshold, "changed_fraction": changed_fraction}, indent=2).encode(),
        "change_detection.json",
        "application/json",
    )
    score_geo = geotiff_bytes(score, profile)
    mask_geo = geotiff_bytes(mask, profile)
    st.download_button(
        "Download score (.tif)" if score_geo else "Download score (.npy)",
        score_geo or npy_bytes(score),
        "change_score.tif" if score_geo else "change_score.npy",
    )
    st.download_button(
        "Download mask (.tif)" if mask_geo else "Download mask (.npy)",
        mask_geo or npy_bytes(mask),
        "change_mask.tif" if mask_geo else "change_mask.npy",
    )


if __name__ == "__main__":
    main()
