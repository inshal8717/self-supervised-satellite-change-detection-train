from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from sat_change.io import read_image
from sat_change.metrics import rank_auc, segmentation_metrics
from sat_change.postprocess import binary_mask
from sat_change.runtime import load_model, score_pair


st.set_page_config(page_title="Satellite Change Detection", page_icon=":satellite:", layout="wide")


def _display_image(arr: np.ndarray) -> np.ndarray:
    image = arr[:3] if arr.shape[0] >= 3 else np.repeat(arr[:1], 3, axis=0)
    image = np.moveaxis(image, 0, -1)
    lo, hi = np.nanpercentile(image, [2, 98])
    return np.clip((image - lo) / max(float(hi - lo), 1e-6), 0, 1)


def _load_uploaded(uploaded: object, suffix: str) -> Path:
    if not uploaded:
        raise ValueError("Both before and after images are required.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded.getvalue())
        return Path(handle.name)


def _npy_bytes(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, arr)
    return buffer.getvalue()


def _geotiff_bytes(arr: np.ndarray, profile: dict) -> bytes | None:
    if not profile:
        return None
    try:
        import rasterio
        from rasterio.io import MemoryFile
    except ImportError:
        return None
    output = dict(profile)
    output.update(driver="GTiff", count=1, dtype="float32", height=arr.shape[0], width=arr.shape[1], compress="deflate")
    with MemoryFile() as memory:
        with memory.open(**output) as dataset:
            dataset.write(arr.astype(np.float32), 1)
        return memory.read()


def _render_map(score: np.ndarray, mask: np.ndarray, profile: dict) -> None:
    if not profile or "transform" not in profile or "crs" not in profile:
        st.info("Geospatial map layers are available for GeoTIFF inputs with CRS metadata.")
        return
    try:
        import pandas as pd
        from rasterio.transform import xy
        from rasterio.warp import transform
        rows, cols = np.where(mask > 0)
        rows, cols = rows[:: max(1, len(rows) // 5000)], cols[:: max(1, len(cols) // 5000)]
        ys, xs = xy(profile["transform"], rows, cols)
        lon, lat = transform(profile["crs"], "EPSG:4326", xs, ys)
        st.map(pd.DataFrame({"lat": lat, "lon": lon}), latitude="lat", longitude="lon", size=5, color="#ef4444")
    except (ImportError, ValueError, TypeError) as exc:
        st.warning(f"Could not render the geospatial layer: {exc}")


def main() -> None:
    st.title("Satellite Change Detection")
    st.caption("Self-supervised feature comparison for registered before/after imagery.")

    checkpoints = sorted(Path("outputs").glob("**/*.pt"))
    with st.sidebar:
        st.header("Inputs")
        checkpoint = st.selectbox("Model checkpoint", checkpoints, format_func=str) if checkpoints else None
        before_upload = st.file_uploader("Before image", type=["npy", "tif", "tiff"])
        after_upload = st.file_uploader("After image", type=["npy", "tif", "tiff"])
        mask_upload = st.file_uploader("Ground-truth mask (optional)", type=["npy", "tif", "tiff"])
        tile_size = st.number_input("Tile size", min_value=16, value=512, step=16)
        overlap = st.number_input("Tile overlap", min_value=0, value=32, step=8)
        threshold_value = st.number_input("Threshold (0 = Otsu)", min_value=0.0, value=0.0, step=0.01)
        run = st.button("Run detection", type="primary", use_container_width=True)

    if not run:
        st.info("Choose a checkpoint and upload a registered before/after pair to begin.")
        return
    if checkpoint is None:
        st.error("No checkpoint was found under outputs/.")
        return

    suffix = Path(before_upload.name).suffix if before_upload else ".npy"
    before_path = _load_uploaded(before_upload, suffix)
    after_path = _load_uploaded(after_upload, Path(after_upload.name).suffix if after_upload else ".npy")
    with st.spinner("Running tiled change detection..."):
        score, profile = score_pair(load_model(checkpoint), before_path, after_path, tile_size=int(tile_size), overlap=int(overlap))
    mask, threshold = binary_mask(score, None if threshold_value == 0 else threshold_value)

    before, _ = read_image(before_path)
    after, _ = read_image(after_path)
    before_path.unlink(missing_ok=True)
    after_path.unlink(missing_ok=True)
    changed_fraction = float(mask.mean())
    st.success("Change detection complete.")
    cards = st.columns(4)
    cards[0].metric("Threshold", f"{threshold:.4f}")
    cards[1].metric("Changed area", f"{changed_fraction:.2%}")
    cards[2].metric("Image size", f"{score.shape[1]} x {score.shape[0]}")
    cards[3].metric("Changed pixels", f"{int(mask.sum()):,}")

    if mask_upload:
        target_path = _load_uploaded(mask_upload, Path(mask_upload.name).suffix)
        target, _ = read_image(target_path)
        target_path.unlink(missing_ok=True)
        metrics = segmentation_metrics(mask, target[0] > 0.5)
        metrics["auroc"] = rank_auc(score, target[0] > 0.5)
        st.subheader("Evaluation")
        metric_cols = st.columns(5)
        for column, name in zip(metric_cols, ("iou", "f1", "precision", "recall", "auroc")):
            column.metric(name.upper(), f"{metrics[name]:.3f}")

    import plotly.graph_objects as go
    st.subheader("Results")
    columns = st.columns(4)
    columns[0].image(_display_image(before), caption="Before")
    columns[1].image(_display_image(after), caption="After")
    columns[2].plotly_chart(go.Figure(go.Heatmap(z=score, colorscale="Turbo")).update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0)), use_container_width=True)
    columns[3].image(mask, caption="Binary change mask")

    st.subheader("Geospatial layer")
    _render_map(score, mask, profile)

    metadata = json.dumps({"threshold": threshold, "changed_fraction": changed_fraction}, indent=2).encode()
    st.download_button("Download metadata", metadata, "change_detection.json", "application/json")
    score_geo = _geotiff_bytes(score, profile)
    mask_geo = _geotiff_bytes(mask, profile)
    st.download_button("Download score (.tif)" if score_geo else "Download score (.npy)", score_geo or _npy_bytes(score), "change_score.tif" if score_geo else "change_score.npy")
    st.download_button("Download mask (.tif)" if mask_geo else "Download mask (.npy)", mask_geo or _npy_bytes(mask), "change_mask.tif" if mask_geo else "change_mask.npy")


if __name__ == "__main__":
    main()
