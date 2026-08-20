"""Fetch real Sentinel-2 L2A pairs from Microsoft Planetary Computer.

Downloads real, open Sentinel-2 surface reflectance imagery for a user-chosen
location and two time windows, co-registers the acquisitions to a common 10 m
UTM grid (B02/B03/B04/B08), and writes them in the layout the pipeline expects:

    <output>/<scene>/images/<scene>_<YYYY-MM-DD>.tif   (before & after)
    <output>/<scene>/masks/<scene>.tif                  (optional change mask)
    <output>/<scene>/manifest.csv                       (path,scene,date,split)

No account or API key is required; access is anonymous (rate-limited).

Example:

    python scripts/fetch_sentinel2.py \
      --center -70.25,-2.9 \
      --size 10 \
      --before 2023-01-01/2023-03-31 \
      --after  2023-07-01/2023-09-30 \
      --scene amazon \
      --output data/real \
      --cloud-max 20

Then pretrain and run change detection exactly as with synthetic data:

    python -m sat_change.train --data data/real --output outputs/pretrain --epochs 10
    python -m sat_change.predict --before data/real/amazon/images/amazon_2023-02-01.tif \
      --after data/real/amazon/images/amazon_2023-08-15.tif \
      --checkpoint outputs/pretrain/best.pt --output outputs/amazon_change.tif
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pystac_client
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, transform

import planetary_computer

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = ["B02", "B03", "B04", "B08"]
# SCL classes that are no-data, clouds, cirrus or shadow: excluded from composites.
BAD_SCL = {0, 3, 7, 8, 9, 10}


def _utm_epsg(lon: float, lat: float) -> int:
    """EPSG code for the UTM zone containing (lon, lat), northern/southern."""
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def _bbox(center: tuple[float, float], size_km: float) -> tuple[float, float, float, float]:
    """Degrees bbox around a (lon, lat) center given an approximate side in km."""
    lon, lat = center
    half = size_km / 2.0
    d_lon = half / (111.32 * max(np.cos(np.radians(lat)), 0.01))
    d_lat = half / 110.574
    return (lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat)


def _target_grid(bbox: tuple[float, float, float, float], dst_crs: int, gsd: int = 10):
    """A GSD-metre UTM grid whose origin is aligned to multiples of GSD."""
    left, bottom, right, top = bbox
    xs, ys = transform("EPSG:4326", f"EPSG:{dst_crs}", [left, left, right, right], [bottom, top, bottom, top])
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    origin_x = np.floor(minx / gsd) * gsd
    origin_y = np.ceil(maxy / gsd) * gsd
    width = int(np.ceil((maxx - origin_x) / gsd))
    height = int(np.ceil((origin_y - miny) / gsd))
    affine = Affine(gsd, 0, origin_x, 0, -gsd, origin_y)
    return affine, height, width


def _search(catalog, bbox, window: str, cloud_max: float) -> list:
    return list(
        catalog.search(
            collections=[COLLECTION],
            bbox=bbox,
            datetime=window,
            query={"eo:cloud_cover": {"lt": cloud_max}},
        ).items()
    )


def _read_band(href: str, dst_crs: int, transform: Affine, height: int, width: int) -> np.ndarray:
    """Read one signed COG band and reproject it onto the shared grid."""
    out = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_crs=src.crs,
            src_transform=src.transform,
            src_nodata=src.nodata,
            dst_crs=f"EPSG:{dst_crs}",
            dst_transform=transform,
            dst_nodata=0,
            resampling=rasterio.enums.Resampling.bilinear,
        )
    return out


def _composite(items: Iterable, dst_crs: int, transform: Affine, height: int, width: int, use_scl: bool):
    """Stack best cloud-free acquisitions for a window into one median composite.

    Returns a (4, H, W) float32 stack in Sentinel-2 integer scale (0..10000).
    Pixels that are clouds/shadow (per SCL) or nodata are excluded from the median.
    """
    stack = []
    for item in items:
        item = planetary_computer.sign(item)
        bands = np.stack([_read_band(item.assets[b].href, dst_crs, transform, height, width) for b in BANDS])
        if use_scl and "SCL" in item.assets:
            scl = np.zeros((height, width), dtype=np.uint8)
            with rasterio.open(item.assets["SCL"].href) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=scl,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    src_nodata=src.nodata,
                    dst_crs=f"EPSG:{dst_crs}",
                    dst_transform=transform,
                    dst_nodata=0,
                    resampling=rasterio.enums.Resampling.nearest,
                )
            bad = np.isin(scl, BAD_SCL)
            bands[:, bad] = 0
        else:
            bands[bands == 0] = np.nan
        stack.append(bands)
    arr = np.stack(stack)  # (n, 4, H, W)
    with np.errstate(invalid="ignore"):
        return np.nanmedian(arr, axis=0).astype(np.float32)


def _write_tiff(path: Path, arr: np.ndarray, crs: int, transform: Affine) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = arr.shape
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": count,
        "height": height,
        "width": width,
        "crs": f"EPSG:{crs}",
        "transform": transform,
        "compress": "deflate",
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--center", required=True, help="lon,lat of the AOI center")
    p.add_argument("--size", type=float, default=10.0, help="AOI side length in km")
    p.add_argument("--before", required=True, help="before window, ISO range 2023-01-01/2023-03-31")
    p.add_argument("--after", required=True, help="after window, ISO range")
    p.add_argument("--scene", required=True, help="scene key used in filenames")
    p.add_argument("--output", default="data/real", help="output root directory")
    p.add_argument("--cloud-max", type=float, default=20.0, help="max eo:cloud_cover per item")
    p.add_argument("--limit", type=int, default=1, help="max acquisitions to median-composite per window (1 = best only)")
    p.add_argument("--no-scl", action="store_true", help="disable SCL cloud masking in composites")
    p.add_argument("--mask", default=None, help="optional ground-truth change-mask GeoTIFF to include for evaluation")
    a = p.parse_args()

    lon, lat = (float(v) for v in a.center.split(","))
    bbox = _bbox((lon, lat), a.size)
    dst_crs = _utm_epsg(lon, lat)
    transform, height, width = _target_grid(bbox, dst_crs, gsd=10)
    print(f"AOI bbox (deg): {bbox}  grid: {height}x{width} px at 10 m, EPSG:{dst_crs}")

    catalog = pystac_client.Client.open(CATALOG_URL)
    scene_dir = Path(a.output) / a.scene
    images_dir = scene_dir / "images"
    masks_dir = scene_dir / "masks"

    rows = []
    for label, window in (("before", a.before), ("after", a.after)):
        items = _search(catalog, bbox, window, a.cloud_max)
        if not items:
            raise SystemExit(f"no cloud-free (<{a.cloud_max}%) Sentinel-2 items in {window} window")
        items = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 999))[: a.limit]
        print(f"{label}: {len(items)} item(s): " + ", ".join(it.id for it in items))
        arr = _composite(items, dst_crs, transform, height, width, use_scl=not a.no_scl)
        date = datetime.fromisoformat(items[0].datetime.isoformat()).date()
        filename = f"{a.scene}_{date.isoformat()}.tif"
        _write_tiff(images_dir / filename, arr, dst_crs, transform)
        rows.append({"path": f"{a.scene}/images/{filename}", "scene": a.scene, "date": date.isoformat(), "split": "train"})
        print(f"  wrote {images_dir / filename}  ({arr.min():.0f}..{arr.max():.0f} DN)")

    if a.mask:
        masks_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(a.mask) as src:
            mask = np.zeros((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=mask,
                src_crs=src.crs,
                src_transform=src.transform,
                src_nodata=src.nodata,
                dst_crs=f"EPSG:{dst_crs}",
                dst_transform=transform,
                dst_nodata=0,
                resampling=rasterio.enums.Resampling.nearest,
            )
        _write_tiff(masks_dir / f"{a.scene}.tif", mask[None], dst_crs, transform)
        print(f"  wrote {masks_dir / f'{a.scene}.tif'}")

    with open(scene_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "scene", "date", "split"])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "scene": a.scene,
        "center": [lon, lat],
        "size_km": a.size,
        "windows": {"before": a.before, "after": a.after},
        "bands": BANDS,
        "grid": {"crs": f"EPSG:{dst_crs}", "transform": list(transform)[:6], "height": height, "width": width},
        "cloud_max": a.cloud_max,
        "scl_masked": not a.no_scl,
    }
    with open(scene_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"done. data at {scene_dir}")


if __name__ == "__main__":
    main()