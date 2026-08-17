"""Authenticated, read-only Earth Engine collection audit."""
from __future__ import annotations

import json
from pathlib import Path

import ee


def describe_collection(collection_id: str, aoi: ee.Geometry) -> dict:
    col = ee.ImageCollection(collection_id)
    first = ee.Image(col.first())
    return {
        "id": collection_id,
        "global_image_count": col.size().getInfo(),
        "aoi_image_count": col.filterBounds(aoi).size().getInfo(),
        "first_band_names": first.bandNames().getInfo(),
        "min_time_start": col.aggregate_min("system:time_start").getInfo(),
        "max_time_start": col.aggregate_max("system:time_start").getInfo(),
    }


def main() -> None:
    ee.Initialize()
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / "logs"
    out.mkdir(parents=True, exist_ok=True)
    target = ee.Geometry.Rectangle([118.00, 28.90, 119.00, 29.70])
    # Broad read-only source envelope; Phase 1 replaces it with exact state boundaries.
    source = ee.Geometry.Rectangle([-85.7, 30.3, -78.4, 35.3])
    result = {
        "alphaearth_source": describe_collection(
            "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL", source),
        "alphaearth_target": describe_collection(
            "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL", target),
        "gedi_source": describe_collection(
            "LARSE/GEDI/GEDI04_A_002_MONTHLY", source),
        "gedi_target": describe_collection(
            "LARSE/GEDI/GEDI04_A_002_MONTHLY", target),
    }
    (out / "phase0_gee_live.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
