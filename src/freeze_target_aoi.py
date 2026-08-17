"""Validate and freeze the already vetted Kaihua County target AOI."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "data" / "interim" / "aoi" / "target_kaihua_county_wgs84.geojson"
    out_dir = root / "data" / "boundaries"
    manifest_dir = root / "outputs" / "manifests"
    log_dir = root / "outputs" / "logs"
    for directory in [out_dir, manifest_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    aoi = gpd.read_file(source)
    if len(aoi) != 1 or aoi.crs.to_epsg() != 4326:
        raise RuntimeError("Kaihua AOI must be one EPSG:4326 feature")
    if aoi.geometry.isna().any() or aoi.geometry.is_empty.any() or not aoi.geometry.is_valid.all():
        raise RuntimeError("Kaihua AOI geometry validation failed")
    if str(aoi.iloc[0]["admin_code"]) != "330824":
        raise RuntimeError("Kaihua administrative code mismatch")
    bounds = aoi.total_bounds.tolist()
    expected = [117.9, 28.8, 118.8, 29.7]
    if not (bounds[0] >= expected[0] and bounds[1] >= expected[1]
            and bounds[2] <= expected[2] and bounds[3] <= expected[3]):
        raise RuntimeError("Kaihua AOI is outside the expected western Zhejiang envelope")
    area_km2 = float(aoi.to_crs("EPSG:32650").geometry.area.sum() / 1e6)
    if not 2100 <= area_km2 <= 2350:
        raise RuntimeError(f"Kaihua AOI area is implausible: {area_km2}")

    output = out_dir / "kaihua_target_aoi.geojson"
    aoi.to_file(output, driver="GeoJSON")
    digest = sha256(output)
    (manifest_dir / "kaihua_target_aoi.sha256").write_text(
        f"{digest}  kaihua_target_aoi.geojson\n", encoding="utf-8")
    record = {
        "status": "frozen", "path": str(output), "sha256": digest,
        "crs": "EPSG:4326", "feature_count": 1, "geometry_valid": True,
        "admin_code": "330824", "bounds_wgs84": bounds,
        "area_km2_utm50n": area_km2, "target_labels_read": False,
        "redistribution_license": "not documented in supplied metadata",
    }
    (log_dir / "frozen_target_aoi.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
