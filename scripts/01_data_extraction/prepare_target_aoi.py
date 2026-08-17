"""Prepare the vetted Kaihua County target AOI without modifying source data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_shapefile", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "data" / "interim" / "aoi"
    log_dir = root / "outputs" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    source = gpd.read_file(args.source_shapefile)
    if source.crs is None:
        raise RuntimeError("Source AOI has no CRS")
    kaihua = source.loc[source["XZQDM"].astype(str) == "330824"].copy()
    if len(kaihua) != 1:
        raise RuntimeError(f"Expected exactly one Kaihua feature, found {len(kaihua)}")
    if kaihua.geometry.isna().any() or kaihua.geometry.is_empty.any():
        raise RuntimeError("Kaihua geometry is null or empty")
    if not kaihua.geometry.is_valid.all():
        raise RuntimeError("Kaihua geometry is invalid")

    kaihua = kaihua[["XZQDM", "XZQMC", "geometry"]].rename(
        columns={"XZQDM": "admin_code", "XZQMC": "name_zh"})
    kaihua["name_en"] = "Kaihua County"
    kaihua["aoi_role"] = "primary_target"
    kaihua["boundary_status"] = "user_supplied_vetted"
    kaihua_wgs84 = kaihua.to_crs("EPSG:4326")

    geojson_path = out_dir / "target_kaihua_county_wgs84.geojson"
    gpkg_path = out_dir / "target_kaihua_county.gpkg"
    kaihua_wgs84.to_file(geojson_path, driver="GeoJSON")
    kaihua_wgs84.to_file(gpkg_path, layer="kaihua_county", driver="GPKG")

    area_km2 = float(kaihua.to_crs("EPSG:32650").geometry.area.sum() / 1e6)
    audit = {
        "source_path": str(args.source_shapefile),
        "source_crs": str(source.crs),
        "source_feature_count": len(source),
        "filter": "XZQDM == '330824'",
        "target_feature_count": 1,
        "target_geometry_type": kaihua_wgs84.geom_type.iloc[0],
        "target_geometry_valid": bool(kaihua_wgs84.geometry.is_valid.iloc[0]),
        "target_crs": "EPSG:4326",
        "bounds_wgs84": kaihua_wgs84.total_bounds.tolist(),
        "area_km2_utm50n": area_km2,
        "official_planning_area_reference_km2": 2232.25,
        "area_difference_percent": (area_km2 - 2232.25) / 2232.25 * 100,
        "redistribution_license": "not documented in supplied metadata",
        "outputs": [str(geojson_path), str(gpkg_path)],
    }
    (log_dir / "target_aoi_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
