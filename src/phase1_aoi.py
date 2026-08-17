"""Phase 1: save exact source geometry and explicitly provisional target geometry."""
from __future__ import annotations

import json
from pathlib import Path

import ee


def save_geojson(path: Path, info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def main() -> None:
    ee.Initialize()
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "interim" / "aoi"
    source = ee.FeatureCollection("TIGER/2018/States").filter(
        ee.Filter.inList("NAME", ["Georgia", "South Carolina"])
    )
    if source.size().getInfo() != 2:
        raise RuntimeError("Expected exactly Georgia and South Carolina")
    target = ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Rectangle([118.00, 28.90, 119.00, 29.70]),
            {"name": "Kaihua-Qianjiangyuan provisional envelope", "status": "provisional"},
        )
    ])
    save_geojson(out / "source_ga_sc_tiger2018.geojson", source.getInfo())
    save_geojson(out / "target_kaihua_qianjiangyuan_provisional.geojson", target.getInfo())
    audit = {
        "source_boundary": "TIGER/2018/States",
        "source_names": source.aggregate_array("NAME").getInfo(),
        "source_area_km2": source.geometry().area().divide(1e6).getInfo(),
        "target_boundary": "provisional bbox",
        "target_bbox_wgs84": [118.00, 28.90, 119.00, 29.70],
        "target_area_km2": target.geometry().area().divide(1e6).getInfo(),
        "gaul_kaihua_match_count": ee.FeatureCollection("FAO/GAUL/2015/level2")
            .filter(ee.Filter.eq("ADM0_NAME", "China"))
            .filter(ee.Filter.eq("ADM2_NAME", "Kaihua")).size().getInfo(),
    }
    save_geojson(root / "outputs" / "logs" / "phase1_aoi_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
