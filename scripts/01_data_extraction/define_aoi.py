"""Create deterministic first-pass AOIs without target labels."""
from __future__ import annotations

import json
from pathlib import Path

SOURCE_STATES = ["Georgia", "South Carolina"]
TARGET_BBOX = [118.00, 28.90, 119.00, 29.70]


def bbox_feature(name: str, bbox: list[float]) -> dict:
    west, south, east, north = bbox
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return {"type": "Feature", "properties": {"name": name, "status": "first_pass"},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "data" / "interim" / "aoi"
    out.mkdir(parents=True, exist_ok=True)
    target = {"type": "FeatureCollection", "features": [bbox_feature(
        "Kaihua-Qianjiangyuan first-pass bounding AOI", TARGET_BBOX)]}
    (out / "target_aoi_first_pass.geojson").write_text(
        json.dumps(target, indent=2), encoding="utf-8")
    (out / "source_scope.json").write_text(
        json.dumps({"states": SOURCE_STATES, "boundary_source": "pending live GEE TIGER/Line audit"}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
