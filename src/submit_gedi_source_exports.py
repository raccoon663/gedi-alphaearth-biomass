"""Submit batch exports for source-only GEDI QA footprints and QA counts."""
from __future__ import annotations

import json
from pathlib import Path

import ee

YEARS = range(2019, 2025)
MONTHLY_ID = "LARSE/GEDI/GEDI04_A_002_MONTHLY"
DRIVE_FOLDER = "GEDI_AlphaEarth_Biomass_Transfer"


def source_geometry() -> ee.Geometry:
    return ee.FeatureCollection("TIGER/2018/States").filter(
        ee.Filter.inList("NAME", ["Georgia", "South Carolina"])).geometry()


def asset_ids(geometry: ee.Geometry, year: int) -> list[str]:
    nested = (ee.ImageCollection(MONTHLY_ID)
              .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
              .filterBounds(geometry).aggregate_array("table_asset_ids").getInfo())
    return sorted({item for group in nested for item in (group or [])})


def stages(raw: ee.FeatureCollection) -> list[tuple[str, ee.FeatureCollection]]:
    algorithm = raw.filter(ee.Filter.eq("algorithm_run_flag", 1))
    fields = algorithm.filter(ee.Filter.notNull(["shot_number", "agbd", "agbd_se"]))
    quality = fields.filter(ee.Filter.eq("l4_quality_flag", 1))
    degrade = quality.filter(ee.Filter.eq("degrade_flag", 0))
    agbd = degrade.filter(ee.Filter.gte("agbd", 0))
    agbd_se = agbd.filter(ee.Filter.gte("agbd_se", 0))
    return [
        ("all_intersecting_shots", raw), ("algorithm_run_flag_eq_1", algorithm),
        ("required_fields_nonnull", fields), ("l4_quality_flag_eq_1", quality),
        ("degrade_flag_eq_0", degrade), ("agbd_nonnegative", agbd),
        ("agbd_se_nonnegative", agbd_se),
    ]


def balanced_merge(collections: list[ee.FeatureCollection]) -> ee.FeatureCollection:
    """Merge with logarithmic expression depth to avoid Python serializer recursion."""
    if not collections:
        return ee.FeatureCollection([])
    current = collections
    while len(current) > 1:
        next_level = []
        for i in range(0, len(current), 2):
            next_level.append(current[i] if i + 1 == len(current)
                              else current[i].merge(current[i + 1]))
        current = next_level
    return current[0]


def main() -> None:
    ee.Initialize()
    root = Path(__file__).resolve().parents[1]
    geometry = source_geometry()
    submitted = []
    existing = {t.status().get("description"): t for t in ee.batch.Task.list()
                if t.status().get("state") in {"READY", "RUNNING", "COMPLETED"}}
    for year in YEARS:
        ids = asset_ids(geometry, year)
        raw_collections = []
        count_features = []
        for asset_id in ids:
            raw_asset = ee.FeatureCollection(asset_id).filterBounds(geometry)
            raw_collections.append(raw_asset)
            count_features.append(ee.Feature(None, {
                "asset_id": asset_id, "year": year,
                **{name: fc.size() for name, fc in stages(raw_asset)},
            }))

        valid = stages(balanced_merge(raw_collections))[-1][1]
        valid = valid.map(lambda f: ee.Feature(f).set({
            "lon": ee.Feature(f).geometry().coordinates().get(0),
            "lat": ee.Feature(f).geometry().coordinates().get(1),
            "year": year,
        }))
        selectors = [
            "shot_number", "lon", "lat", "delta_time", "year", "agbd", "agbd_se",
            "algorithm_run_flag", "l2_quality_flag", "l4_quality_flag", "degrade_flag",
            "sensitivity", "predictor_limit_flag", "response_limit_flag", "pft_class",
            "region_class", "beam", "orbit_number",
        ]
        footprint_task = ee.batch.Export.table.toDrive(
            collection=valid, description=f"gedi_source_valid_{year}",
            folder=DRIVE_FOLDER, fileNamePrefix=f"gedi_source_valid_{year}",
            fileFormat="CSV", selectors=selectors,
        )
        count_task = ee.batch.Export.table.toDrive(
            collection=ee.FeatureCollection(count_features),
            description=f"gedi_source_qa_counts_{year}", folder=DRIVE_FOLDER,
            fileNamePrefix=f"gedi_source_qa_counts_{year}", fileFormat="CSV",
        )
        tasks = [("valid_footprints", footprint_task), ("qa_counts", count_task)]
        ids_out = []
        for kind, task in tasks:
            description = task.config["description"]
            actual = existing.get(description)
            if actual is None:
                task.start()
                actual = task
            submitted.append({"year": year, "kind": kind, "task_id": actual.id,
                              "description": description, "asset_count": len(ids)})
            ids_out.append(actual.id)
        print(year, len(ids), *ids_out, flush=True)

    out = root / "outputs" / "logs" / "gedi_source_export_tasks.json"
    out.write_text(json.dumps(submitted, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
