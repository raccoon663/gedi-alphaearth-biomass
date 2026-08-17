"""Source-only GEDI L4A QA, evaluated per official vector granule."""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import ee
import pandas as pd

YEARS = range(2019, 2025)
MONTHLY_ID = "LARSE/GEDI/GEDI04_A_002_MONTHLY"
STAGE_NAMES = [
    "all_intersecting_shots", "algorithm_run_flag_eq_1", "required_fields_nonnull",
    "l4_quality_flag_eq_1", "degrade_flag_eq_0", "agbd_nonnegative",
    "agbd_se_nonnegative",
]


def source_geometry() -> ee.Geometry:
    return ee.FeatureCollection("TIGER/2018/States").filter(
        ee.Filter.inList("NAME", ["Georgia", "South Carolina"])).geometry()


def asset_ids_for_year(geometry: ee.Geometry, year: int) -> list[str]:
    images = (ee.ImageCollection(MONTHLY_ID)
              .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
              .filterBounds(geometry))
    nested = images.aggregate_array("table_asset_ids").getInfo()
    return sorted({asset for group in nested for asset in (group or [])})


def count_asset(asset_id: str, geometry: ee.Geometry) -> dict:
    raw = ee.FeatureCollection(asset_id).filterBounds(geometry)
    algorithm = raw.filter(ee.Filter.eq("algorithm_run_flag", 1))
    fields = algorithm.filter(ee.Filter.notNull(["shot_number", "agbd", "agbd_se"]))
    quality = fields.filter(ee.Filter.eq("l4_quality_flag", 1))
    degrade = quality.filter(ee.Filter.eq("degrade_flag", 0))
    agbd = degrade.filter(ee.Filter.gte("agbd", 0))
    agbd_se = agbd.filter(ee.Filter.gte("agbd_se", 0))
    collections = [raw, algorithm, fields, quality, degrade, agbd, agbd_se]
    result = ee.Dictionary({name: fc.size() for name, fc in zip(STAGE_NAMES, collections)}).getInfo()
    return {"asset_id": asset_id, **{k: int(v) for k, v in result.items()}}


def main() -> None:
    ee.Initialize()
    root = Path(__file__).resolve().parents[1]
    geometry = source_geometry()
    granule_rows = []
    for year in YEARS:
        ids = asset_ids_for_year(geometry, year)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(count_asset, asset_id, geometry): asset_id for asset_id in ids}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                row = future.result()
                row["year"] = year
                granule_rows.append(row)
                if i % 25 == 0 or i == len(ids):
                    print(f"{year}: {i}/{len(ids)} granules", flush=True)

    granules = pd.DataFrame(granule_rows)
    totals = granules[STAGE_NAMES].sum()
    rows, previous = [], None
    for i, name in enumerate(STAGE_NAMES):
        n = int(totals[name])
        rows.append({"stage": i, "filter_stage": name, "n_after": n,
                     "removed_at_stage": None if previous is None else previous - n})
        previous = n
    by_year = granules.groupby("year", as_index=False)[STAGE_NAMES].sum()

    tables = root / "outputs" / "tables"
    logs = root / "outputs" / "logs"
    tables.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables / "gedi_qa.csv", index=False)
    by_year.to_csv(tables / "gedi_qa_by_year.csv", index=False)
    granules.to_csv(tables / "gedi_qa_by_granule.csv", index=False)
    metadata = {
        "domain": "source_only_Georgia_South_Carolina",
        "years": list(YEARS),
        "source": "official vector granules listed in monthly raster table_asset_ids",
        "stable_id": "shot_number",
        "duplicate_audit": "performed after table export; not assumed from summed granule counts",
        "target_labels_read": False,
    }
    (logs / "gedi_qa_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
