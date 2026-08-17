"""Submit the frozen source GEDI sampling policy as reproducible GEE exports.

Policy:
- assign every QA-valid footprint to a 50 km EPSG:5070 block first;
- strata are spatial_block_id x year;
- deterministic uniform random order keyed by shot_number, seed 42;
- retain min(N, 500) per stratum;
- never use AGBD for sampling and never access target-domain labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import ee

YEARS = range(2019, 2025)
SEED = 42
BLOCK_METERS = 50_000
MAX_PER_STRATUM = 500
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


def balanced_merge(collections: list[ee.FeatureCollection]) -> ee.FeatureCollection:
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


def qa_valid(raw: ee.FeatureCollection) -> ee.FeatureCollection:
    return (raw.filter(ee.Filter.eq("algorithm_run_flag", 1))
            .filter(ee.Filter.notNull(["shot_number", "agbd", "agbd_se"]))
            .filter(ee.Filter.eq("l4_quality_flag", 1))
            .filter(ee.Filter.eq("degrade_flag", 0))
            .filter(ee.Filter.gte("agbd", 0))
            .filter(ee.Filter.gte("agbd_se", 0)))


def add_frozen_block(feature: ee.Feature, year: int) -> ee.Feature:
    feature = ee.Feature(feature)
    xy = feature.geometry().transform("EPSG:5070", 1).coordinates()
    block_x = ee.Number(xy.get(0)).divide(BLOCK_METERS).floor().toInt()
    block_y = ee.Number(xy.get(1)).divide(BLOCK_METERS).floor().toInt()
    block_id = (ee.String("EPSG5070_50KM_")
                .cat(block_x.format("%d")).cat("_").cat(block_y.format("%d")))
    lonlat = feature.geometry().coordinates()
    return feature.set({
        "spatial_block_id": block_id,
        "spatial_block_x": block_x,
        "spatial_block_y": block_y,
        "spatial_block_crs": "EPSG:5070",
        "spatial_block_size_m": BLOCK_METERS,
        "year": year,
        "lon": lonlat.get(0),
        "lat": lonlat.get(1),
    })


def sample_and_audit(valid: ee.FeatureCollection):
    randomized = valid.randomColumn(
        columnName="sampling_random", seed=SEED,
        distribution="uniform", rowKeys=["shot_number"])
    block_ids = ee.List(randomized.aggregate_array("spatial_block_id")).distinct().sort()

    def sample_block(block_id):
        subset = randomized.filter(ee.Filter.eq("spatial_block_id", block_id))
        return subset.sort("sampling_random").limit(MAX_PER_STRATUM)

    sampled = ee.FeatureCollection(block_ids.map(sample_block)).flatten()

    def audit_block(block_id):
        raw_n = randomized.filter(ee.Filter.eq("spatial_block_id", block_id)).size()
        sampled_n = ee.Number(raw_n).min(MAX_PER_STRATUM).toInt()
        return ee.Feature(None, {
            "spatial_block_id": block_id,
            "raw_qa_valid_n": raw_n,
            "sampled_n": sampled_n,
            "sampling_fraction": ee.Number(sampled_n).divide(raw_n),
            "max_per_stratum": MAX_PER_STRATUM,
            "seed": SEED,
        })

    audit = ee.FeatureCollection(block_ids.map(audit_block))
    return sampled, audit


def distribution_properties(prefix: str, fc: ee.FeatureCollection) -> dict:
    percentile_names = ["p01", "p05", "p25", "p50", "p75", "p95", "p99"]
    percentiles = fc.reduceColumns(
        ee.Reducer.percentile([1, 5, 25, 50, 75, 95, 99], percentile_names),
        ["agbd"])
    props = {
        f"{prefix}_n": fc.size(),
        f"{prefix}_agbd_mean": fc.aggregate_mean("agbd"),
        f"{prefix}_agbd_sd": fc.aggregate_total_sd("agbd"),
        f"{prefix}_agbd_min": fc.aggregate_min("agbd"),
        f"{prefix}_agbd_max": fc.aggregate_max("agbd"),
    }
    props.update({f"{prefix}_agbd_{name}": percentiles.get(name)
                  for name in percentile_names})
    return props


def task_record(task: ee.batch.Task, year: int, kind: str, asset_count: int) -> dict:
    return {"year": year, "kind": kind, "task_id": task.id,
            "description": task.config["description"], "asset_count": asset_count}


def main() -> None:
    ee.Initialize()
    root = Path(__file__).resolve().parents[1]
    geometry = source_geometry()
    existing = {t.status().get("description"): t for t in ee.batch.Task.list()
                if t.status().get("state") in {"READY", "RUNNING", "COMPLETED"}}
    records = []

    for year in YEARS:
        ids = asset_ids(geometry, year)
        raw = balanced_merge([ee.FeatureCollection(i).filterBounds(geometry) for i in ids])
        valid = qa_valid(raw).map(lambda f, y=year: add_frozen_block(f, y))
        sampled, strata = sample_and_audit(valid)
        strata = strata.map(lambda f, y=year: ee.Feature(f).set("year", y))

        annual_stats = ee.FeatureCollection([ee.Feature(None, {
            "year": year,
            "sampling_policy": "EPSG5070_50km_x_year_max500_seed42",
            **distribution_properties("raw", valid),
            **distribution_properties("sampled", sampled),
        })])

        manifest_task = ee.batch.Export.table.toDrive(
            collection=sampled,
            description=f"gedi_sampled_manifest_{year}", folder=DRIVE_FOLDER,
            fileNamePrefix=f"gedi_sampled_manifest_{year}", fileFormat="CSV",
            selectors=["shot_number", "lon", "lat", "delta_time", "year",
                       "spatial_block_id", "spatial_block_x", "spatial_block_y",
                       "sampling_random", "agbd", "agbd_se"],
        )
        strata_task = ee.batch.Export.table.toDrive(
            collection=strata,
            description=f"gedi_sampling_strata_{year}", folder=DRIVE_FOLDER,
            fileNamePrefix=f"gedi_sampling_strata_{year}", fileFormat="CSV",
        )
        stats_task = ee.batch.Export.table.toDrive(
            collection=annual_stats,
            description=f"gedi_sampling_annual_stats_{year}", folder=DRIVE_FOLDER,
            fileNamePrefix=f"gedi_sampling_annual_stats_{year}", fileFormat="CSV",
        )

        for kind, task in [("sampled_manifest", manifest_task),
                           ("stratum_audit", strata_task),
                           ("annual_agbd_audit", stats_task)]:
            actual = existing.get(task.config["description"])
            if actual is None:
                task.start()
                actual = task
            records.append(task_record(actual, year, kind, len(ids)))
        print(year, len(ids), [r["task_id"] for r in records[-3:]], flush=True)

    log_path = root / "outputs" / "logs" / "source_sampling_export_tasks.json"
    log_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
