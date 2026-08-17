"""Submit direct 200-cap manifests only for legacy tasks that never started."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee

from submit_source_sampling_exports import (
    DRIVE_FOLDER,
    SEED,
    add_frozen_block,
    asset_ids,
    balanced_merge,
    qa_valid,
    source_geometry,
)

MAX_PER_STRATUM = 200


def sample_200(valid: ee.FeatureCollection) -> ee.FeatureCollection:
    randomized = valid.randomColumn(
        columnName="sampling_random", seed=SEED,
        distribution="uniform", rowKeys=["shot_number"],
    )
    block_ids = ee.List(
        randomized.aggregate_array("spatial_block_id")
    ).distinct().sort()

    def sample_block(block_id):
        return (randomized.filter(ee.Filter.eq("spatial_block_id", block_id))
                .sort("sampling_random").limit(MAX_PER_STRATUM))

    return ee.FeatureCollection(block_ids.map(sample_block)).flatten()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("years", nargs="+", type=int)
    args = parser.parse_args()
    invalid = sorted(set(args.years) - set(range(2019, 2025)))
    if invalid:
        raise SystemExit(f"Unsupported years: {invalid}")

    ee.Initialize()
    root = Path(__file__).resolve().parents[2]
    geometry = source_geometry()
    existing = {
        task.status().get("description"): task
        for task in ee.batch.Task.list()
        if task.status().get("state") in {"READY", "RUNNING", "COMPLETED"}
    }
    records = []
    for year in sorted(set(args.years)):
        ids = asset_ids(geometry, year)
        raw = balanced_merge([
            ee.FeatureCollection(asset_id).filterBounds(geometry)
            for asset_id in ids
        ])
        valid = qa_valid(raw).map(lambda feature, y=year: add_frozen_block(feature, y))
        sampled = sample_200(valid)
        description = f"gedi_sampled_manifest_200_{year}"
        task = existing.get(description)
        if task is None:
            task = ee.batch.Export.table.toDrive(
                collection=sampled,
                description=description,
                folder=DRIVE_FOLDER,
                fileNamePrefix=description,
                fileFormat="CSV",
                selectors=[
                    "shot_number", "lon", "lat", "delta_time", "year",
                    "spatial_block_id", "spatial_block_x", "spatial_block_y",
                    "sampling_random", "agbd", "agbd_se",
                ],
            )
            task.start()
        records.append({
            "year": year,
            "kind": "sampled_manifest_200",
            "task_id": task.id,
            "description": description,
            "asset_count": len(ids),
            "max_per_block_year": MAX_PER_STRATUM,
            "seed": SEED,
        })
        print(year, task.id, flush=True)

    path = root / "outputs" / "logs" / "source_manifest_200_tasks.json"
    prior = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    by_description = {item["description"]: item for item in prior + records}
    path.write_text(
        json.dumps(list(by_description.values()), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
