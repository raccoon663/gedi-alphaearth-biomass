"""Submit optimized replacements for timed-out source sampling audit exports.

This does not resample GEDI.  It only computes the two audit products without
embedding the expensive per-block sampled collection in the same GEE graph.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee

from submit_source_sampling_exports import (
    DRIVE_FOLDER,
    add_frozen_block,
    asset_ids,
    balanced_merge,
    distribution_properties,
    qa_valid,
    source_geometry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("years", nargs="+", type=int)
    parser.add_argument(
        "--kinds", nargs="+", choices=["block_counts", "raw_agbd_audit"],
        default=["block_counts", "raw_agbd_audit"],
    )
    args = parser.parse_args()
    invalid = sorted(set(args.years) - set(range(2019, 2025)))
    if invalid:
        raise SystemExit(f"Unsupported years: {invalid}")

    ee.Initialize()
    root = Path(__file__).resolve().parents[1]
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

        # One record containing a JSON dictionary is much cheaper than mapping
        # one full-collection filter over every spatial block.
        block_counts = ee.FeatureCollection([ee.Feature(None, {
            "year": year,
            "block_counts_json": valid.aggregate_histogram("spatial_block_id"),
        })])
        raw_stats = ee.FeatureCollection([ee.Feature(None, {
            "year": year,
            **distribution_properties("raw", valid),
        })])

        specs = [
            (
                "block_counts",
                f"gedi_sampling_block_counts_recovery_{year}",
                block_counts,
            ),
            (
                "raw_agbd_audit",
                f"gedi_sampling_raw_stats_recovery_{year}",
                raw_stats,
            ),
        ]
        for kind, description, collection in specs:
            if kind not in args.kinds:
                continue
            task = existing.get(description)
            if task is None:
                task = ee.batch.Export.table.toDrive(
                    collection=collection,
                    description=description,
                    folder=DRIVE_FOLDER,
                    fileNamePrefix=description,
                    fileFormat="CSV",
                )
                task.start()
            records.append({
                "year": year,
                "kind": kind,
                "task_id": task.id,
                "description": description,
                "asset_count": len(ids),
            })
            print(year, kind, task.id, flush=True)

    path = root / "outputs" / "logs" / "source_sampling_audit_recovery_tasks.json"
    prior = []
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
    by_description = {item["description"]: item for item in prior + records}
    path.write_text(
        json.dumps(list(by_description.values()), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
