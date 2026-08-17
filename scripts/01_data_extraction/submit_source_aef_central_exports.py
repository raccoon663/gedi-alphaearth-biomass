"""Submit deterministic, resumable full-source central AEF chunks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import ee
import pandas as pd

DRIVE_FOLDER = "GEDI_AlphaEarth_Biomass_Transfer"
BANDS = [f"A{i:02d}" for i in range(64)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annual_image(year: int) -> ee.Image:
    collection = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
                  .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
                  .select(BANDS))
    return collection.mosaic().toFloat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--max-new-tasks", type=int, default=None)
    args = parser.parse_args()
    if not 1 <= args.chunk_size <= 1000:
        raise ValueError("chunk-size must be within the frozen 500–1000 range")
    root = Path(__file__).resolve().parents[2]
    path = root / "outputs" / "manifests" / "source_aef_central_input.csv"
    meta = json.loads((root / "outputs" / "manifests" /
                       "source_aef_central_input.json").read_text(encoding="utf-8"))
    freeze = json.loads((root / "outputs" / "manifests" /
                         "aef_method_freeze.json").read_text(encoding="utf-8"))
    if freeze["selected_method"] != "central_pixel_10m":
        raise RuntimeError("Central method is not frozen")
    digest = sha256(path)
    if digest != meta["csv_sha256"]:
        raise RuntimeError("AEF input checksum mismatch")
    data = pd.read_csv(path, dtype={"shot_number": "string", "spatial_block_id": "string"})
    data = data.sort_values(["year", "spatial_block_id", "shot_number"]).reset_index(drop=True)
    log_path = root / "outputs" / "logs" / "source_aef_central_export_tasks.json"
    prior = json.loads(log_path.read_text()) if log_path.exists() else []
    existing = {item["description"]: item for item in prior}
    records, new_tasks = [], 0
    ee.Initialize()
    selectors = ["shot_number", "year", "spatial_block_id", "spatial_fold", "lon", "lat", *BANDS]
    for year, yearly in data.groupby("year", sort=True):
        yearly = yearly.reset_index(drop=True)
        image = annual_image(int(year))
        for start in range(0, len(yearly), args.chunk_size):
            chunk = yearly.iloc[start:start + args.chunk_size]
            index = start // args.chunk_size
            description = f"source_aef_central_y{int(year)}_c{index:03d}"
            record = existing.get(description)
            if record is None:
                if args.max_new_tasks is not None and new_tasks >= args.max_new_tasks:
                    print(f"new-task limit reached at {new_tasks}")
                    log_path.write_text(json.dumps(records + [r for r in prior if r["description"] not in {x["description"] for x in records}], indent=2), encoding="utf-8")
                    return
                points = ee.FeatureCollection([
                    ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]), {
                        "shot_number": str(row.shot_number), "year": int(row.year),
                        "spatial_block_id": str(row.spatial_block_id),
                        "spatial_fold": int(row.spatial_fold),
                        "lon": float(row.lon), "lat": float(row.lat),
                    }) for row in chunk.itertuples(index=False)
                ])
                sampled = image.sampleRegions(
                    collection=points,
                    properties=["shot_number", "year", "spatial_block_id", "spatial_fold", "lon", "lat"],
                    scale=10, geometries=False, tileScale=4)
                task = ee.batch.Export.table.toDrive(
                    collection=sampled, description=description, folder=DRIVE_FOLDER,
                    fileNamePrefix=description, fileFormat="CSV", selectors=selectors)
                task.start()
                record = {"year": int(year), "chunk": index, "expected_n": len(chunk),
                          "chunk_size": args.chunk_size, "task_id": task.id,
                          "description": description, "input_manifest_sha256": digest,
                          "method": "central_pixel", "scale_m": 10}
                new_tasks += 1
            records.append(record)
            log_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(description, len(chunk), record["task_id"], flush=True)
    print(f"tasks={len(records)} new_tasks={new_tasks} expected_rows={sum(r['expected_n'] for r in records)}")


if __name__ == "__main__":
    main()
