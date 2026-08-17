"""Extract Copernicus GLO-30 terrain variables for the frozen cap-200 manifest.

The exact frozen points are embedded in small deterministic batches. This
avoids rescanning GEDI and avoids requiring a Cloud Storage table upload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import ee
import pandas as pd

DRIVE_FOLDER = "GEDI_AlphaEarth_Biomass_Transfer"
DEFAULT_CHUNK_SIZE = 1000


def terrain_image() -> ee.Image:
    elevation = (ee.ImageCollection("COPERNICUS/DEM/GLO30")
                 .select("DEM").mosaic().rename("elevation"))
    slope = ee.Terrain.slope(elevation).rename("slope")
    aspect_radians = ee.Terrain.aspect(elevation).multiply(math.pi).divide(180)
    aspect_sin = aspect_radians.sin().rename("aspect_sin")
    aspect_cos = aspect_radians.cos().rename("aspect_cos")
    return ee.Image.cat([elevation, slope, aspect_sin, aspect_cos]).toFloat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.chunk_size > 5000:
        raise SystemExit("chunk-size must be between 1 and 5000")

    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "outputs" / "manifests" / "frozen_source_manifest_200.csv"
    freeze_path = root / "outputs" / "manifests" / "frozen_source_manifest_200.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != freeze["manifest_csv_sha256"]:
        raise RuntimeError("Frozen manifest checksum mismatch; refusing DEM extraction")
    manifest = pd.read_csv(
        manifest_path,
        dtype={"shot_number": "string", "spatial_block_id": "string"},
        usecols=["shot_number", "lon", "lat", "year", "spatial_block_id",
                 "spatial_fold"],
    ).sort_values(["year", "spatial_block_id", "shot_number"]).reset_index(drop=True)
    if len(manifest) != freeze["sample_n"]:
        raise RuntimeError("Frozen manifest row count mismatch")

    ee.Initialize()
    log = root / "outputs" / "logs" / "source_dem_export_tasks.json"
    prior = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    existing = {item["description"]: item for item in prior}
    image = terrain_image()
    records = []
    for year, yearly in manifest.groupby("year", sort=True):
        yearly = yearly.reset_index(drop=True)
        for start in range(0, len(yearly), args.chunk_size):
            chunk = yearly.iloc[start:start + args.chunk_size]
            chunk_index = start // args.chunk_size
            description = f"source_dem_y{int(year)}_c{chunk_index:03d}"
            features = [
                ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]), {
                    "shot_number": str(row.shot_number),
                    "year": int(row.year),
                    "spatial_block_id": str(row.spatial_block_id),
                    "spatial_fold": int(row.spatial_fold),
                })
                for row in chunk.itertuples(index=False)
            ]
            sampled = image.sampleRegions(
                collection=ee.FeatureCollection(features),
                properties=["shot_number", "year", "spatial_block_id", "spatial_fold"],
                scale=30,
                geometries=False,
                tileScale=4,
            )
            prior_record = existing.get(description)
            if prior_record is None:
                task = ee.batch.Export.table.toDrive(
                    collection=sampled,
                    description=description,
                    folder=DRIVE_FOLDER,
                    fileNamePrefix=description,
                    fileFormat="CSV",
                    selectors=["shot_number", "year", "spatial_block_id", "spatial_fold",
                               "elevation", "slope", "aspect_sin", "aspect_cos"],
                )
                task.start()
                task_id = task.id
            else:
                task_id = prior_record["task_id"]
            record = {
                "year": int(year), "chunk": chunk_index,
                "expected_n": len(chunk), "task_id": task_id,
                "description": description,
                "manifest_csv_sha256": digest,
                "chunk_size": args.chunk_size,
            }
            records.append(record)
            # Persist after every task so an interruption is exactly resumable.
            log.write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(description, len(chunk), task_id, flush=True)

    log.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"submitted_or_reused={len(records)} expected_n={sum(r['expected_n'] for r in records)}")


if __name__ == "__main__":
    main()
