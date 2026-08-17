"""Submit symmetric central and 25 m AEF development extraction chunks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import ee
import pandas as pd

DRIVE_FOLDER = "GEDI_AlphaEarth_Biomass_Transfer"
BANDS = [f"A{i:02d}" for i in range(64)]


def annual_images(year: int) -> dict[str, ee.Image]:
    central = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
               .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
               .select(BANDS).mosaic().toFloat())
    mean25 = central.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.circle(radius=25, units="meters", normalize=False),
        skipMasked=True,
    ).rename(BANDS)
    norm = mean25.pow(2).reduce(ee.Reducer.sum()).sqrt()
    aggregate = mean25.divide(norm).rename(BANDS).toFloat()
    return {"central": central, "aggregate25": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "outputs" / "manifests" / "aef_development_common_manifest.csv"
    freeze = json.loads((root / "outputs" / "manifests" /
                         "aef_development_common_manifest.json").read_text())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != freeze["csv_sha256"]:
        raise RuntimeError("AEF development common manifest checksum mismatch")
    manifest = pd.read_csv(
        manifest_path, dtype={"shot_number": "string", "spatial_block_id": "string"},
        usecols=["shot_number", "lon", "lat", "year", "spatial_block_id", "spatial_fold"],
    ).sort_values(["year", "spatial_block_id", "shot_number"]).reset_index(drop=True)
    log = root / "outputs" / "logs" / "aef_development_export_tasks.json"
    prior = json.loads(log.read_text()) if log.exists() else []
    existing = {item["description"]: item for item in prior}
    records = []
    ee.Initialize()
    selectors = ["shot_number", "year", "spatial_block_id", "spatial_fold", *BANDS]
    for year, yearly in manifest.groupby("year", sort=True):
        yearly = yearly.reset_index(drop=True)
        images = annual_images(int(year))
        for start in range(0, len(yearly), args.chunk_size):
            chunk = yearly.iloc[start:start + args.chunk_size]
            chunk_index = start // args.chunk_size
            points = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]), {
                    "shot_number": str(row.shot_number), "year": int(row.year),
                    "spatial_block_id": str(row.spatial_block_id),
                    "spatial_fold": int(row.spatial_fold),
                }) for row in chunk.itertuples(index=False)
            ])
            for method, image in images.items():
                description = f"aefdev_{method}_y{int(year)}_c{chunk_index:03d}"
                prior_record = existing.get(description)
                if prior_record is None:
                    sampled = image.sampleRegions(
                        collection=points,
                        properties=["shot_number", "year", "spatial_block_id", "spatial_fold"],
                        scale=10, geometries=False, tileScale=4)
                    task = ee.batch.Export.table.toDrive(
                        collection=sampled, description=description,
                        folder=DRIVE_FOLDER, fileNamePrefix=description,
                        fileFormat="CSV", selectors=selectors)
                    task.start()
                    task_id = task.id
                else:
                    task_id = prior_record["task_id"]
                records.append({
                    "year": int(year), "chunk": chunk_index, "method": method,
                    "expected_n": len(chunk), "chunk_size": args.chunk_size,
                    "task_id": task_id, "description": description,
                    "development_manifest_sha256": digest,
                })
                log.write_text(json.dumps(records, indent=2), encoding="utf-8")
                print(description, len(chunk), task_id, flush=True)
    print(f"submitted_or_reused={len(records)} expected_method_rows="
          f"{sum(item['expected_n'] for item in records)}")


if __name__ == "__main__":
    main()
