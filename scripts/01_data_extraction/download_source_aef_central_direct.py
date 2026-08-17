"""Download frozen full-source central AEF chunks directly from Earth Engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import requests

BANDS = [f"A{i:02d}" for i in range(64)]
META = ["shot_number", "year", "spatial_block_id", "spatial_fold", "lon", "lat"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path: Path, expected: pd.DataFrame) -> dict:
    frame = pd.read_csv(path, dtype={"shot_number": "string", "spatial_block_id": "string"})
    required = META + BANDS
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing columns in {path.name}: {missing}")
    if len(frame) != len(expected):
        raise RuntimeError(f"Row mismatch {path.name}: {len(frame)} != {len(expected)}")
    if frame["shot_number"].duplicated().any():
        raise RuntimeError(f"Duplicate shot_number: {path.name}")
    if set(frame["shot_number"].astype(str)) != set(expected["shot_number"].astype(str)):
        raise RuntimeError(f"Shot set mismatch: {path.name}")
    values = frame[BANDS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.all(values == 0, axis=1).any():
        raise RuntimeError(f"Invalid embeddings: {path.name}")
    if frame[required].isna().any().any():
        raise RuntimeError(f"Missing required values: {path.name}")
    if set(frame["year"].astype(int)) != set(expected["year"].astype(int)):
        raise RuntimeError(f"Year mismatch: {path.name}")
    return {"n": len(frame), "sha256": sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--request-size", type=int, default=100)
    parser.add_argument("--max-new-chunks", type=int, default=None)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.chunk_size <= 1000:
        raise ValueError("chunk-size must be 1..1000")
    if not 1 <= args.request_size <= args.chunk_size:
        raise ValueError("request-size must be 1..chunk-size")
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "outputs" / "manifests" / "source_aef_central_input.csv"
    meta = json.loads((root / "outputs" / "manifests" /
                       "source_aef_central_input.json").read_text(encoding="utf-8"))
    if sha256(manifest_path) != meta["csv_sha256"]:
        raise RuntimeError("Frozen input checksum mismatch")
    data = pd.read_csv(manifest_path, dtype={"shot_number": "string",
                                             "spatial_block_id": "string"})
    data = data.sort_values(["year", "spatial_block_id", "shot_number"]).reset_index(drop=True)
    out = root / "data" / "raw" / "source_aef_central_chunks"
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "outputs" / "logs" / "source_aef_central_direct_downloads.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    existing = {x["description"]: x for x in ledger}
    records, new_count = [], 0
    ee.Initialize()
    for year, yearly in data.groupby("year", sort=True):
        yearly = yearly.reset_index(drop=True)
        image = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
                 .filterDate(f"{int(year)}-01-01", f"{int(year)+1}-01-01")
                 .select(BANDS).mosaic().toFloat())
        for start in range(0, len(yearly), args.chunk_size):
            chunk = yearly.iloc[start:start + args.chunk_size]
            index = start // args.chunk_size
            description = f"source_aef_central_y{int(year)}_c{index:03d}"
            path = out / f"{description}.csv"
            record = existing.get(description)
            if path.exists():
                audit = validate(path, chunk)
                if record and audit["sha256"] != record.get("sha256"):
                    raise RuntimeError(f"Existing chunk hash changed: {description}")
            else:
                if args.max_new_chunks is not None and new_count >= args.max_new_chunks:
                    ledger_path.write_text(json.dumps(records + [x for x in ledger if x["description"] not in {r["description"] for r in records}], indent=2), encoding="utf-8")
                    print(f"new-chunk limit reached: {new_count}")
                    return
                subframes = []
                for sub_start in range(0, len(chunk), args.request_size):
                    subchunk = chunk.iloc[sub_start:sub_start + args.request_size]
                    points = ee.FeatureCollection([
                        ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]), {
                            "shot_number": str(row.shot_number), "year": int(row.year),
                            "spatial_block_id": str(row.spatial_block_id),
                            "spatial_fold": int(row.spatial_fold), "lon": float(row.lon),
                            "lat": float(row.lat),
                        }) for row in subchunk.itertuples(index=False)
                    ])
                    sampled = image.sampleRegions(
                        collection=points, properties=META, scale=10,
                        geometries=False, tileScale=4)
                    for attempt in range(1, args.retries + 1):
                        try:
                            url = sampled.getDownloadURL(filetype="CSV", selectors=META + BANDS)
                            response = requests.get(url, timeout=180)
                            response.raise_for_status()
                            subpath = path.with_name(f"{path.stem}_s{sub_start // args.request_size:02d}.part.csv")
                            subpath.write_bytes(response.content)
                            subframes.append(pd.read_csv(
                                subpath, dtype={"shot_number": "string", "spatial_block_id": "string"}))
                            subpath.unlink()
                            break
                        except Exception:
                            if attempt == args.retries:
                                raise
                            time.sleep(min(60, 5 * 2 ** (attempt - 1)))
                temporary = path.with_suffix(".csv.part")
                pd.concat(subframes, ignore_index=True).to_csv(temporary, index=False)
                temporary.replace(path)
                audit = validate(path, chunk)
                new_count += 1
                print(description, audit, flush=True)
            record = {"description": description, "year": int(year), "chunk": index,
                      "expected_n": len(chunk), "chunk_size": args.chunk_size,
                      "request_size": args.request_size,
                      "input_manifest_sha256": meta["csv_sha256"], **audit}
            records.append(record)
            ledger_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"complete chunks={len(records)} new={new_count} rows={sum(x['n'] for x in records)}")


if __name__ == "__main__":
    main()
