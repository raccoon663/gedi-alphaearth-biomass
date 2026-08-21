"""Restart-safe exact-year central-pixel PALSAR extraction for frozen source shots."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from utilities.radar_extractors import Palsar2FeatureExtractor  # noqa: E402

SOURCE_META = ["shot_number", "year", "spatial_block_id", "spatial_fold"]
TARGET_META = ["shot_number", "year"]
BANDS = ["palsar_hh_db", "palsar_hv_db", "palsar_hh_minus_hv_db", "palsar_rfdi",
         "palsar_angle", "palsar_epoch", "palsar_qa"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path: Path, expected: pd.DataFrame, metadata: list[str]) -> dict:
    data = pd.read_csv(path, dtype={"shot_number": "string", "spatial_block_id": "string"})
    if data.duplicated(["shot_number", "year"]).any():
        raise RuntimeError(f"Duplicate shot/year in {path}")
    if not set(metadata + BANDS) <= set(data):
        raise RuntimeError(f"Unexpected PALSAR schema in {path}")
    allowed = set(zip(expected.shot_number.astype(str), expected.year.astype(int)))
    found = set(zip(data.shot_number.astype(str), data.year.astype(int)))
    if not found <= allowed:
        raise RuntimeError(f"PALSAR output contains shots outside frozen manifest: {path}")
    return {"rows": len(data), "sha256": sha256(path), "bytes": path.stat().st_size}


def download(root: Path, manifest: Path, availability: Path, out_dir: Path,
             ledger_path: Path, chunk_size: int, retries: int, project: str | None) -> None:
    import ee
    ee.Initialize(project=project) if project else ee.Initialize()
    freeze = json.loads(availability.read_text())
    years = {int(x) for x in freeze["palsar_common_years"]}
    if manifest.suffix.lower() == ".parquet":
        source = pd.read_parquet(manifest)
        source["shot_number"] = source.shot_number.astype("string")
    else:
        source = pd.read_csv(manifest, dtype={"shot_number": "string", "spatial_block_id": "string"})
    metadata = SOURCE_META if "spatial_block_id" in source else TARGET_META
    prefix = "source" if metadata == SOURCE_META else "kaihua"
    missing_metadata = set(metadata) - set(source)
    if missing_metadata:
        raise RuntimeError(f"Manifest missing frozen metadata: {sorted(missing_metadata)}")
    sort_columns = (["year", "spatial_block_id", "shot_number"]
                    if metadata == SOURCE_META else ["year", "shot_number"])
    source = source[source.year.isin(years)].sort_values(sort_columns)
    if source.duplicated(["shot_number", "year"]).any():
        raise RuntimeError("Frozen source manifest has duplicate shot/year")
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "dataset": Palsar2FeatureExtractor.spec.dataset, "input_manifest_sha256": sha256(manifest),
        "availability_freeze_sha256": sha256(availability), "chunks": {}}
    if ledger["input_manifest_sha256"] != sha256(manifest):
        raise RuntimeError("Restart ledger belongs to a different frozen manifest")
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = Palsar2FeatureExtractor()
    for year, yearly in source.groupby("year", sort=True):
        image = extractor.image_for_year(int(year))
        for start in range(0, len(yearly), chunk_size):
            chunk = yearly.iloc[start:start+chunk_size]
            key = f"{int(year)}_{start:07d}"
            out = out_dir / f"{prefix}_palsar_{key}.csv"
            if out.exists() and key in ledger["chunks"]:
                audit = validate(out, chunk, metadata)
                if audit["sha256"] != ledger["chunks"][key]["sha256"]:
                    raise RuntimeError(f"Restart hash mismatch: {out}")
                continue
            features = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                                   {c: getattr(r, c) for c in metadata})
                        for r in chunk.itertuples(index=False)]
            sampled = image.sampleRegions(ee.FeatureCollection(features), properties=metadata,
                                          scale=extractor.spec.scale_m, geometries=False,
                                          tileScale=4)
            url = sampled.getDownloadURL(filetype="CSV", selectors=metadata+BANDS)
            for attempt in range(retries):
                response = requests.get(url, timeout=180)
                if response.ok:
                    out.write_bytes(response.content)
                    break
                if attempt + 1 == retries:
                    response.raise_for_status()
                time.sleep(2 ** attempt)
            audit = validate(out, chunk, metadata)
            ledger["chunks"][key] = audit
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
            print(f"completed {key}: {audit['rows']} rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/manifests/frozen_source_manifest_200.csv")
    parser.add_argument("--availability", type=Path, default=ROOT / "outputs/manifests/palsar_availability_freeze.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/source_palsar")
    parser.add_argument("--ledger", type=Path, default=ROOT / "outputs/logs/source_palsar_download_status.json")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--project")
    args = parser.parse_args()
    download(ROOT, args.manifest, args.availability, args.output_dir, args.ledger,
             args.chunk_size, args.retries, args.project)


if __name__ == "__main__":
    main()
