"""Download frozen annual-median S1/S2 features directly from Earth Engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import ee
import pandas as pd
import requests

S2 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
IDX = ["NDVI", "EVI", "NDMI", "NBR", "NDRE"]
S1 = ["VV", "VH", "VV_minus_VH"]
FEATURES = S1 + S2 + IDX
META = ["shot_number", "year", "spatial_block_id", "spatial_fold"]
MISSING = -9999.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_s2(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    valid = (scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9))
             .And(scl.neq(10)).And(scl.neq(11)))
    qa = image.select("QA60")
    valid = valid.And(qa.bitwiseAnd(1 << 10).eq(0)).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(valid).select(S2).multiply(0.0001).toFloat()


def yearly_image(year: int) -> ee.Image:
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
          .map(mask_s2).median().select(S2))
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = s2.expression("2.5*(nir-red)/(nir+6*red-7.5*blue+1)", {
        "nir": s2.select("B8"), "red": s2.select("B4"), "blue": s2.select("B2")}).rename("EVI")
    ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")
    nbr = s2.normalizedDifference(["B8", "B12"]).rename("NBR")
    ndre = s2.normalizedDifference(["B8A", "B5"]).rename("NDRE")
    s1c = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .select(["VV", "VH"]).median().toFloat())
    diff = s1c.select("VV").subtract(s1c.select("VH")).rename("VV_minus_VH")
    return s1c.addBands(diff).addBands(s2).addBands([ndvi, evi, ndmi, nbr, ndre]).select(FEATURES).unmask(MISSING)


def validate(path: Path, expected: pd.DataFrame) -> dict:
    frame = pd.read_csv(path, dtype={"shot_number": "string", "spatial_block_id": "string"})
    required = META + FEATURES
    if sorted(set(required) - set(frame.columns)):
        raise RuntimeError(f"Missing columns: {path.name}")
    if len(frame) != len(expected) or frame["shot_number"].duplicated().any():
        raise RuntimeError(f"Row/uniqueness failure: {path.name}")
    if set(frame["shot_number"].astype(str)) != set(expected["shot_number"].astype(str)):
        raise RuntimeError(f"Shot-set failure: {path.name}")
    return {"n": len(frame), "sha256": sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--request-size", type=int, default=100)
    parser.add_argument("--max-new-chunks", type=int, default=None)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--year-start", type=int, default=2019)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument("--ledger-suffix", default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "outputs" / "manifests" / "source_aef_central_input.csv"
    meta = json.loads((root / "outputs" / "manifests" / "source_aef_central_input.json").read_text())
    if sha256(manifest_path) != meta["csv_sha256"]:
        raise RuntimeError("Frozen common input checksum mismatch")
    data = pd.read_csv(manifest_path, dtype={"shot_number": "string", "spatial_block_id": "string"})
    data = data.sort_values(["year", "spatial_block_id", "shot_number"]).reset_index(drop=True)
    data = data.loc[data["year"].between(args.year_start, args.year_end)].copy()
    if data.empty:
        raise RuntimeError("Selected year range is empty")
    out = root / "data" / "raw" / "source_conventional_chunks"
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.ledger_suffix}" if args.ledger_suffix else ""
    ledger_path = root / "outputs" / "logs" / f"source_conventional_direct_downloads{suffix}.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    existing = {x["description"]: x for x in ledger}
    records, new_count = [], 0
    ee.Initialize()
    for year, yearly in data.groupby("year", sort=True):
        yearly = yearly.reset_index(drop=True)
        image = yearly_image(int(year))
        for start in range(0, len(yearly), args.chunk_size):
            chunk = yearly.iloc[start:start + args.chunk_size]
            desc = f"source_conventional_y{int(year)}_c{start // args.chunk_size:03d}"
            path = out / f"{desc}.csv"
            prior = existing.get(desc)
            if path.exists():
                audit = validate(path, chunk)
                if prior and prior.get("sha256") != audit["sha256"]:
                    raise RuntimeError(f"Existing hash changed: {desc}")
            else:
                if args.max_new_chunks is not None and new_count >= args.max_new_chunks:
                    ledger_path.write_text(json.dumps(records + [x for x in ledger if x["description"] not in {r["description"] for r in records}], indent=2))
                    print(f"new-chunk limit reached: {new_count}")
                    return
                parts = []
                for sub_start in range(0, len(chunk), args.request_size):
                    sub = chunk.iloc[sub_start:sub_start + args.request_size]
                    points = ee.FeatureCollection([ee.Feature(
                        ee.Geometry.Point([float(r.lon), float(r.lat)]),
                        {"shot_number": str(r.shot_number), "year": int(r.year),
                         "spatial_block_id": str(r.spatial_block_id), "spatial_fold": int(r.spatial_fold)})
                        for r in sub.itertuples(index=False)])
                    sampled = image.sampleRegions(collection=points, properties=META, scale=10,
                                                  geometries=False, tileScale=4)
                    for attempt in range(1, args.retries + 1):
                        try:
                            url = sampled.getDownloadURL(filetype="CSV", selectors=META + FEATURES)
                            response = requests.get(url, timeout=240)
                            response.raise_for_status()
                            part = path.with_name(f"{path.stem}_s{sub_start // args.request_size:02d}.part.csv")
                            part.write_bytes(response.content)
                            parts.append(pd.read_csv(part, dtype={"shot_number": "string", "spatial_block_id": "string"}))
                            part.unlink()
                            break
                        except Exception:
                            if attempt == args.retries:
                                raise
                            time.sleep(min(60, 5 * 2 ** (attempt - 1)))
                temporary = path.with_suffix(".csv.part")
                pd.concat(parts, ignore_index=True).to_csv(temporary, index=False)
                temporary.replace(path)
                audit = validate(path, chunk)
                new_count += 1
                print(desc, audit, flush=True)
            record = {"description": desc, "year": int(year), "chunk": start // args.chunk_size,
                      "expected_n": len(chunk), "chunk_size": args.chunk_size,
                      "request_size": args.request_size, "input_manifest_sha256": meta["csv_sha256"], **audit}
            records.append(record)
            ledger_path.write_text(json.dumps(records, indent=2))
    print(f"complete chunks={len(records)} new={new_count} rows={sum(x['n'] for x in records)}")


if __name__ == "__main__":
    main()
