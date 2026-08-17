"""Build Kaihua predictors while the GEDI L4A target label lock is active.

The GEDI collection is filtered server-side with the already frozen source QA
policy.  Only footprint metadata are selected for download; AGBD and AGBD_SE
are never materialized by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import ee
import joblib
import numpy as np
import pandas as pd
import requests
from shapely import intersects_xy
from shapely.geometry import shape

YEARS = range(2019, 2025)
GEDI_MONTHLY = "LARSE/GEDI/GEDI04_A_002_MONTHLY"
AEF_BANDS = [f"A{i:02d}" for i in range(64)]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
INDICES = ["NDVI", "EVI", "NDMI", "NBR", "NDRE"]
S1_BANDS = ["VV", "VH", "VV_minus_VH"]
CONVENTIONAL = S1_BANDS + S2_BANDS + INDICES
DEM_BANDS = ["elevation", "slope", "aspect_sin", "aspect_cos"]
META = ["shot_number", "year", "lon", "lat"]
MISSING = -9999.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_csv(collection: ee.FeatureCollection, selectors: list[str], retries: int = 6) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            url = collection.getDownloadURL(filetype="CSV", selectors=selectors)
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            from io import BytesIO
            return pd.read_csv(BytesIO(response.content), dtype={"shot_number": "string"})
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    raise AssertionError("unreachable")


def balanced_merge(collections: list[ee.FeatureCollection]) -> ee.FeatureCollection:
    if not collections:
        return ee.FeatureCollection([])
    current = collections
    while len(current) > 1:
        current = [current[i] if i + 1 == len(current) else current[i].merge(current[i + 1])
                   for i in range(0, len(current), 2)]
    return current[0]


def gedi_assets(aoi: ee.Geometry, year: int, month: int) -> list[str]:
    # Kaihua (118.02--118.63 E, 28.91--29.50 N) is wholly contained in
    # the fixed GEDI monthly catalogue tile 114E_024N. Direct tile access
    # avoids an expensive global ImageCollection spatial catalogue scan.
    index = f"{year}{month:02d}_114E_024N"
    collection = ee.ImageCollection(GEDI_MONTHLY).filter(ee.Filter.eq("system:index", index))
    if collection.size().getInfo() == 0:
        return []
    nested = collection.aggregate_array("table_asset_ids").getInfo()
    return sorted({item for group in nested for item in (group or [])})


def source_qa(raw: ee.FeatureCollection) -> ee.FeatureCollection:
    # Exact frozen source QA policy. Labels are used only in server-side validity
    # predicates and are not selected, downloaded, logged, or summarized.
    return (raw.filter(ee.Filter.eq("algorithm_run_flag", 1))
            .filter(ee.Filter.notNull(["shot_number", "agbd", "agbd_se"]))
            .filter(ee.Filter.eq("l4_quality_flag", 1))
            .filter(ee.Filter.eq("degrade_flag", 0))
            .filter(ee.Filter.gte("agbd", 0))
            .filter(ee.Filter.gte("agbd_se", 0)))


def terrain_image() -> ee.Image:
    elevation = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().rename("elevation")
    slope = ee.Terrain.slope(elevation).rename("slope")
    radians = ee.Terrain.aspect(elevation).multiply(math.pi).divide(180)
    return ee.Image.cat([elevation, slope, radians.sin().rename("aspect_sin"),
                         radians.cos().rename("aspect_cos")]).toFloat()


def mask_s2(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    valid = (scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9))
             .And(scl.neq(10)).And(scl.neq(11)))
    qa = image.select("QA60")
    valid = valid.And(qa.bitwiseAnd(1 << 10).eq(0)).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(valid).select(S2_BANDS).multiply(0.0001).toFloat()


def conventional_image(year: int) -> ee.Image:
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(f"{year}-01-01", f"{year + 1}-01-01").map(mask_s2)
          .median().select(S2_BANDS))
    derived = [
        s2.normalizedDifference(["B8", "B4"]).rename("NDVI"),
        s2.expression("2.5*(nir-red)/(nir+6*red-7.5*blue+1)", {
            "nir": s2.select("B8"), "red": s2.select("B4"), "blue": s2.select("B2")}).rename("EVI"),
        s2.normalizedDifference(["B8", "B11"]).rename("NDMI"),
        s2.normalizedDifference(["B8", "B12"]).rename("NBR"),
        s2.normalizedDifference(["B8A", "B5"]).rename("NDRE"),
    ]
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]).median().toFloat())
    diff = s1.select("VV").subtract(s1.select("VH")).rename("VV_minus_VH")
    return s1.addBands(diff).addBands(s2).addBands(derived).select(CONVENTIONAL).unmask(MISSING)


def points(frame: pd.DataFrame) -> ee.FeatureCollection:
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {
            "shot_number": str(r.shot_number), "year": int(r.year),
            "lon": float(r.lon), "lat": float(r.lat)})
        for r in frame.itertuples(index=False)
    ])


def download_chunks(root: Path, shots: pd.DataFrame, kind: str, bands: list[str],
                    image_fn, scale: int, chunk_size: int, request_size: int,
                    ledger_suffix: str = "") -> None:
    out = root / "data" / "raw" / f"kaihua_{kind}_locked_chunks"
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{ledger_suffix}" if ledger_suffix else ""
    ledger_path = root / "outputs" / "logs" / f"kaihua_{kind}_locked_downloads{suffix}.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    prior = {x["description"]: x for x in ledger}
    records = []
    for year, yearly in shots.groupby("year", sort=True):
        yearly = yearly.reset_index(drop=True)
        image = image_fn(int(year)) if kind != "dem" else image_fn()
        for start in range(0, len(yearly), chunk_size):
            chunk = yearly.iloc[start:start + chunk_size]
            desc = f"kaihua_{kind}_locked_y{int(year)}_c{start // chunk_size:03d}"
            path = out / f"{desc}.csv"
            if path.exists():
                frame = pd.read_csv(path, dtype={"shot_number": "string"})
                if len(frame) != len(chunk) or set(frame.shot_number) != set(chunk.shot_number):
                    raise RuntimeError(f"Existing chunk mismatch: {path}")
                digest = sha256(path)
                if desc in prior and digest != prior[desc]["sha256"]:
                    raise RuntimeError(f"Existing chunk hash changed: {path}")
            else:
                parts = []
                for sub_start in range(0, len(chunk), request_size):
                    sub = chunk.iloc[sub_start:sub_start + request_size]
                    sampled = image.sampleRegions(collection=points(sub), properties=META,
                                                  scale=scale, geometries=False, tileScale=4)
                    parts.append(get_csv(sampled, META + bands))
                frame = pd.concat(parts, ignore_index=True)
                if len(frame) != len(chunk) or set(frame.shot_number) != set(chunk.shot_number):
                    raise RuntimeError(f"Downloaded row/set mismatch: {desc}")
                temporary = path.with_suffix(".csv.part")
                frame.to_csv(temporary, index=False)
                temporary.replace(path)
                digest = sha256(path)
                print(desc, len(frame), flush=True)
            records.append({"description": desc, "year": int(year), "expected_n": len(chunk),
                            "n": len(frame), "sha256": digest, "bytes": path.stat().st_size,
                            "label_locked": True})
            ledger_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--request-size", type=int, default=100)
    parser.add_argument("--year-start", type=int, default=2019)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument("--kinds", default="dem,aef,conventional")
    parser.add_argument("--ledger-suffix", default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "outputs" / "manifests" / "frozen_source_model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("target_label_locked"):
        raise RuntimeError("TARGET_LABEL_LOCKED is not true")
    verification = {"verified_at": datetime.now().astimezone().isoformat(),
                    "SOURCE_MODEL_FREEZE_VERIFIED": True, "TARGET_LABEL_LOCKED": True}
    for branch in ["alphaearth", "conventional"]:
        spec = manifest["selected_models"][branch]
        model_path = root / spec["model_file"]
        actual = sha256(model_path)
        if actual != spec["model_sha256"]:
            raise RuntimeError(f"HARD STOP: {branch} model checksum mismatch")
        model = joblib.load(model_path)
        names = list(model.feature_names_in_)
        if names != manifest["features"][branch]:
            raise RuntimeError(f"HARD STOP: {branch} model feature order mismatch")
        verification[f"{branch}_model_sha256"] = actual

    aoi_path = root / "data" / "boundaries" / "kaihua_target_aoi.geojson"
    aoi_hash_record = (root / "outputs" / "manifests" / "kaihua_target_aoi.sha256").read_text().split()[0]
    geojson = json.loads(aoi_path.read_text(encoding="utf-8"))
    feature = geojson["features"][0]
    geom = shape(feature["geometry"])
    if sha256(aoi_path) != aoi_hash_record or not geom.is_valid:
        raise RuntimeError("HARD STOP: target AOI hash/geometry mismatch")
    if str(feature["properties"].get("admin_code")) != "330824":
        raise RuntimeError("HARD STOP: target AOI administrative code mismatch")
    verification.update({"target_aoi_sha256": aoi_hash_record, "target_aoi_valid": True,
                         "target_admin_code": "330824", "target_crs": "OGC:CRS84"})
    verify_path = root / "outputs" / "manifests" / "source_model_freeze_verification.json"
    verify_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")

    ee.Initialize()
    aoi = ee.Geometry(feature["geometry"])
    query_aoi = ee.Geometry.Rectangle(list(geom.bounds), proj="EPSG:4326", geodesic=False)
    manifest_dir = root / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    shot_path = manifest_dir / "kaihua_target_shots_locked.csv"
    count_path = root / "outputs" / "tables" / "kaihua_target_shot_counts_locked.csv"
    count_path.parent.mkdir(parents=True, exist_ok=True)
    if not shot_path.exists():
        all_years, counts = [], []
        for year in YEARS:
            annual_frames, annual_ids = [], set()
            raw_n = valid_n = 0
            for month in range(1, 13):
                ids = gedi_assets(aoi, year, month)
                annual_ids.update(ids)
                if not ids:
                    continue
                raw = balanced_merge([ee.FeatureCollection(asset).filterBounds(query_aoi) for asset in ids])
                bbox_raw_n = raw.size().getInfo()
                if bbox_raw_n == 0:
                    print("GEDI", year, f"{month:02d}", 0, 0, flush=True)
                    continue
                raw_meta = raw.map(lambda f: ee.Feature(f).set({
                    "lon": ee.Feature(f).geometry().coordinates().get(0),
                    "lat": ee.Feature(f).geometry().coordinates().get(1), "year": year}))
                valid = source_qa(raw).map(lambda f: ee.Feature(f).set({
                    "lon": ee.Feature(f).geometry().coordinates().get(0),
                    "lat": ee.Feature(f).geometry().coordinates().get(1), "year": year}))
                bbox_valid_n = valid.size().getInfo()
                raw_frame = get_csv(raw_meta.select(META), META)
                frame = (get_csv(valid.select(META), META) if bbox_valid_n
                         else pd.DataFrame(columns=META))
                raw_frame = raw_frame.loc[intersects_xy(
                    geom, raw_frame["lon"].to_numpy(float), raw_frame["lat"].to_numpy(float))].copy()
                frame = frame.loc[intersects_xy(
                    geom, frame["lon"].to_numpy(float), frame["lat"].to_numpy(float))].copy()
                month_raw, month_valid = len(raw_frame), len(frame)
                raw_n += month_raw
                valid_n += month_valid
                annual_frames.append(frame[META])
                print("GEDI", year, f"{month:02d}", month_raw, month_valid, flush=True)
            frame = pd.concat(annual_frames, ignore_index=True) if annual_frames else pd.DataFrame(columns=META)
            all_years.append(frame)
            counts.append({"year": year, "asset_n": len(annual_ids), "raw_intersecting_n": raw_n,
                           "qa_valid_n": valid_n, "labels_downloaded": False})
            print("GEDI ANNUAL", year, raw_n, valid_n, flush=True)
        shots = pd.concat(all_years, ignore_index=True)
        shots["shot_number"] = shots["shot_number"].astype("string")
        shots = shots.sort_values(["year", "shot_number"]).reset_index(drop=True)
        if shots.shot_number.duplicated().any() or shots[META].isna().any().any():
            raise RuntimeError("Target shot manifest duplicate/missing metadata")
        shots.to_csv(shot_path, index=False)
        pd.DataFrame(counts).to_csv(count_path, index=False)
    else:
        shots = pd.read_csv(shot_path, dtype={"shot_number": "string"})
        if sorted(shots.columns) != sorted(META):
            raise RuntimeError("Locked target shot manifest contains unexpected fields")
    shots = shots.loc[shots.year.between(args.year_start, args.year_end)].copy()
    kinds = {x.strip() for x in args.kinds.split(",") if x.strip()}
    if not kinds <= {"dem", "aef", "conventional"} or not len(shots):
        raise ValueError("Invalid kinds or empty year range")
    if "dem" in kinds:
        download_chunks(root, shots, "dem", DEM_BANDS, terrain_image, 30,
                        args.chunk_size, args.request_size, args.ledger_suffix)
    if "aef" in kinds:
        download_chunks(root, shots, "aef", AEF_BANDS,
                        lambda y: (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
                                   .filterDate(f"{y}-01-01", f"{y + 1}-01-01")
                                   .select(AEF_BANDS).mosaic().toFloat()),
                        10, args.chunk_size, args.request_size, args.ledger_suffix)
    if "conventional" in kinds:
        download_chunks(root, shots, "conventional", CONVENTIONAL, conventional_image, 10,
                        args.chunk_size, args.request_size, args.ledger_suffix)
    print(f"LOCKED PREDICTOR DOWNLOAD COMPLETE: target_shots={len(shots)}")


if __name__ == "__main__":
    main()
