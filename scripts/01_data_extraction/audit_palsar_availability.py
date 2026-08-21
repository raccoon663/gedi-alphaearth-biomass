"""Audit exact-year PALSAR-2 availability before any paired model is trained.

The output contains aggregate coverage/QA statistics only.  Per-footprint shot IDs and
coordinates stay in ignored local inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from utilities.radar_extractors import Palsar2FeatureExtractor  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_points(ee, image, frame: pd.DataFrame, scale: int) -> pd.DataFrame:
    properties = [c for c in ["shot_number", "year", "spatial_block_id",
                              "target_block_id"] if c in frame]
    def native(value):
        return value.item() if hasattr(value, "item") else value
    features = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                           {c: native(getattr(r, c)) for c in properties})
                for r in frame.itertuples(index=False)]
    sampled = image.sampleRegions(collection=ee.FeatureCollection(features),
                                  properties=properties, scale=scale,
                                  geometries=False, tileScale=4)
    rows = sampled.getInfo().get("features", [])
    return pd.DataFrame([x["properties"] for x in rows])


def audit_domain(ee, extractor, name: str, manifest: Path, years: list[int],
                 chunk_size: int) -> tuple[list[dict], set[int]]:
    if manifest.suffix.lower() == ".parquet":
        source = pd.read_parquet(manifest)
        source["shot_number"] = source.shot_number.astype("string")
    else:
        source = pd.read_csv(manifest, dtype={"shot_number": "string",
                                             "spatial_block_id": "string",
                                             "target_block_id": "string"})
    required = {"shot_number", "year", "lon", "lat"}
    if not required <= set(source):
        raise ValueError(f"{manifest} lacks {sorted(required-set(source))}")
    if source.duplicated(["shot_number", "year"]).any():
        raise ValueError(f"Duplicate shot/year rows in {manifest}")
    records, valid_years = [], set()
    for year in years:
        subset = source[source.year == year].copy()
        collection = (ee.ImageCollection(extractor.spec.dataset)
                      .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))
        image_count = int(collection.size().getInfo())
        parts = []
        if image_count:
            image = extractor.raw_image_for_year(year)
            for start in range(0, len(subset), chunk_size):
                parts.append(sample_points(ee, image, subset.iloc[start:start+chunk_size],
                                           extractor.spec.scale_m))
        sampled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        qa = pd.to_numeric(sampled.get("qa"), errors="coerce")
        unexpected = extractor.unexpected_qa_values(qa.dropna().unique()) if len(sampled) else set()
        if unexpected:
            raise RuntimeError(
                f"Undocumented PALSAR QA classes for {name} {year}: {sorted(unexpected)}"
            )
        land = qa.isin(extractor.valid_qa_values) if len(sampled) else pd.Series(dtype=bool)
        valid = sampled[land &
                        (pd.to_numeric(sampled.get("HH"), errors="coerce") > 0) &
                        (pd.to_numeric(sampled.get("HV"), errors="coerce") > 0)] if len(sampled) else sampled
        if len(valid):
            valid_years.add(year)
        records.append({"scope": "year", "domain": name, "year": year,
                        "image_count": image_count, "frozen_sample_n": len(subset),
                        "sampled_n": len(sampled), "valid_hh_n": int((pd.to_numeric(sampled.get("HH"), errors="coerce") > 0).sum()) if len(sampled) else 0,
                        "valid_hv_n": int((pd.to_numeric(sampled.get("HV"), errors="coerce") > 0).sum()) if len(sampled) else 0,
                        "qa_land_n": int(land.sum()) if len(sampled) else 0,
                        "valid_palsar_n": len(valid),
                        "missing_fraction": float(1-len(valid)/len(subset)) if len(subset) else np.nan})
        block = "spatial_block_id" if name == "source" else "target_block_id"
        if block in subset:
            keys = valid[["shot_number", "year"]].drop_duplicates() if len(valid) else pd.DataFrame(columns=["shot_number", "year"])
            marked = subset.merge(keys.assign(received=True), on=["shot_number", "year"], how="left")
            for block_id, group in marked.groupby(block, dropna=False):
                records.append({"scope": "block", "domain": name, "year": year,
                                "block_id_hash": hashlib.sha256(str(block_id).encode()).hexdigest(),
                                "frozen_sample_n": len(group),
                                "valid_palsar_n": int(group.received.fillna(False).sum()),
                                "missing_fraction": float(1-group.received.fillna(False).mean())})
        if len(sampled) and "qa" in sampled:
            for qa, count in sampled.qa.value_counts(dropna=False).items():
                records.append({"scope": "qa_class", "domain": name, "year": year,
                                "qa_class": qa, "qa_count": int(count)})
        print(f"audited {name} {year}: {len(valid)}/{len(subset)} valid", flush=True)
    return records, valid_years


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-folds", type=Path,
                        help="Optional private CSV providing target_block_id by shot_number")
    parser.add_argument("--year-start", type=int, default=2019)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--project")
    args = parser.parse_args()
    import ee
    ee.Initialize(project=args.project) if args.project else ee.Initialize()
    extractor = Palsar2FeatureExtractor()
    years = list(range(args.year_start, args.year_end + 1))
    target_manifest = args.target_manifest
    if args.target_folds:
        target = pd.read_parquet(target_manifest) if target_manifest.suffix == ".parquet" else pd.read_csv(target_manifest)
        target["shot_number"] = target.shot_number.astype("string")
        folds = pd.read_csv(args.target_folds, dtype={"shot_number": "string",
                                                      "target_block_id": "string"})
        target = target.merge(folds[["shot_number", "target_block_id"]], on="shot_number",
                              how="left", validate="one_to_one")
        if target.target_block_id.isna().any():
            raise RuntimeError("Target block join incomplete")
        target_manifest = ROOT / "data/interim/kaihua_palsar_audit_private.parquet"
        target_manifest.parent.mkdir(parents=True, exist_ok=True)
        target.to_parquet(target_manifest, index=False)
    source_rows, source_years = audit_domain(ee, extractor, "source",
                                             args.source_manifest, years, args.chunk_size)
    target_rows, target_years = audit_domain(ee, extractor, "target",
                                             target_manifest, years, args.chunk_size)
    common = sorted(source_years & target_years)
    out = ROOT / "outputs/tables/audits/palsar_availability_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows + target_rows).to_csv(out, index=False)
    freeze = {"status": "PALSAR_AVAILABILITY_FROZEN", "frozen_at": datetime.now(timezone.utc).isoformat(),
              "dataset": extractor.spec.dataset, "year_match": "exact",
              "valid_qa_values": sorted(extractor.valid_qa_values),
              "qa_semantics_source": "JAXA PALSAR-2 ScanSAR/stripmap mosaic v2.4 product description",
              "nearest_year_substitution": False, "audited_years": years,
              "palsar_common_years": common,
              "source_manifest_sha256": sha256(args.source_manifest),
              "target_manifest_sha256": sha256(args.target_manifest),
              "target_fold_manifest_sha256": sha256(args.target_folds) if args.target_folds else None,
              "audit_csv_sha256": sha256(out), "audit_rows": len(source_rows)+len(target_rows),
              "private_sample_rows_committed": False}
    path = ROOT / "outputs/manifests/palsar_availability_freeze.json"
    path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
