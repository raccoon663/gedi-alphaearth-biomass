"""Finalize conventional features and build the frozen common representation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

S1 = ["VV", "VH", "VV_minus_VH"]
S2 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
IDX = ["NDVI", "EVI", "NDMI", "NBR", "NDRE"]
FEATURES = S1 + S2 + IDX
DEM = ["elevation", "slope", "aspect_sin", "aspect_cos"]
AEF = [f"A{i:02d}" for i in range(64)]
META = ["shot_number", "year", "spatial_block_id", "spatial_fold"]
MISSING = -9999.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    manifests = root / "outputs" / "manifests"
    processed = root / "data" / "processed"
    tables = root / "outputs" / "tables" / "diagnostics"
    reports = root / "outputs" / "tables" / "diagnostics"
    source_path = manifests / "source_aef_central_input.csv"
    source_meta = json.loads((manifests / "source_aef_central_input.json").read_text())
    if sha256(source_path) != source_meta["csv_sha256"]:
        raise RuntimeError("Frozen eligible manifest hash mismatch")
    source = pd.read_csv(source_path, dtype={"shot_number": "string", "spatial_block_id": "string"})
    source = source.sort_values(["year", "spatial_block_id", "shot_number"]).reset_index(drop=True)
    raw_dir = root / "data" / "raw" / "source_conventional_chunks"
    frames, chunk_audit, expected_files = [], [], set()
    for year, yearly in source.groupby("year", sort=True):
        yearly = yearly.reset_index(drop=True)
        for start in range(0, len(yearly), 500):
            expected = yearly.iloc[start:start + 500]
            name = f"source_conventional_y{int(year)}_c{start // 500:03d}.csv"
            expected_files.add(name)
            path = raw_dir / name
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, dtype={"shot_number": "string", "spatial_block_id": "string"})
            required = META + FEATURES
            if sorted(set(required) - set(frame.columns)):
                raise RuntimeError(f"Missing fields: {name}")
            if len(frame) != len(expected) or frame["shot_number"].duplicated().any():
                raise RuntimeError(f"Chunk row/uniqueness failure: {name}")
            if set(frame["shot_number"].astype(str)) != set(expected["shot_number"].astype(str)):
                raise RuntimeError(f"Chunk shot-set failure: {name}")
            frames.append(frame[required])
            chunk_audit.append({"file": name, "expected_n": len(expected),
                                "observed_n": len(frame), "sha256": sha256(path), "status": "PASS"})
    unexpected = sorted(p.name for p in raw_dir.glob("*.csv") if p.name not in expected_files)
    if unexpected:
        raise RuntimeError(f"Unexpected chunk outputs: {unexpected[:5]}")
    raw = pd.concat(frames, ignore_index=True)
    if len(raw) != len(source) or raw["shot_number"].duplicated().any():
        raise RuntimeError("Global row or uniqueness failure")
    if set(raw["shot_number"].astype(str)) != set(source["shot_number"].astype(str)):
        raise RuntimeError("Global shot-set failure")
    check = raw[META].merge(source[META], on="shot_number", validate="one_to_one",
                            suffixes=("_eo", "_frozen"), indicator=True)
    if not (check["_merge"] == "both").all():
        raise RuntimeError("Frozen metadata join failure")
    for field in ["year", "spatial_block_id", "spatial_fold"]:
        if not (check[f"{field}_eo"].astype(str).to_numpy() ==
                check[f"{field}_frozen"].astype(str).to_numpy()).all():
            raise RuntimeError(f"Metadata mismatch: {field}")

    values = raw[FEATURES].apply(pd.to_numeric, errors="coerce")
    sentinel = values.eq(MISSING)
    nonfinite = ~np.isfinite(values.to_numpy(dtype=float))
    invalid = sentinel.any(axis=1).to_numpy() | nonfinite.any(axis=1)
    per_feature_missing = sentinel.sum().to_dict()
    values = values.mask(sentinel)
    raw[FEATURES] = values
    raw = raw.sort_values("shot_number").reset_index(drop=True)
    raw_path = processed / "source_conventional_raw.parquet"
    raw.to_parquet(raw_path, index=False)

    valid = raw.loc[~invalid].copy()
    bounded_indices = ["NDVI", "NDMI", "NBR", "NDRE"]
    bounds_bad = {band: int((valid[band].abs() > 1.000001).sum()) for band in bounded_indices}
    finite_valid = np.isfinite(valid[FEATURES].to_numpy(dtype=float)).all()
    if not finite_valid or any(bounds_bad.values()):
        raise RuntimeError(f"Conventional mathematical validity failed: {bounds_bad}")
    valid_path = processed / "source_conventional_valid.parquet"
    valid.to_parquet(valid_path, index=False)

    aef = pd.read_parquet(processed / "source_aef_central.parquet")
    dem = pd.read_parquet(processed / "source_dem_features.parquet")
    aef["shot_number"] = aef["shot_number"].astype("string")
    dem["shot_number"] = dem["shot_number"].astype("string")
    common_ids = set(valid["shot_number"].astype(str)) & set(aef["shot_number"].astype(str)) & set(dem["shot_number"].astype(str))
    common_meta = source.loc[source["shot_number"].astype(str).isin(common_ids),
                             ["shot_number", "agbd", "agbd_se", "year", "spatial_block_id", "spatial_fold"]]
    conventional = common_meta.merge(valid[["shot_number", *FEATURES]], on="shot_number", validate="one_to_one")
    conventional = conventional.merge(dem[["shot_number", *DEM]], on="shot_number", validate="one_to_one")
    alphaearth = common_meta.merge(aef[["shot_number", *AEF]], on="shot_number", validate="one_to_one")
    alphaearth = alphaearth.merge(dem[["shot_number", *DEM]], on="shot_number", validate="one_to_one")
    conventional = conventional.sort_values("shot_number").reset_index(drop=True)
    alphaearth = alphaearth.sort_values("shot_number").reset_index(drop=True)
    if not conventional["shot_number"].equals(alphaearth["shot_number"]):
        raise RuntimeError("Representation shot order differs")
    for field in ["agbd", "agbd_se", "year", "spatial_block_id", "spatial_fold"]:
        if not conventional[field].equals(alphaearth[field]):
            raise RuntimeError(f"Representation shared field differs: {field}")
    conv_path = processed / "source_common_conventional.parquet"
    aef_path = processed / "source_common_alphaearth.parquet"
    conventional.to_parquet(conv_path, index=False)
    alphaearth.to_parquet(aef_path, index=False)
    common_manifest = common_meta.sort_values("shot_number").reset_index(drop=True)
    common_path = processed / "source_common_representation.parquet"
    common_manifest.to_parquet(common_path, index=False)

    annual = source.groupby("year").size().rename("aef_dem_eligible_n").to_frame()
    annual["conventional_valid_n"] = valid.groupby("year").size()
    annual["final_common_n"] = common_manifest.groupby("year").size()
    annual = annual.fillna(0).astype(int).reset_index()
    annual.to_csv(tables / "common_representation_audit.csv", index=False)
    pd.DataFrame(chunk_audit).to_csv(tables / "conventional_chunk_audit.csv", index=False)
    qc_rows = [
        {"check": "frozen_gedi_n", "value": 127622, "status": "DESCRIPTIVE"},
        {"check": "dem_valid_n", "value": len(source), "status": "PASS"},
        {"check": "aef_valid_n", "value": len(aef), "status": "PASS"},
        {"check": "conventional_valid_n", "value": len(valid), "status": "PASS"},
        {"check": "final_common_n", "value": len(common_manifest), "status": "PASS"},
        {"check": "conventional_missing_rows", "value": int(invalid.sum()), "status": "DESCRIPTIVE"},
    ]
    for feature, count in per_feature_missing.items():
        qc_rows.append({"check": f"missing_{feature}", "value": int(count), "status": "DESCRIPTIVE"})
    pd.DataFrame(qc_rows).to_csv(tables / "conventional_qc.csv", index=False)
    hashes = {
        "source_conventional_raw_parquet_sha256": sha256(raw_path),
        "source_conventional_valid_parquet_sha256": sha256(valid_path),
        "source_common_representation_parquet_sha256": sha256(common_path),
        "source_common_conventional_parquet_sha256": sha256(conv_path),
        "source_common_alphaearth_parquet_sha256": sha256(aef_path),
    }
    (processed / "source_common_representation.sha256").write_text(
        "\n".join(f"{value}  {key}" for key, value in hashes.items()) + "\n")
    report = f"""# Conventional EO Extraction and Common Intersection

- Frozen AEF+DEM eligible N: {len(source):,}.
- Conventional complete finite N: {len(valid):,}.
- Removed for at least one S1/S2/index missing value: {int(invalid.sum()):,}.
- Final common representation N: {len(common_manifest):,}.
- Both branches have exactly the same ordered shot numbers, AGBD, years and folds.
- Missing annual-composite values were not imputed and no temporal window was expanded.
- NDVI, NDMI, NBR and NDRE bound violations among retained rows: {sum(bounds_bad.values())}.
- Longitude and latitude are not included as predictors.

No Zhejiang labels or target performance were used.
"""
    (reports / "conventional_extraction.md").write_text(report, encoding="utf-8")
    result = {"source_n": len(source), "conventional_valid_n": len(valid),
              "removed_missing_n": int(invalid.sum()), "common_n": len(common_manifest),
              "hashes": hashes}
    (tables / "common_representation_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
