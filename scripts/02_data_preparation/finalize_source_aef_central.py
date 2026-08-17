"""Strictly finalize full-source central AEF and join frozen DEM/AGBD."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BANDS = [f"A{i:02d}" for i in range(64)]
META = ["shot_number", "year", "spatial_block_id", "spatial_fold", "lon", "lat"]
DEM = ["elevation", "slope", "aspect_sin", "aspect_cos"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "outputs" / "manifests" / "source_aef_central_input.csv"
    manifest_meta = json.loads((root / "outputs" / "manifests" /
                                "source_aef_central_input.json").read_text(encoding="utf-8"))
    if sha256(manifest_path) != manifest_meta["csv_sha256"]:
        raise RuntimeError("Input manifest checksum mismatch")
    manifest = pd.read_csv(manifest_path, dtype={"shot_number": "string",
                                                 "spatial_block_id": "string"})
    manifest = manifest.sort_values(["year", "spatial_block_id", "shot_number"]).reset_index(drop=True)
    raw = root / "data" / "raw" / "source_aef_central_chunks"
    frames, chunk_rows = [], []
    expected_files = set()
    for year, yearly in manifest.groupby("year", sort=True):
        yearly = yearly.reset_index(drop=True)
        for start in range(0, len(yearly), 500):
            chunk = yearly.iloc[start:start + 500]
            description = f"source_aef_central_y{int(year)}_c{start // 500:03d}"
            path = raw / f"{description}.csv"
            expected_files.add(path.name)
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, dtype={"shot_number": "string",
                                             "spatial_block_id": "string"})
            required = META + BANDS
            if sorted(set(required) - set(frame.columns)):
                raise RuntimeError(f"Missing fields: {path.name}")
            if len(frame) != len(chunk):
                raise RuntimeError(f"Chunk row mismatch: {path.name}")
            if set(frame["shot_number"].astype(str)) != set(chunk["shot_number"].astype(str)):
                raise RuntimeError(f"Chunk shot mismatch: {path.name}")
            frames.append(frame[required])
            chunk_rows.append({"file": path.name, "expected_n": len(chunk), "observed_n": len(frame),
                               "sha256": sha256(path), "status": "PASS"})
    unexpected = sorted(p.name for p in raw.glob("*.csv") if p.name not in expected_files)
    if unexpected:
        raise RuntimeError(f"Unexpected duplicate chunk outputs: {unexpected[:5]}")
    data = pd.concat(frames, ignore_index=True)
    if len(data) != len(manifest) or data["shot_number"].duplicated().any():
        raise RuntimeError("Global row count or uniqueness failure")
    if set(data["shot_number"].astype(str)) != set(manifest["shot_number"].astype(str)):
        raise RuntimeError("Global expected shot set failure")
    audit_join = data[META].merge(manifest[META], on="shot_number", how="outer",
                                  suffixes=("_aef", "_frozen"), indicator=True,
                                  validate="one_to_one")
    if not (audit_join["_merge"] == "both").all():
        raise RuntimeError("Metadata join mismatch")
    for field in ["year", "spatial_block_id", "spatial_fold", "lon", "lat"]:
        left, right = audit_join[f"{field}_aef"], audit_join[f"{field}_frozen"]
        if field in ("lon", "lat"):
            ok = np.isclose(left.astype(float), right.astype(float), atol=1e-9, rtol=0)
        else:
            ok = left.astype(str).to_numpy() == right.astype(str).to_numpy()
        if not np.all(ok):
            raise RuntimeError(f"Frozen metadata mismatch: {field}")
    values = data[BANDS].to_numpy(dtype=np.float64)
    if data[META + BANDS].isna().any().any() or not np.isfinite(values).all():
        raise RuntimeError("Missing or non-finite AEF values")
    if np.all(values == 0, axis=1).any():
        raise RuntimeError("All-zero AEF vector")
    norms = np.linalg.norm(values, axis=1)
    norm_stats = {"mean": float(norms.mean()), "sd": float(norms.std(ddof=1)),
                  "min": float(norms.min()), "p01": float(np.quantile(norms, .01)),
                  "median": float(np.median(norms)), "p99": float(np.quantile(norms, .99)),
                  "max": float(norms.max())}
    norm_flag = np.abs(norms - 1.0) > 0.02
    # Norm deviations are audited and retained, never silently deleted.
    data = data.sort_values("shot_number").reset_index(drop=True)
    processed = root / "data" / "processed"
    tables = root / "outputs" / "tables"
    reports = root / "reports"
    csv_path = processed / "source_aef_central.csv"
    parquet_path = processed / "source_aef_central.parquet"
    data.to_csv(csv_path, index=False)
    data.to_parquet(parquet_path, index=False)
    (processed / "source_aef_central.sha256").write_text(
        f"{sha256(csv_path)}  {csv_path.name}\n{sha256(parquet_path)}  {parquet_path.name}\n",
        encoding="utf-8")

    qc = pd.DataFrame([
        {"check": "expected_shot_set", "value": len(data), "status": "PASS"},
        {"check": "global_unique_shot_number", "value": data["shot_number"].nunique(), "status": "PASS"},
        {"check": "complete_A00_A63", "value": int(np.isfinite(values).all()), "status": "PASS"},
        {"check": "all_zero_vectors", "value": int(np.all(values == 0, axis=1).sum()), "status": "PASS"},
        {"check": "year_consistency", "value": 1, "status": "PASS"},
        {"check": "block_fold_consistency", "value": 1, "status": "PASS"},
        {"check": "norm_abs_deviation_gt_0.02", "value": int(norm_flag.sum()),
         "status": "WARNING" if norm_flag.any() else "PASS"},
    ])
    for key, value in norm_stats.items():
        qc.loc[len(qc)] = {"check": f"l2_norm_{key}", "value": value, "status": "DESCRIPTIVE"}
    qc.to_csv(tables / "source_aef_qc.csv", index=False)
    pd.DataFrame(chunk_rows).to_csv(tables / "source_aef_chunk_audit.csv", index=False)

    dem = pd.read_parquet(processed / "source_dem_features.parquet")
    dem["shot_number"] = dem["shot_number"].astype("string")
    joined = manifest[["shot_number", "agbd", "agbd_se", "year", "spatial_block_id",
                       "spatial_fold"]].merge(data[["shot_number", *BANDS]], on="shot_number",
                                             validate="one_to_one")
    joined = joined.merge(dem[["shot_number", *DEM]], on="shot_number", validate="one_to_one")
    if len(joined) != len(manifest) or joined["shot_number"].duplicated().any():
        raise RuntimeError("AEF+DEM join multiplication or loss")
    if joined[["agbd", "agbd_se", "year", "spatial_block_id", "spatial_fold", *BANDS, *DEM]].isna().any().any():
        raise RuntimeError("AEF+DEM joined required value missing")
    dataset_path = processed / "source_aef_dem_dataset.parquet"
    joined.sort_values("shot_number").to_parquet(dataset_path, index=False)
    summary = {
        "status": "PASS" if not norm_flag.any() else "PASS_WITH_NORM_WARNING",
        "finalized_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "n": len(data), "chunk_n": len(chunk_rows), "bands_n": len(BANDS),
        "norm": norm_stats, "norm_abs_deviation_gt_0.02_n": int(norm_flag.sum()),
        "source_aef_csv_sha256": sha256(csv_path),
        "source_aef_parquet_sha256": sha256(parquet_path),
        "source_aef_dem_parquet_sha256": sha256(dataset_path),
        "input_manifest_sha256": manifest_meta["csv_sha256"],
    }
    (tables / "source_aef_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# Full-source AlphaEarth Central Extraction

Status: **{summary['status']}**

- Frozen input N: {len(manifest):,}; finalized N: {len(data):,}.
- Dataset: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`; exact GEDI-year match.
- Method: central pixel at 10 m; A00–A63; no buffer, averaging or renormalization.
- All expected shots are present exactly once. Year, block and fold match the frozen input.
- Longitude and latitude are metadata only and are excluded from the modeling dataset.
- No NaN, Inf or all-zero embeddings occurred.
- L2 norm: mean {norm_stats['mean']:.8f}, SD {norm_stats['sd']:.8f}, min {norm_stats['min']:.8f}, p01 {norm_stats['p01']:.8f}, median {norm_stats['median']:.8f}, p99 {norm_stats['p99']:.8f}, max {norm_stats['max']:.8f}.
- Samples with absolute norm deviation above 0.02: {int(norm_flag.sum()):,}; these were audited and retained.
- One-to-one AEF/DEM/AGBD join N: {len(joined):,}; no row multiplication or target loss.

No Zhejiang predictors, labels or performance were used.
"""
    (reports / "source_aef_extraction.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
