"""Freeze the AEF development subset after the common DEM availability gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    development = pd.read_parquet(
        root / "outputs" / "manifests" / "aef_development_subset_75.parquet")
    dem = pd.read_parquet(root / "data" / "processed" / "source_dem_features.parquet")
    valid = set(dem["shot_number"].astype(str))
    common = development.loc[
        development["shot_number"].astype(str).isin(valid)
    ].sort_values(["year", "spatial_block_id", "shot_number"]).reset_index(drop=True)
    out = root / "outputs" / "manifests"
    csv_path = out / "aef_development_common_manifest.csv"
    parquet_path = out / "aef_development_common_manifest.parquet"
    common.to_csv(csv_path, index=False)
    common.to_parquet(parquet_path, index=False)
    audit = {
        "status": "frozen", "sample_n": len(common),
        "source_development_n": len(development),
        "dem_masked_removed_n": len(development) - len(common),
        "spatial_block_n": int(common["spatial_block_id"].nunique()),
        "stratum_n": int(common.groupby(["spatial_block_id", "year"]).ngroups),
        "years": sorted(common["year"].astype(int).unique().tolist()),
        "all_blocks_single_fold": bool(
            common.groupby("spatial_block_id")["spatial_fold"].nunique().max() == 1),
        "csv_sha256": sha256(csv_path), "parquet_sha256": sha256(parquet_path),
    }
    (out / "aef_development_common_manifest.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
