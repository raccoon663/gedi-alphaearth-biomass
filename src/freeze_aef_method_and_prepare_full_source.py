"""Freeze the selected AEF method and materialize its DEM-valid source input."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifests = root / "outputs" / "manifests"
    source_path = manifests / "frozen_source_manifest_200.parquet"
    dem_path = root / "data" / "processed" / "source_dem_features.parquet"
    decision_path = root / "outputs" / "tables" / "aef_aggregation_decision.json"

    source = pd.read_parquet(source_path)
    dem = pd.read_parquet(dem_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    source["shot_number"] = source["shot_number"].astype("string")
    dem["shot_number"] = dem["shot_number"].astype("string")
    if source["shot_number"].duplicated().any() or dem["shot_number"].duplicated().any():
        raise RuntimeError("Duplicate shot_number in frozen source or DEM")
    if not set(dem["shot_number"]).issubset(set(source["shot_number"])):
        raise RuntimeError("DEM includes shots outside frozen source manifest")

    fields = ["shot_number", "lon", "lat", "year", "spatial_block_id",
              "spatial_fold", "agbd", "agbd_se"]
    eligible = source.loc[source["shot_number"].isin(set(dem["shot_number"])), fields]
    eligible = eligible.sort_values(["year", "spatial_block_id", "shot_number"]).reset_index(drop=True)
    if len(eligible) != len(dem) or eligible[fields].isna().any().any():
        raise RuntimeError("DEM-valid AEF manifest completeness failure")
    if not eligible["year"].between(2019, 2024).all():
        raise RuntimeError("Unexpected year")
    if eligible.groupby("spatial_block_id")["spatial_fold"].nunique().max() != 1:
        raise RuntimeError("Block-level fold mapping violated")

    csv_path = manifests / "source_aef_central_input.csv"
    parquet_path = manifests / "source_aef_central_input.parquet"
    eligible.to_csv(csv_path, index=False)
    eligible.to_parquet(parquet_path, index=False)
    input_meta = {
        "status": "frozen_input",
        "n": len(eligible),
        "unique_shot_number_n": int(eligible["shot_number"].nunique()),
        "years": sorted(eligible["year"].astype(int).unique().tolist()),
        "spatial_blocks": int(eligible["spatial_block_id"].nunique()),
        "folds": sorted(eligible["spatial_fold"].astype(int).unique().tolist()),
        "source_manifest_parquet_sha256": sha256(source_path),
        "dem_parquet_sha256": sha256(dem_path),
        "csv_sha256": sha256(csv_path),
        "parquet_sha256": sha256(parquet_path),
        "feature_extraction": {
            "dataset": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
            "bands": [f"A{i:02d}" for i in range(64)],
            "year_match": "exact",
            "method": "central_pixel",
            "scale_m": 10,
        },
        "lon_lat_predictor_use_forbidden": True,
    }
    (manifests / "source_aef_central_input.json").write_text(
        json.dumps(input_meta, indent=2), encoding="utf-8")

    summary = {row["method"]: row for row in decision["summary"]}
    freeze = {
        "status": "frozen",
        "selected_method": "central_pixel_10m",
        "retired_main_experiment_method": "footprint_mean_25m_l2_renormalized",
        "development_sample_N": int(decision["sample_n"]),
        "mean_R2_central": summary["central"]["mean_r2"],
        "mean_R2_aggregate": summary["aggregate25"]["mean_r2"],
        "mean_RMSE_central": summary["central"]["mean_rmse"],
        "mean_RMSE_aggregate": summary["aggregate25"]["mean_rmse"],
        "decision_rule": {
            "practically_similar_if": "abs(mean R2 difference) < 0.01 and relative mean RMSE difference < 0.01",
            "tie_break": "select lower-compute central pixel",
            "observed_abs_R2_difference": decision["r2_absolute_difference"],
            "observed_relative_RMSE_difference": decision["relative_rmse_difference"],
            "practically_similar": decision["practically_similar_by_frozen_rule"],
        },
        "decision_timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_only_confirmation": True,
        "zhejiang_predictors_used": False,
        "zhejiang_labels_used": False,
        "target_performance_may_change_decision": False,
        "full_source_dual_extraction_forbidden": True,
    }
    freeze_path = manifests / "aef_method_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps({"freeze": str(freeze_path), "input": input_meta}, indent=2))


if __name__ == "__main__":
    main()
