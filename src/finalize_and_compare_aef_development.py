"""Finalize paired AEF development features and run source-only RF spatial CV."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BANDS = [f"A{i:02d}" for i in range(64)]
DEM = ["elevation", "slope", "aspect_sin", "aspect_cos"]
METHODS = ["central", "aggregate25"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("drive_export_folder", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ledger = json.loads((root / "outputs" / "logs" /
                         "aef_development_export_tasks.json").read_text())
    common_path = root / "outputs" / "manifests" / "aef_development_common_manifest.csv"
    common_freeze = json.loads((root / "outputs" / "manifests" /
                                "aef_development_common_manifest.json").read_text())
    if sha256(common_path) != common_freeze["csv_sha256"]:
        raise RuntimeError("Development common manifest checksum mismatch")
    common = pd.read_csv(common_path, dtype={"shot_number": "string",
                                             "spatial_block_id": "string"})
    expected_shots = set(common["shot_number"].astype(str))
    processed = root / "data" / "processed"
    tables = root / "outputs" / "tables"
    reports = root / "reports"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    finalized = {}
    feature_hashes = {}

    for method in METHODS:
        records = [item for item in ledger if item["method"] == method]
        frames = []
        for record in records:
            path = args.drive_export_folder / f"{record['description']}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, dtype={"shot_number": "string",
                                             "spatial_block_id": "string"})
            if len(frame) != int(record["expected_n"]):
                raise RuntimeError(f"Row-count mismatch: {path.name}")
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True)
        required = ["shot_number", "year", "spatial_block_id", "spatial_fold", *BANDS]
        if sorted(set(required) - set(data.columns)):
            raise RuntimeError(f"Missing {method} fields")
        if int(data[required].isna().sum().sum()):
            raise RuntimeError(f"Null {method} fields")
        if int(data["shot_number"].duplicated().sum()):
            raise RuntimeError(f"Duplicate {method} shot_number")
        if set(data["shot_number"].astype(str)) != expected_shots:
            raise RuntimeError(f"{method} shot set differs from frozen development common set")
        values = data[BANDS].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite {method} embedding")
        norms = np.linalg.norm(values, axis=1)
        norm_audit = {
            "min": float(norms.min()), "mean": float(norms.mean()),
            "max": float(norms.max()), "p01": float(np.quantile(norms, 0.01)),
            "p99": float(np.quantile(norms, 0.99)),
        }
        # Both official central embeddings and explicitly re-normalized aggregate
        # embeddings should remain essentially unit length.
        # GEE's nominal 10 m sampling can introduce small resampling/float
        # deviations even for the published unit-length central embeddings.
        # Keep the exported vectors unchanged and reject only deviations above
        # 2%, while preserving the complete norm distribution in the audit.
        if np.max(np.abs(norms - 1.0)) > 2e-2:
            raise RuntimeError(f"{method} L2 norm audit failed: {norm_audit}")
        data = data.sort_values("shot_number").reset_index(drop=True)
        parquet = processed / f"aef_development_{method}.parquet"
        data.to_parquet(parquet, index=False)
        finalized[method] = data
        feature_hashes[method] = {"parquet_sha256": sha256(parquet),
                                  "norm_audit": norm_audit, "n": len(data)}

    if not finalized["central"]["shot_number"].equals(
        finalized["aggregate25"]["shot_number"]
    ):
        raise RuntimeError("Central/aggregate ordered shot sets differ")
    dem = pd.read_parquet(processed / "source_dem_features.parquet")
    dem["shot_number"] = dem["shot_number"].astype(str)
    targets = common[["shot_number", "agbd", "spatial_fold", "spatial_block_id"]].copy()
    targets["shot_number"] = targets["shot_number"].astype(str)
    base = targets.merge(dem[["shot_number", *DEM]], on="shot_number",
                         validate="one_to_one", how="inner")
    if len(base) != len(common):
        raise RuntimeError("Development common set lost DEM rows")

    all_folds = []
    start_time = time.time()
    rf_params = {
        "n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 5,
        "max_depth": None, "random_state": 42, "n_jobs": -1,
    }
    for method in METHODS:
        aef = finalized[method][["shot_number", *BANDS]]
        dataset = base.merge(aef, on="shot_number", validate="one_to_one")
        features = [*BANDS, *DEM]
        for fold in range(5):
            train = dataset["spatial_fold"].ne(fold)
            test = dataset["spatial_fold"].eq(fold)
            model = RandomForestRegressor(**rf_params)
            model.fit(dataset.loc[train, features], dataset.loc[train, "agbd"])
            pred = model.predict(dataset.loc[test, features])
            truth = dataset.loc[test, "agbd"].to_numpy()
            all_folds.append({
                "method": method, "fold": fold,
                "train_n": int(train.sum()), "test_n": int(test.sum()),
                "test_blocks": int(dataset.loc[test, "spatial_block_id"].nunique()),
                "r2": float(r2_score(truth, pred)),
                "rmse": float(mean_squared_error(truth, pred) ** 0.5),
                "mae": float(mean_absolute_error(truth, pred)),
            })
    folds = pd.DataFrame(all_folds)
    folds.to_csv(tables / "aef_aggregation_spatial_cv_folds.csv", index=False)
    summary = folds.groupby("method").agg(
        mean_r2=("r2", "mean"), r2_fold_sd=("r2", "std"),
        mean_rmse=("rmse", "mean"), rmse_fold_sd=("rmse", "std"),
        mean_mae=("mae", "mean"), mae_fold_sd=("mae", "std"),
    ).reset_index()
    summary.to_csv(tables / "aef_aggregation_spatial_cv_summary.csv", index=False)
    metrics = summary.set_index("method")
    r2_difference = abs(metrics.loc["central", "mean_r2"] -
                        metrics.loc["aggregate25", "mean_r2"])
    relative_rmse_difference = abs(
        metrics.loc["central", "mean_rmse"] - metrics.loc["aggregate25", "mean_rmse"]
    ) / metrics.loc["central", "mean_rmse"]
    practically_similar = bool(r2_difference < 0.01 and relative_rmse_difference < 0.01)
    rule_winner = ("central" if practically_similar else
                   summary.sort_values(["mean_rmse", "mean_mae"]).iloc[0]["method"])
    decision = {
        "status": "awaiting_human_freeze_review",
        "sample_n": len(base), "target_data_used": False,
        "rf_params": rf_params, "folds": 5,
        "r2_absolute_difference": float(r2_difference),
        "relative_rmse_difference": float(relative_rmse_difference),
        "practically_similar_by_frozen_rule": practically_similar,
        "rule_implied_winner": rule_winner,
        "feature_audits": feature_hashes,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "summary": summary.to_dict(orient="records"),
    }
    (tables / "aef_aggregation_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8")
    report = [
        "# AEF Central vs 25 m Aggregation Decision\n",
        "Status: awaiting human freeze review.\n",
        f"Source-only common development N: {len(base):,}. No Zhejiang data were used.\n",
        "## Five-fold block spatial CV\n",
        summary.to_markdown(index=False, floatfmt=".5f") + "\n",
        "## Frozen decision rule\n",
        f"- Absolute mean R² difference: {r2_difference:.6f}\n",
        f"- Relative mean RMSE difference: {relative_rmse_difference:.4%}\n",
        f"- Practically similar: {practically_similar}\n",
        f"- Rule-implied winner: **{rule_winner}**\n",
        "No full-source AEF extraction should begin until this result is reviewed and frozen.\n",
    ]
    (reports / "aef_aggregation_decision.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
