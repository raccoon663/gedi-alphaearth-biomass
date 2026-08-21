"""Run the eight-way PALSAR-common source benchmark with nested spatial CV."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from utilities.guards import assert_safe_predictors, assert_target_not_used  # noqa: E402
from utilities.radar_benchmark import REPRESENTATIONS, validate_paired_frames  # noqa: E402

SEED = 42
FOLDS = 5
TRAIN_CAP = 60000
CANDIDATES = [("random_forest", "rf_sqrt"), ("random_forest", "rf_half"),
              ("xgboost", "xgb_depth6"), ("xgboost", "xgb_depth8")]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return {"R2": float(r2_score(y, pred)),
            "RMSE": float(mean_squared_error(y, pred) ** .5),
            "MAE": float(mean_absolute_error(y, pred)),
            "Bias": float(np.mean(pred-y)), "N": int(len(y))}


def model(family: str, config: str, seed: int = SEED):
    if family == "random_forest":
        return RandomForestRegressor(n_estimators=350, min_samples_leaf=5,
                                     max_features="sqrt" if config == "rf_sqrt" else .5,
                                     random_state=seed, n_jobs=-1)
    params = dict(n_estimators=700, learning_rate=.05, subsample=.8,
                  colsample_bytree=.8, reg_lambda=1., objective="reg:squarederror",
                  tree_method="hist", random_state=seed, n_jobs=6)
    params.update(max_depth=6, min_child_weight=5) if config == "xgb_depth6" else params.update(
        max_depth=8, min_child_weight=10)
    from xgboost import XGBRegressor
    return XGBRegressor(**params)


def select_inner(data: pd.DataFrame, features: list[str]) -> tuple[tuple[str, str], list[dict]]:
    rows = []
    for hold in sorted(data.spatial_fold.unique()):
        train, test = data[data.spatial_fold != hold], data[data.spatial_fold == hold]
        for family, config in CANDIDATES:
            estimator = model(family, config)
            estimator.fit(train[features], train.agbd)
            rows.append({"inner_fold": int(hold), "model": family, "config": config,
                         **metrics(test.agbd, estimator.predict(test[features]))})
    frame = pd.DataFrame(rows)
    winner = (frame.groupby(["model", "config"], as_index=False).RMSE.mean()
              .sort_values(["RMSE", "model", "config"]).iloc[0])
    return (winner.model, winner.config), rows


def load_frames(common_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {name: pd.read_parquet(common_dir / f"source_{name}.parquet")
              for name in REPRESENTATIONS}
    for frame in frames.values():
        frame["shot_number"] = frame.shot_number.astype("string")
    validate_paired_frames(frames, domain="source")
    return frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--common-dir", type=Path, default=ROOT / "data/processed/palsar_common")
    p.add_argument("--train-cap", type=int, default=TRAIN_CAP)
    p.add_argument("--quick", action="store_true", help="Smoke-test one outer fold and one RF config")
    args = p.parse_args()
    assert_target_not_used(stage="model_tuning", target_labels_loaded=False)
    frames = load_frames(args.common_dir)
    if set(frames["dem"].spatial_fold.unique()) != set(range(FOLDS)):
        raise RuntimeError("Expected deterministic source folds 0..4")
    for name, features in REPRESENTATIONS.items():
        assert_safe_predictors(features)
    folds = [0] if args.quick else list(range(FOLDS))
    global CANDIDATES
    if args.quick:
        CANDIDATES = [("random_forest", "rf_sqrt")]
    fold_rows, tuning_rows, importance_rows, summaries, deployments = [], [], [], [], {}
    oof_frames = []
    models_dir = ROOT / "outputs/models/palsar_benchmark"
    tables = ROOT / "outputs/tables"
    models_dir.mkdir(parents=True, exist_ok=True)
    (tables / "diagnostics").mkdir(parents=True, exist_ok=True)
    (tables / "main_results").mkdir(parents=True, exist_ok=True)
    for name, data in frames.items():
        features = REPRESENTATIONS[name]
        oof = np.full(len(data), np.nan)
        selected = []
        for outer in folds:
            mask = data.spatial_fold == outer
            train = data.loc[~mask]
            if len(train) > args.train_cap:
                train = train.sample(args.train_cap, random_state=SEED)
            (family, config), inner = select_inner(train, features)
            selected.append((family, config))
            for row in inner:
                tuning_rows.append({"representation": name, "outer_fold": outer, **row})
            estimator = model(family, config)
            estimator.fit(train[features], train.agbd)
            pred = estimator.predict(data.loc[mask, features])
            oof[np.flatnonzero(mask)] = pred
            fold_rows.append({"representation": name, "outer_fold": outer,
                              "model": family, "config": config,
                              **metrics(data.loc[mask, "agbd"], pred)})
            held = data.loc[mask]
            if len(held) > 5000:
                held = held.sample(5000, random_state=SEED + outer)
            importance = permutation_importance(
                estimator, held[features], held.agbd,
                scoring="neg_root_mean_squared_error", n_repeats=3,
                random_state=SEED + outer, n_jobs=1)
            for feature, mean, sd in zip(features, importance.importances_mean,
                                         importance.importances_std):
                importance_rows.append({"representation": name, "outer_fold": outer,
                                        "feature": feature,
                                        "importance_mean": float(mean),
                                        "importance_sd": float(sd),
                                        "heldout_n": len(held)})
        evaluated = np.isfinite(oof)
        pooled = metrics(data.loc[evaluated, "agbd"], oof[evaluated])
        rep_folds = [x for x in fold_rows if x["representation"] == name]
        summaries.append({"representation": name, **pooled,
                          **{f"{m}_std": float(np.std([x[m] for x in rep_folds]))
                             for m in ["R2", "RMSE", "MAE", "Bias"]},
                          "evaluation": "nested_source_spatial_cv",
                          "common_sample": True, "quick_smoke": args.quick})
        oof_frames.append(pd.DataFrame({"representation": name,
                                        "agbd": data.loc[evaluated, "agbd"].to_numpy(),
                                        "prediction": oof[evaluated],
                                        "residual": oof[evaluated] - data.loc[evaluated, "agbd"].to_numpy(),
                                        "spatial_fold": data.loc[evaluated, "spatial_fold"].to_numpy()}))
        # Deployment selection is a separate source-only CV on all common rows.
        (family, config), deployment_tuning = select_inner(data, features)
        estimator = model(family, config)
        estimator.fit(data[features], data.agbd)
        path = models_dir / f"{name}_{family}.joblib"
        joblib.dump(estimator, path, compress=3)
        deployments[name] = {"model": family, "config": config, "model_file": path.relative_to(ROOT).as_posix(),
                             "model_sha256": sha256(path), "features": features,
                             "feature_schema_sha256": hashlib.sha256(json.dumps(features).encode()).hexdigest(),
                             "selection_fold_rows": len(deployment_tuning)}
    pd.DataFrame(fold_rows).to_csv(tables / "diagnostics/palsar_source_spatial_cv_folds.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(tables / "diagnostics/palsar_source_model_tuning.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(tables / "diagnostics/radar_feature_importance.csv", index=False)
    pd.concat(oof_frames, ignore_index=True).to_parquet(
        tables / "diagnostics/palsar_source_oof_predictions.parquet", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(tables / "main_results/palsar_source_spatial_cv.csv", index=False)
    freeze = {"status": "PALSAR_COMMON_SOURCE_MODELS_FROZEN", "frozen_at": datetime.now(timezone.utc).isoformat(),
              "quick_smoke": args.quick, "selection_metric": "source_spatial_cv_RMSE",
              "target_labels_loaded": False, "target_metrics_used": False,
              "folds": folds, "train_cap": args.train_cap, "seed": SEED,
              "candidate_budget": CANDIDATES, "representations": deployments,
              "source_table_sha256": sha256(tables / "main_results/palsar_source_spatial_cv.csv"),
              "software": {"python": platform.python_version(), "platform": platform.platform()}}
    out = ROOT / "outputs/manifests/palsar_source_model_freeze.json"
    out.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
