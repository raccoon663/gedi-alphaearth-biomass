"""Source-only balanced-budget selection, CV, diagnostics, and model freeze."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import sklearn
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SEED = 42
DEM = ["elevation", "slope", "aspect_sin", "aspect_cos"]
AEF = [f"A{i:02d}" for i in range(64)] + DEM
CONVENTIONAL = ["VV", "VH", "VV_minus_VH", "B2", "B3", "B4", "B5", "B6",
                "B7", "B8", "B8A", "B11", "B12", "NDVI", "EVI", "NDMI",
                "NBR", "NDRE"] + DEM


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"R2": float(r2_score(y, pred)),
            "RMSE": float(mean_squared_error(y, pred) ** 0.5),
            "MAE": float(mean_absolute_error(y, pred)),
            "Bias": float(np.mean(pred - y)), "N": int(len(y))}


def build(model: str, config: str):
    if model == "random_forest":
        params = dict(n_estimators=350, min_samples_leaf=5, random_state=SEED,
                      n_jobs=-1, max_depth=None)
        params["max_features"] = "sqrt" if config == "rf_sqrt" else 0.5
        return RandomForestRegressor(**params)
    if model == "xgboost":
        params = dict(n_estimators=700, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0, objective="reg:squarederror",
                      tree_method="hist", random_state=SEED, n_jobs=6)
        if config == "xgb_depth6":
            params.update(max_depth=6, min_child_weight=5)
        else:
            params.update(max_depth=8, min_child_weight=10)
        return XGBRegressor(**params)
    if model == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    raise ValueError(model)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed, tables, reports = root / "data/processed", root / "outputs/tables", root / "reports"
    models_dir, figures = root / "outputs/models", root / "outputs/figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    paths = {"alphaearth": processed / "source_common_alphaearth.parquet",
             "conventional": processed / "source_common_conventional.parquet"}
    features = {"alphaearth": AEF, "conventional": CONVENTIONAL}
    datasets = {k: pd.read_parquet(v) for k, v in paths.items()}
    if not datasets["alphaearth"]["shot_number"].equals(datasets["conventional"]["shot_number"]):
        raise RuntimeError("Common shot order differs")
    for field in ["agbd", "year", "spatial_block_id", "spatial_fold"]:
        if not datasets["alphaearth"][field].equals(datasets["conventional"][field]):
            raise RuntimeError(f"Shared field differs: {field}")
    if set(datasets["alphaearth"]["spatial_fold"].unique()) != set(range(5)):
        raise RuntimeError("Expected frozen folds 0..4")

    tune_rows = []
    candidates = {"random_forest": ["rf_sqrt", "rf_half"],
                  "xgboost": ["xgb_depth6", "xgb_depth8"]}
    selected = {}
    # Fixed fold 0 is the source-only development validation fold. Cap training
    # at 60k with deterministic sampling for equal, compute-efficient budgets.
    for representation, df in datasets.items():
        xcols = features[representation]
        train = df[df.spatial_fold != 0]
        if len(train) > 60000:
            train = train.sample(60000, random_state=SEED)
        valid = df[df.spatial_fold == 0]
        for family, configs in candidates.items():
            for config in configs:
                start = time.time()
                estimator = build(family, config)
                estimator.fit(train[xcols], train.agbd)
                row = {"representation": representation, "model": family,
                       "config": config, "split_type": "source_tuning_fold0",
                       **metrics(valid.agbd.to_numpy(), estimator.predict(valid[xcols])),
                       "elapsed_seconds": round(time.time() - start, 2)}
                tune_rows.append(row)
            family_rows = [r for r in tune_rows if r["representation"] == representation and r["model"] == family]
            selected[(representation, family)] = min(family_rows, key=lambda r: r["RMSE"])["config"]
    pd.DataFrame(tune_rows).to_csv(tables / "source_model_tuning.csv", index=False)

    rows, predictions = [], {}
    for representation, df in datasets.items():
        xcols = features[representation]
        y = df.agbd.to_numpy()
        # Mean and Ridge sanity baselines under identical spatial folds.
        for fold in range(5):
            tr, te = df.spatial_fold != fold, df.spatial_fold == fold
            mean_pred = np.full(te.sum(), df.loc[tr, "agbd"].mean())
            rows.append({"representation": representation, "model": "mean",
                         "config": "train_mean", "split_type": "block_5fold", "fold": fold,
                         **metrics(df.loc[te, "agbd"].to_numpy(), mean_pred)})
            ridge = build("ridge", "ridge_alpha10")
            ridge.fit(df.loc[tr, xcols], df.loc[tr, "agbd"])
            rows.append({"representation": representation, "model": "ridge",
                         "config": "ridge_alpha10", "split_type": "block_5fold", "fold": fold,
                         **metrics(df.loc[te, "agbd"].to_numpy(), ridge.predict(df.loc[te, xcols]))})
        for family in ["random_forest", "xgboost"]:
            config = selected[(representation, family)]
            oof = np.empty(len(df), dtype=np.float64)
            for fold in range(5):
                tr, te = df.spatial_fold != fold, df.spatial_fold == fold
                estimator = build(family, config)
                estimator.fit(df.loc[tr, xcols], df.loc[tr, "agbd"])
                pred = estimator.predict(df.loc[te, xcols])
                oof[np.flatnonzero(te.to_numpy())] = pred
                rows.append({"representation": representation, "model": family,
                             "config": config, "split_type": "block_5fold", "fold": fold,
                             **metrics(df.loc[te, "agbd"].to_numpy(), pred)})
            predictions[(representation, family)] = oof
            tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED)
            estimator = build(family, config)
            estimator.fit(df.iloc[tr_idx][xcols], df.iloc[tr_idx].agbd)
            rows.append({"representation": representation, "model": family,
                         "config": config, "split_type": "random_80_20", "fold": 0,
                         **metrics(df.iloc[te_idx].agbd.to_numpy(), estimator.predict(df.iloc[te_idx][xcols]))})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(tables / "source_model_comparison.csv", index=False)
    summary = (comparison[comparison.split_type == "block_5fold"]
               .groupby(["representation", "model", "config"])
               .agg({"R2": ["mean", "std"], "RMSE": ["mean", "std"],
                     "MAE": ["mean", "std"], "Bias": ["mean", "std"], "N": "sum"})
               .reset_index())
    summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns]
    summary.to_csv(tables / "source_model_comparison_summary.csv", index=False)

    best = {}
    for representation in datasets:
        eligible = summary[(summary.representation == representation) &
                           (summary.model.isin(["random_forest", "xgboost"]))]
        winner = eligible.loc[eligible.RMSE_mean.idxmin()].to_dict()
        family, config = winner["model"], winner["config"]
        best[representation] = {"model": family, "config": config,
                                "hyperparameters": build(family, config).get_params(),
                                "source_cv": {k: float(winner[k]) for k in
                                              ["R2_mean", "R2_std", "RMSE_mean", "RMSE_std",
                                               "MAE_mean", "MAE_std", "Bias_mean", "Bias_std"]}}
        df, xcols = datasets[representation], features[representation]
        final_model = build(family, config)
        final_model.fit(df[xcols], df.agbd)
        model_path = models_dir / f"frozen_source_{representation}_{family}.joblib"
        joblib.dump(final_model, model_path, compress=3)
        best[representation]["model_sha256"] = sha256(model_path)
        best[representation]["model_file"] = str(model_path.relative_to(root))

        oof = predictions[(representation, family)]
        residual = oof - df.agbd.to_numpy()
        diag = pd.DataFrame({"shot_number": df.shot_number, "agbd": df.agbd,
                             "prediction": oof, "residual": residual,
                             "absolute_error": np.abs(residual), "spatial_fold": df.spatial_fold})
        diag.to_parquet(tables / f"source_{representation}_best_oof.parquet", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hexbin(df.agbd, oof, gridsize=60, mincnt=1); axes[0].set(xlabel="Observed AGBD", ylabel="OOF predicted", title=representation)
        axes[1].hist(residual, bins=80); axes[1].set(xlabel="Residual (pred-observed)", title="Residual distribution")
        axes[2].hexbin(oof, residual, gridsize=60, mincnt=1); axes[2].axhline(0, color="red"); axes[2].set(xlabel="Predicted", ylabel="Residual")
        fig.tight_layout(); fig.savefig(figures / f"source_{representation}_diagnostics.png", dpi=160); plt.close(fig)

        # Source-only interpretation on a deterministic 5k subset with 5 repeats.
        sample = df.sample(min(5000, len(df)), random_state=SEED)
        imp = permutation_importance(final_model, sample[xcols], sample.agbd,
                                     scoring="neg_root_mean_squared_error", n_repeats=5,
                                     random_state=SEED, n_jobs=1)
        pd.DataFrame({"feature": xcols, "importance_mean": imp.importances_mean,
                      "importance_sd": imp.importances_std}).sort_values(
                          "importance_mean", ascending=False).to_csv(
                              tables / f"source_{representation}_permutation_importance.csv", index=False)

    packages = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                "xgboost": xgboost.__version__, "pyarrow": pyarrow.__version__,
                "joblib": joblib.__version__}
    common_path = processed / "source_common_representation.parquet"
    source_manifest = root / "outputs/manifests/frozen_source_manifest_200.csv"
    freeze = {
        "status": "FROZEN_SOURCE_MODELS_TARGET_LABEL_LOCKED",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_sample_sha256": sha256(source_manifest),
        "common_sample_sha256": sha256(common_path),
        "common_sample_n": len(datasets["alphaearth"]),
        "aef_extraction_method": "central_pixel_10m_exact_year",
        "conventional_temporal_rule": "exact_year_calendar_year_median",
        "features": features, "spatial_block_km": 50, "spatial_block_crs": "EPSG:5070",
        "folds": 5, "fold_field": "spatial_fold", "random_seed": SEED,
        "tuning_budget": "2 RF + 2 XGBoost configs per representation on source fold 0; 60k train cap",
        "selected_models": best, "package_versions": packages,
        "target_label_locked": True, "zhejiang_labels_used": False,
        "zhejiang_performance_used": False,
    }
    freeze_path = root / "outputs/manifests/frozen_source_model_manifest.json"
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    report = "# Source Model Freeze\n\nStatus: **FROZEN; TARGET_LABEL_LOCKED = True**\n\n"
    report += f"Common N: {len(datasets['alphaearth']):,}. Both representations use identical shots, folds and AGBD.\n\n"
    for rep, info in best.items():
        cv = info["source_cv"]
        report += f"- {rep}: {info['model']} / {info['config']}; mean block-CV R2 {cv['R2_mean']:.4f}, RMSE {cv['RMSE_mean']:.3f}, MAE {cv['MAE_mean']:.3f}, Bias {cv['Bias_mean']:.3f}.\n"
    report += "\nNo Zhejiang predictors, labels or performance were used. A00–A63 are learned latent features and are not assigned physical meanings.\n"
    (reports / "source_model_freeze.md").write_text(report, encoding="utf-8")
    print(json.dumps({"selected": best, "manifest": str(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
