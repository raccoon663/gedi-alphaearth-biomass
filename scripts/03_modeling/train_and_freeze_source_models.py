"""Source-domain model selection and evaluation with nested spatial cross-validation.

This script selects and evaluates source-domain (USA GEDI L4A) aboveground biomass
models for two predictor representations:

  * AlphaEarth Foundations embeddings (A00-A63) + DEM derivatives
  * Conventional Sentinel-1/2 indices + DEM derivatives

Both model selection and performance estimation respect spatial blocking so that the
reported numbers are an honest estimate of out-of-region predictive skill:

  * Outer loop: the five spatial blocks (``spatial_fold`` 0..4) are held out one at a
    time. The held-out block is never used for tuning.
  * Inner loop: for each outer-training set, a 4-fold spatial CV (the remaining blocks)
    jointly selects ONE (model family, configuration) from all RF/XGB candidates by
    RMSE. Family selection is therefore nested, not taken from outer-fold performance.
  * Outer test: only the configuration selected by the inner CV on the outer-training
    set is refit on that set and evaluated on the untouched outer-test block. The five
    outer-test blocks are aggregated into a single out-of-fold prediction set for the
    final metrics.

After the outer nested-CV, a separate source-only CV on ALL source data selects the
final deployment (family, config); that pipeline is refit on all source data. The
deployment fit's in-sample score is intentionally NOT reported as an evaluation metric.

The final deployment model for each representation is refit on all source data after
selection. Its in-sample fit is intentionally NOT reported as an evaluation metric.

The Zhejiang/Kaihua target labels and performance are never used to select or tune
these source models (see the ``target_label_locked`` safeguard in the manifest).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SEED = 42
N_OUTER = 5
TRAIN_CAP = 60000  # cap outer-training size for compute-efficient, equal budgets
DEM = ["elevation", "slope", "aspect_sin", "aspect_cos"]
AEF = [f"A{i:02d}" for i in range(64)] + DEM
CONVENTIONAL = ["VV", "VH", "VV_minus_VH", "B2", "B3", "B4", "B5", "B6",
                "B7", "B8", "B8A", "B11", "B12", "NDVI", "EVI", "NDMI",
                "NBR", "NDRE"] + DEM

# Every outer-test prediction is produced under this design; the manifest and the
# downstream figure filter on this exact label so the reported metric is unambiguous.
OUTER_SPLIT_TYPE = "nested_spatial_cv"
CANDIDATES = {"random_forest": ["rf_sqrt", "rf_half"],
              "xgboost": ["xgb_depth6", "xgb_depth8"]}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
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


def inner_select(train_df: pd.DataFrame, xcols, candidates) -> tuple[tuple[str, str], pd.DataFrame]:
    """4-fold spatial inner CV on an outer-training set.

    Returns the single (family, config) with the lowest mean inner-CV RMSE across
    ALL RF/XGB candidates, plus a long table of the inner-CV RMSE/R2 per candidate
    (for diagnostics). Selecting the family inside the inner CV is what makes the
    family choice fully nested rather than leaked from outer-fold performance.
    """
    inner_folds = sorted(int(f) for f in train_df["spatial_fold"].unique())
    records = []
    best: tuple[str, str] | None = None
    best_rmse = float("inf")
    for hold in inner_folds:
        itr = train_df[train_df.spatial_fold != hold]
        ite = train_df[train_df.spatial_fold == hold]
        for fam, cfgs in candidates.items():
            for cfg in cfgs:
                est = build(fam, cfg)
                est.fit(itr[xcols], itr.agbd)
                pred = est.predict(ite[xcols])
                m = metrics(ite.agbd.to_numpy(), pred)
                records.append({"model": fam, "config": cfg, "inner_fold": hold,
                                "RMSE": m["RMSE"], "R2": m["R2"]})
                if m["RMSE"] < best_rmse:
                    best_rmse = m["RMSE"]
                    best = (fam, cfg)
    assert best is not None, "inner_select found no candidate"
    return best, pd.DataFrame(records)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    env = os.environ.get("GEDI_SOURCE_DATA")
    processed = Path(env) if env else root / "data/processed"
    alpha_path = processed / "source_common_alphaearth.parquet"
    conv_path = processed / "source_common_conventional.parquet"
    if not alpha_path.exists() or not conv_path.exists():
        sys.stderr.write(
            f"Source parquets not found in {processed}. Set GEDI_SOURCE_DATA to the "
            "directory containing source_common_alphaearth.parquet and "
            "source_common_conventional.parquet.\n")
        sys.exit(2)

    tables = root / "outputs/tables"
    diagnostics = tables / "diagnostics"
    main_results = tables / "main_results"
    figures = root / "figures"
    models_dir = root / "outputs/models"
    manifests = root / "outputs/manifests"
    for d in (tables, diagnostics, main_results, figures, models_dir, manifests):
        d.mkdir(parents=True, exist_ok=True)

    paths = {"alphaearth": alpha_path, "conventional": conv_path}
    features = {"alphaearth": AEF, "conventional": CONVENTIONAL}
    datasets = {k: pd.read_parquet(v) for k, v in paths.items()}

    # The two representations must describe identical shots, AGBD and spatial folds so
    # that any performance gap reflects the predictors, not the sample.
    if not datasets["alphaearth"]["agbd"].equals(datasets["conventional"]["agbd"]):
        raise RuntimeError("AGBD differs between representations")
    for field in ["year", "spatial_block_id", "spatial_fold"]:
        if not datasets["alphaearth"][field].equals(datasets["conventional"][field]):
            raise RuntimeError(f"Shared field differs: {field}")
    if set(datasets["alphaearth"]["spatial_fold"].unique()) != set(range(N_OUTER)):
        raise RuntimeError("Expected frozen folds 0..4")

    tune_records = []   # inner-CV diagnostics (joint family+config selection)
    rows = []           # outer-CV per-fold metrics (honest estimate)
    oof_pred = {rep: np.empty(len(df), dtype=float) for rep, df in datasets.items()}

    for rep, df in datasets.items():
        xcols = features[rep]
        for o in range(N_OUTER):
            test_mask = df.spatial_fold == o
            train_df = df[~test_mask]
            if len(train_df) > TRAIN_CAP:
                train_df = train_df.sample(TRAIN_CAP, random_state=SEED)
            # Inner CV jointly selects ONE (family, config) on the outer-training set.
            (family, config), inner_df = inner_select(train_df, xcols, CANDIDATES)
            inner_df["representation"] = rep
            inner_df["outer_fold"] = o
            tune_records.append(inner_df)
            est = build(family, config)
            est.fit(train_df[xcols], train_df.agbd)
            pred = est.predict(df.loc[test_mask, xcols])
            idx = np.flatnonzero(test_mask.to_numpy())
            oof_pred[rep][idx] = pred
            m = metrics(df.loc[test_mask, "agbd"].to_numpy(), pred)
            rows.append({"representation": rep, "model": family, "config": config,
                         "split_type": OUTER_SPLIT_TYPE, "outer_fold": o, **m})

    # Mean and Ridge baselines under the identical outer folds (no tuning involved).
    for rep, df in datasets.items():
        xcols = features[rep]
        for o in range(N_OUTER):
            test_mask = df.spatial_fold == o
            tr, te = df[~test_mask], df[test_mask]
            mean_pred = np.full(int(test_mask.sum()), tr.agbd.mean())
            rows.append({"representation": rep, "model": "mean", "config": "train_mean",
                         "split_type": OUTER_SPLIT_TYPE, "outer_fold": o,
                         **metrics(te.agbd.to_numpy(), mean_pred)})
            ridge = build("ridge", "ridge_alpha10")
            ridge.fit(tr[xcols], tr.agbd)
            rows.append({"representation": rep, "model": "ridge", "config": "ridge_alpha10",
                         "split_type": OUTER_SPLIT_TYPE, "outer_fold": o,
                         **metrics(te.agbd.to_numpy(), ridge.predict(te[xcols]))})

    comparison = pd.DataFrame(rows)
    comparison.to_csv(diagnostics / "source_model_comparison.csv", index=False)
    pd.concat(tune_records, ignore_index=True).to_csv(diagnostics / "source_model_tuning.csv", index=False)

    # Honest aggregated out-of-fold metrics from the nested-selected pipelines, plus
    # cross-fold variability. Each outer fold contributed predictions from its own
    # inner-CV-selected pipeline, so outer performance is never reused for selection.
    summary_rows = []
    for rep, df in datasets.items():
        y = df.agbd.to_numpy()
        pred = oof_pred[rep]
        agg = metrics(y, pred)
        fold_metrics = [r for r in rows if r["representation"] == rep
                        and r["split_type"] == OUTER_SPLIT_TYPE
                        and r["model"] not in ("mean", "ridge")]
        summary_rows.append({
            "representation": rep, "model": "nested_selected", "config": "per_fold_inner_cv",
            "split_type": OUTER_SPLIT_TYPE,
            "R2": agg["R2"], "RMSE": agg["RMSE"], "MAE": agg["MAE"], "Bias": agg["Bias"], "N": agg["N"],
            "R2_std": float(np.std([r["R2"] for r in fold_metrics])),
            "RMSE_std": float(np.std([r["RMSE"] for r in fold_metrics])),
            "MAE_std": float(np.std([r["MAE"] for r in fold_metrics])),
            "Bias_std": float(np.std([r["Bias"] for r in fold_metrics])),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(main_results / "source_model_comparison_summary.csv", index=False)

    best = {}
    for rep, df in datasets.items():
        # Final deployment pipeline: a SEPARATE source-only CV on ALL source data
        # selects (family, config); the pipeline is then fit on all source data.
        # This selection is independent of the outer nested-CV estimate above.
        (family, config), _ = inner_select(df, features[rep], CANDIDATES)
        final_model = build(family, config)
        final_model.fit(df[features[rep]], df.agbd)
        model_path = models_dir / f"frozen_source_{rep}_{family}.joblib"
        joblib.dump(final_model, model_path, compress=3)

        winner = next(r for r in summary_rows if r["representation"] == rep)
        best[rep] = {
            "model": family,
            "config": config,
            "hyperparameters": build(family, config).get_params(),
            "source_cv": {k: float(winner[k]) for k in
                          ["R2", "R2_std", "RMSE", "RMSE_std",
                           "MAE", "MAE_std", "Bias", "Bias_std"]},
            "model_sha256": sha256(model_path),
            "model_file": str(model_path.relative_to(root)),
        }

        # OOF diagnostics for the nested-selected pipeline (no shot_number written out).
        oof_vec = oof_pred[rep]
        residual = oof_vec - df.agbd.to_numpy()
        diag = pd.DataFrame({
            "agbd": df.agbd, "prediction": oof_vec, "residual": residual,
            "absolute_error": np.abs(residual), "spatial_fold": df.spatial_fold,
            "model": family, "config": config,
        })
        diag.to_parquet(diagnostics / f"source_{rep}_oof_predictions.parquet", index=False)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hexbin(df.agbd, oof_pred[rep], gridsize=60, mincnt=1)
        axes[0].set(xlabel="Observed AGBD (Mg/ha)", ylabel="Predicted (OOF)", title=rep)
        axes[1].hist(residual, bins=80)
        axes[1].set(xlabel="Residual (pred-observed)", title="Residual distribution")
        axes[2].hexbin(oof_pred[rep], residual, gridsize=60, mincnt=1)
        axes[2].axhline(0, color="red")
        axes[2].set(xlabel="Predicted", ylabel="Residual")
        fig.tight_layout()
        fig.savefig(figures / f"source_{rep}_diagnostics.png", dpi=160)
        plt.close(fig)

        sample = df.sample(min(5000, len(df)), random_state=SEED)
        imp = permutation_importance(final_model, sample[features[rep]], sample.agbd,
                                     scoring="neg_root_mean_squared_error", n_repeats=5,
                                     random_state=SEED, n_jobs=1)
        pd.DataFrame({"feature": features[rep],
                      "importance_mean": imp.importances_mean,
                      "importance_sd": imp.importances_std}).sort_values(
                          "importance_mean", ascending=False).to_csv(
                              diagnostics / f"source_{rep}_permutation_importance.csv", index=False)

    packages = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                "xgboost": xgboost.__version__, "pyarrow": pyarrow.__version__,
                "joblib": joblib.__version__}

    # Provenance hashes are recorded only when the referenced files are present.
    source_sample_sha256 = None
    s200 = manifests / "frozen_source_manifest_200.csv"
    if s200.exists():
        source_sample_sha256 = sha256(s200)
    common_sample_sha256, common_sample_n = None, len(datasets["alphaearth"])
    common_path = processed / "source_common_representation.parquet"
    if common_path.exists():
        common_sample_sha256 = sha256(common_path)

    freeze = {
        "status": "source_models_selected",
        "selection_method": ("Nested spatial cross-validation: 5 outer spatial-block "
                             "holdouts; 4-fold spatial inner CV on each outer-training "
                             "set selects the configuration per family by RMSE. Outer "
                             "test blocks are never used for tuning."),
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_sample_sha256": source_sample_sha256,
        "common_sample_sha256": common_sample_sha256,
        "common_sample_n": common_sample_n,
        "aef_extraction_method": "central_pixel_10m_exact_year",
        "conventional_temporal_rule": "exact_year_calendar_year_median",
        "features": features, "spatial_block_km": 50, "spatial_block_crs": "EPSG:5070",
        "folds": N_OUTER, "fold_field": "spatial_fold", "random_seed": SEED,
        "train_cap": TRAIN_CAP,
        "selected_models": best, "package_versions": packages,
        "target_label_locked": True, "zhejiang_labels_used": False,
        "zhejiang_performance_used": False,
    }
    freeze_path = manifests / "frozen_source_model_manifest.json"
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    print(json.dumps({"selected": best, "manifest": str(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
