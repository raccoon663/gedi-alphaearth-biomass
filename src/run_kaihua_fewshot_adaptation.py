"""Run frozen Kaihua bias, affine and local-representation few-shot experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

REPRESENTATIONS = ["alphaearth", "conventional"]
METHODS = ["bias_only", "affine", "local_xgboost"]
THRESHOLDS = [0.0, 0.2, 0.4, 0.5]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"R2": float(r2_score(y, p)),
            "RMSE": float(mean_squared_error(y, p) ** 0.5),
            "MAE": float(mean_absolute_error(y, p)),
            "Bias": float(np.mean(p-y))}


def model(protocol: dict, seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=protocol["n_estimators"], learning_rate=protocol["learning_rate"],
        max_depth=protocol["max_depth"], min_child_weight=protocol["min_child_weight"],
        subsample=protocol["subsample"], colsample_bytree=protocol["colsample_bytree"],
        reg_lambda=protocol["reg_lambda"], objective=protocol["objective"],
        tree_method=protocol["tree_method"], n_jobs=protocol["n_jobs"],
        random_state=seed, verbosity=0)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifests, tables, figures = root / "outputs/manifests", root / "outputs/tables", root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    design_path = manifests / "kaihua_fewshot_design_freeze.json"
    design = json.loads(design_path.read_text())
    if design["status"] != "KAIHUA_FEWSHOT_DESIGN_FROZEN_BEFORE_REPRESENTATION_RESULTS":
        raise RuntimeError("Few-shot design is not frozen")
    for key, filename in [("block_manifest_sha256", "kaihua_target_spatial_blocks.csv"),
                          ("fold_manifest_sha256", "kaihua_fewshot_spatial_folds.csv"),
                          ("label_draw_manifest_sha256", "kaihua_fewshot_label_draws.csv")]:
        if sha256(manifests / filename) != design[key]:
            raise RuntimeError(f"HARD STOP: frozen design hash mismatch: {filename}")
    evaluation_path = root / "data/processed/kaihua_zero_shot_evaluation.parquet"
    if sha256(evaluation_path) != design["input_evaluation_sha256"]:
        raise RuntimeError("HARD STOP: evaluation dataset hash mismatch")
    source_manifest = json.loads((manifests / "frozen_source_model_manifest.json").read_text())
    data = pd.read_parquet(evaluation_path)
    data["shot_number"] = data.shot_number.astype("string")
    folds = pd.read_csv(manifests / "kaihua_fewshot_spatial_folds.csv",
                        dtype={"shot_number": "string"})
    data = data.merge(folds[["shot_number", "target_block_id", "spatial_fold"]],
                      on="shot_number", how="left", validate="one_to_one")
    if data.spatial_fold.isna().any() or data.groupby("target_block_id").spatial_fold.nunique().max() != 1:
        raise RuntimeError("Spatial fold join/integrity failure")
    draws = pd.read_csv(manifests / "kaihua_fewshot_label_draws.csv",
                        dtype={"shot_number": "string"})
    features = {"alphaearth": source_manifest["features"]["alphaearth"],
                "conventional": source_manifest["features"]["conventional"]}
    zero_cols = {"alphaearth": "prediction_alphaearth",
                 "conventional": "prediction_conventional"}
    protocol = design["common_model_protocol"]

    rows, parameter_rows = [], []
    for rep in REPRESENTATIONS:
        y = data.agbd.to_numpy(float); p = data[zero_cols[rep]].to_numpy(float)
        rows.append({"representation": rep, "method": "zero_shot", "budget": 0,
                     "fold": -1, "seed": 0, "N_train": 0, "N_test": len(data), **metrics(y, p)})

    grouped_draws = {(int(f), int(b), int(s)): g.shot_number.to_numpy()
                     for (f, b, s), g in draws.groupby(["fold", "budget", "seed"], sort=True)}
    data_indexed = data.set_index("shot_number", drop=False)
    total = len(grouped_draws)
    for iteration, ((fold, budget, seed), train_shots) in enumerate(grouped_draws.items(), 1):
        train = data_indexed.loc[train_shots]
        test = data.loc[data.spatial_fold == fold]
        if len(train) != budget or (train.spatial_fold == fold).any():
            raise RuntimeError(f"Training draw/fold failure: {fold}/{budget}/{seed}")
        y_train, y_test = train.agbd.to_numpy(float), test.agbd.to_numpy(float)
        for rep in REPRESENTATIONS:
            p_train = train[zero_cols[rep]].to_numpy(float)
            p_test = test[zero_cols[rep]].to_numpy(float)
            correction = float(np.mean(y_train-p_train))
            bias_pred = p_test + correction
            rows.append({"representation": rep, "method": "bias_only", "budget": budget,
                         "fold": fold, "seed": seed, "N_train": len(train), "N_test": len(test),
                         **metrics(y_test, bias_pred)})
            slope, intercept = np.polyfit(p_train, y_train, 1)
            affine_pred = intercept + slope*p_test
            rows.append({"representation": rep, "method": "affine", "budget": budget,
                         "fold": fold, "seed": seed, "N_train": len(train), "N_test": len(test),
                         **metrics(y_test, affine_pred)})
            parameter_rows.append({"representation": rep, "budget": budget, "fold": fold,
                                   "seed": seed, "bias_correction": correction,
                                   "affine_intercept": float(intercept), "affine_slope": float(slope)})
            estimator = model(protocol, seed)
            estimator.fit(train[features[rep]], y_train)
            local_pred = estimator.predict(test[features[rep]])
            rows.append({"representation": rep, "method": "local_xgboost", "budget": budget,
                         "fold": fold, "seed": seed, "N_train": len(train), "N_test": len(test),
                         **metrics(y_test, local_pred)})
        if iteration % 5 == 0 or iteration == total:
            pd.DataFrame(rows).to_csv(tables / "kaihua_fewshot_label_efficiency.csv", index=False)
            pd.DataFrame(parameter_rows).to_csv(tables / "kaihua_fewshot_calibration_parameters.csv", index=False)
            print(f"completed design cells {iteration}/{total}", flush=True)

    detailed = pd.DataFrame(rows)
    detailed.to_csv(tables / "kaihua_fewshot_label_efficiency.csv", index=False)
    adapted = detailed[detailed.budget > 0]
    summary = (adapted.groupby(["representation", "method", "budget"], as_index=False)
               .agg(runs=("R2", "size"), N_train=("N_train", "median"),
                    N_test_mean=("N_test", "mean"),
                    R2_mean=("R2", "mean"), R2_sd=("R2", "std"), R2_median=("R2", "median"),
                    RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"), RMSE_median=("RMSE", "median"),
                    MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"), MAE_median=("MAE", "median"),
                    Bias_mean=("Bias", "mean"), Bias_sd=("Bias", "std"), Bias_median=("Bias", "median")))
    zero = detailed[detailed.budget == 0].copy()
    zero_summary = pd.DataFrame([{
        "representation": r.representation, "method": "zero_shot", "budget": 0,
        "runs": 1, "N_train": 0, "N_test_mean": r.N_test,
        "R2_mean": r.R2, "R2_sd": np.nan, "R2_median": r.R2,
        "RMSE_mean": r.RMSE, "RMSE_sd": np.nan, "RMSE_median": r.RMSE,
        "MAE_mean": r.MAE, "MAE_sd": np.nan, "MAE_median": r.MAE,
        "Bias_mean": r.Bias, "Bias_sd": np.nan, "Bias_median": r.Bias}
        for r in zero.itertuples(index=False)])
    summary = pd.concat([zero_summary, summary], ignore_index=True)
    summary.to_csv(tables / "kaihua_fewshot_summary.csv", index=False)

    threshold_rows = []
    for rep in REPRESENTATIONS:
        for method in METHODS:
            subset = summary[(summary.representation == rep) & (summary.method == method)].sort_values("budget")
            for threshold in THRESHOLDS:
                reached = subset[subset.R2_mean > threshold]
                threshold_rows.append({"representation": rep, "method": method,
                                       "R2_threshold": threshold,
                                       "labels_required": (int(reached.iloc[0].budget) if len(reached) else "not reached"),
                                       "criterion": "mean spatial-holdout R2 across 5 folds x 3 seeds"})
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(tables / "kaihua_label_thresholds.csv", index=False)

    colors = {"alphaearth": "#286f9b", "conventional": "#d47a23"}
    styles = {"bias_only": ":", "affine": "--", "local_xgboost": "-"}
    labels = {"bias_only": "bias", "affine": "affine", "local_xgboost": "local XGB"}
    zero_lookup = zero.set_index("representation")
    for metric, ylabel, filename in [("R2", "Spatial holdout R²", "kaihua_label_efficiency.png"),
                                      ("RMSE", "Spatial holdout RMSE (Mg/ha)", "kaihua_label_efficiency_rmse.png")]:
        fig, ax = plt.subplots(figsize=(9, 6))
        for rep in REPRESENTATIONS:
            for method in METHODS:
                z = summary[(summary.representation == rep) & (summary.method == method)].sort_values("budget")
                x = np.r_[0, z.budget.to_numpy(float)]
                mean = np.r_[zero_lookup.loc[rep, metric], z[f"{metric}_mean"].to_numpy(float)]
                sd = np.r_[0, z[f"{metric}_sd"].fillna(0).to_numpy(float)]
                ax.plot(x, mean, color=colors[rep], ls=styles[method], marker="o", ms=3,
                        label=f"{'AEF' if rep == 'alphaearth' else 'Conventional'} {labels[method]}")
                ax.fill_between(x, mean-sd, mean+sd, color=colors[rep], alpha=.08)
        if metric == "R2": ax.axhline(0, color="black", lw=1, alpha=.6)
        ax.set_xscale("symlog", linthresh=25, linscale=1)
        ax.set_xticks([0, 25, 50, 100, 250, 500, 1000, 2500])
        ax.set_xticklabels(["0", "25", "50", "100", "250", "500", "1000", "2500"])
        ax.set_xlabel("Number of Kaihua labels"); ax.set_ylabel(ylabel)
        ax.grid(alpha=.2); ax.legend(ncol=2, fontsize=8); fig.tight_layout()
        fig.savefig(figures / filename, dpi=240); plt.close(fig)
    print(summary.to_string(index=False))
    print("\nThresholds\n", thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
