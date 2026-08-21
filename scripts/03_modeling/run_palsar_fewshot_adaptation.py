"""Shared-draw Kaihua label-efficiency experiment for all radar representations."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from utilities.radar_benchmark import REPRESENTATIONS  # noqa: E402

BUDGETS = [25, 50, 100, 250, 500, 1000, 2500]
SEEDS = [42, 43, 44]
PROTOCOL = {"n_estimators": 300, "learning_rate": .03, "max_depth": 3,
            "min_child_weight": 5, "subsample": .8, "colsample_bytree": .8,
            "reg_lambda": 10., "objective": "reg:squarederror", "tree_method": "hist",
            "n_jobs": 6}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return {"R2": float(r2_score(y, pred)), "RMSE": float(mean_squared_error(y, pred) ** .5),
            "MAE": float(mean_absolute_error(y, pred)), "Bias": float(np.mean(pred-y))}


def deterministic_draws(data: pd.DataFrame) -> dict[tuple[int, int, int], np.ndarray]:
    draws = {}
    for fold in range(5):
        pool = data.loc[data.spatial_fold != fold, "shot_number"].sort_values().to_numpy()
        for seed in SEEDS:
            order = np.random.default_rng(seed + fold * 100_000).permutation(len(pool))
            for budget in BUDGETS:
                if budget <= len(pool):
                    draws[(fold, budget, seed)] = pool[order[:budget]]
    return draws


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--common-dir", type=Path, default=ROOT / "data/processed/palsar_common")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--draws", type=Path, help="Optional frozen legacy draw CSV; otherwise identical algorithm is regenerated")
    args = p.parse_args()
    from xgboost import XGBRegressor
    labels = pd.read_parquet(args.labels) if args.labels.suffix == ".parquet" else pd.read_csv(args.labels)
    labels["shot_number"] = labels.shot_number.astype("string")
    frames = {}
    expected = None
    for name, features in REPRESENTATIONS.items():
        data = pd.read_parquet(args.common_dir / f"target_{name}.parquet")
        data["shot_number"] = data.shot_number.astype("string")
        data = data.merge(labels[["shot_number", "year", "agbd"]], on=["shot_number", "year"],
                          how="left", validate="one_to_one")
        data = data.sort_values(["year", "shot_number"]).reset_index(drop=True)
        if data.agbd.isna().any() or data.groupby("target_block_id").spatial_fold.nunique().max() != 1:
            raise RuntimeError(f"Target labels/folds invalid for {name}")
        keys = data[["shot_number", "year", "agbd", "target_block_id", "spatial_fold"]]
        if expected is not None and not keys.equals(expected):
            raise RuntimeError(f"Few-shot rows differ for {name}")
        expected, frames[name] = keys, data
    base = frames["dem"]
    if args.draws:
        raw = pd.read_csv(args.draws, dtype={"shot_number": "string"})
        draws = {(int(f), int(b), int(s)): g.shot_number.to_numpy()
                 for (f, b, s), g in raw.groupby(["fold", "budget", "seed"], sort=True)}
    else:
        draws = deterministic_draws(base)
    rows = []
    for (fold, budget, seed), shot_ids in sorted(draws.items()):
        for name, features in REPRESENTATIONS.items():
            data = frames[name].set_index("shot_number", drop=False)
            train, test = data.loc[shot_ids], data[data.spatial_fold == fold]
            if len(train) != budget or (train.spatial_fold == fold).any():
                raise RuntimeError(f"Draw integrity failure: {fold}/{budget}/{seed}")
            estimator = XGBRegressor(**PROTOCOL, random_state=seed, verbosity=0)
            estimator.fit(train[features], train.agbd)
            rows.append({"representation": name, "budget": budget, "fold": fold, "seed": seed,
                         "N_train": len(train), "N_test": len(test),
                         **metrics(test.agbd, estimator.predict(test[features]))})
    detailed = pd.DataFrame(rows)
    summary = (detailed.groupby(["representation", "budget"], as_index=False)
               .agg(runs=("R2", "size"), R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                    RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
                    MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
                    Bias_mean=("Bias", "mean"), Bias_sd=("Bias", "std")))
    first_positive = (summary[summary.R2_mean > 0].sort_values("budget")
                      .groupby("representation", as_index=False).first()[["representation", "budget"]]
                      .rename(columns={"budget": "first_positive_R2_labels"}))
    summary = summary.merge(first_positive, on="representation", how="left")
    # Compare at the same budget with both legacy representations rerun on common rows.
    comparisons = []
    for name in REPRESENTATIONS:
        for legacy in ["sentinel1_sentinel2", "alphaearth"]:
            for budget in BUDGETS:
                own = summary[(summary.representation == name) & (summary.budget == budget)]
                ref = summary[(summary.representation == legacy) & (summary.budget == budget)]
                if len(own) and len(ref) and own.iloc[0].R2_mean > ref.iloc[0].R2_mean:
                    comparisons.append({"representation": name, "legacy_reference": legacy,
                                        "fewest_labels_to_exceed": budget})
                    break
    out = ROOT / "outputs/tables/main_results/"
    out.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(out / "palsar_fewshot_label_efficiency_detailed.csv", index=False)
    summary.to_csv(out / "palsar_fewshot_label_efficiency.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "palsar_fewshot_legacy_thresholds.csv", index=False)
    manifest = {"status": "PALSAR_FEWSHOT_COMPLETE", "budgets": BUDGETS, "seeds": SEEDS,
                "folds": 5, "model_protocol": PROTOCOL, "shared_draws": True,
                "target_specific_tuning": False, "labels_sha256": sha256(args.labels),
                "summary_sha256": sha256(out / "palsar_fewshot_label_efficiency.csv")}
    (ROOT / "outputs/manifests/palsar_fewshot_results_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
