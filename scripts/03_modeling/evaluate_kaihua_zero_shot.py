"""Unlock Kaihua GEDI labels only after prediction freeze and evaluate transfer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from shapely import intersects_xy
from shapely.geometry import shape
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from run_kaihua_locked_predictors import (META, YEARS, balanced_merge, gedi_assets,
                                          get_csv, source_qa)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"R2": float(r2_score(y, p)), "RMSE": float(mean_squared_error(y, p) ** .5),
            "MAE": float(mean_absolute_error(y, p)), "Bias": float(np.mean(p-y)), "N": len(y)}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    manifests, processed = root / "outputs/manifests", root / "data/processed"
    tables = root / "outputs/tables"
    main_results, diagnostics = tables / "main_results", tables / "diagnostics"
    figures = root / "figures"
    for d in (main_results, diagnostics, figures):
        d.mkdir(parents=True, exist_ok=True)
    freeze_path = manifests / "kaihua_zero_shot_freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("HARD STOP: zero-shot predictions are not frozen; labels remain locked")
    freeze = json.loads(freeze_path.read_text())
    if not freeze.get("labels_locked_during_predictor_construction_and_prediction"):
        raise RuntimeError("HARD STOP: label-lock provenance missing")
    pred_paths = {"alphaearth": root / "outputs/predictions/kaihua_zero_shot_aef.csv",
                  "conventional": root / "outputs/predictions/kaihua_zero_shot_conventional.csv"}
    for branch, path in pred_paths.items():
        if sha256(path) != freeze["prediction_file_hashes"][branch]:
            raise RuntimeError(f"HARD STOP: frozen {branch} prediction hash mismatch")

    label_path = processed / "kaihua_gedi_labels_unlocked.csv"
    geojson = json.loads((root / "data/boundaries/kaihua_target_aoi.geojson").read_text(encoding="utf-8"))
    geom = shape(geojson["features"][0]["geometry"])
    if not label_path.exists():
        ee.Initialize()
        bbox = ee.Geometry.Rectangle(list(geom.bounds), proj="EPSG:4326", geodesic=False)
        frames = []
        for year in YEARS:
            for month in range(1, 13):
                ids = gedi_assets(bbox, year, month)
                if not ids:
                    continue
                raw = balanced_merge([ee.FeatureCollection(x).filterBounds(bbox) for x in ids])
                if raw.size().getInfo() == 0:
                    continue
                valid = source_qa(raw).map(lambda f: ee.Feature(f).set({
                    "lon": ee.Feature(f).geometry().coordinates().get(0),
                    "lat": ee.Feature(f).geometry().coordinates().get(1), "year": year}))
                n = valid.size().getInfo()
                if not n:
                    continue
                frame = get_csv(valid, META + ["agbd", "agbd_se"])
                frame = frame.loc[intersects_xy(geom, frame.lon.to_numpy(float),
                                                 frame.lat.to_numpy(float))].copy()
                frames.append(frame[META + ["agbd", "agbd_se"]])
                print("LABEL UNLOCK", year, f"{month:02d}", len(frame), flush=True)
        labels = pd.concat(frames, ignore_index=True)
        labels["shot_number"] = labels.shot_number.astype("string")
        labels = labels.sort_values(["year", "shot_number"]).reset_index(drop=True)
        if labels.shot_number.duplicated().any() or labels[["agbd", "agbd_se"]].isna().any().any():
            raise RuntimeError("Unlocked labels duplicate/missing failure")
        if (labels.agbd < 0).any():
            raise RuntimeError("Unlocked labels contain negative AGBD")
        labels.to_csv(label_path, index=False)
    else:
        labels = pd.read_csv(label_path, dtype={"shot_number": "string"})

    common_path = processed / "kaihua_common_predictors_locked.parquet"
    if sha256(common_path) != freeze["target_predictor_hashes"]["common"]:
        raise RuntimeError("HARD STOP: common predictor hash mismatch after prediction freeze")
    common = pd.read_parquet(common_path)
    common["shot_number"] = common.shot_number.astype("string")
    evaluation = common.merge(labels[["shot_number", "agbd", "agbd_se"]], on="shot_number",
                              how="left", validate="one_to_one")
    if len(evaluation) != freeze["target_common_n"] or evaluation.agbd.isna().any():
        raise RuntimeError("HARD STOP: exact target shot-label join failure")
    for branch, path in pred_paths.items():
        p = pd.read_csv(path, dtype={"shot_number": "string"})
        evaluation = evaluation.merge(p[["shot_number", "prediction_agbd"]].rename(
            columns={"prediction_agbd": f"prediction_{branch}"}), on="shot_number",
            how="left", validate="one_to_one")
    if evaluation[["prediction_alphaearth", "prediction_conventional"]].isna().any().any():
        raise RuntimeError("Prediction join failure")
    evaluation.to_parquet(processed / "kaihua_zero_shot_evaluation.parquet", index=False)

    source = json.loads((manifests / "frozen_source_model_manifest.json").read_text())
    result = {b: metrics(evaluation.agbd.to_numpy(), evaluation[f"prediction_{b}"].to_numpy())
              for b in ["alphaearth", "conventional"]}
    rows = []
    for b in ["alphaearth", "conventional"]:
        src = source["selected_models"][b]["source_cv"]
        m = result[b]
        rows.append({"representation": "AlphaEarth" if b == "alphaearth" else "Conventional",
                     "source_R2": src["R2"], "target_R2": m["R2"],
                     "R2_drop": src["R2"]-m["R2"], "source_RMSE": src["RMSE"],
                     "target_RMSE": m["RMSE"], "RMSE_increase": m["RMSE"]-src["RMSE"],
                     "RMSE_increase_percent": 100*(m["RMSE"]/src["RMSE"]-1),
                     "target_RMSE_over_source_RMSE": m["RMSE"]/src["RMSE"],
                     "target_MAE": m["MAE"], "target_Bias": m["Bias"], "N": m["N"]})
    summary = pd.DataFrame(rows)
    assert len(summary) == 2, f"expected 2 rows (AlphaEarth, Conventional), got {len(summary)}"
    summary.to_csv(main_results / "representation_transfer_summary.csv", index=False)

    year_rows = []
    for year, group in evaluation.groupby("year"):
        if len(group) < 100:
            continue
        for b in ["alphaearth", "conventional"]:
            year_rows.append({"year": year, "representation": b,
                              **metrics(group.agbd.to_numpy(), group[f"prediction_{b}"].to_numpy())})
    pd.DataFrame(year_rows).to_csv(diagnostics / "kaihua_zero_shot_metrics_by_year.csv", index=False)

    distance = pd.read_parquet(diagnostics / "kaihua_aef_nearest_source_distance.parquet")
    evaluation = evaluation.merge(distance, on="shot_number", validate="one_to_one")
    evaluation["aef_absolute_error"] = abs(evaluation.prediction_alphaearth-evaluation.agbd)
    rho = spearmanr(evaluation.nearest_source_aef_euclidean_distance,
                    evaluation.aef_absolute_error)
    evaluation["distance_decile"] = pd.qcut(evaluation.nearest_source_aef_euclidean_distance,
                                             10, labels=False, duplicates="drop") + 1
    deciles = evaluation.groupby("distance_decile").apply(lambda g: pd.Series({
        "N": len(g), "distance_min": g.nearest_source_aef_euclidean_distance.min(),
        "distance_max": g.nearest_source_aef_euclidean_distance.max(),
        "MAE": mean_absolute_error(g.agbd, g.prediction_alphaearth),
        "RMSE": mean_squared_error(g.agbd, g.prediction_alphaearth) ** .5}),
        include_groups=False).reset_index()
    deciles["spearman_rho_all"] = rho.statistic
    deciles["spearman_pvalue_all"] = rho.pvalue
    deciles.to_csv(diagnostics / "kaihua_aef_distance_vs_error.csv", index=False)

    diag_rows = []
    evaluation["observed_decile"] = pd.qcut(evaluation.agbd, 10, labels=False, duplicates="drop") + 1
    for b in ["alphaearth", "conventional"]:
        for decile, g in evaluation.groupby("observed_decile"):
            diag_rows.append({"representation": b, "observed_agbd_decile": decile, "N": len(g),
                              "observed_mean": g.agbd.mean(),
                              "prediction_mean": g[f"prediction_{b}"].mean(),
                              "bias": (g[f"prediction_{b}"]-g.agbd).mean(),
                              "MAE": mean_absolute_error(g.agbd, g[f"prediction_{b}"])})
    pd.DataFrame(diag_rows).to_csv(diagnostics / "kaihua_error_by_observed_biomass_decile.csv", index=False)

    for b, title in [("alphaearth", "AlphaEarth"), ("conventional", "Conventional")]:
        pred = evaluation[f"prediction_{b}"]
        residual = pred-evaluation.agbd
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].scatter(evaluation.agbd, pred, s=3, alpha=.12, rasterized=True)
        lo = min(evaluation.agbd.min(), pred.min()); hi = max(evaluation.agbd.max(), pred.max())
        axes[0].plot([lo, hi], [lo, hi], "k--", lw=1); axes[0].set(xlabel="Observed AGBD", ylabel="Predicted AGBD")
        axes[1].hist(residual, bins=80); axes[1].set(xlabel="Residual (pred-observed)", ylabel="N")
        axes[2].scatter(evaluation.agbd, residual, s=3, alpha=.12, rasterized=True)
        axes[2].axhline(0, color="k", ls="--", lw=1); axes[2].set(xlabel="Observed AGBD", ylabel="Residual")
        fig.suptitle(f"Kaihua zero-shot: {title}"); fig.tight_layout()
        fig.savefig(figures / f"kaihua_zero_shot_{b}_diagnostics.png", dpi=220); plt.close(fig)

    a, c = summary.iloc[0], summary.iloc[1]
    if a.target_R2 > c.target_R2 and a.target_RMSE < c.target_RMSE and a.R2_drop < c.R2_drop:
        interpretation = "AlphaEarth embeddings retained substantially stronger predictive performance under USA-to-Zhejiang transfer than conventional Sentinel features."
    elif abs(a.target_R2-c.target_R2) < .05 or abs(a.target_RMSE-c.target_RMSE)/c.target_RMSE < .05:
        interpretation = "AlphaEarth showed stronger source-domain prediction, but cross-domain transfer performance was broadly comparable."
    else:
        interpretation = "AlphaEarth improved source-domain biomass prediction but did not provide superior cross-region transfer to Zhejiang."
    report = f"""# Kaihua zero-shot transfer

## Frozen source result

Source common N = {source['common_sample_n']:,}. Frozen AlphaEarth source CV R² = {a.source_R2:.4f}; conventional = {c.source_R2:.4f}. Model hashes were verified before target predictor construction.

## Target predictor construction and label-lock proof

The frozen Kaihua AOI (330824) and exact 2019–2024 GEDI metadata shot manifest were used. DEM, exact-year central-pixel A00–A63, and frozen exact-calendar-year Sentinel features were constructed while `TARGET_LABEL_LOCKED = True`. Neither AGBD nor AGBD_SE appeared in predictor or prediction files. Predictions were SHA-256 frozen before labels were unlocked. Target common N = {len(evaluation):,}.

## Target metrics and transfer degradation

AlphaEarth: R² {a.target_R2:.4f}, RMSE {a.target_RMSE:.3f}, MAE {a.target_MAE:.3f}, Bias {a.target_Bias:.3f}; R² drop {a.R2_drop:.4f}; target/source RMSE {a.target_RMSE_over_source_RMSE:.3f}.

Conventional: R² {c.target_R2:.4f}, RMSE {c.target_RMSE:.3f}, MAE {c.target_MAE:.3f}, Bias {c.target_Bias:.3f}; R² drop {c.R2_drop:.4f}; target/source RMSE {c.target_RMSE_over_source_RMSE:.3f}.

## Core transfer conclusion

{interpretation}

## Predictor-domain shift

Predictor-only diagnostics include conventional KS/Wasserstein statistics, source-fit AEF PCA, a descriptive USA-vs-Kaihua domain-classifier AUROC, and exact nearest-source 64-D Euclidean distances. These diagnostics were not used for model selection or adaptation.

## AEF distance versus error

Spearman rho = {rho.statistic:.4f} (p = {rho.pvalue:.3g}). Decile-level N, MAE, and RMSE are saved in `outputs/tables/diagnostics/kaihua_aef_distance_vs_error.csv`.

## Limitations

This is footprint-level zero-shot evaluation, not wall-to-wall mapping. No Zhejiang retraining, calibration, feature selection, temporal adjustment, or target-dependent thresholding was performed. Downstream spatial leakage was controlled, while AlphaEarth retains pretraining provenance overlap with GEDI-derived forest structure. AlphaEarth pretraining incorporated GEDI L2A relative-height information; the downstream target here is GEDI L4A AGBD, so this is not direct target-label leakage but it is provenance overlap.

Wall-to-wall map exports remain stopped pending human review.
"""
    (diagnostics / "kaihua_zero_shot_transfer.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False)); print(interpretation)


if __name__ == "__main__":
    main()
