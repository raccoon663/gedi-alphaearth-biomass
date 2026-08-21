"""Paired source-target separability, distance and zero-shot error diagnostics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from utilities.radar_benchmark import REPRESENTATIONS  # noqa: E402

REPS = ["sentinel1_c", "palsar_l", "sentinel1_c_palsar_l", "sentinel1_sentinel2",
        "palsar_sentinel2", "alphaearth"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--common-dir", type=Path, default=ROOT / "data/processed/palsar_common")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--max-domain-n", type=int, default=20000)
    p.add_argument("--max-source-distance-n", type=int, default=20000)
    args = p.parse_args()
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(args.labels) if args.labels.suffix == ".parquet" else pd.read_csv(args.labels)
    labels["shot_number"] = labels.shot_number.astype("string")
    rows, pca_rows = [], []
    fig, axes = plt.subplots(2, 3, figsize=(15, 9)); axes = axes.ravel()
    for ax, rep in zip(axes, REPS):
        features = REPRESENTATIONS[rep]
        source = pd.read_parquet(args.common_dir / f"source_{rep}.parquet")
        target = pd.read_parquet(args.common_dir / f"target_{rep}.parquet")
        target["shot_number"] = target.shot_number.astype("string")
        pred = pd.read_csv(ROOT / f"outputs/predictions/palsar_benchmark/target_{rep}.csv",
                           dtype={"shot_number": "string"})
        target = (target.merge(labels[["shot_number", "year", "agbd"]], on=["shot_number", "year"],
                               validate="one_to_one")
                  .merge(pred, on=["shot_number", "year"], validate="one_to_one"))
        source_sample = source.sample(min(args.max_domain_n, len(source)), random_state=42)
        target_sample = target.sample(min(args.max_domain_n, len(target)), random_state=42)
        x = np.vstack([source_sample[features], target_sample[features]])
        domain = np.r_[np.zeros(len(source_sample)), np.ones(len(target_sample))]
        classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
        probability = cross_val_predict(classifier, x, domain,
                                        cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                        method="predict_proba", n_jobs=1)[:, 1]
        auroc = roc_auc_score(domain, probability)
        scaler = StandardScaler().fit(source_sample[features])
        source_scaled = scaler.transform(source_sample[features])
        target_scaled = scaler.transform(target_sample[features])
        pca = PCA(n_components=2, random_state=42).fit(source_scaled)
        source_pc, target_pc = pca.transform(source_scaled), pca.transform(target_scaled)
        ax.scatter(source_pc[:, 0], source_pc[:, 1], s=3, alpha=.12, label="USA source")
        ax.scatter(target_pc[:, 0], target_pc[:, 1], s=3, alpha=.12, label="Kaihua target")
        ax.set(title=f"{rep}\nAUROC={auroc:.3f}", xlabel="source-fit PC1", ylabel="source-fit PC2")
        if rep == REPS[0]: ax.legend(frameon=False, markerscale=3)
        # Nearest-source distance uses a deterministic source subset and a
        # source-fit PCA projection.  This keeps the diagnostic tractable for
        # 68-dimensional AlphaEarth while preserving a common, source-only
        # Euclidean geometry.  It is descriptive and never enters selection.
        distance_source = source.sample(min(args.max_source_distance_n, len(source)), random_state=42)
        distance_scaler = StandardScaler().fit(distance_source[features])
        source_distance_scaled = distance_scaler.transform(distance_source[features])
        target_distance_scaled = distance_scaler.transform(target[features])
        distance_components = min(12, len(features), len(distance_source) - 1)
        distance_pca = PCA(n_components=distance_components, random_state=42).fit(
            source_distance_scaled)
        source_distance_projected = distance_pca.transform(source_distance_scaled)
        target_distance_projected = distance_pca.transform(target_distance_scaled)
        nearest = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1).fit(
            source_distance_projected)
        distance = nearest.kneighbors(target_distance_projected,
                                      return_distance=True)[0][:, 0]
        absolute_error = np.abs(target.prediction - target.agbd)
        rho, pvalue = spearmanr(distance, absolute_error)
        rows.append({"representation": rep, "domain_classifier_AUROC": auroc,
                     "domain_classifier_n_source": len(source_sample),
                     "domain_classifier_n_target": len(target_sample),
                     "nearest_source_reference_n": len(distance_source),
                     "nearest_source_distance_space": "source-standardized PCA",
                     "nearest_source_distance_components": distance_components,
                     "nearest_source_distance_variance_fraction": float(
                         distance_pca.explained_variance_ratio_.sum()),
                     "nearest_source_distance_mean": float(np.mean(distance)),
                     "nearest_source_distance_median": float(np.median(distance)),
                     "distance_absolute_error_spearman_rho": float(rho),
                     "distance_absolute_error_spearman_p": float(pvalue),
                     "zero_shot_MAE": float(absolute_error.mean()),
                     "note": "AUROC describes domain separability, not biomass accuracy"})
        for domain_name, pc in [("source", source_pc[:2000]), ("target", target_pc[:2000])]:
            pca_rows.extend({"representation": rep, "domain": domain_name,
                             "PC1": float(a), "PC2": float(b)} for a, b in pc)
    table = pd.DataFrame(rows)
    out = ROOT / "outputs/tables/diagnostics/radar_domain_shift.csv"
    out.parent.mkdir(parents=True, exist_ok=True); table.to_csv(out, index=False)
    pd.DataFrame(pca_rows).to_csv(ROOT / "outputs/tables/diagnostics/radar_domain_pca_points.csv", index=False)
    fig.suptitle("C-band/L-band representation domain shift (descriptive only)")
    fig.tight_layout(); fig.savefig(ROOT / "figures/radar_domain_shift.png", dpi=220); plt.close(fig)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
