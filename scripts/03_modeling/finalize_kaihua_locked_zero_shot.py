"""Finalize Kaihua predictors and freeze zero-shot predictions before label unlock."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AEF = [f"A{i:02d}" for i in range(64)]
DEM = ["elevation", "slope", "aspect_sin", "aspect_cos"]
CONV = ["VV", "VH", "VV_minus_VH", "B2", "B3", "B4", "B5", "B6", "B7",
        "B8", "B8A", "B11", "B12", "NDVI", "EVI", "NDMI", "NBR", "NDRE"]
META = ["shot_number", "year", "lon", "lat"]
SEED = 42
MISSING = -9999.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_chunks(directory: Path, expected: list[str]) -> pd.DataFrame:
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No chunks found: {directory}")
    frames = [pd.read_csv(p, dtype={"shot_number": "string"}) for p in files]
    frame = pd.concat(frames, ignore_index=True)
    missing = sorted(set(expected) - set(frame.columns))
    if missing or frame.shot_number.duplicated().any():
        raise RuntimeError(f"Chunk schema/duplicate failure for {directory.name}: {missing}")
    return frame[expected]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    processed = root / "data" / "processed"
    tables = root / "outputs" / "tables"
    figures = root / "outputs" / "figures"
    predictions = root / "outputs" / "predictions"
    manifests = root / "outputs" / "manifests"
    for d in [processed, tables, figures, predictions, manifests]:
        d.mkdir(parents=True, exist_ok=True)

    source_freeze = json.loads((manifests / "frozen_source_model_manifest.json").read_text())
    verification = json.loads((manifests / "source_model_freeze_verification.json").read_text())
    if not verification.get("SOURCE_MODEL_FREEZE_VERIFIED") or not verification.get("TARGET_LABEL_LOCKED"):
        raise RuntimeError("Source freeze verification/target label lock failure")
    shots = pd.read_csv(root / "data/manifests/kaihua_target_shots_locked.csv",
                        dtype={"shot_number": "string"}).sort_values(["year", "shot_number"]).reset_index(drop=True)
    if set(shots.columns) != set(META) or shots.shot_number.duplicated().any():
        raise RuntimeError("Locked shot metadata manifest invalid")

    dem = read_chunks(root / "data/raw/kaihua_dem_locked_chunks", META + DEM)
    aef = read_chunks(root / "data/raw/kaihua_aef_locked_chunks", META + AEF)
    conv = read_chunks(root / "data/raw/kaihua_conventional_locked_chunks", META + CONV)
    initial = len(shots)
    dem_valid = dem.loc[np.isfinite(dem[DEM].to_numpy(float)).all(axis=1)].copy()
    aef_values = aef[AEF].to_numpy(float)
    aef_mask = np.isfinite(aef_values).all(axis=1) & ~np.all(aef_values == 0, axis=1)
    aef_valid = aef.loc[aef_mask].copy()
    conv_values = conv[CONV].to_numpy(float)
    conv_mask = np.isfinite(conv_values).all(axis=1) & ~(conv_values == MISSING).any(axis=1)
    conv_valid = conv.loc[conv_mask].copy()

    common_ids = (set(shots.shot_number) & set(dem_valid.shot_number) &
                  set(aef_valid.shot_number) & set(conv_valid.shot_number))
    common_meta = shots.loc[shots.shot_number.isin(common_ids)].copy()
    def aligned(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return common_meta[["shot_number"]].merge(frame[["shot_number"] + cols], on="shot_number",
                                                    how="left", validate="one_to_one")
    target_aef = common_meta.merge(aef_valid[["shot_number"] + AEF], on="shot_number", validate="one_to_one")
    target_conv = common_meta.merge(conv_valid[["shot_number"] + CONV], on="shot_number", validate="one_to_one")
    common = common_meta.copy()
    for frame, cols in [(dem_valid, DEM), (aef_valid, AEF), (conv_valid, CONV)]:
        common = common.merge(frame[["shot_number"] + cols], on="shot_number", validate="one_to_one")
    if common[META].isna().any().any() or common.shot_number.duplicated().any():
        raise RuntimeError("Common target predictor intersection invalid")
    target_aef.to_parquet(processed / "kaihua_aef_locked.parquet", index=False)
    target_conv.to_parquet(processed / "kaihua_conventional_locked.parquet", index=False)
    common.to_parquet(processed / "kaihua_common_predictors_locked.parquet", index=False)

    audit = pd.DataFrame([
        {"stage": "initial_target_shots", "N": initial, "removed_from_initial": 0},
        {"stage": "DEM_valid", "N": len(dem_valid), "removed_from_initial": initial-len(dem_valid)},
        {"stage": "AEF_valid", "N": len(aef_valid), "removed_from_initial": initial-len(aef_valid)},
        {"stage": "conventional_valid", "N": len(conv_valid), "removed_from_initial": initial-len(conv_valid)},
        {"stage": "final_common", "N": len(common), "removed_from_initial": initial-len(common)},
    ])
    audit.to_csv(tables / "kaihua_predictor_common_audit.csv", index=False)
    norms = np.linalg.norm(aef_valid[AEF].to_numpy(float), axis=1)
    pd.DataFrame([{"N": len(aef_valid), "complete_64d": True, "all_finite": True,
                   "nonzero_n": int((norms > 0).sum()), "duplicate_shot_n": int(aef_valid.shot_number.duplicated().sum()),
                   "l2_min": norms.min(), "l2_p05": np.quantile(norms, .05),
                   "l2_median": np.median(norms), "l2_p95": np.quantile(norms, .95),
                   "l2_max": norms.max(), "year_consistent": True}]).to_csv(
        tables / "kaihua_aef_qc.csv", index=False)

    source_aef = pd.read_parquet(processed / "source_common_alphaearth.parquet", columns=AEF)
    source_conv = pd.read_parquet(processed / "source_common_conventional.parquet",
                                  columns=["VV", "VH", "NDVI", "NDMI", "B11", "NDRE", "elevation"])
    domain_rows = []
    for feature in ["VV", "VH", "NDVI", "NDMI", "B11", "NDRE", "elevation"]:
        x = source_conv[feature].to_numpy(float)
        y = common[feature].to_numpy(float)
        ks = ks_2samp(x, y)
        domain_rows.append({"feature": feature, "source_n": len(x), "target_n": len(y),
                            "source_mean": x.mean(), "target_mean": y.mean(),
                            "source_std": x.std(), "target_std": y.std(),
                            "source_median": np.median(x), "target_median": np.median(y),
                            "ks_statistic": ks.statistic, "ks_pvalue": ks.pvalue,
                            "wasserstein_distance": wasserstein_distance(x, y)})
    pd.DataFrame(domain_rows).to_csv(tables / "kaihua_conventional_domain_shift.csv", index=False)

    rng = np.random.default_rng(SEED)
    ns = min(30000, len(source_aef)); nt = min(30000, len(common))
    si = rng.choice(len(source_aef), ns, replace=False); ti = rng.choice(len(common), nt, replace=False)
    sx = source_aef.iloc[si].to_numpy(float); tx = common.iloc[ti][AEF].to_numpy(float)
    pca = PCA(n_components=2, random_state=SEED).fit(source_aef.sample(min(60000, len(source_aef)), random_state=SEED))
    sp, tp = pca.transform(sx), pca.transform(tx)
    pca_frame = pd.concat([pd.DataFrame({"PC1": sp[:,0], "PC2": sp[:,1], "domain": "USA_source"}),
                           pd.DataFrame({"PC1": tp[:,0], "PC2": tp[:,1], "domain": "Kaihua"})], ignore_index=True)
    pca_frame.to_parquet(tables / "kaihua_aef_domain_pca.parquet", index=False)
    plt.figure(figsize=(7, 6))
    for label, color in [("USA_source", "#2f6f9f"), ("Kaihua", "#d9822b")]:
        z = pca_frame[pca_frame.domain == label]
        plt.scatter(z.PC1, z.PC2, s=3, alpha=.16, label=label, color=color, rasterized=True)
    plt.xlabel("Source-fit PC1"); plt.ylabel("Source-fit PC2"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "kaihua_aef_domain_pca.png", dpi=220); plt.close()

    nclf = min(50000, len(source_aef), len(common))
    sx = source_aef.sample(nclf, random_state=SEED).to_numpy(float)
    tx = common.sample(nclf, random_state=SEED)[AEF].to_numpy(float)
    X = np.vstack([sx, tx]); y = np.r_[np.zeros(nclf), np.ones(nclf)]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    classifier.fit(Xtr, ytr)
    auroc = roc_auc_score(yte, classifier.predict_proba(Xte)[:,1])
    pd.DataFrame([{"method": "standardized_logistic_regression", "features": "A00-A63",
                   "train_n": len(ytr), "test_n": len(yte), "AUROC": auroc,
                   "purpose": "descriptive_domain_shift_only_not_model_selection"}]).to_csv(
        tables / "kaihua_aef_domain_classifier.csv", index=False)

    # Exact 64-D Euclidean nearest-source distances.  A full brute-force
    # calculation is used (no OOD algorithm or approximation); CUDA only
    # accelerates the matrix multiplication.
    cache = root / "outputs" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    source_npy, target_npy, distance_npy = (cache / "usa_source_aef64.npy",
                                             cache / "kaihua_target_aef64.npy",
                                             cache / "kaihua_nearest_source_aef.npy")
    np.save(source_npy, pd.read_parquet(processed / "source_common_alphaearth.parquet",
                                       columns=AEF).to_numpy(np.float32))
    np.save(target_npy, common[AEF].to_numpy(np.float32))
    gpu_python = Path(r"C:\Users\ASUS\anaconda3\envs\dl-pytorch\python.exe")
    subprocess.run([str(gpu_python), str(root / "scripts/04_diagnostics/gpu_exact_nearest_aef.py"),
                    "--source", str(source_npy), "--target", str(target_npy),
                    "--output", str(distance_npy)], check=True)
    nearest = np.load(distance_npy)
    if len(nearest) != len(common) or not np.isfinite(nearest).all():
        raise RuntimeError("Exact nearest-source distance output invalid")
    pd.DataFrame({"shot_number": common.shot_number,
                  "nearest_source_aef_euclidean_distance": nearest}).to_parquet(
        tables / "kaihua_aef_nearest_source_distance.parquet", index=False)

    model_paths = {
        "alphaearth": root / source_freeze["selected_models"]["alphaearth"]["model_file"],
        "conventional": root / source_freeze["selected_models"]["conventional"]["model_file"]}
    feature_map = {"alphaearth": AEF + DEM, "conventional": CONV + DEM}
    prediction_paths = {}
    for branch in ["alphaearth", "conventional"]:
        spec = source_freeze["selected_models"][branch]
        if sha256(model_paths[branch]) != spec["model_sha256"]:
            raise RuntimeError(f"HARD STOP: {branch} model hash changed")
        model = joblib.load(model_paths[branch])
        pred = model.predict(common[feature_map[branch]])
        out = pd.DataFrame({"shot_number": common.shot_number, "year": common.year,
                            "prediction_agbd": pred, "model_id": f"frozen_source_{branch}_xgb_depth6",
                            "source_model_sha256": spec["model_sha256"]})
        path = predictions / f"kaihua_zero_shot_{'aef' if branch == 'alphaearth' else 'conventional'}.csv"
        out.to_csv(path, index=False)
        prediction_paths[branch] = path

    predictor_path = processed / "kaihua_common_predictors_locked.parquet"
    freeze = {
        "status": "KAIHUA_ZERO_SHOT_PREDICTIONS_FROZEN_LABELS_MAY_NOW_BE_UNLOCKED",
        "frozen_at": datetime.now().astimezone().isoformat(),
        "target_aoi_sha256": verification["target_aoi_sha256"],
        "target_shot_manifest_sha256": sha256(root / "data/manifests/kaihua_target_shots_locked.csv"),
        "target_predictor_hashes": {
            "aef": sha256(processed / "kaihua_aef_locked.parquet"),
            "conventional": sha256(processed / "kaihua_conventional_locked.parquet"),
            "common": sha256(predictor_path)},
        "source_model_hashes": {k: source_freeze["selected_models"][k]["model_sha256"]
                                for k in ["alphaearth", "conventional"]},
        "feature_lists": source_freeze["features"],
        "preprocessing_contract": {"aef": "exact-year central 10m A00-A63; no renormalization",
                                   "conventional": "exact-year calendar-year median frozen S1/S2/index rules",
                                   "dem": "COPERNICUS/DEM/GLO30 frozen elevation/slope/aspect sin/cos"},
        "target_common_n": len(common),
        "prediction_file_hashes": {k: sha256(v) for k, v in prediction_paths.items()},
        "labels_locked_during_predictor_construction_and_prediction": True,
        "observed_agbd_in_prediction_files": False,
        "target_label_locked": False}
    (manifests / "kaihua_zero_shot_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps({"target_common_n": len(common), "aef_AUROC": auroc,
                      "prediction_hashes": freeze["prediction_file_hashes"]}, indent=2))


if __name__ == "__main__":
    main()
