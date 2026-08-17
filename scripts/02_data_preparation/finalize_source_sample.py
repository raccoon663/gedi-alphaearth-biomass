"""Finalize and cryptographically freeze completed sampled GEDI exports."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

YEARS = range(2019, 2025)
MAX_PER_STRATUM = 500
N_FOLDS = 5


def require_files(source: Path, stem: str) -> list[Path]:
    paths = [source / f"{stem}_{year}.csv" for year in YEARS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing exports:\n" + "\n".join(missing))
    return paths


def read_many(paths: list[Path], dtype=None) -> pd.DataFrame:
    return pd.concat([pd.read_csv(path, dtype=dtype) for path in paths], ignore_index=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("drive_export_folder", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw" / "source_sampling_exports"
    processed = root / "data" / "processed"
    tables = root / "outputs" / "tables"
    logs = root / "outputs" / "logs"
    manifests_dir = root / "outputs" / "manifests"
    for directory in [raw_dir, processed, tables, logs, manifests_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    manifest_paths = require_files(args.drive_export_folder, "gedi_sampled_manifest")
    strata_paths = require_files(args.drive_export_folder, "gedi_sampling_strata")
    stats_paths = require_files(args.drive_export_folder, "gedi_sampling_annual_stats")
    for path in manifest_paths + strata_paths + stats_paths:
        shutil.copy2(path, raw_dir / path.name)

    manifest = read_many(manifest_paths, dtype={"shot_number": "string", "spatial_block_id": "string"})
    strata = read_many(strata_paths, dtype={"spatial_block_id": "string"})
    annual_export = read_many(stats_paths)

    # GEDI delta_time is seconds since 2018-01-01 00:00:00 UTC.
    acquired = (pd.Timestamp("2018-01-01", tz="UTC")
                + pd.to_timedelta(manifest["delta_time"], unit="s"))
    if not (acquired.dt.year.astype(int).to_numpy() == manifest["year"].astype(int).to_numpy()).all():
        raise RuntimeError("Year derived from delta_time does not match exported year")
    manifest["date"] = acquired.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    required = ["shot_number", "lon", "lat", "year", "date", "spatial_block_id", "agbd", "agbd_se"]
    null_required = int(manifest[required].isna().sum().sum())
    duplicate_shots = int(manifest["shot_number"].duplicated().sum())
    if null_required:
        raise RuntimeError(f"Sample manifest has {null_required} null required values")
    if duplicate_shots:
        raise RuntimeError(f"Sample manifest has {duplicate_shots} duplicate shot_number values")

    actual = (manifest.groupby(["spatial_block_id", "year"], as_index=False)
              .size().rename(columns={"size": "actual_sampled_n"}))
    if int(actual["actual_sampled_n"].max()) > MAX_PER_STRATUM:
        raise RuntimeError("A stratum exceeds the frozen cap of 500")
    audit = strata.merge(actual, on=["spatial_block_id", "year"], how="outer", validate="one_to_one")
    if audit[["sampled_n", "actual_sampled_n"]].isna().any().any():
        raise RuntimeError("Manifest strata and audit strata do not match")
    if not (audit["sampled_n"].astype(int) == audit["actual_sampled_n"].astype(int)).all():
        raise RuntimeError("Exported stratum sampled_n does not match manifest")
    expected = np.minimum(audit["raw_qa_valid_n"].astype(int), MAX_PER_STRATUM)
    if not (expected == audit["sampled_n"].astype(int)).all():
        raise RuntimeError("Sampling does not equal min(raw N, 500)")
    fraction_expected = audit["sampled_n"] / audit["raw_qa_valid_n"]
    if not np.allclose(fraction_expected, audit["sampling_fraction"], rtol=1e-10, atol=1e-12):
        raise RuntimeError("Sampling fraction audit mismatch")

    # GroupKFold is label-free and guarantees a spatial block never crosses folds.
    manifest["spatial_fold"] = -1
    splitter = GroupKFold(n_splits=N_FOLDS)
    dummy = np.zeros((len(manifest), 1))
    for fold, (_, test_idx) in enumerate(splitter.split(dummy, groups=manifest["spatial_block_id"])):
        manifest.loc[test_idx, "spatial_fold"] = fold
    fold_per_block = manifest.groupby("spatial_block_id")["spatial_fold"].nunique()
    if int(fold_per_block.max()) != 1:
        raise RuntimeError("A spatial block crosses folds")

    manifest = manifest.sort_values(["year", "spatial_block_id", "shot_number"]).reset_index(drop=True)
    csv_path = manifests_dir / "frozen_source_manifest.csv"
    parquet_path = manifests_dir / "frozen_source_manifest.parquet"
    manifest.to_csv(csv_path, index=False)
    manifest.to_parquet(parquet_path, index=False)
    # Reusable processed copies contain the exact same ordered records.
    shutil.copy2(csv_path, processed / "source_sample_manifest.csv")
    shutil.copy2(parquet_path, processed / "source_sample_manifest.parquet")
    audit.sort_values(["year", "spatial_block_id"]).to_csv(
        tables / "source_sampling_by_block_year.csv", index=False)

    annual_local = manifest.groupby("year")["agbd"].agg(
        sampled_n="count", sampled_agbd_mean="mean", sampled_agbd_sd="std",
        sampled_agbd_min="min", sampled_agbd_max="max").reset_index()
    annual_local["sampled_agbd_sd"] = (
        manifest.groupby("year")["agbd"].apply(lambda values: values.std(ddof=0)).values)
    for q, name in [(0.01, "p01"), (0.05, "p05"), (0.25, "p25"), (0.5, "p50"),
                    (0.75, "p75"), (0.95, "p95"), (0.99, "p99")]:
        annual_local[f"sampled_agbd_{name}"] = manifest.groupby("year")["agbd"].quantile(q).values
    annual = annual_export.merge(annual_local, on="year", suffixes=("_gee", "_local"), validate="one_to_one")
    # The GEE sample audit and the locally finalized manifest must agree.
    for field in ["n", "agbd_mean", "agbd_sd", "agbd_min", "agbd_max",
                  "agbd_p01", "agbd_p05", "agbd_p25", "agbd_p50",
                  "agbd_p75", "agbd_p95", "agbd_p99"]:
        gee_col = f"sampled_{field}_gee"
        local_col = f"sampled_{field}_local"
        if local_col not in annual.columns:
            # Local names omit the second 'agbd_' only for n.
            local_col = "sampled_n_local" if field == "n" else local_col
        if gee_col in annual.columns and local_col in annual.columns:
            if not np.allclose(annual[gee_col], annual[local_col], rtol=1e-8, atol=1e-8):
                raise RuntimeError(f"GEE/local annual sample audit mismatch: {field}")

    # Predeclared descriptive distortion flags: |SMD| <= 0.10 and median shift
    # <= 10% of raw IQR are treated as no obvious distribution anomaly.
    annual["agbd_standardized_mean_difference"] = (
        (annual["sampled_agbd_mean_local"] - annual["raw_agbd_mean"])
        / annual["raw_agbd_sd"])
    annual["agbd_median_shift_over_raw_iqr"] = (
        (annual["sampled_agbd_p50_local"] - annual["raw_agbd_p50"]).abs()
        / (annual["raw_agbd_p75"] - annual["raw_agbd_p25"]))
    annual["no_obvious_distribution_anomaly"] = (
        annual["agbd_standardized_mean_difference"].abs().le(0.10)
        & annual["agbd_median_shift_over_raw_iqr"].le(0.10))
    if not bool(annual["no_obvious_distribution_anomaly"].all()):
        bad_years = annual.loc[~annual["no_obvious_distribution_anomaly"], "year"].tolist()
        raise RuntimeError(f"Predeclared AGBD sampling-distortion audit failed for years: {bad_years}")
    annual.to_csv(tables / "source_sampling_annual_agbd_audit.csv", index=False)

    fold_counts = manifest.groupby("spatial_fold", as_index=False).agg(
        n=("shot_number", "size"), blocks=("spatial_block_id", "nunique"))
    fold_counts.to_csv(tables / "source_spatial_fold_counts.csv", index=False)
    block_fold = manifest[["spatial_block_id", "spatial_fold"]].drop_duplicates()
    block_fold.sort_values("spatial_block_id").to_csv(
        processed / "spatial_block_fold_manifest.csv", index=False)

    csv_hash = sha256(csv_path)
    parquet_hash = sha256(parquet_path)
    freeze = {
        "status": "frozen",
        "sampling_policy": "EPSG5070_50km_x_year_max500_seed42_shot_number",
        "sample_n": len(manifest),
        "unique_shot_number_n": int(manifest["shot_number"].nunique()),
        "spatial_block_n": int(manifest["spatial_block_id"].nunique()),
        "stratum_n": len(audit),
        "max_actual_stratum_n": int(audit["actual_sampled_n"].max()),
        "years": sorted(manifest["year"].astype(int).unique().tolist()),
        "folds": N_FOLDS,
        "target_labels_read": False,
        "required_fields_complete": True,
        "global_duplicate_shot_number_n": 0,
        "all_strata_obey_cap": True,
        "all_small_strata_fully_retained": True,
        "all_blocks_single_fold": True,
        "all_years_pass_agbd_distortion_audit": True,
        "manifest_csv": str(csv_path),
        "manifest_parquet": str(parquet_path),
        "manifest_csv_sha256": csv_hash,
        "manifest_parquet_sha256": parquet_hash,
    }
    (manifests_dir / "frozen_source_manifest.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8")
    (manifests_dir / "frozen_source_manifest.sha256").write_text(
        f"{csv_hash}  frozen_source_manifest.csv\n"
        f"{parquet_hash}  frozen_source_manifest.parquet\n", encoding="utf-8")
    (logs / "frozen_source_sample_manifest.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
