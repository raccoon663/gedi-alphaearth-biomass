"""Merge, validate, and freeze completed source DEM chunk exports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["elevation", "slope", "aspect_sin", "aspect_cos"]


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
    root = Path(__file__).resolve().parents[1]
    logs = root / "outputs" / "logs"
    records = json.loads((logs / "source_dem_export_tasks.json").read_text())
    manifest_path = root / "outputs" / "manifests" / "frozen_source_manifest_200.csv"
    freeze = json.loads((root / "outputs" / "manifests" /
                         "frozen_source_manifest_200.json").read_text())
    if sha256(manifest_path) != freeze["manifest_csv_sha256"]:
        raise RuntimeError("Frozen manifest checksum mismatch")

    manifest = pd.read_csv(manifest_path, dtype={"shot_number": "string",
                                                "spatial_block_id": "string"})
    ordered = manifest.sort_values(
        ["year", "spatial_block_id", "shot_number"]
    ).reset_index(drop=True)
    frames = []
    selected_files = []
    rejected_candidates = []
    for record in records:
        candidates = sorted(args.drive_export_folder.glob(
            f"{record['description']}*.csv"))
        yearly = ordered.loc[ordered["year"].eq(record["year"])].reset_index(drop=True)
        start = int(record["chunk"]) * int(record["chunk_size"])
        expected_shots = set(
            yearly.iloc[start:start + int(record["expected_n"])]
            ["shot_number"].astype(str)
        )
        exact_matches = []
        subset_candidates = []
        for path in candidates:
            candidate = pd.read_csv(path, dtype={"shot_number": "string",
                                                 "spatial_block_id": "string"})
            candidate_shots = set(candidate["shot_number"].astype(str))
            if len(candidate) == int(record["expected_n"]) and candidate_shots == expected_shots:
                exact_matches.append((path, candidate))
            elif candidate_shots.issubset(expected_shots):
                subset_candidates.append((path, candidate, candidate_shots))
            else:
                rejected_candidates.append(str(path))
        if len(exact_matches) == 1:
            selected_files.append(str(exact_matches[0][0]))
            frames.append(exact_matches[0][1])
        elif len(exact_matches) > 1:
            raise RuntimeError(f"Multiple exact candidates for {record['description']}")
        elif subset_candidates:
            # sampleRegions omits points where the DEM is masked. Choose the
            # candidate with maximum valid coverage, requiring no foreign shot.
            subset_candidates.sort(key=lambda item: len(item[2]), reverse=True)
            selected_files.append(str(subset_candidates[0][0]))
            frames.append(subset_candidates[0][1])
            rejected_candidates.extend(str(item[0]) for item in subset_candidates[1:])
        else:
            raise RuntimeError(
                f"No valid Drive candidate for {record['description']} among "
                f"{len(candidates)} files"
            )
    dem = pd.concat(frames, ignore_index=True)
    expected_n = sum(int(item["expected_n"]) for item in records)
    if int(dem["shot_number"].duplicated().sum()):
        raise RuntimeError("Duplicate shot_number values in DEM export")
    expected = manifest[["shot_number", "year", "spatial_block_id", "spatial_fold"]]
    joined = expected.merge(dem, on="shot_number", how="left",
                            suffixes=("_manifest", "_dem"), indicator=True,
                            validate="one_to_one")
    if joined["_merge"].eq("right_only").any():
        raise RuntimeError("DEM contains shots outside the frozen manifest")
    available = joined.loc[joined["_merge"].eq("both")].copy()
    masked = joined.loc[joined["_merge"].eq("left_only")].copy()
    for field in ["year", "spatial_fold"]:
        left = pd.to_numeric(available[f"{field}_manifest"], errors="raise").astype(int)
        right = pd.to_numeric(available[f"{field}_dem"], errors="raise").astype(int)
        if not np.array_equal(left.to_numpy(), right.to_numpy()):
            raise RuntimeError(f"DEM {field} differs from frozen manifest")
    if not np.array_equal(
        available["spatial_block_id_manifest"].astype(str).to_numpy(),
        available["spatial_block_id_dem"].astype(str).to_numpy(),
    ):
        raise RuntimeError("DEM spatial_block_id differs from frozen manifest")
    values = available[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("DEM features contain missing or non-finite values")
    if ((available["aspect_sin"].abs() > 1.000001) |
            (available["aspect_cos"].abs() > 1.000001)).any():
        raise RuntimeError("Aspect sin/cos outside [-1, 1]")

    output = available[["shot_number", *FEATURES]].sort_values("shot_number")
    masked_output = masked[["shot_number", "year_manifest", "spatial_block_id_manifest",
                            "spatial_fold_manifest"]].rename(columns={
        "year_manifest": "year", "spatial_block_id_manifest": "spatial_block_id",
        "spatial_fold_manifest": "spatial_fold",
    }).sort_values("shot_number")
    masked_output["missing_reason"] = "DEM_MASKED"
    processed = root / "data" / "processed"
    tables = root / "outputs" / "tables"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    csv_path = processed / "source_dem_features.csv"
    parquet_path = processed / "source_dem_features.parquet"
    output.to_csv(csv_path, index=False)
    output.to_parquet(parquet_path, index=False)
    masked_path = tables / "source_dem_masked_shots.csv"
    masked_output.to_csv(masked_path, index=False)
    audit = {
        "status": "frozen_with_documented_mask", "manifest_n": expected_n,
        "sample_n": len(output), "dem_masked_n": len(masked_output),
        "dem_availability_fraction": len(output) / expected_n,
        "unique_shot_number_n": int(output["shot_number"].nunique()),
        "manifest_csv_sha256": freeze["manifest_csv_sha256"],
        "features": FEATURES, "required_fields_complete": True,
        "dem_shots_are_strict_manifest_subset": True,
        "missing_shots_documented_for_common_intersection": True,
        "duplicate_shot_number_n": 0,
        "selected_chunk_file_n": len(selected_files),
        "rejected_obsolete_candidate_file_n": len(set(rejected_candidates)),
        "csv_sha256": sha256(csv_path),
        "parquet_sha256": sha256(parquet_path),
        "masked_manifest_sha256": sha256(masked_path),
    }
    (tables / "source_dem_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    (processed / "source_dem_features.sha256").write_text(
        f"{audit['csv_sha256']}  {csv_path.name}\n"
        f"{audit['parquet_sha256']}  {parquet_path.name}\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
