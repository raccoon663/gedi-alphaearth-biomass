"""Freeze Kaihua 5-km blocks, spatial folds, label draws and model protocol."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

CRS = "EPSG:32650"
GRID_M = 5000
FOLDS = 5
FOLD_SEED = 42
BUDGETS = [25, 50, 100, 250, 500, 1000, 2500]
SEEDS = [42, 43, 44]
MODEL_PROTOCOL = {
    "model": "XGBRegressor",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 10.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": 6,
    "random_state": "label_draw_seed",
    "decision_basis": "Conservative fixed configuration frozen before representation results; chosen for 25/50-label stability, with no target hyperparameter tuning.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    evaluation_path = root / "data/processed/kaihua_zero_shot_evaluation.parquet"
    freeze_path = root / "outputs/manifests/kaihua_zero_shot_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    for branch, filename in [("alphaearth", "kaihua_zero_shot_aef.csv"),
                             ("conventional", "kaihua_zero_shot_conventional.csv")]:
        path = root / "outputs/predictions" / filename
        if sha256(path) != freeze["prediction_file_hashes"][branch]:
            raise RuntimeError(f"HARD STOP: frozen {branch} zero-shot hash mismatch")
    # Explicitly load metadata only; labels are irrelevant to block/fold/draw design.
    data = pd.read_parquet(evaluation_path, columns=["shot_number", "year", "lon", "lat"])
    data["shot_number"] = data.shot_number.astype("string")
    data = data.sort_values("shot_number").reset_index(drop=True)
    if len(data) != freeze["target_common_n"] or data.shot_number.duplicated().any():
        raise RuntimeError("Frozen target common metadata mismatch")

    transformer = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    east, north = transformer.transform(data.lon.to_numpy(), data.lat.to_numpy())
    data["easting_m"] = east
    data["northing_m"] = north
    data["block_e_index"] = np.floor(data.easting_m / GRID_M).astype(np.int64)
    data["block_n_index"] = np.floor(data.northing_m / GRID_M).astype(np.int64)
    data["target_block_id"] = ("E" + data.block_e_index.astype(str) +
                               "_N" + data.block_n_index.astype(str))

    blocks = (data.groupby(["target_block_id", "block_e_index", "block_n_index"], as_index=False)
              .agg(sample_n=("shot_number", "size")))
    blocks["tie_hash"] = blocks.target_block_id.map(
        lambda x: hashlib.sha256(f"{FOLD_SEED}|{x}".encode()).hexdigest())
    blocks = blocks.sort_values(["sample_n", "tie_hash"], ascending=[False, True]).reset_index(drop=True)
    totals = [0] * FOLDS
    fold_map = {}
    for row in blocks.itertuples(index=False):
        fold = min(range(FOLDS), key=lambda f: (totals[f], f))
        fold_map[row.target_block_id] = fold
        totals[fold] += int(row.sample_n)
    blocks["spatial_fold"] = blocks.target_block_id.map(fold_map).astype(int)
    data["spatial_fold"] = data.target_block_id.map(fold_map).astype(int)
    if data.groupby("target_block_id").spatial_fold.nunique().max() != 1:
        raise RuntimeError("Block split across folds")

    blocks["e_min_m"] = blocks.block_e_index * GRID_M
    blocks["n_min_m"] = blocks.block_n_index * GRID_M
    blocks["e_max_m"] = blocks.e_min_m + GRID_M
    blocks["n_max_m"] = blocks.n_min_m + GRID_M
    blocks["geometry_wkt"] = blocks.apply(
        lambda r: (f"POLYGON (({r.e_min_m} {r.n_min_m}, {r.e_max_m} {r.n_min_m}, "
                   f"{r.e_max_m} {r.n_max_m}, {r.e_min_m} {r.n_max_m}, "
                   f"{r.e_min_m} {r.n_min_m}))"), axis=1)
    blocks["crs"] = CRS
    blocks["grid_size_m"] = GRID_M
    blocks["grid_origin_e_m"] = 0
    blocks["grid_origin_n_m"] = 0
    blocks["fold_seed"] = FOLD_SEED
    blocks["fold_algorithm"] = "sample-count-balanced greedy whole-block assignment; descending block N; SHA256(seed|block_id) tie break"
    blocks = blocks.sort_values("target_block_id").reset_index(drop=True)

    manifests = root / "outputs/manifests"
    block_path = manifests / "kaihua_target_spatial_blocks.csv"
    fold_path = manifests / "kaihua_fewshot_spatial_folds.csv"
    draw_path = manifests / "kaihua_fewshot_label_draws.csv"
    blocks.drop(columns="tie_hash").to_csv(block_path, index=False)
    folds = data[["shot_number", "year", "lon", "lat", "target_block_id", "spatial_fold"]]
    folds.to_csv(fold_path, index=False)

    draw_frames = []
    for fold in range(FOLDS):
        pool = data.loc[data.spatial_fold != fold, "shot_number"].sort_values().to_numpy()
        for seed in SEEDS:
            rng = np.random.default_rng(seed + fold * 100_000)
            order = rng.permutation(len(pool))
            for budget in BUDGETS:
                if budget > len(pool):
                    continue
                selected = pool[order[:budget]]
                draw_frames.append(pd.DataFrame({"fold": fold, "budget": budget,
                                                 "seed": seed, "shot_number": selected,
                                                 "representation_independent_inclusion": True}))
    draws = pd.concat(draw_frames, ignore_index=True)
    expected = sum(BUDGETS) * len(SEEDS) * FOLDS
    if len(draws) != expected:
        raise RuntimeError(f"Label draw count mismatch: {len(draws)} != {expected}")
    draws.to_csv(draw_path, index=False)

    protocol_path = manifests / "kaihua_fewshot_design_freeze.json"
    protocol = {
        "status": "KAIHUA_FEWSHOT_DESIGN_FROZEN_BEFORE_REPRESENTATION_RESULTS",
        "frozen_at": datetime.now().astimezone().isoformat(),
        "input_evaluation_sha256": sha256(evaluation_path),
        "input_n": len(data),
        "target_block_crs": CRS,
        "target_block_size_m": GRID_M,
        "target_block_n": len(blocks),
        "folds": FOLDS,
        "fold_seed": FOLD_SEED,
        "fold_sample_n": {str(i): int((data.spatial_fold == i).sum()) for i in range(FOLDS)},
        "fold_assignment_uses_labels_or_errors": False,
        "budgets": BUDGETS,
        "label_draw_seeds": SEEDS,
        "draws_are_nested_within_fold_seed": True,
        "draws_use_agbd_stratification_or_errors": False,
        "common_model_protocol": MODEL_PROTOCOL,
        "block_manifest_sha256": sha256(block_path),
        "fold_manifest_sha256": sha256(fold_path),
        "label_draw_manifest_sha256": sha256(draw_path),
        "zero_shot_prediction_hashes_verified": True,
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
