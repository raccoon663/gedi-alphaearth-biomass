"""Two-phase zero-shot runner: freeze predictions first, then unlock labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from utilities.guards import assert_safe_predictors, verify_frozen_prediction  # noqa: E402
from utilities.radar_benchmark import REPRESENTATIONS, canonical_hash  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return {"R2": float(r2_score(y, pred)), "RMSE": float(mean_squared_error(y, pred) ** .5),
            "MAE": float(mean_absolute_error(y, pred)), "Bias": float(np.mean(pred-y)), "N": len(y)}


def ordered_target_frames(common_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    expected = None
    for name, features in REPRESENTATIONS.items():
        assert_safe_predictors(features)
        data = pd.read_parquet(common_dir / f"target_{name}.parquet")
        if "agbd" in data:
            raise RuntimeError("HARD STOP: target AGBD present during zero-shot prediction")
        data["shot_number"] = data.shot_number.astype("string")
        data = data.sort_values(["year", "shot_number"]).reset_index(drop=True)
        if data.duplicated(["shot_number", "year"]).any() or data[features].isna().any().any():
            raise RuntimeError(f"Invalid locked target frame: {name}")
        keys = data[["shot_number", "year"]]
        if expected is not None and not keys.equals(expected):
            raise RuntimeError(f"Target ordered rows differ: {name}")
        expected = keys
        frames[name] = data
    return frames


def predict(common_dir: Path) -> None:
    source_freeze_path = ROOT / "outputs/manifests/palsar_source_model_freeze.json"
    source = json.loads(source_freeze_path.read_text())
    if source["status"] != "PALSAR_COMMON_SOURCE_MODELS_FROZEN" or source.get("quick_smoke"):
        raise RuntimeError("Full source models must be frozen before zero-shot prediction")
    frames = ordered_target_frames(common_dir)
    out_dir = ROOT / "outputs/predictions/palsar_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, data in frames.items():
        model_info = source["representations"][name]
        model_path = ROOT / model_info["model_file"]
        if sha256(model_path) != model_info["model_sha256"]:
            raise RuntimeError(f"Frozen source model changed: {name}")
        estimator = joblib.load(model_path)
        prediction = estimator.predict(data[REPRESENTATIONS[name]])
        path = out_dir / f"target_{name}.csv"
        pd.DataFrame({"shot_number": data.shot_number, "year": data.year,
                      "prediction": prediction}).to_csv(path, index=False)
        hashes[name] = sha256(path)
    first = frames[next(iter(frames))]
    freeze = {"status": "PALSAR_ZERO_SHOT_PREDICTIONS_FROZEN_BEFORE_LABEL_UNLOCK",
              "frozen_at": datetime.now(timezone.utc).isoformat(),
              "source_model_freeze_sha256": sha256(source_freeze_path),
              "target_n": len(first), "target_ordered_row_hash": canonical_hash(first, ["shot_number", "year"]),
              "feature_schemas": REPRESENTATIONS, "prediction_file_hashes": hashes,
              "target_labels_loaded": False, "target_metrics_used": False}
    path = ROOT / "outputs/manifests/palsar_zero_shot_freeze.json"
    path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps(freeze, indent=2))


def evaluate(labels_path: Path) -> None:
    freeze_path = ROOT / "outputs/manifests/palsar_zero_shot_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze["status"] != "PALSAR_ZERO_SHOT_PREDICTIONS_FROZEN_BEFORE_LABEL_UNLOCK":
        raise RuntimeError("Prediction freeze is missing")
    labels = pd.read_parquet(labels_path) if labels_path.suffix == ".parquet" else pd.read_csv(labels_path)
    labels["shot_number"] = labels.shot_number.astype("string")
    if not {"shot_number", "year", "agbd"} <= set(labels) or labels.duplicated(["shot_number", "year"]).any():
        raise RuntimeError("Invalid unlocked target labels")
    rows = []
    pred_dir = ROOT / "outputs/predictions/palsar_benchmark"
    for name in REPRESENTATIONS:
        path = pred_dir / f"target_{name}.csv"
        verify_frozen_prediction(path, freeze["prediction_file_hashes"][name])
        pred = pd.read_csv(path, dtype={"shot_number": "string"})
        joined = pred.merge(labels[["shot_number", "year", "agbd"]], on=["shot_number", "year"],
                            how="left", validate="one_to_one")
        if joined.agbd.isna().any() or len(joined) != freeze["target_n"]:
            raise RuntimeError(f"Target label join failure: {name}")
        rows.append({"representation": name, **metrics(joined.agbd, joined.prediction),
                     "prediction_sha256": freeze["prediction_file_hashes"][name]})
    out = ROOT / "outputs/tables/main_results/palsar_zero_shot.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    evaluation = {"status": "PALSAR_ZERO_SHOT_EVALUATED_AFTER_LABEL_UNLOCK",
                  "evaluated_at": datetime.now(timezone.utc).isoformat(),
                  "prediction_freeze_sha256": sha256(freeze_path),
                  "label_file_sha256": sha256(labels_path), "result_sha256": sha256(out)}
    (ROOT / "outputs/manifests/palsar_zero_shot_evaluation.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("predict")
    freeze.add_argument("--common-dir", type=Path, default=ROOT / "data/processed/palsar_common")
    unlock = sub.add_parser("evaluate")
    unlock.add_argument("--labels", type=Path, required=True)
    args = p.parse_args()
    predict(args.common_dir) if args.command == "predict" else evaluate(args.labels)


if __name__ == "__main__":
    main()
