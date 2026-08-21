"""Hard guards against target-label leakage and predictor shortcuts."""
from __future__ import annotations

import json
from pathlib import Path

TARGET_LABEL_LOCKED = True

FORBIDDEN_FEATURE_TOKENS = {
    "agbd", "agbd_se", "shot_number", "lon", "lat", "longitude", "latitude",
    "spatial_block_id", "target_block_id", "spatial_fold", "block_e_index",
    "block_n_index", "palsar_angle", "palsar_epoch", "palsar_acquisition_date",
    "palsar_qa", "qa", "epoch", "angle",
}


def assert_safe_predictors(features: list[str]) -> None:
    """Reject labels, identifiers, coordinates, blocks and PALSAR QC metadata."""
    normalized = {str(x).strip().lower() for x in features}
    bad = sorted(normalized & FORBIDDEN_FEATURE_TOKENS)
    if bad:
        raise RuntimeError(f"HARD STOP: forbidden model predictors: {bad}")


def assert_target_not_used(*, stage: str, target_labels_loaded: bool,
                           target_metrics_used: bool = False) -> None:
    """Protect source selection, tuning, sampling and feature engineering stages."""
    protected = {"source_sampling", "feature_engineering", "representation_selection",
                 "model_tuning", "palsar_aggregation_selection"}
    if stage in protected and (target_labels_loaded or target_metrics_used):
        raise RuntimeError(f"HARD STOP: target information entered protected stage {stage}")


def verify_frozen_prediction(path: Path, expected_sha256: str) -> None:
    """Prevent label-unlock evaluation if a zero-shot file changed after freeze."""
    import hashlib
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"HARD STOP: frozen zero-shot prediction changed: {path}")


def require_target_labels_unlocked(project_root: Path) -> dict:
    """Return the frozen manifest or fail before any target-label read."""
    manifest_path = project_root / "outputs" / "models" / "frozen_source_manifest.json"
    if TARGET_LABEL_LOCKED:
        raise RuntimeError(
            "TARGET_LABEL_LOCKED=True: target AGBD cannot be read during source development. "
            "Use the separate target evaluation stage after freezing source decisions."
        )
    if not manifest_path.exists():
        raise RuntimeError("Target evaluation requires frozen_source_manifest.json")
    return json.loads(manifest_path.read_text(encoding="utf-8"))
