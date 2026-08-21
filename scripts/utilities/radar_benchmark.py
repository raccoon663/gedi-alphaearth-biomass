"""Shared definitions and integrity checks for the PALSAR-common benchmark.

This module intentionally contains no target-model selection logic.  It defines the
frozen predictor schemas and checks used by extraction, preparation and modelling.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

SEED = 42
ID_COLUMNS = ["shot_number", "year"]
SOURCE_BLOCK = "spatial_block_id"
TARGET_BLOCK = "target_block_id"
FOLD = "spatial_fold"
RESPONSE = "agbd"

DEM = ["elevation", "slope", "aspect_sin", "aspect_cos"]
S1 = ["VV", "VH", "VV_minus_VH"]
S2 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_INDICES = ["NDVI", "EVI", "NDMI", "NBR", "NDRE"]
PALSAR = ["palsar_hh_db", "palsar_hv_db", "palsar_hh_minus_hv_db", "palsar_rfdi"]
ALPHAEARTH = [f"A{i:02d}" for i in range(64)]
PALSAR_QC = ["palsar_angle", "palsar_epoch", "palsar_acquisition_date", "palsar_qa"]

REPRESENTATIONS = {
    "dem": DEM,
    "sentinel1_c": S1 + DEM,
    "palsar_l": PALSAR + DEM,
    "sentinel1_c_palsar_l": S1 + PALSAR + DEM,
    "sentinel1_sentinel2": S1 + S2 + S2_INDICES + DEM,
    "palsar_sentinel2": PALSAR + S2 + S2_INDICES + DEM,
    "sentinel1_palsar_sentinel2": S1 + PALSAR + S2 + S2_INDICES + DEM,
    "alphaearth": ALPHAEARTH + DEM,
}

FORBIDDEN_PREDICTORS = {
    RESPONSE, "agbd_se", "lon", "lat", "longitude", "latitude", "geometry",
    "shot_number", SOURCE_BLOCK, TARGET_BLOCK, FOLD, "block_e_index",
    "block_n_index", "palsar_angle", "palsar_epoch", "palsar_acquisition_date",
    "palsar_qa", "qa", "epoch", "angle",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    """Hash values in ordered rows without writing private identifiers."""
    cols = list(columns)
    payload = frame.loc[:, cols].to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def schema_hash(features: Iterable[str]) -> str:
    payload = json.dumps(list(features), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def dn_to_gamma0_db(dn: np.ndarray | pd.Series) -> np.ndarray:
    """Convert positive ALOS mosaic DN to gamma-naught dB: 20 log10(DN) - 83."""
    values = np.asarray(dn, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values > 0)
    out[valid] = 20.0 * np.log10(values[valid]) - 83.0
    return out


def rfdi_from_dn(hh_dn: np.ndarray | pd.Series,
                 hv_dn: np.ndarray | pd.Series) -> np.ndarray:
    """Compute RFDI from linear power (DN squared), never from dB values."""
    hh = np.asarray(hh_dn, dtype=float)
    hv = np.asarray(hv_dn, dtype=float)
    hh_power, hv_power = np.square(hh), np.square(hv)
    denominator = hh_power + hv_power
    out = np.full(np.broadcast(hh, hv).shape, np.nan, dtype=float)
    valid = (np.isfinite(hh_power) & np.isfinite(hv_power) &
             (hh > 0) & (hv > 0) & (denominator > 0))
    out[valid] = (hh_power[valid] - hv_power[valid]) / denominator[valid]
    return out


def add_palsar_features(frame: pd.DataFrame, *, drop_raw: bool = True) -> pd.DataFrame:
    """Apply the frozen PALSAR physics transform and validate its output."""
    required = {"HH", "HV"}
    if not required <= set(frame.columns):
        raise ValueError(f"Missing raw PALSAR columns: {sorted(required-set(frame.columns))}")
    result = frame.copy()
    result["palsar_hh_db"] = dn_to_gamma0_db(result["HH"])
    result["palsar_hv_db"] = dn_to_gamma0_db(result["HV"])
    result["palsar_hh_minus_hv_db"] = result.palsar_hh_db - result.palsar_hv_db
    result["palsar_rfdi"] = rfdi_from_dn(result["HH"], result["HV"])
    validate_palsar_physics(result, raw_hh=result["HH"], raw_hv=result["HV"])
    if drop_raw:
        result = result.drop(columns=["HH", "HV"])
    return result


def validate_palsar_physics(frame: pd.DataFrame, *, raw_hh=None, raw_hv=None) -> None:
    """Hard-stop on invalid logarithms, non-finite RFDI, or implausible dB values."""
    if raw_hh is not None and ((pd.Series(raw_hh).dropna() <= 0).any() or
                               (pd.Series(raw_hv).dropna() <= 0).any()):
        raise ValueError("PALSAR HH/HV DN must be > 0 before logarithm")
    finite = frame[PALSAR].replace([np.inf, -np.inf], np.nan)
    if finite.isna().any().any():
        raise ValueError("PALSAR primary features contain missing or infinite values")
    # Broad sensor sanity range; it is a corruption guard, not a scientific filter.
    for column in ["palsar_hh_db", "palsar_hv_db"]:
        if not frame[column].between(-60.0, 20.0).all():
            raise ValueError(f"{column} outside broad plausible range [-60, 20] dB")
    if not frame["palsar_rfdi"].between(-1.0, 1.0).all():
        raise ValueError("RFDI outside its physical [-1, 1] range")


def validate_feature_schema(name: str, features: Iterable[str]) -> list[str]:
    features = list(features)
    expected = REPRESENTATIONS[name]
    if features != expected:
        raise ValueError(f"Feature schema mismatch for {name}: {features} != {expected}")
    overlap = sorted(set(features) & FORBIDDEN_PREDICTORS)
    if overlap:
        raise RuntimeError(f"HARD STOP: forbidden predictors in {name}: {overlap}")
    if len(features) != len(set(features)):
        raise ValueError(f"Duplicate predictors in {name}")
    return features


def validate_paired_frames(frames: Mapping[str, pd.DataFrame], *, domain: str) -> None:
    """Ensure all paired representations contain identical ordered samples/labels."""
    if set(frames) != set(REPRESENTATIONS):
        raise ValueError("All eight frozen representations are required")
    block = SOURCE_BLOCK if domain == "source" else TARGET_BLOCK
    shared = ID_COLUMNS + [RESPONSE, block, FOLD]
    reference_name = next(iter(frames))
    reference = frames[reference_name]
    if reference.shot_number.astype("string").duplicated().any():
        raise ValueError("shot_number is not unique")
    for name, frame in frames.items():
        validate_feature_schema(name, REPRESENTATIONS[name])
        missing = set(shared + REPRESENTATIONS[name]) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        if len(frame) != len(reference):
            raise ValueError(f"Paired row count differs: {name}")
        for column in shared:
            left = reference[column].reset_index(drop=True)
            right = frame[column].reset_index(drop=True)
            if not left.equals(right):
                raise ValueError(f"Paired ordered field differs: {name}.{column}")
        if frame[REPRESENTATIONS[name]].replace([np.inf, -np.inf], np.nan).isna().any().any():
            raise ValueError(f"Non-finite predictor in {name}")
        if frame.groupby(block)[FOLD].nunique().max() != 1:
            raise RuntimeError(f"Spatial block crosses folds in {name}")


def public_freeze_summary(frame: pd.DataFrame, *, features: Mapping[str, list[str]],
                          exclusions: Mapping[str, int], domain: str) -> dict:
    """Build a freeze summary that stores hashes/counts but no shot IDs/coordinates."""
    block = SOURCE_BLOCK if domain == "source" else TARGET_BLOCK
    ordered = frame.sort_values(ID_COLUMNS).reset_index(drop=True)
    return {
        "domain": domain,
        "n": int(len(ordered)),
        "years": sorted(int(x) for x in ordered.year.unique()),
        "block_count": int(ordered[block].nunique()),
        "fold_counts": {str(k): int(v) for k, v in ordered[FOLD].value_counts().sort_index().items()},
        "sample_ids_hash": canonical_hash(ordered.sort_values("shot_number"), ["shot_number"]),
        "ordered_row_hash": canonical_hash(ordered, ID_COLUMNS),
        "response_hash": canonical_hash(ordered, ID_COLUMNS + [RESPONSE]),
        "feature_schema_hashes": {k: schema_hash(v) for k, v in features.items()},
        "exclusion_counts": {str(k): int(v) for k, v in exclusions.items()},
        "private_rows_committed": False,
    }
