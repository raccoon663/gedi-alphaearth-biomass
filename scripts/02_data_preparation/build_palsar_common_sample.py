"""Build identical-row source/target inputs for all eight radar representations."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from utilities.radar_benchmark import (ALPHAEARTH, DEM, FOLD, PALSAR, REPRESENTATIONS,
                                       RESPONSE, S1, S2, S2_INDICES, public_freeze_summary,
                                       validate_paired_frames)  # noqa: E402

KEYS = ["shot_number", "year"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    data["shot_number"] = data.shot_number.astype("string")
    if data.duplicated(KEYS).any():
        raise RuntimeError(f"Duplicate shot/year: {path}")
    return data


def merge_one_to_one(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    add = [c for c in columns if c not in KEYS and c not in left and c in right]
    return left.merge(right[KEYS + add], on=KEYS, how="left", validate="one_to_one")


def build(domain: str, conventional_path: Path, alphaearth_path: Path, palsar_path: Path,
          output_dir: Path, freeze_path: Path, folds_path: Path | None = None) -> dict:
    conventional, alphaearth, palsar = map(read, [conventional_path, alphaearth_path, palsar_path])
    block = "spatial_block_id" if domain == "source" else "target_block_id"
    if domain == "target" and (block not in conventional or FOLD not in conventional):
        if folds_path is None:
            raise RuntimeError("Locked target predictors require --folds metadata")
        folds = pd.read_csv(folds_path, dtype={"shot_number": "string",
                                              "target_block_id": "string"})
        if folds.shot_number.duplicated().any():
            raise RuntimeError("Target folds contain duplicate shot_number")
        conventional = conventional.merge(
            folds[["shot_number", block, FOLD]], on="shot_number", how="left",
            validate="one_to_one")
        if conventional[[block, FOLD]].isna().any().any():
            raise RuntimeError("Target fold join is incomplete")
    metadata = [RESPONSE, block, FOLD]
    if domain == "target" and RESPONSE not in conventional:
        metadata.remove(RESPONSE)
    required_conv = KEYS + metadata + S1 + S2 + S2_INDICES + DEM
    # DEM is supplied by the conventional frame and shared by every representation;
    # the locked target AlphaEarth file therefore need not duplicate it.
    required_aef = KEYS + ALPHAEARTH
    required_palsar = KEYS + PALSAR + ["palsar_valid"]
    for label, frame, required in [("conventional", conventional, required_conv),
                                    ("alphaearth", alphaearth, required_aef),
                                    ("palsar", palsar, required_palsar)]:
        missing = set(required) - set(frame)
        if missing:
            raise RuntimeError(f"{label} missing {sorted(missing)}")
    n_before = len(conventional)
    common = conventional[required_conv].copy()
    common = merge_one_to_one(common, alphaearth, ALPHAEARTH)
    common = merge_one_to_one(common, palsar, PALSAR + ["palsar_valid", "palsar_angle",
                                                        "palsar_epoch", "palsar_acquisition_date",
                                                        "palsar_acquisition_month", "palsar_qa"])
    palsar_valid = common.palsar_valid.fillna(False).astype(bool)
    reasons = {
        "not_in_palsar_common_years": int((~common.year.isin(palsar.year.unique())).sum()),
        "invalid_or_missing_palsar": int((~palsar_valid).sum()),
        "missing_s1": int(common[S1].isna().any(axis=1).sum()),
        "missing_s2": int(common[S2 + S2_INDICES].isna().any(axis=1).sum()),
        "missing_dem": int(common[DEM].isna().any(axis=1).sum()),
        "missing_alphaearth": int(common[ALPHAEARTH].isna().any(axis=1).sum()),
    }
    required_all = S1 + S2 + S2_INDICES + DEM + ALPHAEARTH + PALSAR
    keep = palsar_valid & common[required_all].replace(
        [np.inf, -np.inf], np.nan).notna().all(axis=1)
    common = common.loc[keep].sort_values(KEYS).reset_index(drop=True)
    if not len(common):
        raise RuntimeError("PALSAR-common sample is empty")
    frames = {}
    for name, features in REPRESENTATIONS.items():
        frames[name] = common[KEYS + metadata + features].copy()
    if RESPONSE in metadata:
        validate_paired_frames(frames, domain=domain)
    else:
        # Pre-label target freeze: identical row/feature checks without reading AGBD.
        response_stub = np.zeros(len(common), dtype=float)
        for frame in frames.values():
            frame[RESPONSE] = response_stub
        validate_paired_frames(frames, domain=domain)
        for frame in frames.values():
            frame.drop(columns=RESPONSE, inplace=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, frame in frames.items():
        path = output_dir / f"{domain}_{name}.parquet"
        frame.to_parquet(path, index=False)
        hashes[name] = sha256(path)
    summary_frame = common if RESPONSE in common else common.assign(agbd=0.0)
    summary = public_freeze_summary(summary_frame, features=REPRESENTATIONS,
                                    exclusions=reasons, domain=domain)
    summary.update({"status": "PALSAR_COMMON_SAMPLE_FROZEN", "frozen_at": datetime.now(timezone.utc).isoformat(),
                    "n_before_masking": n_before, "n_after_masking": len(common),
                    "exact_year_matching": True, "nearest_year_substitution": False,
                    "input_hashes": {"conventional": sha256(conventional_path),
                                     "alphaearth": sha256(alphaearth_path),
                                     "palsar": sha256(palsar_path)},
                    "representation_file_hashes": hashes,
                    "target_labels_loaded": RESPONSE in metadata,
                    "private_per_sample_outputs_committed": False})
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    combined = {"status": "PALSAR_COMMON_SAMPLE_FROZEN", "domains": {}}
    if freeze_path.exists():
        prior = json.loads(freeze_path.read_text())
        if isinstance(prior.get("domains"), dict):
            combined = prior
    combined["domains"][domain] = summary
    combined["updated_at"] = datetime.now(timezone.utc).isoformat()
    combined["paired_representations"] = list(REPRESENTATIONS)
    combined["private_per_sample_outputs_committed"] = False
    freeze_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=["source", "target"], required=True)
    p.add_argument("--conventional", type=Path, required=True)
    p.add_argument("--alphaearth", type=Path, required=True)
    p.add_argument("--palsar", type=Path, required=True)
    p.add_argument("--folds", type=Path,
                   help="Private target whole-block fold CSV when predictors lack block/fold")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/palsar_common")
    p.add_argument("--freeze", type=Path, default=ROOT / "outputs/manifests/palsar_common_sample_freeze.json")
    args = p.parse_args()
    print(json.dumps(build(args.domain, args.conventional, args.alphaearth, args.palsar,
                           args.output_dir, args.freeze, args.folds), indent=2))


if __name__ == "__main__":
    main()
