"""Finalize aggregate PALSAR source chunks and retain acquisition metadata for QC."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ["palsar_hh_db", "palsar_hv_db", "palsar_hh_minus_hv_db", "palsar_rfdi"]
QC = ["palsar_angle", "palsar_epoch", "palsar_qa"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(input_dir: Path, output: Path, manifest: Path, domain: str) -> dict:
    paths = sorted(input_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No PALSAR chunks in {input_dir}")
    dtype = {"shot_number": "string", "spatial_block_id": "string", "target_block_id": "string"}
    data = pd.concat([pd.read_csv(p, dtype=dtype) for p in paths], ignore_index=True)
    if data.duplicated(["shot_number", "year"]).any():
        raise RuntimeError("Duplicate PALSAR shot/year across chunks")
    if not set(PRIMARY + QC) <= set(data):
        raise RuntimeError("Final PALSAR chunks lack primary or QC bands")
    data = data.replace([np.inf, -np.inf], np.nan)
    valid_qa_values = {1, 255}
    unexpected = set(pd.to_numeric(data.palsar_qa, errors="coerce").dropna().astype(int)) - {
        0, 1, 2, 3, 4, 50, 100, 150, 255
    }
    if unexpected:
        raise RuntimeError(f"Undocumented PALSAR QA classes: {sorted(unexpected)}")
    valid = (data.palsar_qa.isin(valid_qa_values) & data[PRIMARY].notna().all(axis=1) &
             data.palsar_hh_db.between(-60, 20) & data.palsar_hv_db.between(-60, 20) &
             data.palsar_rfdi.between(-1, 1))
    data["palsar_valid"] = valid
    data["palsar_acquisition_date"] = pd.to_datetime(
        data.palsar_epoch.where(data.palsar_epoch > 0), unit="ms", utc=True,
        errors="coerce").dt.strftime("%Y-%m-%d")
    data["palsar_acquisition_month"] = pd.to_datetime(
        data.palsar_acquisition_date, errors="coerce").dt.month.astype("Int64")
    data = data.sort_values(["year", "shot_number"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False)
    summary = {"status": f"{domain.upper()}_PALSAR_FINALIZED",
               "created_at": datetime.now(timezone.utc).isoformat(), "domain": domain,
               "input_chunks": len(paths), "input_rows": len(data),
               "valid_rows": int(valid.sum()), "invalid_rows": int((~valid).sum()),
               "valid_fraction": float(valid.mean()), "years": sorted(map(int, data.year.unique())),
               "valid_qa_values": sorted(valid_qa_values),
               "feature_columns": PRIMARY, "qc_columns": QC + ["palsar_acquisition_date",
                                                                  "palsar_acquisition_month"],
               "epoch_units": "milliseconds since 1970-01-01",
               "output_sha256": sha256(output), "private_output_committed": False,
               "chunk_hashes": {p.name: sha256(p) for p in paths}}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=ROOT / "data/raw/source_palsar")
    p.add_argument("--output", type=Path, default=ROOT / "data/processed/source_palsar.parquet")
    p.add_argument("--manifest", type=Path, default=ROOT / "outputs/logs/source_palsar_finalize.json")
    args = p.parse_args()
    print(json.dumps(finalize(args.input_dir, args.output, args.manifest, "source"), indent=2))


if __name__ == "__main__":
    main()
