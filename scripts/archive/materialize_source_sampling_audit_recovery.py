"""Convert optimized audit recovery exports to the frozen standard CSV schema."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAX_PER_STRATUM = 500
PERCENTILES = [(0.01, "p01"), (0.05, "p05"), (0.25, "p25"),
               (0.50, "p50"), (0.75, "p75"), (0.95, "p95"),
               (0.99, "p99")]


def parse_dictionary(value: object) -> dict[str, int]:
    if isinstance(value, dict):
        parsed = value
    else:
        text = str(value)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                stripped = text.strip()
                if not (stripped.startswith("{") and stripped.endswith("}")):
                    raise ValueError("Unsupported block-count dictionary format")
                parsed = {}
                body = stripped[1:-1].strip()
                if body:
                    for item in body.split(","):
                        key, separator, count = item.strip().partition("=")
                        if not separator or not key:
                            raise ValueError(
                                f"Malformed block-count item: {item!r}"
                            )
                        parsed[key.strip()] = int(count.strip())
    return {str(key): int(count) for key, count in parsed.items()}


def sampled_stats(manifest: pd.DataFrame) -> dict[str, float | int]:
    values = manifest["agbd"].to_numpy(dtype=float)
    props: dict[str, float | int] = {
        "sampled_n": int(values.size),
        "sampled_agbd_mean": float(values.mean()),
        "sampled_agbd_sd": float(values.std(ddof=0)),
        "sampled_agbd_min": float(values.min()),
        "sampled_agbd_max": float(values.max()),
    }
    for quantile, name in PERCENTILES:
        props[f"sampled_agbd_{name}"] = float(np.quantile(values, quantile))
    return props


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("drive_export_folder", type=Path)
    parser.add_argument("year", type=int)
    parser.add_argument(
        "--kinds", nargs="+", choices=["block_counts", "raw_agbd_audit"],
        default=["block_counts", "raw_agbd_audit"],
    )
    args = parser.parse_args()
    drive = args.drive_export_folder
    year = args.year
    manifest_path = drive / f"gedi_sampled_manifest_{year}.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(
        manifest_path,
        dtype={"shot_number": "string", "spatial_block_id": "string"},
    )

    if "block_counts" in args.kinds:
        source = drive / f"gedi_sampling_block_counts_recovery_{year}.csv"
        recovery = pd.read_csv(source)
        if len(recovery) != 1:
            raise RuntimeError(f"Expected one block-count record, got {len(recovery)}")
        counts = parse_dictionary(recovery.loc[0, "block_counts_json"])
        sampled_counts = manifest.groupby("spatial_block_id").size().to_dict()
        rows = []
        for block_id, raw_n in sorted(counts.items()):
            sampled_n = int(sampled_counts.get(block_id, 0))
            expected = min(raw_n, MAX_PER_STRATUM)
            if sampled_n != expected:
                raise RuntimeError(
                    f"{year} {block_id}: manifest {sampled_n}, expected {expected}"
                )
            rows.append({
                "spatial_block_id": block_id,
                "raw_qa_valid_n": raw_n,
                "sampled_n": sampled_n,
                "sampling_fraction": sampled_n / raw_n,
                "max_per_stratum": MAX_PER_STRATUM,
                "seed": 42,
                "year": year,
            })
        if set(sampled_counts) != set(counts):
            raise RuntimeError("Recovery block IDs do not match sampled manifest")
        pd.DataFrame(rows).to_csv(
            drive / f"gedi_sampling_strata_{year}.csv", index=False
        )

    if "raw_agbd_audit" in args.kinds:
        source = drive / f"gedi_sampling_raw_stats_recovery_{year}.csv"
        raw = pd.read_csv(source)
        if len(raw) != 1:
            raise RuntimeError(f"Expected one raw-stat record, got {len(raw)}")
        row = raw.iloc[0].to_dict()
        row.update(sampled_stats(manifest))
        row["year"] = year
        row["sampling_policy"] = "EPSG5070_50km_x_year_max500_seed42"
        pd.DataFrame([row]).to_csv(
            drive / f"gedi_sampling_annual_stats_{year}.csv", index=False
        )


if __name__ == "__main__":
    main()
