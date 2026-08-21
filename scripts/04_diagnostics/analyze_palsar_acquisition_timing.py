"""Audit PALSAR mosaic acquisition timing without using it as a predictor."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def read(path: Path, domain: str) -> pd.DataFrame:
    data = pd.read_parquet(path, columns=["year", "palsar_epoch", "palsar_valid"])
    data = data[data.palsar_valid].copy()
    data["date"] = pd.to_datetime(data.palsar_epoch, unit="ms", utc=True, errors="coerce")
    data = data.dropna(subset=["date"])
    data["month"] = data.date.dt.month
    data["day_of_year"] = data.date.dt.dayofyear
    data["domain"] = domain
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=ROOT / "data/processed/source_palsar.parquet")
    p.add_argument("--target", type=Path, default=ROOT / "data/processed/kaihua_palsar_locked.parquet")
    args = p.parse_args()
    source, target = read(args.source, "source"), read(args.target, "target")
    limits = (source.groupby("year").day_of_year.quantile([.05, .95]).unstack()
              .rename(columns={.05: "source_doy_p05", .95: "source_doy_p95"}))
    all_data = pd.concat([source, target], ignore_index=True).merge(limits, on="year", how="left")
    all_data["unusual_vs_source_timing"] = ((all_data.day_of_year < all_data.source_doy_p05) |
                                             (all_data.day_of_year > all_data.source_doy_p95))
    rows = (all_data.groupby(["domain", "year", "month"], as_index=False)
            .agg(N=("date", "size"), unusual_n=("unusual_vs_source_timing", "sum")))
    totals = rows.groupby(["domain", "year"]).N.transform("sum")
    rows["fraction"] = rows.N / totals
    rows["unusual_fraction"] = rows.unusual_n / rows.N
    out = ROOT / "outputs/tables/audits/palsar_acquisition_timing.csv"
    out.parent.mkdir(parents=True, exist_ok=True); rows.to_csv(out, index=False)
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
