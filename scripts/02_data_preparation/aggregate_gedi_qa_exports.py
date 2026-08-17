"""Aggregate completed per-granule GEDI QA count exports locally."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

STAGES = [
    "all_intersecting_shots", "algorithm_run_flag_eq_1", "required_fields_nonnull",
    "l4_quality_flag_eq_1", "degrade_flag_eq_0", "agbd_nonnegative",
    "agbd_se_nonnegative",
]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw" / "gedi_exports"
    files = sorted(raw_dir.glob("gedi_source_qa_counts_*.csv"))
    if len(files) != 6:
        raise RuntimeError(f"Expected six yearly QA exports, found {len(files)}")

    yearly = []
    for path in files:
        frame = pd.read_csv(path)
        year = int(frame["year"].dropna().iloc[0])
        row = {"year": year, "granule_count": len(frame)}
        row.update({stage: int(frame[stage].fillna(0).sum()) for stage in STAGES})
        yearly.append(row)
    yearly_df = pd.DataFrame(yearly).sort_values("year")
    totals = yearly_df[STAGES].sum()

    rows = []
    previous = None
    for i, stage in enumerate(STAGES):
        n = int(totals[stage])
        removed = None if previous is None else previous - n
        rows.append({
            "stage": i, "filter_stage": stage, "n_after": n,
            "removed_at_stage": removed,
            "retained_from_raw_percent": n / int(totals[STAGES[0]]) * 100,
        })
        previous = n

    tables = root / "outputs" / "tables"
    logs = root / "outputs" / "logs"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables / "gedi_qa.csv", index=False)
    yearly_df.to_csv(tables / "gedi_qa_by_year.csv", index=False)

    valid_2023_path = raw_dir / "gedi_source_valid_2023.csv"
    validation = {"available": valid_2023_path.exists()}
    if valid_2023_path.exists():
        valid = pd.read_csv(valid_2023_path)
        validation.update({
            "rows": len(valid),
            "expected_from_counts": int(yearly_df.loc[yearly_df.year == 2023, "agbd_se_nonnegative"].iloc[0]),
            "duplicate_shot_number": int(valid["shot_number"].duplicated().sum()),
            "null_required_fields": int(valid[["shot_number", "lon", "lat", "agbd", "agbd_se"]].isna().sum().sum()),
            "agbd_mean": float(valid["agbd"].mean()),
            "agbd_sd": float(valid["agbd"].std()),
            "agbd_percentiles": {str(q): float(valid["agbd"].quantile(q))
                                 for q in [0, .01, .05, .25, .5, .75, .95, .99, 1]},
        })
        if validation["rows"] != validation["expected_from_counts"]:
            raise RuntimeError("2023 valid export row count does not match QA count export")
    (logs / "gedi_qa_local_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
