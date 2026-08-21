"""Create a public aggregate audit of selection induced by PALSAR validity masking."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["shot_number", "year"]


def read(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    data["shot_number"] = data.shot_number.astype("string")
    return data


def block_hash(value: object) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def distribution_rows(domain: str, before: pd.DataFrame, after: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for sample, frame in [("before", before), ("after", after)]:
        agbd = pd.to_numeric(frame.agbd, errors="coerce").dropna()
        q = agbd.quantile([.1, .25, .5, .75, .9, .95])
        stats = {"mean": agbd.mean(), "median": agbd.median(), "sd": agbd.std(ddof=1),
                 "p10": q.loc[.1], "p25": q.loc[.25], "p50": q.loc[.5],
                 "p75": q.loc[.75], "p90": q.loc[.9], "p95": q.loc[.95]}
        rows.extend({"domain": domain, "comparison": "agbd", "sample": sample,
                     "category": metric, "n": len(agbd), "value": float(value)}
                    for metric, value in stats.items())
        for year, n in frame.groupby("year").size().items():
            rows.append({"domain": domain, "comparison": "year_distribution",
                         "sample": sample, "category": str(int(year)), "n": int(n),
                         "value": float(n / len(frame))})
        block = "spatial_block_id" if domain == "source" else "target_block_id"
        if block in frame:
            for value, n in frame.groupby(block, dropna=False).size().items():
                rows.append({"domain": domain, "comparison": "block_distribution",
                             "sample": sample, "category": block_hash(value), "n": int(n),
                             "value": float(n / len(frame))})
    for comparison in ["year_distribution", "block_distribution"]:
        table = pd.DataFrame(rows)
        subset = table[table.comparison == comparison]
        if subset.empty:
            continue
        pivot = subset.pivot(index="category", columns="sample", values="value").fillna(0)
        tv = .5 * (pivot.get("before", 0) - pivot.get("after", 0)).abs().sum()
        rows.append({"domain": domain, "comparison": f"{comparison}_total_variation",
                     "sample": "before_vs_after", "category": "total_variation_distance",
                     "n": len(after), "value": float(tv)})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-before", type=Path, required=True)
    p.add_argument("--source-after", type=Path, required=True)
    p.add_argument("--target-before", type=Path, required=True)
    p.add_argument("--target-after", type=Path, required=True)
    p.add_argument("--target-labels", type=Path,
                   help="Labels joined only after the locked target experiment is complete")
    p.add_argument("--target-folds", type=Path,
                   help="Private shot-to-block mapping used only for aggregate block distributions")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/tables/audits/palsar_common_sample_selection_bias.csv")
    args = p.parse_args()
    frames = {name: read(path) for name, path in {
        "source_before": args.source_before, "source_after": args.source_after,
        "target_before": args.target_before, "target_after": args.target_after}.items()}
    if "agbd" not in frames["target_before"]:
        if not args.target_labels:
            raise RuntimeError("Target labels are required for the post-unlock audit")
        labels = read(args.target_labels)
        if labels.duplicated(KEYS).any():
            raise RuntimeError("Duplicate target labels")
        for name in ("target_before", "target_after"):
            frames[name] = frames[name].merge(labels[KEYS + ["agbd"]], on=KEYS,
                                               how="left", validate="one_to_one")
    if "target_block_id" not in frames["target_before"]:
        if not args.target_folds:
            raise RuntimeError("Target folds are required for the block-distribution audit")
        folds = read(args.target_folds)
        if folds.shot_number.duplicated().any():
            raise RuntimeError("Duplicate target fold assignments")
        frames["target_before"] = frames["target_before"].merge(
            folds[["shot_number", "target_block_id"]], on="shot_number", how="left",
            validate="one_to_one")
    rows = distribution_rows("source", frames["source_before"], frames["source_after"])
    rows += distribution_rows("target", frames["target_before"], frames["target_after"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
