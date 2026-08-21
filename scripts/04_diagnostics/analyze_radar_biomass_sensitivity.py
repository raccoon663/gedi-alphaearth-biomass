"""Source-only biomass-range, response and held-out importance diagnostics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

RADAR_REPS = ["sentinel1_c", "palsar_l", "sentinel1_c_palsar_l"]
RESPONSES = [("VV", "S1 VV"), ("VH", "S1 VH"),
             ("palsar_hh_db", "PALSAR HH"), ("palsar_hv_db", "PALSAR HV"),
             ("palsar_rfdi", "PALSAR RFDI")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--common-dir", type=Path, default=ROOT / "data/processed/palsar_common")
    p.add_argument("--bins", type=int, default=5)
    args = p.parse_args()
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    oof = pd.read_parquet(ROOT / "outputs/tables/diagnostics/palsar_source_oof_predictions.parquet")
    reference = pd.read_parquet(args.common_dir / "source_sentinel1_c_palsar_l.parquet")
    edges = np.unique(np.quantile(reference.agbd, np.linspace(0, 1, args.bins + 1)))
    if len(edges) != args.bins + 1:
        raise RuntimeError("Source AGBD quantiles are not unique")
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    for rep in RADAR_REPS:
        data = oof[oof.representation == rep].copy()
        data["biomass_bin"] = pd.cut(data.agbd, edges, labels=False, include_lowest=True)
        for bin_id, group in data.groupby("biomass_bin"):
            residual = group.prediction - group.agbd
            rows.append({"representation": rep, "biomass_bin": int(bin_id),
                         "source_edge_low": float(edges[int(bin_id)]),
                         "source_edge_high": float(edges[int(bin_id)+1]), "N": len(group),
                         "RMSE": float(mean_squared_error(group.agbd, group.prediction) ** .5),
                         "Bias": float(residual.mean()),
                         "R2": float(r2_score(group.agbd, group.prediction)) if len(group) >= 20 else np.nan,
                         "residual_q25": float(residual.quantile(.25)),
                         "residual_median": float(residual.median()),
                         "residual_q75": float(residual.quantile(.75))})
    table = pd.DataFrame(rows)
    out = ROOT / "outputs/tables/diagnostics/radar_biomass_bin_performance.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    colors = {"sentinel1_c": "#2878B5", "palsar_l": "#D97706",
              "sentinel1_c_palsar_l": "#2E8B57"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for rep in RADAR_REPS:
        data = table[table.representation == rep]
        axes[0].plot(data.biomass_bin + 1, data.RMSE, marker="o", label=rep, color=colors[rep])
        axes[1].plot(data.biomass_bin + 1, data.Bias, marker="o", label=rep, color=colors[rep])
    axes[0].set(ylabel="RMSE (Mg/ha)", xlabel="Source-defined AGBD quantile bin")
    axes[1].set(ylabel="Bias (Mg/ha)", xlabel="Source-defined AGBD quantile bin")
    axes[1].axhline(0, color="black", lw=1)
    for ax in axes: ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.suptitle("Radar sensitivity across source-defined biomass ranges")
    fig.tight_layout(); fig.savefig(ROOT / "figures/radar_biomass_range_sensitivity.png", dpi=240); plt.close(fig)

    # Source-only response diagnostics: hexbin plus fixed-quantile medians/IQR.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9)); axes = axes.ravel()
    for ax, (feature, title) in zip(axes, RESPONSES):
        data = reference[[feature, "agbd"]].dropna()
        ax.hexbin(data[feature], data.agbd, gridsize=45, bins="log", mincnt=1, cmap="Blues")
        bins = pd.qcut(data[feature], 20, duplicates="drop")
        binned = data.groupby(bins, observed=True).agg(x=(feature, "median"), y=("agbd", "median"),
                                                       q25=("agbd", lambda x: x.quantile(.25)),
                                                       q75=("agbd", lambda x: x.quantile(.75)))
        ax.plot(binned.x, binned.y, color="#C43C39", lw=2)
        ax.fill_between(binned.x, binned.q25, binned.q75, color="#C43C39", alpha=.18)
        ax.set(title=title, xlabel=feature, ylabel="GEDI L4A AGBD (Mg/ha)")
    axes[-1].axis("off")
    fig.suptitle("Source-only radar response diagnostics (descriptive, no fitted threshold)")
    fig.tight_layout(); fig.savefig(ROOT / "figures/radar_response_diagnostics.png", dpi=220); plt.close(fig)

    importance = pd.read_csv(ROOT / "outputs/tables/diagnostics/radar_feature_importance.csv")
    imp = (importance[importance.representation.isin(RADAR_REPS)]
           .groupby(["representation", "feature"], as_index=False)
           .agg(importance_mean=("importance_mean", "mean"), importance_sd=("importance_mean", "std")))
    chosen = (imp.sort_values(["representation", "importance_mean"], ascending=[True, False])
              .groupby("representation").head(10))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False)
    for ax, rep in zip(axes, RADAR_REPS):
        data = chosen[chosen.representation == rep].sort_values("importance_mean")
        ax.barh(data.feature, data.importance_mean, xerr=data.importance_sd.fillna(0), color=colors[rep])
        ax.set_title(rep); ax.set_xlabel("Held-out permutation importance\n(increase in RMSE when permuted)")
    fig.tight_layout(); fig.savefig(ROOT / "figures/radar_feature_importance.png", dpi=240); plt.close(fig)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
