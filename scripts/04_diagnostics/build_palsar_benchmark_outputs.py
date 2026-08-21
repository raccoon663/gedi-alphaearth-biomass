"""Build paired headline tables, effect-size contrasts and portfolio figures."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "outputs/tables/main_results/"
DIAG = ROOT / "outputs/tables/diagnostics/"
FIGURES = ROOT / "figures"
CONTRASTS = [("palsar_l", "sentinel1_c"),
             ("sentinel1_c_palsar_l", "sentinel1_c"),
             ("sentinel1_c_palsar_l", "palsar_l"),
             ("palsar_sentinel2", "sentinel1_sentinel2"),
             ("sentinel1_palsar_sentinel2", "sentinel1_sentinel2"),
             ("alphaearth", "palsar_l")]
COLORS = ["#6B7280", "#2878B5", "#D97706", "#2E8B57", "#7C3AED", "#C2417A", "#0F766E", "#111827"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison_table() -> pd.DataFrame:
    source = pd.read_csv(MAIN / "palsar_source_spatial_cv.csv")
    target = pd.read_csv(MAIN / "palsar_zero_shot.csv")
    left = source[["representation", "R2", "RMSE", "MAE", "Bias", "N"]].rename(
        columns={x: f"source_cv_{x}" for x in ["R2", "RMSE", "MAE", "Bias", "N"]})
    right = target[["representation", "R2", "RMSE", "MAE", "Bias", "N"]].rename(
        columns={x: f"target_zero_shot_{x}" for x in ["R2", "RMSE", "MAE", "Bias", "N"]})
    result = left.merge(right, on="representation", validate="one_to_one")
    result.to_csv(MAIN / "radar_representation_comparison.csv", index=False)
    return result


def paired_contrasts() -> pd.DataFrame:
    folds = pd.read_csv(DIAG / "palsar_source_spatial_cv_folds.csv")
    rows = []
    for first, second in CONTRASTS:
        a = folds[folds.representation == first]
        b = folds[folds.representation == second]
        paired = a.merge(b, on="outer_fold", suffixes=("_first", "_second"), validate="one_to_one")
        if len(paired) != 5:
            raise RuntimeError(f"Expected five paired folds: {first} - {second}")
        for r in paired.itertuples(index=False):
            rows.append({"contrast": f"{first} - {second}", "outer_fold": r.outer_fold,
                         "delta_R2": r.R2_first-r.R2_second,
                         "delta_RMSE": r.RMSE_first-r.RMSE_second,
                         "delta_MAE": r.MAE_first-r.MAE_second,
                         "delta_Bias": r.Bias_first-r.Bias_second})
    out = pd.DataFrame(rows)
    out.to_csv(DIAG / "radar_paired_fold_contrasts.csv", index=False)
    (out.groupby("contrast", as_index=False)
     .agg(folds=("outer_fold", "size"), mean_delta_R2=("delta_R2", "mean"),
          mean_delta_RMSE=("delta_RMSE", "mean"),
          folds_favoring_first_R2=("delta_R2", lambda x: int((x > 0).sum())),
          folds_favoring_first_RMSE=("delta_RMSE", lambda x: int((x < 0).sum())))
     .to_csv(DIAG / "radar_paired_contrast_summary.csv", index=False))
    return out


def bar_figure(data: pd.DataFrame, prefix: str, title: str, filename: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = data.representation.tolist(); x = np.arange(len(names))
    axes[0].bar(x, data[f"{prefix}_R2"], color=COLORS); axes[0].axhline(0, color="black", lw=1)
    axes[1].bar(x, data[f"{prefix}_RMSE"], color=COLORS)
    for ax, label in zip(axes, ["R²", "RMSE (Mg/ha)"]):
        ax.set_xticks(x, names, rotation=35, ha="right"); ax.set_ylabel(label); ax.grid(axis="y", alpha=.2)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(FIGURES / filename, dpi=240); plt.close(fig)


def fewshot_figure() -> None:
    data = pd.read_csv(MAIN / "palsar_fewshot_label_efficiency.csv")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for (rep, group), color in zip(data.groupby("representation", sort=False), COLORS):
        group = group.sort_values("budget")
        ax.plot(group.budget, group.R2_mean, marker="o", lw=2, label=rep, color=color)
        ax.fill_between(group.budget, group.R2_mean-group.R2_sd, group.R2_mean+group.R2_sd,
                        color=color, alpha=.10)
    ax.axhline(0, color="black", lw=1); ax.set_xscale("symlog", linthresh=25)
    ax.set_xticks([25, 50, 100, 250, 500, 1000, 2500], ["25", "50", "100", "250", "500", "1000", "2500"])
    ax.set_xlim(24.5, 2500)  # Keep 25 leftmost without clipping its markers.
    ax.set(xlabel="Kaihua labels", ylabel="Mean spatial-holdout R²",
           title="PALSAR-common Kaihua label efficiency (mean ± SD; shared draws)")
    ax.legend(ncol=2, fontsize=8, frameon=False); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "radar_fewshot_label_efficiency.png", dpi=240); plt.close(fig)


def workflow() -> None:
    fig, ax = plt.subplots(figsize=(13, 8)); ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.axis("off")
    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.04",
                                    facecolor=color, edgecolor="#374151", linewidth=1.2))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=10)
    box(4.4, 7, 4.2, .65, "GEDI L4A AGBD response", "#E5E7EB")
    for x, text, color in [(0.5, "Sentinel-1\nC-band VV/VH", "#DBEAFE"),
                           (3.4, "PALSAR-2\nL-band HH/HV", "#FFEDD5"),
                           (6.3, "Sentinel-2 + DEM", "#DCFCE7"),
                           (9.2, "AlphaEarth + DEM", "#EDE9FE")]:
        box(x, 5.4, 2.4, .9, text, color)
    box(2.3, 3.8, 5.1, .75, "C + L radar and optical fusion representations", "#D1FAE5")
    box(7.8, 3.8, 3.0, .75, "Identical common rows", "#F3F4F6")
    box(4.5, 2.3, 4.0, .75, "RF / XGBoost; spatial blocks", "#E5E7EB")
    for x, text in [(1, "Source CV"), (4.8, "Zero-shot transfer"), (8.8, "Few-shot adaptation")]:
        box(x, .65, 3.1, .75, text, "#F9FAFB")
    arrows = [((6.5, 7), (6.5, 3.05)),
              ((1.7, 5.4), (4.2, 4.55)), ((4.6, 5.4), (5.0, 4.55)),
              ((7.5, 5.4), (6.0, 4.55)), ((10.4, 5.4), (9.3, 4.55)),
              ((7.4, 4.18), (7.8, 4.18)), ((9.3, 3.8), (7.5, 3.05)),
              ((6.5, 2.3), (2.55, 1.4)), ((6.5, 2.3), (6.35, 1.4)), ((6.5, 2.3), (10.35, 1.4))]
    for start, end in arrows: ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#4B5563"))
    fig.tight_layout(); fig.savefig(FIGURES / "final_workflow_v2.png", dpi=240); plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = comparison_table(); paired_contrasts()
    bar_figure(data, "source_cv", "PALSAR-common source spatial CV", "radar_representation_source_comparison.png")
    bar_figure(data, "target_zero_shot", "Frozen USA → Kaihua zero-shot transfer", "radar_zero_shot_comparison.png")
    fewshot_figure(); workflow()
    best = data.loc[data.target_zero_shot_R2.idxmax()]
    gate = {"evaluated_at": datetime.now(timezone.utc).isoformat(),
            "best_representation": best.representation, "best_target_R2": float(best.target_zero_shot_R2),
            "best_target_Bias": float(best.target_zero_shot_Bias),
            "minimum_R2_preference": .20,
            "positive_R2": bool(best.target_zero_shot_R2 > 0),
            "wall_to_wall_map_authorized": bool(best.target_zero_shot_R2 >= .20 and abs(best.target_zero_shot_Bias) < best.target_zero_shot_RMSE),
            "withheld_statement": "Wall-to-wall mapping was withheld because target-domain validation remained insufficient."}
    (ROOT / "outputs/manifests/palsar_wall_to_wall_decision_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8")
    manifest = {"status": "PALSAR_BENCHMARK_OUTPUTS_BUILT", "legacy_results_overwritten": False,
                "comparison_sha256": sha256(MAIN / "radar_representation_comparison.csv"),
                "figures": ["radar_representation_source_comparison.png", "radar_zero_shot_comparison.png",
                            "radar_fewshot_label_efficiency.png", "radar_biomass_range_sensitivity.png",
                            "radar_feature_importance.png", "radar_domain_shift.png", "final_workflow_v2.png"]}
    (ROOT / "outputs/manifests/palsar_benchmark_outputs_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(data.to_string(index=False)); print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
