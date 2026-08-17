"""Verify frozen results, build final master table, and generate portfolio figures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

BLUE = "#21618C"
ORANGE = "#D9791F"
GREEN = "#2E8B57"
DARK = "#263238"
LIGHT = "#F4F7F9"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> tuple[dict, dict, dict]:
    manifests = root / "outputs/manifests"
    source = json.loads((manifests / "frozen_source_model_manifest.json").read_text())
    zero = json.loads((manifests / "kaihua_zero_shot_freeze.json").read_text())
    design = json.loads((manifests / "kaihua_fewshot_design_freeze.json").read_text())
    few = json.loads((manifests / "kaihua_fewshot_results_manifest.json").read_text())
    checks = [
        (root / source["selected_models"]["alphaearth"]["model_file"], source["selected_models"]["alphaearth"]["model_sha256"]),
        (root / source["selected_models"]["conventional"]["model_file"], source["selected_models"]["conventional"]["model_sha256"]),
        (root / "outputs/predictions/kaihua_zero_shot_aef.csv", zero["prediction_file_hashes"]["alphaearth"]),
        (root / "outputs/predictions/kaihua_zero_shot_conventional.csv", zero["prediction_file_hashes"]["conventional"]),
        (root / "data/processed/kaihua_zero_shot_evaluation.parquet", design["input_evaluation_sha256"]),
        (root / "outputs/manifests/kaihua_target_spatial_blocks.csv", design["block_manifest_sha256"]),
        (root / "outputs/manifests/kaihua_fewshot_spatial_folds.csv", design["fold_manifest_sha256"]),
        (root / "outputs/tables/main_results/kaihua_fewshot_label_efficiency.csv", few["detailed_results_sha256"]),
        (root / "outputs/tables/main_results/kaihua_fewshot_summary.csv", few["summary_sha256"]),
        (root / "outputs/tables/main_results/kaihua_label_thresholds.csv", few["thresholds_sha256"]),
    ]
    for path, expected in checks:
        if sha256(path) != expected:
            raise RuntimeError(f"Frozen hash mismatch: {path}")
    return source, zero, design


def master_table(root: Path, source: dict) -> pd.DataFrame:
    zero = pd.read_csv(root / "outputs/tables/main_results/representation_transfer_summary.csv")
    few = pd.read_csv(root / "outputs/tables/main_results/kaihua_fewshot_summary.csv")
    folds = pd.read_parquet(root / "data/processed/source_common_representation.parquet",
                            columns=["spatial_fold"])
    fold_n = folds.spatial_fold.value_counts().sort_index().to_numpy()
    source_n = len(folds)
    rows = []
    for rep, label in [("alphaearth", "AlphaEarth"), ("conventional", "Conventional")]:
        cv = source["selected_models"][rep]["source_cv"]
        rows.append({"stage": "source_spatial_cv", "representation": label,
                     "labels": source_n, "R2": cv["R2"], "RMSE": cv["RMSE"],
                     "MAE": cv["MAE"], "Bias": cv["Bias"],
                     "evaluation_design": "USA 50-km block 5-fold spatial CV",
                     "N_train": float(np.mean(source_n-fold_n)), "N_test": float(np.mean(fold_n)),
                     "notes": "Formal five-fold mean; not a random split"})
        z = zero[zero.representation == label].iloc[0]
        rows.append({"stage": "kaihua_zero_shot", "representation": label,
                     "labels": 0, "R2": z.target_R2, "RMSE": z.target_RMSE,
                     "MAE": z.target_MAE, "Bias": z.target_Bias,
                     "evaluation_design": "Frozen USA model on all Kaihua common footprints",
                     "N_train": 0, "N_test": int(z.N),
                     "notes": "Immutable zero-shot baseline; no Zhejiang adaptation"})
        f = few[(few.representation == rep) & (few.method == "local_xgboost")]
        for r in f.sort_values("budget").itertuples(index=False):
            rows.append({"stage": "kaihua_fewshot_local", "representation": label,
                         "labels": int(r.budget), "R2": r.R2_mean, "RMSE": r.RMSE_mean,
                         "MAE": r.MAE_mean, "Bias": r.Bias_mean,
                         "evaluation_design": "Kaihua 5-km block 5-fold spatial holdout; 3 seeds",
                         "N_train": int(r.N_train), "N_test": r.N_test_mean,
                         "notes": f"Mean across {int(r.runs)} fold-seed runs; fixed shared XGBoost"})
    out = pd.DataFrame(rows)
    out.to_csv(root / "outputs/tables/main_results/final_project_summary.csv", index=False)
    return out


def setup_style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": DARK,
                         "text.color": DARK, "figure.facecolor": "white"})


def workflow(root: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
    def box(x, y, w, h, text, color, fontsize=10):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                               facecolor=color, edgecolor=DARK, linewidth=1.1)
        ax.add_patch(patch); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))
    box(2.3, 10.8, 5.4, .75, "Georgia + South Carolina\nGEDI L4A AGBD source labels", "#E8EEF3", 11)
    arrow(5, 10.8, 5, 10.25)
    box(.7, 8.9, 3.8, 1.0, "Branch A\nAlphaEarth A00–A63 + DEM", "#DCEEF8", 11)
    box(5.5, 8.9, 3.8, 1.0, "Branch B\nSentinel-1 + Sentinel-2\nindices + DEM", "#FBE8D5", 11)
    arrow(5, 10.25, 2.6, 9.9); arrow(5, 10.25, 7.4, 9.9)
    box(2.4, 7.55, 5.2, .7, "50 km block spatial cross-validation", "#E7F3E8", 11)
    arrow(2.6, 8.9, 4.2, 8.25); arrow(7.4, 8.9, 5.8, 8.25)
    stages = [(6.4, "Frozen USA source models"), (5.25, "Kaihua predictor extraction"),
              (4.1, "Zero-shot transfer"), (2.95, "Target-label unlock"),
              (1.8, "5 km spatial-holdout few-shot adaptation"),
              (.65, "Label-efficiency comparison")]
    last_y = 7.55
    for y, text in stages:
        arrow(5, last_y, 5, y+.72); box(2.4, y, 5.2, .72, text, "#F5F7F8", 10.5); last_y = y
    ax.text(5, 11.9, "Cross-region GEDI biomass transfer workflow", ha="center", fontsize=16, weight="bold")
    ax.text(5, .15, "Formal wall-to-wall mapping was not pursued because target performance remained modest.",
            ha="center", fontsize=9, style="italic", color="#566573")
    fig.tight_layout(); fig.savefig(root / "figures/final_workflow.png", dpi=260, bbox_inches="tight"); plt.close(fig)


def source_comparison(root: Path) -> None:
    manifests = root / "outputs/manifests"
    source = json.loads((manifests / "frozen_source_model_manifest.json").read_text())
    order = ["alphaearth", "conventional"]
    records = []
    for rep in order:
        cv = source["selected_models"][rep]["source_cv"]
        records.append({"representation": rep, "R2": cv["R2"], "R2_sd": cv.get("R2_std", 0.0),
                        "RMSE": cv["RMSE"], "RMSE_sd": cv.get("RMSE_std", 0.0)})
    summary = pd.DataFrame(records).set_index("representation").loc[order]
    colors = [BLUE, ORANGE]; labels = ["AlphaEarth + DEM", "Conventional + DEM"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric, ylabel in [(axes[0], "R2", "R²"), (axes[1], "RMSE", "RMSE (Mg/ha)")]:
        vals, errs = summary[metric], summary[f"{metric}_sd"]
        bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=colors, width=.62)
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.2)
        ax.tick_params(axis="x", rotation=10)
        for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}" if metric=="R2" else f"{v:.1f}", ha="center", va="bottom")
    fig.suptitle("USA source-domain spatial CV (nested)", fontsize=14, weight="bold")
    fig.tight_layout(); fig.savefig(root / "figures/source_representation_comparison.png", dpi=260); plt.close(fig)


def zero_transfer(root: Path) -> None:
    df = pd.read_csv(root / "outputs/tables/main_results/representation_transfer_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    x = [0, 1]
    for _, r in df.iterrows():
        color = BLUE if r.representation == "AlphaEarth" else ORANGE
        axes[0].plot(x, [r.source_R2, r.target_R2], marker="o", lw=2.4, color=color, label=r.representation)
        axes[1].plot(x, [r.source_RMSE, r.target_RMSE], marker="o", lw=2.4, color=color, label=r.representation)
    for ax in axes:
        ax.set_xticks(x, ["USA source\nspatial CV", "Kaihua\nzero-shot"]); ax.grid(axis="y", alpha=.2)
    axes[0].axhline(0, color=DARK, lw=1); axes[0].set_ylabel("R²"); axes[0].set_title("Predictive performance")
    axes[1].set_ylabel("RMSE (Mg/ha)"); axes[1].set_title("Prediction error")
    axes[1].legend(frameon=False)
    fig.suptitle("Severe zero-shot transfer degradation", fontsize=14, weight="bold")
    fig.tight_layout(); fig.savefig(root / "figures/zero_shot_transfer.png", dpi=260); plt.close(fig)


def label_efficiency(root: Path) -> None:
    few = pd.read_csv(root / "outputs/tables/main_results/kaihua_fewshot_summary.csv")
    zero = pd.read_csv(root / "outputs/tables/main_results/representation_transfer_summary.csv")
    fig, ax = plt.subplots(figsize=(8.8, 5.7))
    for rep, label, color, zlabel in [("alphaearth", "AlphaEarth local model", BLUE, "AlphaEarth"),
                                      ("conventional", "Conventional local model", ORANGE, "Conventional")]:
        d = few[(few.representation == rep) & (few.method == "local_xgboost")].sort_values("budget")
        z = zero.loc[zero.representation == zlabel, "target_R2"].iloc[0]
        x = np.r_[0, d.budget]; y = np.r_[z, d.R2_mean]; sd = np.r_[0, d.R2_sd]
        ax.plot(x, y, marker="o", lw=2.5, color=color, label=label)
        ax.fill_between(x, y-sd, y+sd, color=color, alpha=.16)
    ax.axhline(0, color=DARK, lw=1); ax.axhline(.2, color="#7F8C8D", lw=1, ls="--")
    ax.text(2450, .205, "R² = 0.2", ha="right", va="bottom", color="#7F8C8D", fontsize=9)
    ax.annotate("AEF first positive\n250 labels", (250, .0374), (140, .11), arrowprops=dict(arrowstyle="->", color=BLUE), color=BLUE)
    ax.annotate("Conventional first positive\n500 labels", (500, .0073), (650, -.11), arrowprops=dict(arrowstyle="->", color=ORANGE), color=ORANGE)
    ax.set_xscale("symlog", linthresh=25); ticks=[0,25,50,100,250,500,1000,2500]
    ax.set_xticks(ticks, [str(x) for x in ticks]); ax.set_xlabel("Number of Kaihua labels")
    ax.set_ylabel("Mean spatial-holdout R²"); ax.set_title("Kaihua target-domain label efficiency")
    ax.grid(alpha=.2); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(root / "figures/kaihua_label_efficiency.png", dpi=280); plt.close(fig)


def calibration(root: Path) -> None:
    few = pd.read_csv(root / "outputs/tables/main_results/kaihua_fewshot_summary.csv")
    zero = pd.read_csv(root / "outputs/tables/main_results/representation_transfer_summary.csv")
    methods = [("bias_only", "Bias-only", ":"), ("affine", "Affine", "--"),
               ("local_xgboost", "Local XGBoost", "-")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, rep, title, color, zlabel in [(axes[0], "alphaearth", "AlphaEarth + DEM", BLUE, "AlphaEarth"),
                                         (axes[1], "conventional", "Conventional + DEM", ORANGE, "Conventional")]:
        z = zero.loc[zero.representation == zlabel, "target_R2"].iloc[0]
        for method, label, ls in methods:
            d = few[(few.representation == rep) & (few.method == method)].sort_values("budget")
            x=np.r_[0,d.budget]; y=np.r_[z,d.R2_mean]; sd=np.r_[0,d.R2_sd]
            ax.plot(x,y,marker="o",ms=3,lw=2,ls=ls,color=color,label=label)
            ax.fill_between(x,y-sd,y+sd,color=color,alpha=.07)
        ax.axhline(0,color=DARK,lw=1); ax.set_xscale("symlog",linthresh=25)
        ticks=[0,25,50,100,250,500,1000,2500]; ax.set_xticks(ticks,[str(x) for x in ticks],rotation=25)
        ax.set_title(title); ax.set_xlabel("Kaihua labels"); ax.grid(alpha=.18); ax.legend(frameon=False,fontsize=8)
    axes[0].set_ylabel("Mean spatial-holdout R²")
    fig.suptitle("Calibration versus local adaptation",fontsize=14,weight="bold")
    fig.tight_layout(); fig.savefig(root / "figures/calibration_vs_local_adaptation.png",dpi=260); plt.close(fig)


def domain_shift(root: Path) -> None:
    pca = pd.read_parquet(root / "outputs/tables/diagnostics/kaihua_aef_domain_pca.parquet")
    clf = pd.read_csv(root / "outputs/tables/diagnostics/kaihua_aef_domain_classifier.csv").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios":[3,1]})
    for domain, color in [("USA_source",BLUE),("Kaihua",ORANGE)]:
        d=pca[pca.domain==domain].sample(min(15000,(pca.domain==domain).sum()),random_state=42)
        axes[0].scatter(d.PC1,d.PC2,s=3,alpha=.12,color=color,label=domain.replace("_"," "),rasterized=True)
    axes[0].set_xlabel("Source-fit PC1"); axes[0].set_ylabel("Source-fit PC2")
    axes[0].set_title("AlphaEarth embedding space"); axes[0].legend(markerscale=4,frameon=False)
    axes[1].bar(["Domain\nclassifier"],[clf.AUROC],color=GREEN,width=.55)
    axes[1].set_ylim(0,1.05); axes[1].set_ylabel("AUROC"); axes[1].text(0,clf.AUROC-.08,f"{clf.AUROC:.3f}",ha="center",color="white",weight="bold")
    axes[1].set_title("USA vs Kaihua")
    fig.suptitle("Extreme predictor-domain separability",fontsize=14,weight="bold")
    fig.tight_layout(); fig.savefig(root / "figures/usa_kaihua_domain_shift.png",dpi=260); plt.close(fig)


def distance_error(root: Path) -> None:
    d = pd.read_csv(root / "outputs/tables/diagnostics/kaihua_aef_distance_vs_error.csv")
    rho=d.spearman_rho_all.iloc[0]
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    ax.plot(d.distance_decile,d.MAE,marker="o",lw=2.2,color=BLUE,label="MAE")
    ax.plot(d.distance_decile,d.RMSE,marker="s",lw=2.2,color=ORANGE,label="RMSE")
    ax.set_xticks(range(1,11)); ax.set_xlabel("Nearest-source AEF distance decile")
    ax.set_ylabel("Zero-shot error (Mg/ha)"); ax.grid(alpha=.2); ax.legend(frameon=False)
    ax.text(.02,.96,f"Spearman ρ = {rho:.3f}",transform=ax.transAxes,ha="left",va="top",
            bbox=dict(boxstyle="round",facecolor="white",edgecolor="#BFC9CA"))
    ax.set_title("Nearest-source distance explains little within-Kaihua error")
    fig.tight_layout(); fig.savefig(root / "figures/aef_distance_vs_error.png",dpi=260); plt.close(fig)


def main() -> None:
    root=Path(__file__).resolve().parents[2]
    (root/"figures").mkdir(exist_ok=True); setup_style()
    source,_,_=verify(root)
    table=master_table(root,source)
    workflow(root); source_comparison(root); zero_transfer(root); label_efficiency(root)
    calibration(root); domain_shift(root); distance_error(root)
    print(table.to_string(index=False))
    print("FINAL_TABLE_ROWS",len(table))


if __name__ == "__main__":
    main()
