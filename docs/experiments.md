# Experiments

This page summarizes the experiments that support the project's conclusions. It
is the companion to [`technical_report.md`](technical_report.md) and
[`methodology.md`](methodology.md). All numbers are taken from the frozen result
tables in `outputs/tables/` and the frozen manifests in `outputs/manifests/`.

The experiments separate three questions that are often conflated: **source-domain
prediction**, **direct geographic transfer**, and **target-domain label
efficiency**. Each is reported under its own evaluation setting (caliber) so that
different evaluation settings are not directly compared against one another.

## 1. Source-domain spatial cross-validation

Identical source footprints and five 50 km block folds; both selected models were
XGBoost depth-6 configurations chosen under the same restricted source-only
tuning budget.

| Representation | R² | RMSE (Mg/ha) | MAE (Mg/ha) | Bias (Mg/ha) |
|---|---:|---:|---:|---:|
| AlphaEarth + DEM | 0.5810 | 68.70 | 41.46 | +0.31 |
| Conventional + DEM | 0.4336 | 79.88 | 52.70 | −0.01 |

AlphaEarth improved source R² by 0.147 and reduced RMSE by ~14%.

![Source representation comparison](../figures/source_representation_comparison.png)

## 2. Zero-shot transfer (target labels locked)

Target labels remained locked during Kaihua predictor construction and model
prediction. DEM, AlphaEarth, and conventional variables were extracted for the
exact same 130,195 target footprints. The two frozen prediction files were hashed
before AGBD and AGBD-SE were unlocked.

| Representation | R² | RMSE (Mg/ha) | MAE (Mg/ha) | Bias (Mg/ha) |
|---|---:|---:|---:|---:|
| AlphaEarth + DEM | −0.5556 | 109.72 | 84.58 | +55.68 |
| Conventional + DEM | −1.1166 | 127.98 | 100.41 | +73.59 |

Both representations suffered severe transfer degradation and large positive bias.
AlphaEarth remained relatively better on every primary metric, but **negative
target R² means zero-shot transfer did not succeed in an absolute predictive
sense**.

![Zero-shot transfer degradation](../figures/zero_shot_transfer.png)

## 3. Domain shift (predictor-only diagnostics)

Predictor-only diagnostics were completed without changing the source models. A
source-fit PCA shows strong displacement between AlphaEarth embeddings from the
USA and Kaihua. A standardized logistic domain classifier achieved **AUROC
1.000**, indicating extreme domain separability. This separability is
*descriptive*, not a target-model selection rule, and does not imply that
classifier probability predicts sample-level biomass error. Exact nearest-source
Euclidean embedding distance had only a negligible association with within-Kaihua
absolute zero-shot error (Spearman ρ = 0.022).

![USA–Kaihua domain shift](../figures/usa_kaihua_domain_shift.png)
![AEF distance versus error](../figures/aef_distance_vs_error.png)

## 4. Few-shot adaptation (label efficiency)

Kaihua footprints were assigned to fixed 5 km blocks in EPSG:32650. Five
whole-block folds contained ~26,000 evaluation samples each. Label budgets were
25, 50, 100, 250, 500, 1,000 and 2,500, with seeds 42–44. Every fold × budget ×
seed used the identical target shot-number draw for both representations, without
AGBD stratification or active learning.

Three adaptation methods were tested: intercept-only bias correction, two-parameter
affine calibration, and a shared conservative local XGBoost configuration frozen
before representation results. Bias-only calibration remained negative at every
budget; affine calibration became slightly positive with 50 labels but plateaued
near R² 0.05.

Local AlphaEarth models first achieved **positive mean spatial-holdout R² with 250
labels**; conventional models required **500**. At 2,500 labels, AlphaEarth
reached **R² 0.1306 ± 0.0168** and RMSE 81.90 Mg/ha, versus R² 0.0594 ± 0.0105
and RMSE 85.19 Mg/ha for conventional features. **No method reached mean R² 0.2.**

![Kaihua label efficiency](../figures/kaihua_label_efficiency_final.png)
![Calibration and local adaptation](../figures/calibration_vs_local_adaptation.png)

## Negative result (reported, not hidden)

Direct zero-shot transfer failed for both representations, and even 2,500 local
labels recovered only modest spatial-holdout performance. The project therefore
reports a restricted positive result (representation-level transfer advantage and
label efficiency) rather than presenting itself as a successful operational
mapping system. Formal Kaihua wall-to-wall biomass mapping was not pursued.
