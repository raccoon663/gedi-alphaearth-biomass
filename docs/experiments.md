# Experiments

## Experiment groups

### Group 1 — legacy frozen experiment

The original 2019–2024 AlphaEarth versus S1/S2 experiment and its headline tables
remain frozen with their original interpretation.

### Group 2 — PALSAR-common paired benchmark

After the availability audit, DEM, S1_C, PALSAR_L, S1_C+PALSAR_L, S1+S2,
PALSAR+S2, S1+PALSAR+S2 and AlphaEarth are evaluated on identical rows for source
CV, frozen zero-shot transfer and shared-draw few-shot adaptation. Required source
contrasts are stored fold by fold. With five folds, interpretation emphasizes effect
size and consistency, not manufactured significance.


This page summarizes the experiments that support the project's conclusions. It
is the companion to [`technical_report.md`](technical_report.md) and
[`methodology.md`](methodology.md). All numbers are taken from the frozen result
tables in `outputs/tables/` and the frozen manifests in `outputs/manifests/`.

The experiments separate three questions that are often conflated: **source-domain
prediction**, **direct geographic transfer**, and **target-domain label
efficiency**. Each is reported under its own evaluation setting (caliber) so that
different evaluation settings are not directly compared against one another.

## 1. Source-domain spatial cross-validation

Identical source footprints and five 50 km block folds; model selection used an
identical restricted source-only tuning budget for both representations, with the
configuration chosen by nested spatial cross-validation.

| Representation | R² | RMSE (Mg/ha) | MAE (Mg/ha) | Bias (Mg/ha) |
|---|---:|---:|---:|---:|
| AlphaEarth + DEM | 0.5761 | 69.11 | 42.23 | +1.08 |
| Conventional + DEM | 0.4251 | 80.49 | 53.48 | +1.09 |

AlphaEarth improved source R² by 0.151 and reduced RMSE by ~14%.

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

Kaihua footprints were first assigned to 117 fixed 5 km × 5 km grid cells in
EPSG:32650, and entire cells were then assigned to five sample-balanced spatial
folds (~26,000 evaluation footprints each) so that no cell crosses a train/test
partition. Label budgets were
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

![Kaihua label efficiency](../figures/kaihua_label_efficiency.png)
![Calibration and local adaptation](../figures/calibration_vs_local_adaptation.png)

## Negative result (reported, not hidden)

Direct zero-shot transfer failed for both representations, and even 2,500 local
labels recovered only modest spatial-holdout performance. The project therefore
reports a restricted positive result (representation-level transfer advantage and
label efficiency) rather than presenting itself as a successful operational
mapping system. Formal Kaihua wall-to-wall biomass mapping was not pursued.

## 5. PALSAR-common paired benchmark

The exact-year common set uses all years 2019–2024, with 109,830 source and
107,809 target footprints. Source spatial-CV R² values were 0.097
(DEM), 0.283 (S1_C), 0.296 (PALSAR_L), 0.347 (C+L), 0.421 (S1+S2), 0.419
(PALSAR+S2), 0.430 (S1+PALSAR+S2) and 0.574 (AlphaEarth). PALSAR_L exceeded
S1_C in all five folds (mean ΔR² +0.0128; mean ΔRMSE −0.81 Mg/ha); C+L exceeded
both single-radar branches in all five folds.

All frozen zero-shot R² values were negative. PALSAR+S2 was least poor at −0.297
(RMSE 99.15 Mg/ha). PALSAR_L (−0.596) remained less poor than S1_C (−0.718),
and C+L reached −0.513, but full fusion was worst at −1.104.

Under the shared-draw local XGBoost protocol, AlphaEarth first achieved positive
mean holdout R² at 50 labels; S1+S2, PALSAR+S2 and full fusion at 500; S1_C,
C+L and PALSAR_L at 1,000; DEM at 2,500. At 2,500 labels AlphaEarth was
best (0.142 ± 0.020), and no representation reached 0.20. The predeclared mapping
gate therefore withheld a Kaihua wall-to-wall product.
