# Project Summary

**Cross-Region Forest Biomass Transfer with GEDI and AlphaEarth**

Sparse GEDI footprints provide broad but discontinuous information on forest
aboveground biomass density (AGBD), motivating the use of continuous
Earth-observation representations for prediction and geographic transfer. This
project compared **AlphaEarth annual embeddings** with **conventional
Sentinel-1/Sentinel-2 features** for transferring GEDI L4A biomass relationships
from Georgia and South Carolina, USA, to **Kaihua County, Zhejiang, China**.

A deterministic spatial/year-balanced source design produced **124,303 common
footprints** after terrain masking, evaluated with shared five-fold 50 km block
cross-validation. AlphaEarth plus topography achieved source **R² 0.581** and
RMSE 68.70 Mg/ha, compared with **R² 0.434** and RMSE 79.88 Mg/ha for
conventional features. Models and target predictors were frozen before Kaihua
labels were unlocked.

Zero-shot transfer failed for both representations: target R² was **−0.556** for
AlphaEarth and **−1.117** for conventional features, despite a consistent relative
AlphaEarth advantage. Predictor diagnostics showed extreme USA–Kaihua separability
(domain-classifier AUROC 1.000). Few-shot adaptation used identical random
target-label draws under five-fold 5 km spatial holdout. Local AlphaEarth models
first reached positive mean R² with **250 labels**, whereas conventional models
required **500**; at 2,500 labels, R² values were **0.131** and **0.059**.

These results indicate stronger AlphaEarth representation-level label efficiency
under severe domain shift, but absolute target performance remained insufficient
for reliable operational biomass mapping.

## Key numbers

| Stage | AlphaEarth + DEM | Conventional + DEM |
|---|---:|---:|
| USA source spatial-CV R² | 0.581 | 0.434 |
| Kaihua zero-shot R² | −0.556 | −1.117 |
| First positive mean few-shot R² | 250 labels | 500 labels |
| Kaihua local R² at 2,500 labels | 0.131 ± 0.017 | 0.059 ± 0.011 |

## Scope and limits

GEDI L4A is not field truth, no independent Chinese field plots were available,
and AlphaEarth retains GEDI-derived structural provenance through pretraining.
Only one source geography and one Chinese target region were evaluated. The
project answers a representation-transfer question; it does not present an
operational mapping product.

See [`README.md`](../README.md), [`docs/methodology.md`](methodology.md),
[`docs/experiments.md`](experiments.md), and
[`docs/research_report.md`](research_report.md) for full detail.
