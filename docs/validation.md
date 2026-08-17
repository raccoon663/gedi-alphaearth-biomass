# Validation

This page documents how the results were validated and the limits of that
validation. It accompanies [`technical_report.md`](technical_report.md) and
[`experiments.md`](experiments.md).

## Validation design

- **Source domain:** strict block-level five-fold spatial cross-validation. Folds
  were mapped at the 50 km block level so that no source block crosses
  train/test partitions (`config.yaml`: `validation.spatial_block_km: 50`,
  `spatial_folds: 5`).
- **Target domain:** five fixed 5 km spatial blocks in EPSG:32650, each with
  ~26,000 evaluation footprints. Label budgets and seeds (42–44) were shared
  identically across both representations to isolate the representation effect.

## GEDI quality control

Source GEDI footprints passed a frozen QA gate (`config.yaml`: `gedi_qa`):
`algorithm_run_flag = 1`, `degrade_flag = 0`, finite AGBD standard error
required, and physically invalid negative AGBD rejected. The QA pass rates and
per-year breakdowns are in `outputs/tables/audits/gedi_qa.csv` and
`outputs/tables/audits/gedi_qa_by_year.csv`.

## Domain diagnostics (not model selection)

- A source-fit PCA of AlphaEarth embeddings shows strong USA–Kaihua displacement.
- A logistic domain classifier reaches AUROC 1.000, i.e. the two domains are
  near-perfectly separable in embedding space.
- Nearest-source Euclidean embedding distance explains almost none of the
  within-Kaihua absolute zero-shot error (Spearman ρ = 0.022).

These diagnostics describe *why* zero-shot transfer failed; they were not used as
a target-model selection rule, and separability does not imply that a sample's
domain probability predicts its biomass error.

## What the validation does NOT establish

- **No independent ground truth.** Both source and target evaluation use GEDI L4A
  AGBD footprint estimates. There are no independent Chinese field plots, so the
  study evaluates transfer relative to a GEDI-derived label, not field-level
  accuracy.
- **One source, one target.** Only Georgia/South Carolina and Kaihua County were
  tested. Results should not be read as general across all ecoregions.
- **Limited adaptation search.** Few-shot adaptation used one fixed lightweight
  model rather than extensive target hyperparameter search.
- **Operational claim withheld.** Absolute target performance remained too weak
  to support a reliable operational biomass map; wall-to-wall Kaihua mapping was
  deliberately not pursued.

## Reproducibility checks

SHA-256 checksums of frozen AOI, sample manifests, model freezes, and prediction
files are recorded in `outputs/manifests/*.sha256` and consolidated in
`outputs/manifests/final_reproducibility_manifest.json`. Re-running the frozen
scripts with `seed: 42` reproduces the exact sample draws and folds.
