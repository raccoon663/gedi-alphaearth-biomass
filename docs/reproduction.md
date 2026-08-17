# Reproduction

This repository ships the **frozen result tables, figures, and reproducibility
manifests**, so most of the scientific conclusions can be inspected without running
anything. The scripts under [`scripts/`](../scripts) are the actual pipeline that
produced those artifacts. They are standalone entry points run in order; there is
no single wrapper command.

Two reproduction levels are supported:

- **Inspect existing results.** No data download and no Earth Engine account
  required. Read `outputs/tables/`, `figures/`, and `docs/`.
- **Full reproduction.** Requires a Google Earth Engine account and the local
  Kaihua AOI input. Some stages submit Earth Engine exports that incur compute
  costs and can run for hours.

Deterministic seeds (`config.yaml`: `seed: 42`, few-shot seeds 42–44) and the
frozen manifests in `outputs/manifests/` reproduce the exact sample draws and
folds. Target labels are unlocked only after zero-shot predictions are frozen
(see `utilities/guards.py`).

## Stage 1 — Authenticate Earth Engine

- **Input:** a Google account with Earth Engine access.
- **Output:** local credentials (`earthengine authenticate`).
- **Script:** none (Earth Engine CLI).
- **GEE needed:** yes (one-time).
- **Cost / duration:** none.
- **Skip with frozen outputs?** yes — only needed for full reproduction.

## Stage 2 — Define source and target AOIs

- **Input:** source state names (`config.yaml`: Georgia, South Carolina); the
  Kaihua boundary file (`data/boundaries/kaihua_target_aoi.geojson`, not
  committed).
- **Output:** source/target AOI assets; a frozen, SHA-256-recorded Kaihua AOI.
- **Scripts:** `scripts/01_data_extraction/define_aoi.py`,
  `prepare_target_aoi.py`, `freeze_target_aoi.py`.
- **GEE needed:** no (boundary validation is local).
- **Cost / duration:** seconds.
- **Skip with frozen outputs?** yes — the frozen AOI hashes are in
  `outputs/manifests/`.

## Stage 3 — Export GEDI and predictor samples

- **Input:** AOIs; public dataset IDs in `config.yaml` (GEDI L4A, AlphaEarth,
  Sentinel-1/2, GLO-30).
- **Output:** Earth Engine export tasks and Drive/asset sample tables.
- **Scripts:** `scripts/01_data_extraction/submit_gedi_source_exports.py`,
  `submit_source_aef_central_exports.py`, `submit_source_dem_exports.py`,
  `submit_source_sampling_exports.py`, `submit_source_manifest_200.py`,
  `download_source_aef_central_direct.py`, `download_source_conventional_direct.py`.
- **GEE needed:** yes.
- **Cost / duration:** paid exports; can run for hours depending on quota.
- **Skip with frozen outputs?** yes — see `outputs/manifests/` and `data/README.md`
  for how to obtain the inputs instead.

## Stage 4 — Build frozen spatial folds

- **Input:** exported samples from Stage 3.
- **Output:** finalized, cryptographically frozen source samples; the 50 km source
  block folds; the Kaihua 5 km few-shot fold design and label-draw budgets; GEDI QA
  tables.
- **Scripts:** `scripts/02_data_preparation/finalize_source_sample.py`,
  `finalize_source_sample_200.py`, `finalize_source_aef_central.py`,
  `finalize_source_conventional.py`, `finalize_source_dem.py`,
  `freeze_aef_method_and_prepare_full_source.py`, `prepare_kaihua_fewshot_design.py`,
  `gedi_qa_audit.py`, `aggregate_gedi_qa_exports.py`.
- **GEE needed:** no (operates on exported tables).
- **Cost / duration:** minutes.
- **Skip with frozen outputs?** yes — the frozen manifests already encode the folds
  and draws.

## Stage 5 — Train source models

- **Input:** frozen source samples.
- **Output:** trained and frozen source models for both representations (AlphaEarth
  + DEM, conventional + DEM); the pre-declared AEF aggregation decision.
- **Scripts:** `scripts/03_modeling/train_and_freeze_source_models.py`,
  `prepare_aef_development_common.py`, `finalize_and_compare_aef_development.py`,
  `submit_aef_development_exports.py`.
- **GEE needed:** no.
- **Cost / duration:** minutes to low-hours of CPU.
- **Skip with frozen outputs?** partly — model metrics are in
  `outputs/tables/main_results/`.

## Stage 6 — Run zero-shot evaluation

- **Input:** frozen source models; frozen Kaihua predictors.
- **Output:** zero-shot predictions (hashed before label unlock); the zero-shot
  transfer table.
- **Scripts:** `scripts/03_modeling/run_kaihua_locked_predictors.py`,
  `evaluate_kaihua_zero_shot.py`, `finalize_kaihua_locked_zero_shot.py`.
- **GEE needed:** no.
- **Cost / duration:** minutes.
- **Skip with frozen outputs?** yes — results are in
  `outputs/tables/main_results/representation_transfer_summary.csv`.

## Stage 7 — Run few-shot adaptation

- **Input:** frozen Kaihua folds and label draws.
- **Output:** few-shot adaptation results across label budgets 25–2,500.
- **Script:** `scripts/03_modeling/run_kaihua_fewshot_adaptation.py`.
- **GEE needed:** no.
- **Cost / duration:** minutes.
- **Skip with frozen outputs?** yes — results are in
  `outputs/tables/main_results/kaihua_fewshot_summary.csv`.

## Stage 8 — Generate figures

- **Input:** frozen results and predictions.
- **Output:** publication-style figures in `figures/` and the final
  reproducibility manifest.
- **Scripts:** `scripts/04_diagnostics/gpu_exact_nearest_aef.py`,
  `build_final_portfolio_outputs.py`, `build_final_reproducibility_manifest.py`.
- **GEE needed:** no.
- **Cost / duration:** minutes (the GPU distance script is optional and needs CUDA).
- **Skip with frozen outputs?** yes — figures and the manifest are already
  committed.

## Data not in this repository

Raw and processed remote-sensing data are excluded (volume, redistribution terms,
Earth Engine workflow). See [`data/README.md`](../data/README.md) for dataset
names, time spans, sources, licensing posture, and how to regenerate local inputs.
Target validation uses GEDI L4A footprint estimates, not independent field plots.
