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

### Version-2 PALSAR sequence

The existing private GEDI/Sentinel/AlphaEarth inputs may be supplied from another
verified checkout; hashes must match the public legacy freeze. Then run:

```bash
python scripts/01_data_extraction/audit_palsar_availability.py \
  --source-manifest <FROZEN_SOURCE_PRIVATE_CSV> \
  --target-manifest <KAIHUA_LOCKED_PREDICTORS> \
  --target-folds <KAIHUA_PRIVATE_FOLDS> --project <EE_PROJECT>
python scripts/01_data_extraction/download_source_palsar_direct.py
python scripts/01_data_extraction/download_kaihua_palsar_direct.py
python scripts/02_data_preparation/finalize_source_palsar.py
python scripts/02_data_preparation/finalize_kaihua_palsar.py
# Build source and locked-target common samples with build_palsar_common_sample.py.
python scripts/03_modeling/run_radar_representation_benchmark.py
python scripts/03_modeling/run_palsar_zero_shot.py predict
# Only after prediction freeze:
python scripts/03_modeling/run_palsar_zero_shot.py evaluate --labels <TARGET_LABEL_FILE>
python scripts/03_modeling/run_palsar_fewshot_adaptation.py --labels <TARGET_LABEL_FILE>
python scripts/04_diagnostics/analyze_radar_biomass_sensitivity.py
python scripts/04_diagnostics/analyze_radar_domain_shift.py --labels <TARGET_LABEL_FILE>
python scripts/04_diagnostics/build_palsar_benchmark_outputs.py
```

Direct downloads maintain ignored restart ledgers under `outputs/logs/`. Public
freezes contain aggregate counts and hashes, never private shot IDs/coordinates.

Deterministic seeds (`config.yaml`: `seed: 42`, few-shot seeds 42–44) and the
  frozen manifests in `outputs/manifests/` reproduce the exact sample draws and
  folds. Target labels are unlocked only after zero-shot predictions are frozen
  (see `utilities/guards.py`).

## Reproduction prerequisites

The three downstream runner scripts have different external dependencies. A failure
to run them in a sandbox is an **environment/dependency** failure, **not** a
scientific or test failure of the reported results (those live in
`outputs/tables/`, `figures/`, and `docs/`).

- **Stage A — `scripts/03_modeling/evaluate_kaihua_zero_shot.py`:** requires the
  dependencies listed in `requirements.txt` **and** an authenticated Google Earth
  Engine account (`earthengine-api`; the script calls `ee.Initialize()`). It is the
  only runner that needs GEE.
- **Stages B/C — `scripts/04_diagnostics/build_summary_outputs.py` and
  `build_final_reproducibility_manifest.py`:** require the *frozen runtime
  artifacts* (intermediate predictions and sample manifests under
  `outputs/predictions/` and `outputs/manifests/`). These large intermediates are
  intentionally **not distributed in Git** (see `.gitignore`); the committed
  `outputs/tables/` and `figures/` already contain the final, validated results.

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

## Stage 3 — Export GEDI source samples

- **Input:** source AOIs; public dataset IDs in `config.yaml` (GEDI L4A).
- **Output:** Earth Engine export tasks and Drive/asset GEDI sample tables.
- **Scripts:** `scripts/01_data_extraction/submit_gedi_source_exports.py`,
  `submit_source_manifest_200.py`.
- **GEE needed:** yes.
- **Cost / duration:** paid exports; can run for hours depending on quota.
- **Skip with frozen outputs?** yes — see `outputs/manifests/` and `data/README.md`
  for how to obtain the inputs instead.

## Stage 4 — Export AlphaEarth source samples

- **Input:** source AOIs; AlphaEarth `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`.
- **Output:** Earth Engine export tasks and Drive/asset AlphaEarth sample tables
  (central-pixel and AEF-development subsets).
- **Scripts:** `scripts/01_data_extraction/submit_source_aef_central_exports.py`,
  `submit_aef_development_exports.py`, `download_source_aef_central_direct.py`.
- **GEE needed:** yes — these scripts call `ee.Initialize()` and submit
  `ee.batch.Export` tasks.
- **Cost / duration:** paid exports; can run for hours depending on quota.
- **Skip with frozen outputs?** yes — see `outputs/manifests/` and `data/README.md`.

## Stage 5 — Export Sentinel-1/2 and DEM source samples

- **Input:** source AOIs; Sentinel-1/2 and Copernicus GLO-30 DEM assets.
- **Output:** Earth Engine export tasks and Drive/asset conventional-feature and
  DEM sample tables.
- **Scripts:** `scripts/01_data_extraction/submit_source_dem_exports.py`,
  `submit_source_sampling_exports.py`, `download_source_conventional_direct.py`.
- **GEE needed:** yes.
- **Cost / duration:** paid exports; can run for hours depending on quota.
- **Skip with frozen outputs?** yes — see `outputs/manifests/` and `data/README.md`.

## Stage 6 — Build frozen spatial folds

- **Input:** exported samples from Stages 3–5.
- **Output:** finalized, frozen source samples; the 50 km source block folds; the
  Kaihua 5 km few-shot fold design and label-draw budgets; GEDI QA tables.
- **Scripts:** `scripts/02_data_preparation/finalize_source_sample.py`,
  `finalize_source_sample_200.py`, `finalize_source_aef_central.py`,
  `finalize_source_conventional.py`, `finalize_source_dem.py`,
  `freeze_aef_method_and_prepare_full_source.py`, `prepare_kaihua_fewshot_design.py`,
  `gedi_qa_audit.py`, `aggregate_gedi_qa_exports.py`.
- **GEE needed:** no (operates on exported tables).
- **Cost / duration:** minutes.
- **Skip with frozen outputs?** yes — the frozen manifests already encode the folds
  and draws.

## Stage 7 — Train and freeze source models

- **Input:** frozen source samples.
- **Output:** trained and frozen source models for both representations (AlphaEarth
  + DEM, conventional + DEM); the pre-declared AEF aggregation decision.
- **Scripts:** `scripts/03_modeling/train_and_freeze_source_models.py`,
  `prepare_aef_development_common.py`, `finalize_and_compare_aef_development.py`.
- **GEE needed:** no.
- **Cost / duration:** minutes to low-hours of CPU.
- **Skip with frozen outputs?** partly — model metrics are in
  `outputs/tables/main_results/`.

## Stage 8 — Run zero-shot evaluation

- **Input:** frozen source models; frozen Kaihua predictors.
- **Output:** zero-shot predictions (hashed before label unlock); the zero-shot
  transfer table.
- **Scripts:** `scripts/03_modeling/run_kaihua_locked_predictors.py`,
  `evaluate_kaihua_zero_shot.py`, `finalize_kaihua_locked_zero_shot.py`.
- **GEE needed:** **yes when target labels must be regenerated.** If
  `data/processed/kaihua_gedi_labels_unlocked.csv` is already present (the frozen
  case), no Earth Engine call is made and the stage runs locally. Otherwise
  `evaluate_kaihua_zero_shot.py` calls `ee.Initialize()` and pulls the Kaihua GEDI
  L4A labels from Earth Engine, so a GEE account is required for a full
  regeneration.
- **Cost / duration:** minutes locally; the label-regeneration path adds Earth
  Engine export time.
- **Skip with frozen outputs?** yes — results are in
  `outputs/tables/main_results/representation_transfer_summary.csv`.

## Stage 9 — Run few-shot adaptation

- **Input:** frozen Kaihua folds and label draws.
- **Output:** few-shot adaptation results across label budgets 25–2,500.
- **Script:** `scripts/03_modeling/run_kaihua_fewshot_adaptation.py`.
- **GEE needed:** no.
- **Cost / duration:** minutes.
- **Skip with frozen outputs?** yes — results are in
  `outputs/tables/main_results/kaihua_fewshot_summary.csv`.

## Stage 10 — Generate figures and reproducibility manifest

- **Input:** frozen results and predictions.
- **Output:** publication-style figures in `figures/` and the final
  reproducibility manifest.
- **Scripts:** `scripts/04_diagnostics/gpu_exact_nearest_aef.py`,
  `build_summary_outputs.py`, `build_final_reproducibility_manifest.py`.
- **GEE needed:** no.
- **Cost / duration:** minutes (the GPU distance script is optional and needs CUDA).
- **Skip with frozen outputs?** yes — figures and the manifest are already
  committed.

## Data not in this repository

Raw and processed remote-sensing data are excluded (volume, redistribution terms,
Earth Engine workflow). See [`data/README.md`](../data/README.md) for dataset
names, time spans, sources, licensing posture, and how to regenerate local inputs.
Target validation uses GEDI L4A footprint estimates, not independent field plots.
