# Scripts

The pipeline is organized as standalone, sequentially-run entry points rather than a
single command. Scripts resolve the repository root from their own location
(`Path(__file__).resolve().parents[2]`), so run them from the repository root,
for example:

```bash
python scripts/03_modeling/train_and_freeze_source_models.py
```

Re-running the full pipeline requires an authenticated Earth Engine account
(`earthengine authenticate`) and will incur Earth Engine export costs. The frozen
result tables, figures, and reproducibility manifests are already included in
`outputs/`, so the conclusions can be inspected without any data download (see
[`docs/reproduction.md`](../docs/reproduction.md)).

## 01 — Data extraction (`01_data_extraction/`)

Earth Engine export and direct-download scripts. These submit or pull the raw
GEDI, AlphaEarth, Sentinel, and DEM inputs.

| Script | Purpose |
|---|---|
| `define_aoi.py` | Create deterministic first-pass source/target AOIs without target labels. |
| `prepare_target_aoi.py` | Prepare the Kaihua target AOI from the user-supplied boundary. |
| `freeze_target_aoi.py` | Validate and freeze the vetted Kaihua AOI (SHA-256 provenance). |
| `submit_gedi_source_exports.py` | Submit GEDI L4A source exports to Earth Engine. |
| `submit_source_aef_central_exports.py` | Submit AlphaEarth central-pixel source exports. |
| `submit_source_dem_exports.py` | Submit Copernicus GLO-30 DEM source exports. |
| `submit_source_sampling_exports.py` | Submit source sampling exports (also imported by the two scripts below). |
| `submit_source_manifest_200.py` | Submit the revised cap-200 source manifest exports. |
| `submit_source_sampling_audit_recovery.py` | One-off export supporting the sampling audit. |
| `download_source_aef_central_direct.py` | Download the frozen full-source central AEF directly from Earth Engine. |
| `download_source_conventional_direct.py` | Download frozen annual-median S1/S2 features directly from Earth Engine. |

## 02 — Data preparation (`02_data_preparation/`)

Finalize, QA, and freeze the extracted samples into the common representation and
the Kaihua few-shot design.

| Script | Purpose |
|---|---|
| `aggregate_gedi_qa_exports.py` | Aggregate per-granule GEDI QA count exports locally. |
| `gedi_qa_audit.py` | Source-only GEDI L4A QA, evaluated per official vector granule. |
| `finalize_source_sample.py` | Finalize and freeze completed sampled GEDI exports. |
| `finalize_source_sample_200.py` | Finalize the cap-200 source manifest and 75-cap AEF subset. |
| `finalize_source_aef_central.py` | Finalize full-source central AEF and join frozen DEM/AGBD. |
| `finalize_source_conventional.py` | Finalize conventional features and build the frozen common representation. |
| `finalize_source_dem.py` | Merge, validate, and freeze completed source DEM chunk exports. |
| `freeze_aef_method_and_prepare_full_source.py` | Freeze the selected AEF method and materialize its DEM-valid source input. |
| `prepare_kaihua_fewshot_design.py` | Build the frozen Kaihua few-shot fold design and label-draw budgets. |

## 03 — Modeling (`03_modeling/`)

Train source models, evaluate zero-shot transfer, and run few-shot adaptation.

| Script | Purpose |
|---|---|
| `train_and_freeze_source_models.py` | Train and freeze the source models for both representations. |
| `prepare_aef_development_common.py` | Build the common AEF development subset. |
| `finalize_and_compare_aef_development.py` | Finalize paired AEF development features and run source-only RF spatial CV. |
| `run_kaihua_locked_predictors.py` | Run the locked Kaihua predictors (imported by `evaluate_kaihua_zero_shot.py`). |
| `evaluate_kaihua_zero_shot.py` | Unlock Kaihua GEDI labels after prediction freeze and evaluate transfer. |
| `run_kaihua_fewshot_adaptation.py` | Run few-shot local adaptation on Kaihua. |
| `finalize_kaihua_locked_zero_shot.py` | Finalize Kaihua predictors and freeze zero-shot predictions before label unlock. |
| `submit_aef_development_exports.py` | Submit AEF development exports. |

## 04 — Diagnostics (`04_diagnostics/`)

Domain-shift diagnostics and final figure/manifest generation.

| Script | Purpose |
|---|---|
| `gpu_exact_nearest_aef.py` | Exact brute-force nearest-source AEF distances on CUDA. |
| `build_summary_outputs.py` | Verify frozen results, build the final master table, and generate summary figures. |
| `build_final_reproducibility_manifest.py` | Build the final compact reproducibility manifest from frozen artifacts. |

## utilities (`utilities/`)

| Script | Purpose |
|---|---|
| `guards.py` | Guards against target-label leakage; enforced at runtime by the modeling scripts. |
