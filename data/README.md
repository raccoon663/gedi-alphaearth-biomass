# Data

This repository distributes **code, frozen result tables, frozen reproducibility
manifests, and figures**. The underlying remote-sensing datasets are **not
uploaded to GitHub** because of their volume, redistribution terms, and
Earth-Engine export workflow. This page explains what the data are, where they
come from, their licensing posture, and how to regenerate the local inputs
required to re-run the pipeline.

## Datasets

| Role | Dataset | Source / ID | Period |
|---|---|---|---|
| Response | GEDI L4A AGBD | NASA GEDI `LARSE/GEDI/GEDI04_A_002` (monthly) via Google Earth Engine | 2019–2024 |
| Representation A | AlphaEarth (Google Satellite Embedding V1, ANNUAL) | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, bands A00–A63 | 2019–2024 |
| Representation B | Sentinel-1 GRD | `COPERNICUS/S1_GRD` (annual-median VV/VH, dB) | 2019–2024 |
| Representation B | Sentinel-2 SR Harmonized | `COPERNICUS/S2_SR_HARMONIZED` (10 bands + 5 indices) | 2019–2024 |
| Topography | Copernicus GLO-30 DEM | `COPERNICUS/DEM/GLO30` (elevation, slope, aspect) | static |

## Why the raw data are not in this repository

- **Volume.** Extracted source/validation tables and intermediate products
  total several hundred megabytes per region; the frozen source manifest alone
  is tens of megabytes. GitHub is not an appropriate distribution channel for
  these.
- **Redistribution terms.** GEDI, Sentinel, and DEM products are distributed
  under NASA / Copernicus terms that do not permit silent re-hosting. AlphaEarth
  (Google Satellite Embedding V1) terms should be reviewed by the user before
  any reuse; the code here only *references* the public Earth Engine asset IDs.
- **Reproducible by design.** Every input is regenerated deterministically from
  public Earth Engine assets using the scripts in `scripts/` and the frozen
  configurations in `config.yaml` and `outputs/manifests/`.

## Licensing posture (verify before reuse)

- GEDI L4A: NASA GEDI data, generally open with attribution.
- Sentinel-1/2: Copernicus freely available with attribution.
- Copernicus GLO-30 DEM: Copernicus, free with attribution.
- **AlphaEarth / Google Satellite Embedding V1:** license terms are set by the
  provider. Confirm acceptance before any derivative product or publication.
  This repository does not bundle AlphaEarth data and only points at the public
  asset ID.

## Obtaining the data and reproducing local inputs

1. Install dependencies from `requirements.txt` / `environment.yml`.
2. Authenticate Google Earth Engine (`earthengine authenticate`) with your own
   project; the extraction scripts call `ee.Initialize()` and require an
   Earth-Engine-enabled account.
3. Run the extraction and freeze scripts in `scripts/` in the order implied by
   `docs/methodology.md` and `docs/reproduction.md`. Deterministic seeds
   (`config.yaml`: `seed: 42`) and frozen manifests reproduce the exact sample
   draws and folds.
4. Place local inputs under the `data/` tree expected by `config.yaml`
   (e.g. `data/boundaries/kaihua_target_aoi.geojson`). Boundary files are
   public administrative boundaries and must be supplied by the user; they are
   intentionally not committed.

## What is shipped instead

- `outputs/tables/*.csv` — frozen result and audit tables (no model weights, no
  raw rasters).
- `outputs/manifests/*.json` and selected `*.csv` — frozen sample designs,
  fold mappings, label draws, and SHA-256 checksums of frozen artifacts.
- `figures/*.png` — publication-style result figures.
- `docs/technical_report.md` — the full methodological and results synthesis.

Large binary products (model weights, prediction rasters, raw/processed rasters,
all `*.parquet` intermediates, and the per-sample manifests that carry GEDI shot
IDs or coordinates, e.g. `source_aef_central_input.csv` and
`kaihua_fewshot_spatial_folds.csv`) are excluded to keep the repository lean and
avoid redistributing sample-level locations. The frozen designs, folds, seeds, and
SHA-256 checksums remain so the experiment can be verified without those files.
