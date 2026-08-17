# Methodology

This page records the design decisions that were frozen before any Kaihua
evaluation. It is the methodological companion to
[`research_report.md`](research_report.md) and the public
[`README.md`](../README.md).

## Study regions

- **Source domain:** Georgia and South Carolina, USA, defined from consistent
  state boundaries. Used exclusively for source sampling, representation
  selection, and source-model development.
- **Target domain:** Kaihua County, Zhejiang, China (administrative code
  `330824`), the sole target. Its vetted boundary was frozen before target
  analysis and protected by SHA-256 provenance records (`kaihua_target_aoi.sha256`).

The design intentionally creates a difficult transfer setting: a model trained
in the southeastern United States is applied to an eastern Chinese forest
landscape **without using target labels until after zero-shot predictions are
frozen**.

## Data

- **GEDI L4A V2.1:** footprint-level AGBD response and quality fields, 2019–2024.
- **AlphaEarth:** `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, bands A00–A63.
- **Sentinel-1:** `COPERNICUS/S1_GRD`, annual-median VV and VH (dB), both orbit
  directions pooled, plus VV−VH.
- **Sentinel-2:** `COPERNICUS/S2_SR_HARMONIZED`, scaled reflectance with frozen
  SCL + QA60 masking, ten bands and five indices (NDVI, EVI, NDMI, NBR, NDRE).
- **Topography:** `COPERNICUS/DEM/GLO30`, elevation, slope, sine/cosine aspect.

GEDI L4A AGBD is treated as the **downstream response product**, not as
field-measured biomass truth. Longitude and latitude are metadata only and are
never model predictors.

## Source sampling

Source GEDI footprints were assigned to 50 km × 50 km blocks in EPSG:5070 before
sampling. Strata were spatial block × year, with deterministic random selection,
**seed 42** and a cap of 200 QA-valid footprints per stratum. Strata below the
cap were retained completely; there was **no oversampling, no AGBD quantile
balancing, and no target-dependent resampling**.

The frozen source manifest contains 127,622 footprints. After the frozen DEM
mask, both representation branches use the identical ordered common set of
**124,303 samples**. Five folds were mapped strictly at block level so that no
source block crosses train/test partitions.

## Representation construction

- **AlphaEarth branch:** A00–A63 plus four DEM variables. Annual embeddings use
  the exact footprint year and the **central 10 m pixel**, without
  nearest-year substitution or target-specific renormalization.
- **Conventional branch:** Sentinel-1 VV/VH/VV−VH; ten Sentinel-2 bands; the
  five indices above; and the same four DEM variables. All optical/radar
  predictors use the exact calendar year and a frozen annual-median temporal
  rule in both domains.

## AlphaEarth aggregation decision (pre-declared)

A source-only development subset compared the central 10 m pixel with a 25 m
component-wise mean followed by L2 renormalization. Under a predeclared source
block-CV rule the two were practically equivalent: central mean R² was 0.56735
and aggregate-25 mean R² was 0.56597, a relative RMSE difference of 0.156%. The
rule therefore selected the less expensive central-pixel method **before full
extraction and before any Zhejiang evaluation** (see
`outputs/tables/aef_aggregation_decision.json`).

## Provenance note

AlphaEarth pretraining incorporates GEDI L2A relative-height information. This is
**GEDI-derived structural provenance overlap, not direct downstream AGBD label
leakage**: the AlphaEarth features used here are the public annual embeddings,
and no AGBD labels entered representation selection.

## Reproducibility artifacts

- Frozen designs, fold mappings, label draws, model freezes, and SHA-256
  checksums: `outputs/manifests/`.
- Exact package versions: `environment.yml`.
- All dataset IDs, AOI/sample/model/prediction hashes, spatial designs, feature
  rules, seeds, and final artifact hashes:
  `outputs/manifests/final_reproducibility_manifest.json`.
