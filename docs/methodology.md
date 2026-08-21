# Methodology

This page records the design decisions that were frozen before any Kaihua
evaluation. It is the methodological companion to
[`technical_report.md`](technical_report.md) and the public
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

### Study design rationale

- **Why Georgia + South Carolina as the source.** The southeastern U.S. is a
  well-sampled temperate forest region with dense, high-quality GEDI L4A
  footprint coverage and a marked ecological contrast to subtropical East Asia.
  Using consistent state boundaries keeps the source region unambiguous and
  reproducible, and the contrast makes a credible hard case for geographic
  transfer rather than an easy within-ecoregion extrapolation.
- **Why Kaihua County as the target.** Kaihua is a subtropical forested county in
  western Zhejiang that includes the Qianjiangyuan protected forest area. It is a
  realistic operational target (a single manageable administrative unit with
  available GEDI coverage) while differing sharply from the source in climate,
  species composition, and disturbance regime — exactly the shift a transfer
  method should be tested against.
- **Why 50 km blocks in the source but 5 km blocks in the target.** Source
  sampling spans a large multistate area, so 50 km blocks yield enough footprints
  per fold for stable spatial cross-validation. The target is a single small
  county: at 50 km the county would collapse into only one or two blocks with too
  few partitions for meaningful spatial holdout. Kaihua GEDI footprints are
  therefore first assigned to 117 fixed 5 km × 5 km grid cells in EPSG:32650, and
  entire grid cells are then assigned to five approximately sample-balanced
  spatial folds (~26,000 evaluation footprints each) so that no grid cell crosses
  a train/test partition.

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
extraction and before any Zhejiang evaluation**. The decision is recorded in the
frozen reproducibility manifest
(`outputs/manifests/final_reproducibility_manifest.json`) alongside the exact
R² and RMSE values.

## Provenance note

AlphaEarth pretraining incorporates GEDI L2A relative-height information. This is
documented in the AlphaEarth Foundations release, whose training-data table lists
"LiDAR GEDI L2A Relative height metrics (rh*)" as one of the model's target
modalities (Brown et al., 2025, *AlphaEarth Foundations: An embedding field
model for accurate and efficient global mapping from sparse label data*,
arXiv:2507.22291, Table S1).

This is **GEDI-derived structural provenance overlap, not direct downstream AGBD
label leakage**: the AlphaEarth features used here are the public annual
embeddings, and no AGBD labels entered representation selection.

## Data sources and references

- GEDI L4A Aboveground Biomass Density — NASA GEDI, distributed by the ORNL DAAC
  (Earth Engine asset `LARSE/GEDI/GEDI04_A_002_MONTHLY`).
- AlphaEarth Foundations annual embeddings — Google DeepMind; Brown, N., et al.
  (2025). *AlphaEarth Foundations: An embedding field model for accurate and
  efficient global mapping from sparse label data.* arXiv:2507.22291.
- Sentinel-1 GRD and Sentinel-2 SR Harmonized — Copernicus programme, ESA (Earth
  Engine assets `COPERNICUS/S1_GRD` and `COPERNICUS/S2_SR_HARMONIZED`).
- Copernicus DEM GLO-30 — Copernicus programme, ESA (Earth Engine asset
  `COPERNICUS/DEM/GLO30`).

Dataset IDs and access paths are also listed in `config.yaml` and
[`data/README.md`](../data/README.md).

## Reproducibility artifacts

- Frozen designs, fold mappings, label draws, model freezes, and SHA-256
  checksums: `outputs/manifests/`.
- Exact package versions: `environment.yml`.
- All dataset IDs, AOI/sample/model/prediction hashes, spatial designs, feature
  rules, seeds, and final artifact hashes:
  `outputs/manifests/final_reproducibility_manifest.json`.

## Radar wavelength benchmark

Version 2 adds a paired sensor/wavelength representation benchmark. L-band may
respond differently to woody structure than C-band, but this is not a perfectly
isolated wavelength experiment: Sentinel-1 and PALSAR-2 also differ in polarization
(VV/VH versus HH/HV), geometry, acquisition strategy, temporal sampling and
preprocessing.

PALSAR is read only from `JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH`. Nonpositive HH/HV DN
are masked before gamma-naught conversion (`20 log10(DN) - 83`). Primary features
are HH dB, HV dB, HH−HV dB and RFDI; RFDI uses squared DN (linear power). `angle`,
`epoch`, derived acquisition date and `qa` are QC metadata and forbidden predictors.
The authenticated QA audit follows the authoritative JAXA v2.4 product description:
`qa=1` is ScanSAR land and `qa=255` is land, so both are valid. Classes 2/3/4 and
50/100/150 remain excluded as layover, shadow or water. Earth Engine's catalog page
omits the ScanSAR 1–4 classes even though they occur in the raw asset. Each exact-year
collection contains one image and is read with `first()`; `mosaic()` is avoided because
it discards the native 25 m default projection even for a one-image collection.

The central 25 m pixel is frozen before target results. No textures, patches,
timing predictors or nearest-year substitutions are permitted. An authenticated
audit fixes `palsar_common_years`; all eight representations are then rerun on one
ordered common sample with whole-block folds and the same tuning budget.

The validity filter retained 109,830 of 124,303 source rows and 107,809 of 130,195
target rows. The aggregate selection audit found only small AGBD shifts (source
mean 117.13→117.88 Mg/ha; target 122.74→119.52) and small year/block distribution
shifts, so the frozen sample was not rebalanced.

GEDI L4A AGBD remains the response; GEDI L2A RH metrics are excluded as predictors.
Zhao et al.'s IGARSS 2023 project is methodological inspiration only; its unlicensed
repository code was not copied. References: [Earth Engine JAXA yearly mosaic](https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_PALSAR_YEARLY_SAR_EPOCH),
[JAXA PALSAR-2 mosaic v2.4 product description](https://www.eorc.jaxa.jp/ALOS/en/dataset/pdf/DatasetDescription_PALSAR2_Mosaic_ver240.pdf),
[Sentinel-1](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD),
[GEDI L4A](https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI04_A_002_MONTHLY),
and [Zhao et al. 2023](https://doi.org/10.1109/IGARSS52108.2023.10282061).
