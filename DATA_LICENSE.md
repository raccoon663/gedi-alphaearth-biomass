# Data licensing notice

The MIT License in [`LICENSE`](LICENSE) applies **only to the source code** in
this repository.

The remote-sensing datasets used by the project — GEDI L4A Aboveground Biomass
Density (NASA ORNL DAAC), AlphaEarth Foundations annual embeddings (Google
DeepMind), Sentinel-1 and Sentinel-2 (Copernicus, ESA), and Copernicus DEM GLO-30
(ESA) — are distributed by their respective providers under separate terms and are
**not** included in this repository. Nothing in this repository should be read as
granting rights to those datasets.

See [`data/README.md`](data/README.md) for dataset names, time spans, sources, and
licensing posture. Per-sample manifests that carry GEDI shot IDs or coordinates are
excluded from the repository; frozen designs, folds, seeds, and SHA-256 checksums
remain in `outputs/manifests/` so the experiment can be verified without them.
