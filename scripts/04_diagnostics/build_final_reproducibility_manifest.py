"""Build the final compact reproducibility manifest from frozen artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

RECORDED_EXPERIMENT_PACKAGES = {
    "python": "3.12.10",
    "earthengine-api": "1.7.38",
    "geemap": "0.38.3",
    "geopandas": "1.1.4",
    "pyogrio": "0.13.0",
    "PyYAML": "6.0.3",
    "pandas": "3.0.3",
    "numpy": "2.5.0",
    "scikit-learn": "1.9.0",
    "xgboost": "3.4.0",
    "matplotlib": "3.11.0",
    "seaborn": "0.13.2",
    "joblib": "1.5.3",
    "pyarrow": "25.0.1",
    "shapely": "2.1.2",
    "pyproj": "3.7.2",
    "requests": "2.34.2",
}


def sha256(relative_path: str) -> str:
    path = ROOT / relative_path
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def hashed_files(paths: list[str]) -> dict[str, dict[str, object]]:
    result = {}
    for relative_path in paths:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative_path] = {
            "sha256": sha256(relative_path),
            "bytes": path.stat().st_size,
        }
    return result


def main() -> None:
    source_sample = load_json("outputs/manifests/frozen_source_manifest_200.json")
    source_models = load_json("outputs/manifests/frozen_source_model_manifest.json")
    zero_shot = load_json("outputs/manifests/kaihua_zero_shot_freeze.json")
    fewshot_design = load_json("outputs/manifests/kaihua_fewshot_design_freeze.json")

    # Recompute and hard-check the most consequential frozen hashes.
    checks = {
        "source_sample": (
            "outputs/manifests/frozen_source_manifest_200.csv",
            source_sample["manifest_csv_sha256"],
        ),
        "source_model_alphaearth": (
            "outputs/models/frozen_source_alphaearth_xgboost.joblib",
            source_models["selected_models"]["alphaearth"]["model_sha256"],
        ),
        "source_model_conventional": (
            "outputs/models/frozen_source_conventional_xgboost.joblib",
            source_models["selected_models"]["conventional"]["model_sha256"],
        ),
        "zero_shot_alphaearth": (
            "outputs/predictions/kaihua_zero_shot_aef.csv",
            zero_shot["prediction_file_hashes"]["alphaearth"],
        ),
        "zero_shot_conventional": (
            "outputs/predictions/kaihua_zero_shot_conventional.csv",
            zero_shot["prediction_file_hashes"]["conventional"],
        ),
        "fewshot_folds": (
            "outputs/manifests/kaihua_fewshot_spatial_folds.csv",
            fewshot_design["fold_manifest_sha256"],
        ),
        "fewshot_label_draws": (
            "outputs/manifests/kaihua_fewshot_label_draws.csv",
            fewshot_design["label_draw_manifest_sha256"],
        ),
    }
    verification = {}
    for name, (path, expected) in checks.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Frozen hash mismatch for {name}: {observed} != {expected}")
        verification[name] = {"path": path, "sha256": observed, "verified": True}

    main_artifacts = [
        "README.md",
        "PROJECT_STATUS.md",
        "environment.yml",
        "reports/final_research_report.md",
        "reports/kaihua_zero_shot_transfer.md",
        "reports/kaihua_fewshot_adaptation.md",
        "docs/cv_project_bullets.md",
        "docs/project_abstract.md",
        "docs/faculty_outreach_summary.md",
        "docs/repository_audit.md",
        "outputs/tables/final_project_summary.csv",
        "outputs/tables/source_model_comparison.csv",
        "outputs/tables/representation_transfer_summary.csv",
        "outputs/tables/kaihua_fewshot_summary.csv",
        "outputs/tables/kaihua_fewshot_label_efficiency.csv",
        "outputs/tables/kaihua_label_thresholds.csv",
        "figures/final_workflow.png",
        "figures/source_representation_comparison.png",
        "figures/zero_shot_transfer.png",
        "figures/kaihua_label_efficiency_final.png",
        "figures/calibration_vs_local_adaptation.png",
        "figures/usa_kaihua_domain_shift.png",
        "figures/aef_distance_vs_error.png",
    ]

    manifest = {
        "status": "CORE_EXPERIMENTS_COMPLETE_FINAL_SYNTHESIS_COMPLETE",
        "created_at": datetime.now().astimezone().isoformat(),
        "scientific_scope": {
            "source_domain": "Georgia and South Carolina, USA",
            "source_aoi_rule": "Google Earth Engine TIGER/2018/States features NAME in {Georgia, South Carolina}",
            "target_domain": "Kaihua County (administrative code 330824), Zhejiang, China",
            "target_aoi_file": "data/boundaries/kaihua_target_aoi.geojson",
            "target_aoi_sha256": zero_shot["target_aoi_sha256"],
            "wall_to_wall_mapping_performed": False,
        },
        "earth_engine_dataset_ids": {
            "alphaearth": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
            "gedi_l4a_monthly_raster": "LARSE/GEDI/GEDI04_A_002_MONTHLY",
            "gedi_l4a_vector_granules": "LARSE/GEDI/GEDI04_A_002/*",
            "sentinel_1": "COPERNICUS/S1_GRD",
            "sentinel_2": "COPERNICUS/S2_SR_HARMONIZED",
            "dem": "COPERNICUS/DEM/GLO30",
        },
        "frozen_feature_rules": {
            "alphaearth": source_models["aef_extraction_method"],
            "conventional": source_models["conventional_temporal_rule"],
            "dem": "Copernicus GLO-30 elevation, slope, aspect_sin, aspect_cos",
        },
        "sampling_and_validation": {
            "source": {
                "manifest_sha256": source_sample["manifest_csv_sha256"],
                "sample_n": source_sample["sample_n"],
                "crs": "EPSG:5070",
                "block_size_km": 50,
                "strata": "spatial_block_id x GEDI year",
                "cap_per_stratum": source_sample["sampling_cap"],
                "seed": 42,
                "folds": 5,
                "no_agbd_based_sampling": True,
                "no_oversampling": True,
                "zhejiang_used_for_source_decisions": False,
            },
            "target": {
                "shot_manifest_sha256": zero_shot["target_shot_manifest_sha256"],
                "evaluation_n": zero_shot["target_common_n"],
                "crs": fewshot_design["target_block_crs"],
                "block_size_km": fewshot_design["target_block_size_m"] / 1000,
                "block_n": fewshot_design["target_block_n"],
                "folds": fewshot_design["folds"],
                "fold_seed": fewshot_design["fold_seed"],
                "label_draw_seeds": fewshot_design["label_draw_seeds"],
                "label_budgets": fewshot_design["budgets"],
            },
        },
        "frozen_asset_verification": verification,
        "environment": {
            "recorded_experiment_package_versions": RECORDED_EXPERIMENT_PACKAGES,
            "manifest_generation_python": platform.python_version(),
            "environment_file": "environment.yml",
        },
        "main_artifacts": hashed_files(main_artifacts),
        "interpretation_guardrails": {
            "target_labels": "GEDI L4A footprint estimates, not independent field plots",
            "zero_shot_transfer_operationally_adequate": False,
            "fewshot_mean_r2_reached_0_2": False,
            "formal_biomass_mapping_recommended": False,
        },
    }

    output = ROOT / "outputs/manifests/final_reproducibility_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Verified {len(verification)} frozen assets; hashed {len(main_artifacts)} final artifacts")


if __name__ == "__main__":
    main()
