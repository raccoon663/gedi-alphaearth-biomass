"""Build a v2 manifest without modifying the canonical legacy freeze."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    legacy = ROOT / "outputs/manifests/final_reproducibility_manifest.json"
    legacy_tree_audit = json.loads((ROOT / "outputs/manifests/legacy_frozen_experiment_audit.json").read_text())
    legacy_blob = next(x["git_blob_sha1"] for x in legacy_tree_audit["inventory"]
                       if x["path"] == "outputs/manifests/final_reproducibility_manifest.json")
    paths = [ROOT / "config.yaml",
             ROOT / "outputs/manifests/legacy_frozen_experiment_audit.json",
             ROOT / "outputs/manifests/legacy_private_input_reuse_audit.json"]
    paths += sorted((ROOT / "scripts").rglob("*palsar*.py"))
    paths += [ROOT / "scripts/utilities/radar_benchmark.py",
              ROOT / "scripts/utilities/radar_extractors.py",
              ROOT / "scripts/03_modeling/run_radar_representation_benchmark.py",
              ROOT / "scripts/04_diagnostics/build_summary_outputs.py",
              ROOT / "scripts/04_diagnostics/analyze_radar_biomass_sensitivity.py",
              ROOT / "scripts/04_diagnostics/analyze_radar_domain_shift.py"]
    optional = [ROOT / "outputs/manifests/palsar_availability_freeze.json",
                ROOT / "outputs/manifests/palsar_asset_metadata_audit.json",
                ROOT / "outputs/manifests/palsar_point_forensic_audit.json",
                ROOT / "outputs/logs/source_palsar_finalize.json",
                ROOT / "outputs/logs/kaihua_palsar_finalize.json",
                ROOT / "outputs/manifests/palsar_common_sample_freeze.json",
                ROOT / "outputs/manifests/palsar_source_model_freeze.json",
                ROOT / "outputs/manifests/palsar_zero_shot_freeze.json",
                ROOT / "outputs/manifests/palsar_zero_shot_evaluation.json",
                ROOT / "outputs/manifests/palsar_fewshot_results_manifest.json",
                ROOT / "outputs/manifests/palsar_benchmark_outputs_manifest.json",
                ROOT / "outputs/manifests/palsar_wall_to_wall_decision_gate.json"]
    result_paths = [
        ROOT / "outputs/tables/audits/palsar_availability_audit.csv",
        ROOT / "outputs/tables/audits/palsar_asset_metadata_2022_2024.csv",
        ROOT / "outputs/tables/audits/palsar_raw_qa_histograms.csv",
        ROOT / "outputs/tables/audits/palsar_common_sample_selection_bias.csv",
        ROOT / "outputs/tables/main_results/palsar_source_spatial_cv.csv",
        ROOT / "outputs/tables/main_results/palsar_zero_shot.csv",
        ROOT / "outputs/tables/main_results/palsar_fewshot_label_efficiency.csv",
        ROOT / "outputs/tables/main_results/radar_representation_comparison.csv",
        ROOT / "outputs/tables/diagnostics/radar_biomass_bin_performance.csv",
        ROOT / "outputs/tables/diagnostics/radar_feature_importance.csv",
        ROOT / "outputs/tables/diagnostics/radar_domain_shift.csv",
        ROOT / "outputs/tables/diagnostics/radar_paired_fold_contrasts.csv",
        ROOT / "outputs/tables/audits/palsar_acquisition_timing.csv",
        ROOT / "figures/source_representation_comparison.png",
        ROOT / "figures/radar_representation_source_comparison.png",
        ROOT / "figures/radar_zero_shot_comparison.png",
        ROOT / "figures/radar_fewshot_label_efficiency.png",
        ROOT / "figures/radar_biomass_range_sensitivity.png",
        ROOT / "figures/radar_feature_importance.png",
        ROOT / "figures/radar_domain_shift.png",
        ROOT / "figures/final_workflow_v2.png",
    ]
    availability = json.loads((ROOT / "outputs/manifests/palsar_availability_freeze.json").read_text())
    common = json.loads((ROOT / "outputs/manifests/palsar_common_sample_freeze.json").read_text())
    qa_audit = json.loads((ROOT / "outputs/manifests/palsar_asset_metadata_audit.json").read_text())
    manifest = {
        "schema_version": 2,
        "status": "complete" if all(p.exists() for p in optional) else "pipeline_implemented_execution_in_progress",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "legacy_experiment": {"name": "legacy_frozen_experiment",
                              "manifest": legacy.relative_to(ROOT).as_posix(),
                              "canonical_git_blob_sha1": legacy_blob,
                              "local_materialization_sha256": sha256(legacy),
                              "modified_on_remote_branch": False},
        "extension": {"name": "palsar_common_paired_benchmark",
                      "dataset": "JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH",
                      "exact_year_only": True, "nearest_year_substitution": False,
                      "target_model_selection_forbidden": True,
                      "future_nisar": "2026 prospective NISAR L-band experiment",
                      "resolved_2023_qa_status": qa_audit["resolved_2023_qa_status"],
                      "valid_qa_values": qa_audit["valid_land_qa_classes"],
                      "final_common_years": availability["palsar_common_years"],
                      "final_sample_n": {
                          domain: int(summary["n_after_masking"])
                          for domain, summary in common["domains"].items()
                      }},
        "implementation_hashes": {p.relative_to(ROOT).as_posix(): sha256(p)
                                  for p in dict.fromkeys(paths) if p.exists()},
        "stage_manifests": {p.relative_to(ROOT).as_posix(): sha256(p)
                            for p in optional if p.exists()},
        "missing_stage_manifests": [p.relative_to(ROOT).as_posix()
                                    for p in optional if not p.exists()],
        "result_artifact_hashes": {p.relative_to(ROOT).as_posix(): sha256(p)
                                   for p in result_paths if p.exists()},
        "missing_result_artifacts": [p.relative_to(ROOT).as_posix()
                                     for p in result_paths if not p.exists()],
        "software": {"python": platform.python_version(), "platform": platform.platform()},
        "private_per_sample_data_committed": False,
    }
    out = ROOT / "outputs/manifests/final_reproducibility_manifest_v2.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
