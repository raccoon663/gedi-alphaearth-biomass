"""Cancel only tasks made obsolete by the approved 200-cap scope reduction."""
from __future__ import annotations

import json
from pathlib import Path
import time

import ee


def main() -> None:
    ee.Initialize()
    root = Path(__file__).resolve().parents[2]
    source = root / "outputs" / "logs" / "source_sampling_export_tasks.json"
    records = json.loads(source.read_text(encoding="utf-8"))
    statuses = {
        item["id"]: item for item in ee.data.getTaskStatus(
            [record["task_id"] for record in records]
        )
    }
    cancelled = []
    retained = []
    for record in records:
        year = int(record["year"])
        kind = record["kind"]
        description = record["description"]
        state = statuses.get(record["task_id"], {}).get("state")

        cancel_reason = None
        # Running manifests are explicitly protected.  Only the two expensive
        # running 2020 audit graphs are obsolete because optimized replacements
        # are being submitted.
        if year == 2020 and kind in {"stratum_audit", "annual_agbd_audit"}:
            if state == "RUNNING":
                cancel_reason = "running_obsolete_audit_with_light_replacement"
        # All audit graphs that have not started are replaced by the optimized
        # histogram/raw-stat workflow.
        elif year in range(2021, 2025) and kind in {
            "stratum_audit", "annual_agbd_audit"
        }:
            if state == "READY":
                cancel_reason = "ready_obsolete_audit"
        # Only unstarted legacy 500-cap manifests are replaced directly by 200.
        elif year in {2022, 2023, 2024} and kind == "sampled_manifest":
            if state == "READY":
                cancel_reason = "ready_legacy_500_manifest_replaced_by_200"

        if kind == "sampled_manifest" and state == "RUNNING" and cancel_reason:
            raise RuntimeError(f"Safety invariant violated for {description}")
        if cancel_reason:
            ee.data.cancelTask(record["task_id"])
            cancelled.append({
                "task_id": record["task_id"], "description": description,
                "year": year, "kind": kind, "state_before": state,
                "reason": cancel_reason,
            })
        else:
            retained.append({
                "task_id": record["task_id"], "description": description,
                "year": year, "kind": kind, "state": state,
            })

    payload = {
        "timestamp_ms": int(time.time() * 1000),
        "policy": "compute-efficient cap-200 transition",
        "cancelled": cancelled,
        "retained": retained,
    }
    path = root / "outputs" / "logs" / "compute_scope_task_transition.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
