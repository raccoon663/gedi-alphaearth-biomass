"""Background monitor: finalize the frozen source sample after GEE exports sync."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import ee

POLL_SECONDS = 60
SYNC_TIMEOUT_SECONDS = 30 * 60
STATUS_RETRY_SECONDS = (15, 30, 60, 120, 300)
RECOVERY_KIND = {
    "stratum_audit": "block_counts",
    "annual_agbd_audit": "raw_agbd_audit",
}


def write_log(path: Path, payload: dict) -> None:
    payload = {"timestamp_ms": int(time.time() * 1000), "pid": os.getpid(), **payload}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_task_status_with_retry(task_ids: list[str], monitor_path: Path) -> list[dict]:
    """Keep transient Earth Engine/network failures from killing the monitor."""
    attempt = 0
    while True:
        try:
            return ee.data.getTaskStatus(task_ids)
        except Exception as exc:
            delay = STATUS_RETRY_SECONDS[min(attempt, len(STATUS_RETRY_SECONDS) - 1)]
            write_log(monitor_path, {
                "status": "status_check_retry",
                "attempt": attempt + 1,
                "retry_in_seconds": delay,
                "error": f"{type(exc).__name__}: {exc}",
            })
            attempt += 1
            time.sleep(delay)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: monitor_and_finalize_source_sampling.py DRIVE_EXPORT_FOLDER")
    drive = Path(sys.argv[1])
    root = Path(__file__).resolve().parents[1]
    task_path = root / "outputs" / "logs" / "source_sampling_export_tasks.json"
    monitor_path = root / "outputs" / "logs" / "source_sampling_monitor.json"
    completion_path = root / "outputs" / "logs" / "source_sampling_completion.json"

    ee.Initialize()
    records = json.loads(task_path.read_text(encoding="utf-8"))
    task_ids = [record["task_id"] for record in records]
    while True:
        statuses = get_task_status_with_retry(task_ids, monitor_path)
        by_id = {status["id"]: status for status in statuses}
        primary_states = {
            record["description"]: by_id.get(record["task_id"], {}).get("state")
            for record in records
        }
        manifest_errors = {
            record["description"]: by_id.get(record["task_id"], {}).get("error_message")
            for record in records
            if record["kind"] == "sampled_manifest"
            and by_id.get(record["task_id"], {}).get("state") == "FAILED"
        }
        if manifest_errors:
            write_log(completion_path, {
                "status": "manifest_export_failed", "errors": manifest_errors,
            })
            return

        failed_audits = [
            record for record in records
            if record["kind"] in RECOVERY_KIND
            and by_id.get(record["task_id"], {}).get("state") == "FAILED"
        ]
        recovery_path = root / "outputs" / "logs" / "source_sampling_audit_recovery_tasks.json"
        recovery_records = (
            json.loads(recovery_path.read_text(encoding="utf-8"))
            if recovery_path.exists() else []
        )
        recovery_by_key = {
            (item["year"], item["kind"]): item for item in recovery_records
        }
        submission_errors = {}
        for record in failed_audits:
            recovery_kind = RECOVERY_KIND[record["kind"]]
            key = (record["year"], recovery_kind)
            if key in recovery_by_key:
                continue
            result = subprocess.run(
                [sys.executable, str(root / "src" / "submit_source_sampling_audit_recovery.py"),
                 str(record["year"]), "--kinds", recovery_kind],
                cwd=root, capture_output=True, text=True, encoding="utf-8",
            )
            if result.returncode != 0:
                submission_errors[f"{record['year']}:{recovery_kind}"] = result.stderr[-4000:]
        if submission_errors:
            write_log(monitor_path, {
                "status": "recovery_submission_retry", "errors": submission_errors,
            })
            time.sleep(POLL_SECONDS)
            continue

        recovery_records = (
            json.loads(recovery_path.read_text(encoding="utf-8"))
            if recovery_path.exists() else []
        )
        needed_keys = {
            (record["year"], RECOVERY_KIND[record["kind"]])
            for record in failed_audits
        }
        needed_recovery = [
            item for item in recovery_records
            if (item["year"], item["kind"]) in needed_keys
        ]
        recovery_statuses = get_task_status_with_retry(
            [item["task_id"] for item in needed_recovery], monitor_path
        ) if needed_recovery else []
        recovery_by_id = {status["id"]: status for status in recovery_statuses}
        recovery_states = {
            item["description"]: recovery_by_id.get(item["task_id"], {}).get("state")
            for item in needed_recovery
        }
        recovery_errors = {
            item["description"]: recovery_by_id.get(item["task_id"], {}).get("error_message")
            for item in needed_recovery
            if recovery_by_id.get(item["task_id"], {}).get("state") == "FAILED"
        }
        if recovery_errors:
            write_log(completion_path, {
                "status": "audit_recovery_failed", "errors": recovery_errors,
            })
            return

        effective_states = {}
        for record in records:
            state = by_id.get(record["task_id"], {}).get("state")
            if state == "FAILED" and record["kind"] in RECOVERY_KIND:
                key = (record["year"], RECOVERY_KIND[record["kind"]])
                replacement = next(
                    (item for item in needed_recovery
                     if (item["year"], item["kind"]) == key), None
                )
                state = (recovery_by_id.get(replacement["task_id"], {}).get("state")
                         if replacement else "RECOVERY_PENDING")
            effective_states[record["description"]] = state
        write_log(monitor_path, {
            "status": "monitoring_with_recovery",
            "primary_states": primary_states,
            "recovery_states": recovery_states,
            "effective_states": effective_states,
        })
        if effective_states and all(
            state == "COMPLETED" for state in effective_states.values()
        ):
            break
        time.sleep(POLL_SECONDS)

    expected = []
    for year in range(2019, 2025):
        expected.append(drive / f"gedi_sampled_manifest_{year}.csv")
    recovered_by_year = {}
    for record in failed_audits:
        recovery_kind = RECOVERY_KIND[record["kind"]]
        recovered_by_year.setdefault(record["year"], []).append(recovery_kind)
        stem = ("gedi_sampling_block_counts_recovery" if recovery_kind == "block_counts"
                else "gedi_sampling_raw_stats_recovery")
        expected.append(drive / f"{stem}_{record['year']}.csv")
    for record in records:
        state = by_id.get(record["task_id"], {}).get("state")
        if record["kind"] == "stratum_audit" and state == "COMPLETED":
            expected.append(drive / f"gedi_sampling_strata_{record['year']}.csv")
        if record["kind"] == "annual_agbd_audit" and state == "COMPLETED":
            expected.append(drive / f"gedi_sampling_annual_stats_{record['year']}.csv")
    deadline = time.time() + SYNC_TIMEOUT_SECONDS
    while time.time() < deadline and not all(path.exists() for path in expected):
        missing = [path.name for path in expected if not path.exists()]
        write_log(monitor_path, {"status": "waiting_for_drive_sync", "missing": missing})
        time.sleep(POLL_SECONDS)
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        write_log(completion_path, {"status": "sync_timeout", "missing": missing})
        return

    for year, kinds in sorted(recovered_by_year.items()):
        result = subprocess.run(
            [sys.executable, str(root / "src" / "materialize_source_sampling_audit_recovery.py"),
             str(drive), str(year), "--kinds", *sorted(set(kinds))],
            cwd=root, capture_output=True, text=True, encoding="utf-8",
        )
        if result.returncode != 0:
            write_log(completion_path, {
                "status": "recovery_materialization_failed", "year": year,
                "stdout": result.stdout[-10000:], "stderr": result.stderr[-10000:],
            })
            return

    result = subprocess.run(
        [sys.executable, str(root / "src" / "finalize_source_sample.py"), str(drive)],
        cwd=root, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        write_log(completion_path, {
            "status": "finalization_failed", "returncode": result.returncode,
            "stdout": result.stdout[-10000:], "stderr": result.stderr[-10000:],
        })
        return
    write_log(completion_path, {"status": "complete", "stdout": result.stdout[-10000:]})
    write_log(monitor_path, {"status": "complete"})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        root = Path(__file__).resolve().parents[1]
        write_log(root / "outputs" / "logs" / "source_sampling_completion.json", {
            "status": "monitor_crashed", "traceback": traceback.format_exc(),
        })
        raise
