"""Monitor the revised cap-200 task set and finalize it after Drive sync."""
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
RETRY_SECONDS = (15, 30, 60, 120, 300)


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps({
        "timestamp_ms": int(time.time() * 1000), "pid": os.getpid(), **payload,
    }, indent=2), encoding="utf-8")


def get_status(ids: list[str], log: Path) -> list[dict]:
    attempt = 0
    while True:
        try:
            return ee.data.getTaskStatus(ids)
        except Exception as exc:
            delay = RETRY_SECONDS[min(attempt, len(RETRY_SECONDS) - 1)]
            write(log, {"status": "network_retry", "attempt": attempt + 1,
                        "retry_seconds": delay, "error": repr(exc)})
            attempt += 1
            time.sleep(delay)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: monitor_and_finalize_source_sampling_200.py DRIVE_FOLDER")
    drive = Path(sys.argv[1])
    root = Path(__file__).resolve().parents[1]
    logs = root / "outputs" / "logs"
    monitor_log = logs / "source_sampling_200_monitor.json"
    completion_log = logs / "source_sampling_200_completion.json"
    original = json.loads(
        (logs / "source_sampling_export_tasks.json").read_text(encoding="utf-8")
    )
    legacy = [item for item in original
              if item["kind"] == "sampled_manifest" and item["year"] <= 2021]
    direct = json.loads(
        (logs / "source_manifest_200_tasks.json").read_text(encoding="utf-8")
    )
    audits = json.loads(
        (logs / "source_sampling_audit_recovery_tasks.json").read_text(encoding="utf-8")
    )
    records = legacy + direct + audits
    ids = [item["task_id"] for item in records]

    ee.Initialize()
    while True:
        status = {item["id"]: item for item in get_status(ids, monitor_log)}
        states = {item["description"]: status.get(item["task_id"], {}).get("state")
                  for item in records}
        errors = {item["description"]: status.get(item["task_id"], {}).get("error_message")
                  for item in records
                  if status.get(item["task_id"], {}).get("state") == "FAILED"}
        write(monitor_log, {"status": "monitoring", "states": states,
                            "errors": errors, "task_count": len(records)})
        if errors:
            write(completion_log, {"status": "failed_at_key_decision", "errors": errors})
            return
        if states and all(state == "COMPLETED" for state in states.values()):
            break
        time.sleep(POLL_SECONDS)

    expected = []
    for year in range(2019, 2025):
        stem = (f"gedi_sampled_manifest_{year}" if year <= 2021
                else f"gedi_sampled_manifest_200_{year}")
        expected.extend([
            drive / f"{stem}.csv",
            drive / f"gedi_sampling_block_counts_recovery_{year}.csv",
            drive / f"gedi_sampling_raw_stats_recovery_{year}.csv",
        ])
    while not all(path.exists() for path in expected):
        write(monitor_log, {"status": "waiting_for_drive_sync",
                            "missing": [path.name for path in expected if not path.exists()]})
        time.sleep(POLL_SECONDS)

    result = subprocess.run(
        [sys.executable, str(root / "src" / "finalize_source_sample_200.py"), str(drive)],
        cwd=root, capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode != 0:
        write(completion_log, {"status": "finalization_failed",
                              "stdout": result.stdout[-10000:],
                              "stderr": result.stderr[-10000:]})
        return
    write(completion_log, {"status": "complete", "stdout": result.stdout[-10000:]})
    write(monitor_log, {"status": "complete"})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        root = Path(__file__).resolve().parents[1]
        write(root / "outputs" / "logs" / "source_sampling_200_completion.json", {
            "status": "monitor_crashed", "traceback": traceback.format_exc(),
        })
        raise
