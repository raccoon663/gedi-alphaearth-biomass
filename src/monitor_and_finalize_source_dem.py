"""Monitor source DEM tasks and finalize after Drive synchronization."""
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
RETRY = (15, 30, 60, 120, 300)


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps({"timestamp_ms": int(time.time() * 1000),
                                "pid": os.getpid(), **payload}, indent=2),
                    encoding="utf-8")


def statuses(ids: list[str], log: Path) -> list[dict]:
    attempt = 0
    while True:
        try:
            return ee.data.getTaskStatus(ids)
        except Exception as exc:
            delay = RETRY[min(attempt, len(RETRY) - 1)]
            write(log, {"status": "network_retry", "retry_seconds": delay,
                        "error": repr(exc)})
            attempt += 1
            time.sleep(delay)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: monitor_and_finalize_source_dem.py DRIVE_FOLDER")
    drive = Path(sys.argv[1])
    root = Path(__file__).resolve().parents[1]
    logs = root / "outputs" / "logs"
    records = json.loads((logs / "source_dem_export_tasks.json").read_text())
    monitor = logs / "source_dem_monitor.json"
    completion = logs / "source_dem_completion.json"
    ids = [item["task_id"] for item in records]
    ee.Initialize()
    while True:
        current = {item["id"]: item for item in statuses(ids, monitor)}
        counts = {}
        errors = {}
        for record in records:
            state = current.get(record["task_id"], {}).get("state", "UNKNOWN")
            counts[state] = counts.get(state, 0) + 1
            if state == "FAILED":
                errors[record["description"]] = current[record["task_id"]].get(
                    "error_message")
        write(monitor, {"status": "monitoring", "state_counts": counts,
                        "errors": errors, "task_count": len(records)})
        if errors:
            write(completion, {"status": "failed_at_key_decision", "errors": errors})
            return
        if counts.get("COMPLETED", 0) == len(records):
            break
        time.sleep(POLL_SECONDS)
    expected = [drive / f"{item['description']}.csv" for item in records]
    while not all(path.exists() for path in expected):
        write(monitor, {"status": "waiting_for_drive_sync",
                        "missing_n": sum(not path.exists() for path in expected)})
        time.sleep(POLL_SECONDS)
    result = subprocess.run(
        [sys.executable, str(root / "src" / "finalize_source_dem.py"), str(drive)],
        cwd=root, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        write(completion, {"status": "finalization_failed",
                           "stdout": result.stdout[-10000:],
                           "stderr": result.stderr[-10000:]})
        return
    write(completion, {"status": "complete", "stdout": result.stdout[-10000:]})
    write(monitor, {"status": "complete"})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        root = Path(__file__).resolve().parents[1]
        write(root / "outputs" / "logs" / "source_dem_completion.json",
              {"status": "monitor_crashed", "traceback": traceback.format_exc()})
        raise
