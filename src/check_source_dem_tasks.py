"""Refresh DEM task states in small network-resilient batches."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import ee

BATCH_SIZE = 20
RETRIES = (5, 15, 30)


def get_batch(ids: list[str]) -> list[dict]:
    last = None
    for delay in (0, *RETRIES):
        if delay:
            time.sleep(delay)
        try:
            return ee.data.getTaskStatus(ids)
        except Exception as exc:
            last = exc
    raise last  # type: ignore[misc]


def main() -> None:
    ee.Initialize()
    root = Path(__file__).resolve().parents[1]
    ledger = root / "outputs" / "logs" / "source_dem_export_tasks.json"
    records = json.loads(ledger.read_text(encoding="utf-8"))
    status_by_id = {}
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        for status in get_batch([item["task_id"] for item in batch]):
            status_by_id[status["id"]] = status
        print(f"checked {min(start + BATCH_SIZE, len(records))}/{len(records)}", flush=True)
    counts = Counter()
    errors = {}
    for record in records:
        status = status_by_id.get(record["task_id"], {})
        state = status.get("state", "UNKNOWN")
        counts[state] += 1
        record["state"] = state
        record["error_message"] = status.get("error_message")
        record["update_timestamp_ms"] = status.get("update_timestamp_ms")
        if state == "FAILED":
            errors[record["description"]] = status.get("error_message")
    ledger.write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {"timestamp_ms": int(time.time() * 1000), "task_count": len(records),
               "expected_n": sum(item["expected_n"] for item in records),
               "state_counts": dict(counts), "errors": errors}
    (root / "outputs" / "logs" / "source_dem_status.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
