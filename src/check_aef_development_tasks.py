"""Refresh AEF development task states in small batches."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import ee

BATCH = 20


def main() -> None:
    ee.Initialize()
    root = Path(__file__).resolve().parents[1]
    ledger = root / "outputs" / "logs" / "aef_development_export_tasks.json"
    records = json.loads(ledger.read_text())
    by_id = {}
    for start in range(0, len(records), BATCH):
        statuses = ee.data.getTaskStatus(
            [item["task_id"] for item in records[start:start + BATCH]])
        by_id.update({item["id"]: item for item in statuses})
        print(f"checked {min(start+BATCH, len(records))}/{len(records)}", flush=True)
    counts = Counter()
    errors = {}
    for record in records:
        status = by_id.get(record["task_id"], {})
        state = status.get("state", "UNKNOWN")
        counts[state] += 1
        record["state"] = state
        record["error_message"] = status.get("error_message")
        if state == "FAILED":
            errors[record["description"]] = status.get("error_message")
    ledger.write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {"timestamp_ms": int(time.time()*1000), "task_count": len(records),
               "state_counts": dict(counts), "errors": errors}
    (root / "outputs" / "logs" / "aef_development_status.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
