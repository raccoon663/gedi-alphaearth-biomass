"""Refresh source sampling Earth Engine task states."""
from __future__ import annotations

import json
from pathlib import Path
import ee


def main() -> None:
    ee.Initialize()
    root = Path(__file__).resolve().parents[1]
    path = root / "outputs" / "logs" / "source_sampling_export_tasks.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    statuses = {item["id"]: item for item in ee.data.getTaskStatus(
        [record["task_id"] for record in records])}
    for record in records:
        status = statuses.get(record["task_id"], {})
        record.update({
            "state": status.get("state"),
            "error_message": status.get("error_message"),
            "update_timestamp_ms": status.get("update_timestamp_ms"),
        })
        print(record["description"], record["state"], record.get("error_message"))
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
