"""Hard guards against target-label leakage."""
from __future__ import annotations

import json
from pathlib import Path

TARGET_LABEL_LOCKED = True


def require_target_labels_unlocked(project_root: Path) -> dict:
    """Return the frozen manifest or fail before any target-label read."""
    manifest_path = project_root / "outputs" / "models" / "frozen_source_manifest.json"
    if TARGET_LABEL_LOCKED:
        raise RuntimeError(
            "TARGET_LABEL_LOCKED=True: target AGBD cannot be read during source development. "
            "Use the separate target evaluation stage after freezing source decisions."
        )
    if not manifest_path.exists():
        raise RuntimeError("Target evaluation requires frozen_source_manifest.json")
    return json.loads(manifest_path.read_text(encoding="utf-8"))
