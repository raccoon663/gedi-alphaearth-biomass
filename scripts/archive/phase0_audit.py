"""Local Phase 0 audit. Does not authenticate or mutate cloud state."""
from __future__ import annotations

import importlib.util
import json
import platform
from pathlib import Path

PACKAGES = ["ee", "geemap", "geopandas", "sklearn", "xgboost", "yaml"]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "logs"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "python": platform.python_version(),
        "packages": {name: importlib.util.find_spec(name) is not None for name in PACKAGES},
        "dataset_ids_to_verify_live": {
            "alphaearth": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
            "gedi_l4a": "LARSE/GEDI/GEDI04_A_002_MONTHLY",
        },
        "target_label_locked": True,
    }
    (out / "phase0_environment.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
