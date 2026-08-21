"""Finalize locked Kaihua PALSAR chunks using the identical source transformation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/02_data_preparation"))
from finalize_source_palsar import finalize  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=ROOT / "data/raw/kaihua_palsar")
    p.add_argument("--output", type=Path, default=ROOT / "data/processed/kaihua_palsar_locked.parquet")
    p.add_argument("--manifest", type=Path, default=ROOT / "outputs/logs/kaihua_palsar_finalize.json")
    args = p.parse_args()
    print(json.dumps(finalize(args.input_dir, args.output, args.manifest, "target"), indent=2))


if __name__ == "__main__":
    main()
