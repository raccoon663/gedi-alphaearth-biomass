"""Restart-safe exact-year central-pixel PALSAR extraction for locked Kaihua shots."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/01_data_extraction"))
from download_source_palsar_direct import download  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/kaihua_target_shots_locked.csv")
    parser.add_argument("--availability", type=Path, default=ROOT / "outputs/manifests/palsar_availability_freeze.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/kaihua_palsar")
    parser.add_argument("--ledger", type=Path, default=ROOT / "outputs/logs/kaihua_palsar_download_status.json")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--project")
    args = parser.parse_args()
    download(ROOT, args.manifest, args.availability, args.output_dir, args.ledger,
             args.chunk_size, args.retries, args.project)


if __name__ == "__main__":
    main()
