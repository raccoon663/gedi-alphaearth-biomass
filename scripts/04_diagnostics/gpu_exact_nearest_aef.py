"""Exact brute-force nearest-source Euclidean AEF distances on CUDA."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the exact 124k x 130k distance calculation")
    torch.backends.cuda.matmul.allow_tf32 = False
    source = torch.from_numpy(np.load(args.source).astype(np.float32, copy=False)).cuda()
    target_np = np.load(args.target).astype(np.float32, copy=False)
    source_sq = (source * source).sum(dim=1).unsqueeze(0)
    result = np.empty(len(target_np), dtype=np.float32)
    for start in range(0, len(target_np), args.batch_size):
        target = torch.from_numpy(target_np[start:start + args.batch_size]).cuda()
        distances_sq = ((target * target).sum(dim=1, keepdim=True) + source_sq
                        - 2.0 * target @ source.T).clamp_min_(0)
        result[start:start + len(target)] = distances_sq.min(dim=1).values.sqrt().cpu().numpy()
        print(f"nearest {start + len(target)}/{len(target_np)}", flush=True)
    np.save(Path(args.output), result)


if __name__ == "__main__":
    main()
