#!/usr/bin/env python3
"""Inventory local NumPy artifacts without loading all arrays into memory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def inspect(root: Path) -> dict:
    files = sorted(root.rglob("*.npy"))
    shapes = Counter()
    records = []
    for path in files:
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            shape = tuple(int(value) for value in array.shape)
            dtype = str(array.dtype)
            shapes[str(shape)] += 1
            records.append({
                "path": str(path.relative_to(root)),
                "shape": shape,
                "dtype": dtype,
                "bytes": path.stat().st_size,
            })
        except (OSError, ValueError) as error:
            records.append({
                "path": str(path.relative_to(root)),
                "error": str(error),
            })

    return {
        "root": str(root),
        "files": len(files),
        "shape_counts": dict(shapes),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing local NumPy artifacts")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = inspect(args.root)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
