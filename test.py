"""Small utility: inspect the distribution of z labels in saved .npz shards.

Usage:
  python test.py "data/**/*.npz"

This file is intentionally standalone and cross-platform.
"""

from __future__ import annotations

import argparse
import glob
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="glob pattern, e.g. data/**/*.npz")
    args = ap.parse_args()

    zs = []
    for f in glob.glob(args.pattern, recursive=True):
        d = np.load(f)
        if "z" in d:
            zs.append(d["z"])

    if not zs:
        print("No files matched or no z arrays found.")
        return

    z = np.concatenate(zs)
    values, counts = np.unique(z, return_counts=True)
    print("z unique:")
    for v, c in zip(values, counts):
        print(f"  {v}: {c}")


if __name__ == "__main__":
    main()
