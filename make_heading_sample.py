"""Recreate the small heading tutorial from the frozen full splits."""

import argparse
import csv
from pathlib import Path

import numpy as np
from data import HeadingDataset, spatial_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261123)
    for split, count in [("train", 512), ("validation", 128)]:
        dataset = HeadingDataset(args.source, split)
        _, first = np.unique(dataset.parents, return_index=True)
        selected = np.sort(rng.choice(first, count, replace=False))
        rows = []
        small = np.empty((count, 3, 16, 16), dtype=np.float32)
        for i, index in enumerate(selected):
            field, _ = dataset[index]
            small[i] = spatial_mean(field[None], 16)[0]
            small[i, 0] *= 16 / field.shape[-1]
            small[i, 1] *= 16 / field.shape[-2]
            row = dataset.rows[index]
            rows.append(dict(array_index=i, source_index=row["array_index"], parent=row["parent"],
                             frame=row["frame"], condition=row["condition"],
                             heading_az_deg=row["heading_az_deg"], heading_el_deg=row["heading_el_deg"]))
        np.save(args.destination / f"{split}_raw.npy", small)
        with (args.destination / f"{split}_manifest.csv").open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
