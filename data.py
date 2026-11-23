"""NumPy loader for the frozen, tutorial and newly generated splits."""

import csv
from pathlib import Path

import numpy as np


def heading_vector(azimuth, elevation):
    """Convert camera heading in degrees to (right, up, forward)."""
    az, el = np.deg2rad(azimuth), np.deg2rad(elevation)
    return np.stack((np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)),
                    axis=-1).astype(np.float32)


class HeadingDataset:
    """Index by integer or slice to receive (flow/depth, unit heading)."""

    def __init__(self, root, split):
        root = Path(root)
        if (root / split / f"{split}_manifest.csv").is_file():
            root = root / split
        with (root / f"{split}_manifest.csv").open(newline="") as source:
            self.rows = list(csv.DictReader(source))
        self.fields = np.load(root / f"{split}_raw.npy", mmap_mode="r", allow_pickle=False)
        self.indices = np.array([int(row["array_index"]) for row in self.rows], dtype=int)
        if (self.fields.ndim != 4 or self.fields.shape[1] != 3 or not len(self.rows)
                or not np.array_equal(np.sort(self.indices), np.arange(len(self.fields)))):
            raise ValueError("Expected one manifest row per three-channel field.")
        self.parents = np.array([row["parent"] for row in self.rows])
        self.targets = heading_vector([float(r["heading_az_deg"]) for r in self.rows],
                                      [float(r["heading_el_deg"]) for r in self.rows])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.fields[self.indices[index]], self.targets[index]


def spatial_mean(fields, grid):
    """Average spatial blocks of a (batch, channels, height, width) array."""
    n, c, h, w = fields.shape
    if grid < 1 or h % grid or w % grid:
        raise ValueError(f"Grid {grid} must divide both spatial dimensions {h} and {w}.")
    return fields.reshape(n, c, grid, h // grid, grid, w // grid).mean(axis=(3, 5))
