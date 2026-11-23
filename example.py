"""Inspect the matched optic-flow sample using only NumPy."""

import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    sample = Path(__file__).resolve().parent / "sample"
    meta = json.loads((sample / "metadata.json").read_text())
    path = sample / "matched_scene.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("The sample checksum differs from the release metadata.")
    print(f"Scene {meta['parent']}, interval {meta['frame']}, dt {meta['dt_seconds']:.5f} s")
    print("Condition  Target area (%)  Mean valid flow error (pixels/interval)")
    with np.load(path, allow_pickle=False) as fields:
        error = np.linalg.norm(fields["raw"][:, :2] - fields["reference_raw"][:, :2], axis=1)
        for i, condition in enumerate(meta["conditions"]):
            target_percent = 100 * fields["occupancy"][i].mean()
            epe = error[i][fields["valid"][i]].mean() * meta["dt_seconds"]
            print(f"{condition:9}  {target_percent:15.2f}  {epe:37.4f}")


if __name__ == "__main__":
    main()
