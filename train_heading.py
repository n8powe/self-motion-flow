"""Fit a supervised ridge regression model to the observer's translation heading."""

import argparse
from pathlib import Path

import numpy as np
from data import HeadingDataset, spatial_mean


def features(fields, grid, flow_only):
    cues = spatial_mean(fields, grid)
    flow = cues[:, :2]
    rms = np.sqrt(np.mean(flow ** 2, axis=(1, 2, 3), keepdims=True))
    flow /= np.maximum(rms, 1e-8)
    cues[:, 2] -= 1
    inputs = flow if flow_only else cues
    return inputs.reshape(len(inputs), -1).astype(np.float64)


def batches(dataset, args):
    for start in range(0, len(dataset), args.batch_size):
        x, y = dataset[start:start + args.batch_size]
        yield features(x, args.grid, args.flow_only), y.astype(np.float64)


def fit(dataset, args):
    # Accumulate sufficient statistics so the full training array can stay on disk.
    width = args.grid ** 2 * (2 if args.flow_only else 3)
    gram = np.zeros((width, width))
    cross = np.zeros((width, 3))
    sum_x = np.zeros(width)
    sum_y = np.zeros(3)
    for x, y in batches(dataset, args):
        gram += x.T @ x
        cross += x.T @ y
        sum_x += x.sum(0)
        sum_y += y.sum(0)
    center, base = sum_x / len(dataset), sum_y / len(dataset)
    covariance = gram / len(dataset) - np.outer(center, center)
    covariance.flat[::width + 1] += args.penalty
    weights = np.linalg.solve(covariance, cross / len(dataset) - np.outer(center, base))
    return center, base, weights


def angular_error(pred, truth):
    norms = np.linalg.norm(pred, axis=1, keepdims=True)
    direction = pred / np.maximum(norms, 1e-12)
    return np.rad2deg(np.arccos(np.clip((direction * truth).sum(axis=1), -1, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "sample/heading")
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--penalty", type=float, default=0.01)
    parser.add_argument("--flow-only", action="store_true")
    parser.add_argument("--save", type=Path, help="Optional output .npz with fitted weights")
    args = parser.parse_args()
    if args.batch_size < 1 or args.grid < 1 or args.penalty <= 0:
        parser.error("Use a positive batch size, grid size and ridge penalty.")
    train = HeadingDataset(args.data, "train")
    validation = HeadingDataset(args.data, "validation")
    if set(train.parents) & set(validation.parents):
        raise ValueError("Training and validation contain overlapping parent scenes.")
    center, base, weights = fit(train, args)
    errors = []
    for x, y in batches(validation, args):
        errors.extend(angular_error((x - center) @ weights + base, y))
    baseline = angular_error(np.broadcast_to(base, validation.targets.shape), validation.targets)
    parents, groups = np.unique(validation.parents, return_inverse=True)
    def scene_mean(values):
        return (np.bincount(groups, weights=values) / np.bincount(groups)).mean()
    print(f"Trained on {len(train):,} fields. Evaluated on {len(parents):,} validation scenes.")
    print(f"Mean 3D angular error (equal scene weights): {scene_mean(errors):.2f} degrees")
    print(f"Constant training-mean direction: {scene_mean(baseline):.2f} degrees")
    if args.save:
        np.savez(args.save, center=center, base=base, weights=weights,
                 grid=args.grid, flow_only=args.flow_only, penalty=args.penalty)


if __name__ == "__main__":
    main()
