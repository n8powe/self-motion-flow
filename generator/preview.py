"""Plot two Blender scene directories at their first moving-target interval."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs=2, type=Path)
    parser.add_argument("--labels", nargs=2, default=["Scene 1", "Scene 2"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    samples = []
    for scene in args.scenes:
        condition = scene / "moving"
        rgb = plt.imread(condition / "0000.png")
        with np.load(condition / "0000_reference.npz") as field:
            samples.append((rgb, field["flow"] / field["dt"], field["depth"],
                            field["occupancy"], field["valid"]))
    depth_max = np.ceil(np.percentile(np.concatenate([s[2].ravel() for s in samples]), 99))
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.6))
    fig.subplots_adjust(left=.13, right=.89, top=.91, bottom=.04, wspace=.08, hspace=.12)
    for ax, title in zip(axes[0], ["Rendered frame", "Reference flow", "Axial depth"]):
        ax.set_title(title, fontsize=14, pad=8)
    for row, (rgb, flow, depth, mask, valid) in enumerate(samples):
        frame, vectors, distances = axes[row]
        frame.imshow(rgb, extent=(-.5, 127.5, 127.5, -.5))
        if mask.any() and not mask.all():
            frame.contour(mask.astype(float), levels=[.5], colors=["#ffdf38"], linewidths=1)
        frame.set_ylabel(args.labels[row], fontsize=13)
        y, x = np.mgrid[4:128:9, 4:128:9]
        keep = valid[y, x]
        u, v = flow[y[keep], x[keep]].T
        speed = max(1., float(np.percentile(np.hypot(u, v), 90)))
        vectors.set_facecolor("#f0f4f6")
        q = vectors.quiver(x[keep], y[keep], u, v, angles="xy", scale_units="xy",
                           scale=speed / 7, color="#17576c", width=.006, pivot="mid")
        vectors.quiverkey(q, .40, -.065, speed, f"{speed:.1f} px/s", labelpos="E",
                          coordinates="axes", fontproperties={"size": 9},
                          labelcolor="#16333b")
        image = distances.imshow(depth, cmap="viridis", vmin=0, vmax=depth_max)
        for ax in axes[row]:
            ax.set(xlim=(-.5, 127.5), ylim=(127.5, -.5), aspect="equal", xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
    color_axis = fig.add_axes([.91, .22, .014, .50])
    fig.colorbar(image, cax=color_axis, extend="max", label="Depth (m)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight", pad_inches=.08)


if __name__ == "__main__":
    main()
