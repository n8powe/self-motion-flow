"""Run inside Blender. Arguments following '--' are passed to this script."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renderer import render_scene
from worlds import make_scene


def generate(args):
    conditions = ["absent", "static", "moving"] if args.split == "test" else ["static"]
    destination = args.out.resolve() / args.split
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(destination.name + ".partial")
    if destination.exists() or stage.exists():
        raise FileExistsError(f"Choose a fresh split destination: {destination}")
    stage.mkdir()
    intervals = args.frames - 1
    count = args.scenes * intervals * len(conditions)
    raw = np.lib.format.open_memmap(stage / f"{args.split}_raw.npy", mode="w+",
                                    dtype=np.float32, shape=(count, 3, 128, 128))
    rows = []
    accepted = 0
    split_id = ["train", "validation", "test"].index(args.split)
    for index in range(10 * args.scenes + 10):
        seed = int(np.random.SeedSequence([args.seed, split_id, index]).generate_state(1)[0])
        scene = make_scene(seed, args.split, index, args.frames)
        directory = stage / scene["id"]
        directory.mkdir()
        (directory / "scene.json").write_text(json.dumps(scene, indent=2))
        try:
            render_scene(scene, directory)
        except ValueError as error:
            if not str(error).startswith("Invalid world/camera"):
                raise
            (directory / "rejected.json").write_text(json.dumps({"reason": str(error)}))
        if (directory / "rejected.json").exists():
            continue
        for condition in conditions:
            for frame in range(intervals):
                name = f"{frame:04d}_reference.npz"
                with np.load(directory / condition / name) as field:
                    inv = 1.0 / field["depth"].astype(np.float64)
                    raw[len(rows), :2] = field["flow"].transpose(2, 0, 1) / field["dt"]
                    raw[len(rows), 2] = inv / inv.mean()
                rows.append(dict(array_index=len(rows), parent=scene["parent"],
                                 frame=frame, condition=condition,
                                 heading_az_deg=scene["heading_az_deg"],
                                 heading_el_deg=scene["heading_el_deg"]))
        accepted += 1
        print(f"Accepted {accepted}/{args.scenes} {args.split} scenes", flush=True)
        if accepted == args.scenes:
            break
    if len(rows) != count:
        raise RuntimeError(f"Only {accepted} scenes passed the camera and visibility checks.")
    raw.flush()
    del raw
    with (stage / f"{args.split}_manifest.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stage.rename(destination)
    print(f"Saved {count} reference fields in {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--frames", type=int, default=6)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if args.frames < 2 or args.scenes < 1 or args.seed < 0:
        parser.error("Use at least one scene, two frames and a nonnegative seed.")
    generate(args)
