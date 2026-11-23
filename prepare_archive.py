"""Materialize selected frozen NumPy splits directly from the release ZIP."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


PREFIX = "self-motion-flow-v1/DATA/"
SPLITS = ("train", "validation", "test", "expanded_test")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def publish_bytes(path, payload):
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Different file already exists: {path}")
        return
    with path.open("xb") as output:
        output.write(payload)


def prepare(archive, out, split, expected):
    manifest = archive.read(PREFIX + split + "_manifest.csv")
    rows = sorted(csv.DictReader(io.StringIO(manifest.decode("utf-8-sig"))),
                  key=lambda row: int(row["array_index"]))
    if len(rows) != expected["fields"]:
        raise ValueError(f"Unexpected row count for {split}")
    path = out / f"{split}_raw.npy"
    if path.exists():
        if digest(path) != expected["sha256"]:
            raise FileExistsError(f"Different array already exists: {path}")
    else:
        partial = path.with_suffix(".partial.npy")
        side = 64 if split == "test" else 128
        with partial.open("xb") as stream:
            header = dict(descr="<f4", fortran_order=False, shape=(len(rows), 3, side, side))
            np.lib.format.write_array_header_1_0(stream, header)
            for i, row in enumerate(rows):
                if int(row["array_index"]) != i:
                    raise ValueError(f"Non-contiguous array indices for {split}")
                field = archive.read(PREFIX + "fields/" + row["key"] + ".npz")
                with np.load(io.BytesIO(field), allow_pickle=False) as arrays:
                    raw = arrays["raw"]
                    if raw.shape != (3, side, side) or raw.dtype != np.dtype("<f4"):
                        raise ValueError(f"Unexpected shape or dtype for {row['key']}")
                    stream.write(raw.tobytes(order="C"))
                if (i + 1) % 5000 == 0:
                    print(f"{split}: {i + 1:,}/{len(rows):,}", flush=True)
        if digest(partial) != expected["sha256"]:
            raise ValueError(f"{split} differs from the frozen training file: {partial}")
        partial.rename(path)
    publish_bytes(out / f"{split}_manifest.csv", manifest)
    print(f"{split}: verified {len(rows):,} fields", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        frozen = json.loads(archive.read(PREFIX + "FROZEN.json"))
        norm = archive.read(PREFIX + "normalization.npz")
        if hashlib.sha256(norm).hexdigest() != frozen["normalization_sha256"]:
            raise ValueError("Normalization checksum mismatch")
        publish_bytes(args.out / "normalization.npz", norm)
        for split in args.splits:
            prepare(archive, args.out, split, frozen["counts"][split])


if __name__ == "__main__":
    main()
