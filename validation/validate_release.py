"""Reproduce the Data Descriptor tables from the public ZIP, without Blender or trained models."""
import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
BASE = "self-motion-flow-v1/DATA/"
SPLITS = ("train", "validation", "test", "expanded_test")
SEED = 23112026


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def angular(a, b):
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def check_pair(scene, frame, condition, ref, field):
    depth = ref["depth"].astype(float)
    h, w = depth.shape
    yy, xx = np.indices((h, w))
    # Stored K uses RGB pixel centres. Apply the half-pixel resize on both axes.
    K = ref["K"]
    sx, sy = w / scene["size"], h / scene["size"]
    fx, fy = K[0, 0] * sx, K[1, 1] * sy
    cx, cy = (K[0, 2] + .5) * sx - .5, (K[1, 2] + .5) * sy - .5
    source = np.stack(((xx-cx)*depth/fx, (yy-cy)*depth/fy, depth), axis=-1)
    cv_to_blender = np.diag([1., -1., -1.])
    angular_step = np.array(scene["rotation_per_frame"])
    pose = lambda t: np.array(scene["camera_R0"]) @ Rotation.from_rotvec(t*angular_step).as_matrix() @ cv_to_blender
    # Work directly in the next camera coordinate system. The world origin cancels.
    destination = source @ (pose(frame+1).T @ pose(frame)).T
    destination -= np.array(scene["translation_per_frame"]) @ pose(frame+1)
    target = ref["occupancy"].astype(bool)
    def project(p):
        return np.stack((fx*p[..., 0]/p[..., 2]+cx-xx, fy*p[..., 1]/p[..., 2]+cy-yy), axis=-1)
    static_flow = project(destination)
    if condition == "moving":
        destination[target] += np.array(scene["object_translation_per_frame"]) @ pose(frame+1)
    predicted = project(destination)
    valid = ref["valid"].astype(bool)
    assert valid.any() and np.isfinite(depth).all() and depth.min() > 0
    epe = np.linalg.norm(predicted-ref["flow"], axis=-1)[valid]
    target_valid = valid & target
    omit = np.linalg.norm(static_flow-ref["flow"], axis=-1)[target_valid]
    corrected = np.linalg.norm(predicted-ref["flow"], axis=-1)[target_valid]
    dt = float(ref["dt"])
    assert np.isclose(dt, 1/90) and np.isclose(field["dt"],dt)
    field_valid = field["valid"].astype(bool)
    raft_error = np.linalg.norm((field["raw"][:2]-field["reference_raw"][:2])*dt,axis=0)
    q = np.quantile(depth, [.05,.5,.95])
    return dict(frame=frame, condition=condition, native_grid=h, model_grid=field_valid.shape[0],
                valid_px=int(valid.sum()), valid_fraction=valid.mean(), target_valid_px=int(target_valid.sum()),
                mean_epe_px=epe.mean(), p99_epe_px=np.quantile(epe,.99), max_epe_px=epe.max(),
                moving_corrected_px=corrected.mean() if corrected.size else "",
                rigid_only_target_px=omit.mean() if omit.size else "",
                depth_p05_m=q[0],depth_p50_m=q[1],depth_p95_m=q[2],
                raft_mean_epe_model_px=raft_error[field_valid].mean())


def main(archive_path, out):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    kinematics, checks, matching = [], [], []
    totals = {}
    parents_seen, source_seen = set(), set()
    with ZipFile(archive_path) as archive:
        for split, requested in zip(SPLITS,[128,80,120,40]):
            with archive.open(BASE+f"scenes/{split}.jsonl") as f:
                scenes = [json.loads(line) for line in f]
            pick = set(rng.choice(sorted(s["id"] for s in scenes), requested, replace=False))
            with archive.open(BASE+f"{split}_manifest.csv") as f:
                rows=list(csv.DictReader(io.TextIOWrapper(f,encoding="utf-8-sig")))
            ids = {s["id"] for s in scenes}
            sources = {s["source_parent"] for s in scenes}
            assert parents_seen.isdisjoint(ids) and source_seen.isdisjoint(sources)
            parents_seen.update(ids)
            source_seen.update(sources)
            lookup={(r["scene_id"],int(r["frame"]),r["condition"]):r for r in rows}
            assert len(lookup)==len(rows) and ids=={r["scene_id"] for r in rows}
            by_id={s["id"]:s for s in scenes}
            for row in rows:
                parent=by_id[row["scene_id"]]
                assert all(abs(float(row[key])-parent[key])<1e-9 for key in ("heading_az_deg","heading_el_deg"))
                assert row["parent"]==parent["parent"] and row["source_parent"]==parent["source_parent"]
            conds=["absent","static","moving"] if "test" in split else ["static"]
            expected_keys={(s["id"], t, c) for s in scenes for t in range(s["frames"]-1) for c in conds}
            assert set(lookup)==expected_keys
            totals[split]=dict(scenes=len(scenes),fields=len(rows),sample=requested)
            for s in scenes:
                sid=s["id"]
                T=np.array(s["translation_per_frame"])
                V=np.array(s["object_translation_per_frame"])
                heading=np.array(s["camera_R0"]).T @ T
                az,el=np.radians([s["heading_az_deg"],s["heading_el_deg"]])
                label=np.array([np.cos(el)*np.cos(az),np.sin(el),-np.cos(el)*np.sin(az)])
                kinematics.append(dict(split=split,scene=sid,world=s.get("world_type","room"),
                    heading_deg=s["heading_az_deg"],elevation_deg=s["heading_el_deg"],
                    label_discrepancy_deg=angular(heading,label),camera_m_s=90*np.linalg.norm(T),
                    rotation_deg_s=90*np.degrees(np.linalg.norm(s["rotation_per_frame"])),
                    moving_target_m_s=90*np.linalg.norm(V) if "test" in split else "",
                    speed_ratio=np.linalg.norm(V)/np.linalg.norm(T) if "test" in split else ""))
                if sid not in pick:
                    continue
                t=int(rng.integers(0,s["frames"]-1))
                for c in conds:
                    row=lookup[sid,t,c]
                    stem=BASE+f"rendered/{sid}/{c}/{t:04d}_reference.npz"
                    with np.load(io.BytesIO(archive.read(stem))) as ref, np.load(io.BytesIO(
                            archive.read(BASE+"fields/"+row["key"]+".npz"))) as field:
                        checks.append(dict(split=split,scene=sid,**check_pair(s,t,c,ref,field)))
                if len(conds)>1:
                    def first_rgb(c):
                        return np.asarray(Image.open(io.BytesIO(archive.read(
                            BASE+f"rendered/{sid}/{c}/0000.png"))).convert("RGB"),dtype=int)
                    difference=np.abs(first_rgb("static")-first_rgb("moving"))
                    matching.append(dict(split=split,scene=sid,max_rgb_difference=difference.max(),
                                         mean_rgb_difference=difference.mean()))
            print(split,totals[split],flush=True)
    write_rows(out/"scene_kinematics.csv",kinematics)
    write_rows(out/"projection_and_depth.csv",checks)
    write_rows(out/"matched_rgb.csv",matching)
    (out/"sampling.json").write_text(json.dumps(dict(seed=SEED,splits=totals),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--out",type=Path,default=HERE/"results")
    args=parser.parse_args()
    main(args.archive,args.out)
