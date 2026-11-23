"""Seeded scene parameters and camera projection from the 100k synthetic dataset."""
import hashlib
import json

import numpy as np

def rotation(v):
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v / angle
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def pose(scene, frame):
    R = np.asarray(scene['camera_R0']) @ rotation(np.asarray(scene['rotation_per_frame']) * frame)
    C = np.asarray(scene['camera_C0']) + np.asarray(scene['translation_per_frame']) * frame
    return (R, C)


def project(scene, points, frame, size):
    R, C = pose(scene, frame)
    local = (points - C) @ R
    depth = -local[..., 2]
    f = size / (2 * np.tan(np.deg2rad(scene['fov_deg'] / 2)))
    safe = np.where(np.abs(depth) < 1e-10, 1e-10, depth)
    return (np.stack([f * local[..., 0] / safe + size / 2 - 0.5, -f * local[..., 1] / safe + size / 2 - 0.5], axis=-1), depth)


def make_scene(seed, split, j, frames):
    g = np.random.default_rng(seed)
    open_floor = j % 4 == 0
    w, l, h = np.exp(g.uniform(np.log([2, 3, 2.5]), np.log([25, 40, 12])))
    C = np.array([0.0, 0.0, g.uniform(0.8, min(2.5, h * 0.7))])
    yaw = g.uniform(-np.pi, np.pi)
    R = rotation([0, 0, yaw]) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    if open_floor:
        R = R @ rotation([-np.deg2rad(60), 0, 0])
    boxes = [dict(center=[0, 0, -0.1], size=[1000, 1000, 0.2]) if open_floor else dict(center=[0, 0, -0.1], size=[2 * w, 2 * l, 0.2])]
    if not open_floor:
        boxes += [dict(center=[0, 0, h + 0.1], size=[2 * w, 2 * l, 0.2]), dict(center=[-w - 0.1, 0, h / 2], size=[0.2, 2 * l, h]), dict(center=[w + 0.1, 0, h / 2], size=[0.2, 2 * l, h]), dict(center=[0, -l - 0.1, h / 2], size=[2 * w, 0.2, h]), dict(center=[0, l + 0.1, h / 2], size=[2 * w, 0.2, h])]
    items = []
    family = ['blob', 'jagged', 'thin', 'layered'][j // 4 % 4]
    number = int(g.choice([0, 2, 5, 10, 20, 35]))
    for k in range(number):
        d = np.exp(g.uniform(np.log(0.15), np.log(min(w, l) * 0.4)))
        size = d * g.uniform(0.4, 1.8, 3)
        center = [g.uniform(-w * 0.85, w * 0.85), g.uniform(-l * 0.85, l * 0.85), size[2] / 2]
        if np.linalg.norm(np.asarray(center) - C) < np.linalg.norm(size) + 0.8:
            continue
        items.append(dict(family=['blob', 'jagged', 'thin', 'layered'][k % 4], center=center, size=size.tolist(), rotation=[0, 0, float(g.uniform(-np.pi, np.pi))], seed=int(g.integers(1, 2 ** 31))))
    az = g.uniform(0, 2 * np.pi)
    el = np.arcsin(g.uniform(-1, 1))
    local = np.array([np.cos(el) * np.cos(az), np.sin(el), -np.cos(el) * np.sin(az)])
    speed = float(np.exp(g.uniform(np.log(0.0005), np.log(0.04))))
    T = R @ local * speed
    axis = g.normal(size=3)
    axis /= np.linalg.norm(axis)
    angular = axis * g.uniform(0.0001, 0.004)
    if g.random() < 0.2:
        angular[:] = 0
    distance = float(g.uniform(1.0, min(w, l, 12)))
    center = C + R @ np.array([g.uniform(-0.55, 0.55) * distance, g.uniform(-0.25, 0.25) * distance, -distance])
    diam = distance * g.uniform(0.06, 0.3)
    center[2] = max(center[2], diam * 0.7)
    objectV = R @ g.normal(size=3)
    objectV /= np.linalg.norm(objectV)
    objectV *= speed * float(np.exp(g.uniform(np.log(0.2), np.log(5))))
    ident = f'{split}_{j:05d}_{seed:08x}'
    s = dict(id=ident, parent=ident, source_parent=ident, split=split, seed=seed, size=256, fov_deg=90, frames=frames, camera_R0=R.tolist(), camera_C0=C.tolist(), translation_per_frame=T.tolist(), rotation_per_frame=angular.tolist(), heading_az_deg=float(np.rad2deg(az) % 360), heading_el_deg=float(np.rad2deg(el)), boxes=boxes, diverse_objects=items, target=dict(center=center.tolist(), size=[diam] * 3), target_family=family, target_mesh_seed=int(g.integers(1, 2 ** 31)), target_rotation=[0, 0, float(g.uniform(-np.pi, np.pi))], geometry_family=family, world_type='open_floor' if open_floor else 'room', room_dimensions=[float(2 * w), float(2 * l), float(h)], condition='static', object_translation_per_frame=objectV.tolist())
    s['scene_hash'] = hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()
    return s
