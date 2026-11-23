"""EEVEE RGB rendering and mesh ray-cast reference flow on a 90-degree camera."""
import json
import math
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from worlds import pose, project

def original_material(rng, name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n = m.node_tree.nodes
    links = m.node_tree.links
    bsdf = n.get('Principled BSDF')
    bsdf.inputs['Roughness'].default_value = 0.85
    bsdf.inputs['Specular IOR Level'].default_value = 0
    bsdf.inputs['Emission Strength'].default_value = 0.25
    coord = n.new('ShaderNodeTexCoord')
    checker = n.new('ShaderNodeTexChecker')
    checker.inputs['Scale'].default_value = float(rng.uniform(8, 24))
    checker.inputs['Color1'].default_value = (*rng.uniform(0.05, 0.3, 3), 1)
    checker.inputs['Color2'].default_value = (*rng.uniform(0.5, 0.9, 3), 1)
    noise = n.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = float(rng.uniform(30, 90))
    noise.inputs['Detail'].default_value = 2
    mix = n.new('ShaderNodeMixRGB')
    mix.blend_type = 'MULTIPLY'
    mix.inputs[0].default_value = 0.6
    links.new(coord.outputs['Generated'], checker.inputs['Vector'])
    links.new(coord.outputs['Generated'], noise.inputs['Vector'])
    links.new(checker.outputs['Color'], mix.inputs[1])
    links.new(noise.outputs['Fac'], mix.inputs[2])
    links.new(mix.outputs[0], bsdf.inputs['Base Color'])
    links.new(mix.outputs[0], bsdf.inputs['Emission Color'])
    return m


def material(rng, name):
    m = original_material(rng, name)
    nodes = m.node_tree.nodes
    links = m.node_tree.links
    geom = nodes.new('ShaderNodeNewGeometry')
    for node in nodes:
        if node.bl_idname in ['ShaderNodeTexChecker', 'ShaderNodeTexNoise']:
            links.new(geom.outputs['Position'], node.inputs['Vector'])
            node.inputs['Scale'].default_value = float(rng.uniform(0.5, 4) if node.bl_idname == 'ShaderNodeTexChecker' else rng.uniform(2, 12))
    return m


def shape(spec):
    family = spec['family']
    rng = np.random.default_rng(spec['seed'])
    if family == 'thin':
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.5, depth=1)
        o = bpy.context.object
    elif family == 'layered':
        bpy.ops.mesh.primitive_torus_add(major_radius=0.34, minor_radius=0.16, major_segments=24, minor_segments=10)
        o = bpy.context.object
    elif family == 'jagged':
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.5)
        o = bpy.context.object
        for v in o.data.vertices:
            v.co *= float(rng.uniform(0.65, 1.5))
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=0.5)
        o = bpy.context.object
        phase = rng.uniform(0, 2 * np.pi, 4)
        for v in o.data.vertices:
            p = v.co
            theta = math.atan2(p.y, p.x)
            phi = math.atan2(math.hypot(p.x, p.y), p.z)
            factor = 1 + 0.19 * math.sin(3 * theta + phase[0]) * math.sin(phi) ** 2 + 0.13 * math.cos(5 * phi + phase[1]) + 0.1 * math.sin(2 * theta + phase[2]) * math.sin(2 * phi)
            if family == 'twisted_star_unseen':
                factor += 0.18 * math.cos(7 * theta + 3 * phi) * math.sin(phi) ** 2
            v.co *= factor
        for polygon in o.data.polygons:
            polygon.use_smooth = True
    o.location = spec['center']
    o.rotation_euler = spec.get('rotation', [0, 0, 0])
    dims = np.array(spec['size']) * (np.array([0.12, 0.12, 3.0]) if family == 'thin' else 1)
    o.dimensions = dims
    bpy.context.view_layer.update()
    return o


def bvh(objects):
    vertices = []
    faces = []
    target_ids = []
    for obj in objects:
        obj.data.calc_loop_triangles()
        offset = len(vertices)
        vertices.extend([obj.matrix_world @ v.co for v in obj.data.vertices])
        tri = [tuple((offset + i for i in t.vertices)) for t in obj.data.loop_triangles]
        faces.extend(tri)
        target_ids.extend([obj.get('target', False)] * len(tri))
    return (BVHTree.FromPolygons(vertices, faces, all_triangles=True), np.asarray(target_ids))


def reference(s, t, tree, target_flags):
    R, C = pose(s, t)
    size = 128
    yy, xx = np.mgrid[:size, :size]
    local = np.stack([(xx + 0.5 - size / 2) / (size / 2), -(yy + 0.5 - size / 2) / (size / 2), -np.ones_like(xx)], -1)
    rays = local @ R.T
    points = np.full((size * size, 3), np.nan)
    ids = np.full(size * size, -1)
    origin = Vector(C)
    for i, r in enumerate(rays.reshape(-1, 3)):
        hit, normal, idx, distance = tree.ray_cast(origin, Vector(r).normalized(), 200)
        if hit is not None:
            points[i] = hit
            ids[i] = idx
    points = points.reshape(size, size, 3)
    Z = -((points - C) @ R)[..., 2]
    mask = np.zeros(size * size, bool)
    take = ids >= 0
    mask[take] = target_flags[ids[take]]
    return (points, Z, mask.reshape(size, size), take.reshape(size, size))


def inside(tree, C):
    direction = Vector((1.0, 0.317, 0.193)).normalized()
    point = Vector(C)
    hits = 0
    for _ in range(100):
        hit, _, _, distance = tree.ray_cast(point, direction, 200)
        if hit is None:
            return hits % 2 == 1
        hits += 1
        point = hit + direction * 1e-05
    raise RuntimeError('Ray parity did not terminate')


def validate(s, conditions=None):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    objects = []
    for b in s['boxes']:
        bpy.ops.mesh.primitive_cube_add(size=1, location=b['center'])
        obj = bpy.context.object
        obj.dimensions = b['size']
        objects.append(obj)
    for spec in s['diverse_objects']:
        objects.append(shape(spec))
    target = shape(dict(family=s['target_family'], seed=s['target_mesh_seed'], center=s['target']['center'], size=s['target']['size'], rotation=s['target_rotation']))
    bpy.context.view_layer.update()
    background = bvh(objects)[0]
    if conditions is None:
        conditions = ['absent', 'static', 'moving'] if s['split'] == 'test' else ['static']
    minimum = float('inf')
    for condition in conditions:
        velocity = np.asarray(s['object_translation_per_frame']) if condition == 'moving' else np.zeros(3)
        for t in range(s['frames']):
            target.location = np.asarray(s['target']['center']) + velocity * t
            bpy.context.view_layer.update()
            active = objects + ([target] if condition != 'absent' else [])
            tree = bvh(active)[0]
            _, C = pose(s, t)
            nearest = tree.find_nearest(Vector(C))
            distance = float(nearest[3]) if nearest[0] is not None else float('inf')
            minimum = min(minimum, distance)
            if distance < 0.06:
                return dict(passed=False, reason='Camera clearance below0.06m', condition=condition, frame=t, minimum_clearance=minimum)
            for obj in active:
                bounds = np.array([obj.matrix_world @ Vector(v) for v in obj.bound_box])
                if ((C >= bounds.min(0)) & (C <= bounds.max(0))).all() and inside(bvh([obj])[0], C):
                    return dict(passed=False, reason='Camera inside closed geometry', condition=condition, frame=t, minimum_clearance=minimum)
            if t < s['frames'] - 1:
                _, end = pose(s, t + 1)
                delta = end - C
                length = float(np.linalg.norm(delta))
                if length > 1e-09 and background.ray_cast(Vector(C), Vector(delta / length), length)[0] is not None:
                    return dict(passed=False, reason='Camera segment crosses stationary geometry', condition=condition, frame=t, minimum_clearance=minimum)
                if condition != 'absent':
                    relative = delta - velocity
                    length = float(np.linalg.norm(relative))
                    if length > 1e-09 and bvh([target])[0].ray_cast(Vector(C), Vector(relative / length), length)[0] is not None:
                        return dict(passed=False, reason='Camera and target trajectories intersect', condition=condition, frame=t, minimum_clearance=minimum)
    return dict(passed=True, minimum_clearance=minimum)


def render_scene(s, root, conditions=None):
    safety = validate(s, conditions)
    if not safety['passed']:
        (root / 'rejected.json').write_text(json.dumps(dict(parent=s['id'], scene_hash=s['scene_hash'], reason=safety)))
        return
    rng = np.random.default_rng(s['seed'])
    if conditions is None:
        conditions = ['absent', 'static', 'moving'] if s['split'] == 'test' else ['static']
    started = time.monotonic()
    for condition in conditions:
        dest = root / condition
        dest.mkdir(exist_ok=True)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        for block in list(bpy.data.materials):
            bpy.data.materials.remove(block)
        for collection in [bpy.data.meshes, bpy.data.cameras, bpy.data.lights]:
            for block in list(collection):
                if block.users == 0:
                    collection.remove(block)
        scene = bpy.context.scene
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
        scene.render.resolution_x = scene.render.resolution_y = 256
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.fps = 90
        scene.world.color = (0.3, 0.3, 0.3)
        scene.view_settings.view_transform = 'Standard'
        rng = np.random.default_rng(s['seed'])
        objects = []
        for i, b in enumerate(s['boxes']):
            bpy.ops.mesh.primitive_cube_add(size=1, location=b['center'])
            o = bpy.context.object
            o.dimensions = b['size']
            o.data.materials.append(material(rng, f'background_{i}'))
            objects.append(o)
        for i, spec in enumerate(s['diverse_objects']):
            o = shape(spec)
            o.data.materials.append(material(rng, f'irregular_{i}'))
            objects.append(o)
        target_mat = material(rng, 'target')
        target = None
        if condition != 'absent':
            target = shape(dict(family=s['target_family'], seed=s['target_mesh_seed'], center=s['target']['center'], size=s['target']['size'], rotation=s['target_rotation']))
            target.data.materials.append(target_mat)
            target['target'] = True
            objects.append(target)
        bpy.ops.object.light_add(type='AREA', location=(0, 0, 3))
        bpy.context.object.data.energy = 250
        bpy.context.object.data.size = 6
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        scene.camera = camera
        camera.data.lens = 18
        camera.data.sensor_width = 36
        camera.data.sensor_fit = 'HORIZONTAL'
        camera.data.clip_start = 0.03
        camera.data.clip_end = 200
        fields = []
        trees = []
        for t in range(s['frames']):
            R, C = pose(s, t)
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = C
            camera.matrix_world = Matrix(M.tolist())
            if target is not None:
                target.location = np.asarray(s['target']['center']) + (np.asarray(s['object_translation_per_frame']) * t if condition == 'moving' else 0)
            bpy.context.view_layer.update()
            tree, flags = bvh(objects)
            fields.append(reference(s, t, tree, flags))
            trees.append(tree)
            scene.render.filepath = str(dest / f'{t:04d}.png')
            bpy.ops.render.render(write_still=True)
        for t in range(s['frames'] - 1):
            points, Z, mask, finite = fields[t]
            moved = points.copy()
            if condition == 'moving':
                moved[mask] += s['object_translation_per_frame']
            end, z = project(s, moved, t + 1, 128)
            yy, xx = np.mgrid[:128, :128]
            flow = end - np.stack([xx, yy], -1)
            R, C = pose(s, t + 1)
            valid = finite & (z > 0) & (end[..., 0] >= 0) & (end[..., 0] <= 127) & (end[..., 1] >= 0) & (end[..., 1] <= 127)
            occluded = np.zeros((128, 128), bool)
            for j in np.flatnonzero(valid):
                p = moved.reshape(-1, 3)[j]
                direction = p - C
                distance = np.linalg.norm(direction)
                hit, _, _, length = trees[t + 1].ray_cast(Vector(C), Vector(direction).normalized(), 200)
                if hit is None or abs(length - distance) > 0.01 + 0.001 * distance:
                    occluded.ravel()[j] = True
            valid &= ~occluded
            if not np.isfinite(flow).all() or not np.isfinite(Z).all() or (Z <= 0).any():
                raise ValueError('Invalid world/camera: reject; do not impute')
            np.savez_compressed(dest / f'{t:04d}_reference.npz', flow=flow.astype('float32'), depth=Z.astype('float32'), occupancy=mask, valid=valid, occluded=occluded, K_rgb=np.array([[128, 0, 127.5], [0, 128, 127.5], [0, 0, 1]]), K_flow=np.array([[64, 0, 63.5], [0, 64, 63.5], [0, 0, 1]]), dt=np.float64(1 / 90))
    (root / 'complete.json').write_text(json.dumps(dict(scene_hash=s['scene_hash'], conditions=conditions, wall_seconds=time.monotonic() - started, blender=bpy.app.version_string, reference='World-space triangulated BVH, source rays and destination visibility, axial Z')))
    print('PARENT_COMPLETE', s['id'], time.monotonic() - started, flush=True)
