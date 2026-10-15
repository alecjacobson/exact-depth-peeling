#!/usr/bin/env python3
"""Cache independent Blender collapse-decimations with a fixed original camera.

Run with Blender's Python (see --help after --). Geometry is converted to Blender's
float32 mesh storage before decimation; the resulting represented coordinates are
then the exact input to both solvers. Each level is decimated from the full input,
not successively from the preceding level. Exact intersection preprocessing is a
separate, cached step in benchmark_global.py.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dragon_scene import read_ply


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', type=Path, default=Path('../sparse-solver-benchmark/xyzrgb_dragon-720K.ply'))
    p.add_argument('--camera', type=Path, default=Path('results/dragon/input.json'))
    p.add_argument('--output', type=Path, default=Path('results/global/dragon-cache'))
    p.add_argument('--sizes', type=int, nargs='+', default=[64, 128, 512, 2048, 8192, 32768])
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    a = p.parse_args(args)
    camera = json.loads(a.camera.read_text())
    key = {'input_sha256': digest(a.mesh), 'camera_sha256': digest(a.camera),
           'generator_sha256': digest(Path(__file__)), 'blender': bpy.app.version_string,
           'method': 'Decimate COLLAPSE, triangulate, independent full-input ratios',
           'coordinate_storage': 'Blender float32, exported as exactly represented binary64'}
    a.output.mkdir(parents=True, exist_ok=True)
    vertices, faces = read_ply(a.mesh)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    mesh = bpy.data.meshes.new('original-dragon')
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    records = []
    for target in sorted(set(a.sizes)):
        if target < 4 or target > len(faces):
            raise ValueError('target face count outside [4, input_faces]')
        directory = a.output / str(target)
        scene = directory / 'raw.json'
        manifest = directory / 'manifest.json'
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if old['cache_key'] == key and scene.exists() and digest(scene) == old['scene_sha256']:
                records.append(old)
                print(f'cache hit: {target}', flush=True)
                continue
            raise RuntimeError(f'Cache identity changed at {directory}; use a new output directory')
        start = time.monotonic()
        obj = bpy.data.objects.new('decimated-dragon', mesh)
        bpy.context.collection.objects.link(obj)
        modifier = obj.modifiers.new('collapse', 'DECIMATE')
        modifier.decimate_type = 'COLLAPSE'
        modifier.ratio = target / len(faces)
        modifier.use_collapse_triangulate = True
        graph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(graph)
        reduced = evaluated.to_mesh()
        reduced.calc_loop_triangles()
        v = np.empty(len(reduced.vertices) * 3, dtype=np.float32)
        reduced.vertices.foreach_get('co', v)
        v = v.reshape(-1, 3).astype(np.float64)
        f = np.empty(len(reduced.loop_triangles) * 3, dtype=np.int32)
        reduced.loop_triangles.foreach_get('vertices', f)
        f = f.reshape(-1, 3)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / 'mesh.npz', vertices=v, faces=f)
        data = {'camera': camera['camera'], 'window': camera['window'],
                'triangles': [{'id': i, 'mesh_face_id': i, 'vertices': v[face].tolist()}
                              for i, face in enumerate(f)]}
        scene.write_text(json.dumps(data, separators=(',', ':')) + '\n')
        record = {'cache_key': key, 'target_faces': target, 'actual_faces': len(f),
                  'vertices': len(v), 'scene_sha256': digest(scene),
                  'mesh_sha256': digest(directory / 'mesh.npz'),
                  'decimation_seconds': time.monotonic() - start}
        manifest.write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
        evaluated.to_mesh_clear()
        bpy.data.objects.remove(obj, do_unlink=True)
        print(json.dumps(record), flush=True)
    (a.output / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
