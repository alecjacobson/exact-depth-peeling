#!/usr/bin/env python3
"""Blender render of exported layers: camera stack, orbit, ray-preserving 3D morph.

Run with Blender --background --python scripts/render_dragon.py -- --assets ...
PNG frames and the editable .blend are generated from actual exported meshes.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=Path("results/dragon"))
    p.add_argument(
        "--output", type=Path, default=Path("/opt/depth-peeling-work/render")
    )
    p.add_argument("--frames", type=int, default=192)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--still-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--start", type=int, default=1)
    return p.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )


def render_frames(scene, a):
    import bpy

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "OPTIX"
    scene.cycles.device = "GPU"

    # Do not reevaluate millions of unchanged shape-key vertices during a pure
    # camera/material animation. Restore the original actions for the morph.
    actions = [
        (k, k.animation_data.action)
        for k in bpy.data.shape_keys
        if k.animation_data and k.animation_data.action
    ]
    visibility = [
        (o, o.animation_data.action)
        for o in bpy.data.objects
        if o.type == "MESH" and o.animation_data and o.animation_data.action
    ]
    visibility_frozen = False
    previous = None
    for frame in range(a.start, a.frames + 1):
        if frame >= round(a.frames * 0.35) and not visibility_frozen:
            for obj, _ in visibility:
                obj.animation_data.action = None
                obj.hide_render = False
            visibility_frozen = True
        if frame == a.frames:
            for obj, _ in visibility:
                obj.hide_render = True
        state = (
            0
            if frame < round(a.frames * 0.60)
            else (2 if frame > round(a.frames * 0.78) else 1)
        )
        if state != previous:
            for key, action in actions:
                key.animation_data.action = action if state == 1 else None
                if state != 1:
                    key.key_blocks[1].value = 0 if state == 0 else 1
            previous = state
        scene.frame_set(frame)
        scene.render.filepath = str(a.output / f"frame-{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
    for key, action in actions:
        key.animation_data.action = action
    for obj, action in visibility:
        obj.animation_data.action = action


def main():
    import bpy
    from mathutils import Matrix, Vector

    a = arguments()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.resume:
        bpy.ops.wm.open_mainfile(filepath=str(a.output / "dragon-layers.blend"))
        render_frames(bpy.context.scene, a)
        return
    data = np.load(a.assets / "meshes.npz")
    display_outlines = np.load(a.assets / "outlines.npz")
    meta = json.loads((a.assets / "input.json").read_text())
    n = int(data["layer_count"])
    matrix = np.array(meta["camera"]["world_to_camera"])
    r = matrix[:, :3]
    trans = matrix[:, 3]
    # Rendering coordinates are camera-aligned and normalized; all solver data is retained unchanged.
    target = np.array(meta["camera_target"])
    target_depth = float((r @ target + trans)[2])
    scale = 100.0
    center_camera = r @ target + trans

    def render_coords(world):
        return (
            ((world @ r.T + trans) - center_camera) / scale * np.array([1.0, 1.0, -1.0])
        )

    acquisition = render_coords(np.array(meta["camera_position"]))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = a.samples
    scene.cycles.use_denoising = True
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    scene.cycles.device = "GPU"
    scene.cycles.transparent_max_bounces = max(32, 2 * n + 4)
    scene.cycles.max_bounces = 8
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.resolution_x = 1100
    scene.render.resolution_y = 780
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (
        0.12,
        0.16,
        0.23,
        1,
    )
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.4
    scene.view_settings.view_transform = "AgX"
    scene.render.fps = 16
    scene.render.use_persistent_data = True
    palette = [
        "5BC4E8",
        "F5AF59",
        "9AD09A",
        "E7869D",
        "A395E8",
        "6CD6C1",
        "E9D27C",
        "89A9EB",
        "EA9C78",
        "A9BD75",
        "C994C3",
        "70BFC1",
    ]

    def rgb(h):
        return tuple((int(h[i : i + 2], 16) / 255) ** 2.2 for i in (0, 2, 4))

    def material(name, color, alpha=1, metallic=0.0):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nodes = m.node_tree.nodes
        links = m.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*color, 1)
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = 0.35
        mix = nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = alpha
        clear = nodes.new("ShaderNodeBsdfTransparent")
        links.new(clear.outputs[0], mix.inputs[1])
        links.new(bsdf.outputs[0], mix.inputs[2])
        links.new(mix.outputs[0], nodes.get("Material Output").inputs[0])
        return m, mix.inputs[0]

    def mesh(name, v, f, mat, edges=None):
        m = bpy.data.meshes.new(name)
        m.from_pydata(v.tolist(), [] if edges is None else edges.tolist(), f.tolist())
        m.update()
        obj = bpy.data.objects.new(name, m)
        bpy.context.collection.objects.link(obj)
        m.materials.append(mat)
        for face in m.polygons:
            face.use_smooth = True
        return obj

    def shape(obj, flat, lifted):
        obj.shape_key_add(name="Camera billboard")
        key = obj.shape_key_add(name="Original 3D surface")
        key.data.foreach_set("co", lifted.astype(np.float32).ravel())
        return key

    def outline_modifier(obj, radius=0.0013):
        g = bpy.data.node_groups.new(obj.name + " outline", "GeometryNodeTree")
        g.interface.new_socket(
            name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        g.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )
        nodes = g.nodes
        links = g.links
        inp = nodes.new("NodeGroupInput")
        out = nodes.new("NodeGroupOutput")
        curve = nodes.new("GeometryNodeMeshToCurve")
        circle = nodes.new("GeometryNodeCurvePrimitiveCircle")
        circle.inputs["Resolution"].default_value = 4
        circle.inputs["Radius"].default_value = radius
        tube = nodes.new("GeometryNodeCurveToMesh")
        setmat = nodes.new("GeometryNodeSetMaterial")
        setmat.inputs["Material"].default_value = obj.data.materials[0]
        links.new(inp.outputs["Geometry"], curve.inputs["Mesh"])
        links.new(curve.outputs["Curve"], tube.inputs["Curve"])
        links.new(circle.outputs["Curve"], tube.inputs["Profile Curve"])
        links.new(tube.outputs["Mesh"], setmat.inputs["Geometry"])
        links.new(setmat.outputs["Geometry"], out.inputs["Geometry"])
        mod = obj.modifiers.new("Boundary edges only", "NODES")
        mod.node_group = g

    layers = []
    outlines = []
    keys = []
    alpha = []
    outline_alpha = []
    for k in range(n):
        world = data[f"vertices_{k}"]
        faces = data[f"faces_{k}"]
        # Weld only render copies. Archive retains every original float64 corner.
        raw_world = world
        world, first_world, inv = np.unique(
            np.round(world, 8), axis=0, return_index=True, return_inverse=True
        )
        faces = inv[faces]
        cam = raw_world[first_world] @ r.T + trans
        uv = cam[:, :2] / cam[:, 2:]
        depth = target_depth + (k - (n - 1) / 2) * min(17.0, 180.0 / max(n - 1, 1))
        flatcam = np.column_stack([uv * depth, np.full(len(uv), depth)])
        flat = (flatcam - center_camera) / scale * np.array([1.0, 1.0, -1.0])
        lifted = render_coords(world)
        mat, opacity = material(
            f"Layer {k + 1:02d}", rgb(palette[k % len(palette)]), 0.40, 0.12
        )
        obj = mesh(f"Rank {k + 1:02d} · {len(faces):,} triangles", flat, faces, mat)
        key = shape(obj, flat, lifted)
        # Display-resolution union outlines avoid magnifying subpixel numerical cracks.
        line_world = display_outlines[f"world_{k}"]
        line_cam = line_world @ r.T + trans
        line_uv = line_cam[:, :2] / line_cam[:, 2:]
        line_flat_cam = np.column_stack([line_uv * depth, np.full(len(line_uv), depth)])
        line_flat = (line_flat_cam - center_camera) / scale * np.array([1.0, 1.0, -1.0])
        edges = display_outlines[f"edges_{k}"]
        line_mat, line_opacity = material(
            f"Rank {k + 1:02d} boundary", rgb(palette[k % len(palette)]), 1, 0.0
        )
        edge = mesh(
            f"Rank {k + 1:02d} projected boundary",
            line_flat,
            np.empty((0, 3), int),
            line_mat,
            edges,
        )
        outline_modifier(edge)
        layers.append(obj)
        outlines.append(edge)
        keys.append(key)
        alpha.append(opacity)
        outline_alpha.append(line_opacity)
        print(
            f"Prepared layer {k + 1}/{n}: {len(faces)} faces, {len(edges)} outline segments",
            flush=True,
        )

    def point_at(obj, target):
        # Explicit Y-up basis: to_track_quat uses world Z-up and is singular at
        # our acquisition view, which looks along Z in rendering coordinates.
        forward = (Vector(target) - obj.location).normalized()
        right = forward.cross(Vector((0, 1, 0))).normalized()
        up = right.cross(forward)
        obj.rotation_euler = Matrix((right, up, -forward)).transposed().to_euler()

    def light(name, position, power, size, color):
        bpy.ops.object.light_add(type="AREA", location=position)
        obj = bpy.context.object
        obj.name = name
        obj.data.energy = power
        obj.data.shape = "DISK"
        obj.data.size = size
        obj.data.color = color
        point_at(obj, (0, 0, 0))
        return obj

    light("Large softbox", (-3, -3, 5), 210, 5, (0.75, 0.87, 1.0))
    light("Warm rim", (4, 1, 3), 280, 3, (1.0, 0.78, 0.56))
    light("Top reflection", (-1, 3, -1), 150, 4, (0.55, 0.74, 1.0))
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Presentation orbit (starts at acquisition camera)"
    scene.camera = camera
    camera.data.clip_start = 0.01
    camera.data.clip_end = 100
    # Camera at exactly the extraction origin, matching the image window aspect.
    window = np.array(meta["window"])
    half = (window[2:] - window[:2]) / 2
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36
    camera.data.lens = 36 / (2 * half[0] * 1.12)

    def view(angle):
        origin = Vector(acquisition)
        pivot = Vector((0, 0, 0))
        rotation = Matrix.Rotation(math.radians(angle), 4, "Y")
        camera.location = pivot + rotation @ (origin - pivot) * (
            1 + 0.35 * math.sin(math.radians(angle))
        )
        point_at(camera, (0, 0, 0))

    def smooth(t):
        t = max(0.0, min(1.0, t))
        return t * t * (3 - 2 * t)

    for frame in range(1, a.frames + 1):
        t = (frame - 1) / (a.frames - 1)
        # Sequential stack 0..38%, orbit 38..57%, morph 60..78%, return 80..96%.
        angle = 58 * smooth((t - 0.38) / 0.19) * (1 - smooth((t - 0.80) / 0.16))
        view(angle)
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_euler", frame=frame)
        fade = 1 - smooth((t - 0.96) / 0.04)
        for k in range(n):
            appear = smooth((t - (0.025 + k * 0.32 / n)) / (0.32 / n * 0.7))
            layers[k].hide_render = outlines[k].hide_render = appear * fade < 1e-5
            layers[k].keyframe_insert("hide_render", frame=frame)
            outlines[k].keyframe_insert("hide_render", frame=frame)
            alpha[k].default_value = 0.40 * appear * fade
            alpha[k].keyframe_insert("default_value", frame=frame)
            outline_alpha[k].default_value = (
                appear * fade * (1 - smooth((t - 0.60) / 0.04))
            )
            outline_alpha[k].keyframe_insert("default_value", frame=frame)
    # Constant key intervals avoid rebuilding millions of vertices during pure camera orbits.
    for key in keys:
        for frame, value in [
            (1, 0),
            (round(a.frames * 0.60), 0),
            (round(a.frames * 0.78), 1),
            (a.frames, 1),
        ]:
            key.value = value
            key.keyframe_insert("value", frame=frame)
    scene.frame_start = 1
    scene.frame_end = a.frames
    # Keep the animation and actual data editable and save compressed.
    scene.frame_set(int(a.frames * 0.58))
    bpy.ops.wm.save_as_mainfile(
        filepath=str(a.output / "dragon-layers.blend"), compress=True
    )
    if a.preview:
        for label, f in [
            ("camera", int(a.frames * 0.36)),
            ("stack", int(a.frames * 0.58)),
            ("scene", int(a.frames * 0.79)),
        ]:
            scene.frame_set(f)
            scene.render.filepath = str(a.output / f"preview-{label}.png")
            bpy.ops.render.render(write_still=True)
        return
    if not a.still_only:
        render_frames(scene, a)
    # Hero: opaque source beside the translucent billboard stack, plus acquisition camera.
    scene.frame_set(int(a.frames * 0.58))
    for o in layers + outlines:
        o.location.x += 1.6
    mat, _ = material("Warm porcelain input", (0.58, 0.50, 0.37), 1, 0.18)
    source = mesh(
        "Original dragon · 721,510 input faces",
        render_coords(data["input_vertices"]),
        data["input_faces"],
        mat,
    )
    source.location.x = -2.0
    camera.animation_data_clear()
    camera.location = (4.4, 2.8, 6.2)
    point_at(camera, (-0.2, -0.05, 0.4))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 6.3
    scene.render.resolution_x = 2000
    scene.render.resolution_y = 1200
    scene.cycles.samples = 64
    # Grounding pedestal and restrained camera frustum in the diagram.
    pedestalmat, _ = material("Slate pedestal", (0.035, 0.055, 0.085), 1, 0.25)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=128,
        radius=1.22,
        depth=0.08,
        location=(-2.0, -0.55, 0),
        rotation=(math.pi / 2, 0, 0),
    )
    bpy.context.object.data.materials.append(pedestalmat)
    iconmat, _ = material("Acquisition camera", (0.14, 0.21, 0.29), 1, 0.65)
    pos = Vector(acquisition) + Vector((1.6, 0, 0))
    bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
    body = bpy.context.object
    body.name = "Acquisition camera icon"
    body.scale = (0.13, 0.09, 0.09)
    body.data.materials.append(iconmat)
    bpy.ops.mesh.primitive_cone_add(
        vertices=32,
        radius1=0.055,
        radius2=0.10,
        depth=0.13,
        location=pos + Vector((0, 0, -0.12)),
    )
    bpy.context.object.data.materials.append(iconmat)
    # Show the actual acquisition frustum to the first billboard depth.
    depth = target_depth - (n - 1) / 2 * min(17.0, 180.0 / max(n - 1, 1))
    corners = np.array(
        [
            [window[0], window[1]],
            [window[2], window[1]],
            [window[2], window[3]],
            [window[0], window[3]],
        ]
    )
    cp = np.column_stack([corners * depth, np.full(4, depth)])
    cp = (cp - center_camera) / scale * np.array([1.0, 1.0, -1.0]) + [1.6, 0.0, 0.0]
    frustum = np.vstack([np.array(pos), cp])
    wiremat, _ = material("Camera frustum", (0.12, 0.22, 0.30), 0.65, 0.0)
    wire = mesh(
        "Acquisition frustum",
        frustum,
        np.empty((0, 3), int),
        wiremat,
        np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]),
    )
    outline_modifier(wire, 0.0012)
    scene.render.filepath = str(a.output / "hero.png")
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(
        filepath=str(a.output / "dragon-hero.blend"), compress=True
    )


if __name__ == "__main__":
    main()
