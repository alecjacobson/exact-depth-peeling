#!/usr/bin/env python3
"""Render the actual input and lifted layer meshes with Blender (no inferred geometry).

python scripts/render_exploded.py --scene SCENE.json --layers OUTPUT --output FIGURE.png
"""

import argparse
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--layers", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inside-blender", action="store_true")
    p.add_argument("--blender", default="blender")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    return p.parse_args(argv)


def read_obj(path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            vertices.append(tuple(map(float, fields[1:4])))
        elif fields and fields[0] == "f":
            faces.append(tuple(int(v.split("/")[0]) - 1 for v in fields[1:4]))
    return vertices, faces


def render(a):
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Matrix, Vector

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.resolution_x = 2400
    scene.render.resolution_y = 1000
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True
    scene.world.color = (0.8, 0.8, 0.8)
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.7
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    palette = [
        "4D7BCB",
        "28A699",
        "E3AA43",
        "D9684E",
        "946BC6",
        "40AAC6",
        "D579A6",
        "88A64C",
        "A97146",
        "66889F",
    ]

    def material(name, rgb):
        color = tuple((int(rgb[i : i + 2], 16) / 255) ** 2.2 for i in (0, 2, 4))
        m = bpy.data.materials.new(name)
        m.diffuse_color = (*color, 1)
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*color, 1)
        bsdf.inputs["Roughness"].default_value = 0.57
        return m

    def mesh(name, vertices, faces, mat, offset=(0, 0, 0)):
        # Weld only render vertices, never the solver's authoritative geometry.
        lookup, vs, fs = {}, [], []
        for face in faces:
            indices = []
            for j in face:
                key = tuple(round(x, 12) for x in vertices[j])
                if key not in lookup:
                    lookup[key] = len(vs)
                    vs.append(key)
                indices.append(lookup[key])
            fs.append(indices)
        data = bpy.data.meshes.new(name)
        data.from_pydata(vs, [], fs)
        data.update()
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = offset
        obj.data.materials.append(mat)
        for face in obj.data.polygons:
            face.use_smooth = True
        return obj

    def line(name, points, mat, radius=0.015):
        curve = bpy.data.curves.new(name, "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = radius
        curve.bevel_resolution = 2
        spline = curve.splines.new("POLY")
        spline.points.add(len(points) - 1)
        for p, q in zip(spline.points, points):
            p.co = (*q, 1)
        obj = bpy.data.objects.new(name, curve)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(mat)

    data = json.loads(a.scene.read_text())
    vertices = [
        tuple(float(Fraction(x)) for x in v)
        for t in data["triangles"]
        for v in t["vertices"]
    ]
    faces = [(i, i + 1, i + 2) for i in range(0, len(vertices), 3)]
    mesh("Original connected mesh", vertices, faces, material("Input", "71849A"))
    files = sorted(
        a.layers.glob("layer-*.obj"), key=lambda p: int(p.stem.split("-")[-1])
    )
    labels = [
        {
            "name": "INPUT MESH",
            "point": [0, 0, -2.8],
            "detail": f"{len(faces):,} triangles",
            "color": "71849A",
        }
    ]
    for k, path in enumerate(files):
        vs, fs = read_obj(path)
        offset = Vector((4.7 * (k + 1), 0, 0))
        color = palette[k % len(palette)]
        mesh(f"Layer {k + 1}", vs, fs, material(f"Rank {k + 1}", color), offset)
        labels.append(
            {
                "name": f"LAYER {k + 1}",
                "point": list(offset + Vector((0, 0, -2.8))),
                "detail": f"{len(fs):,} faces",
                "color": color,
                "fragment_point": list(offset + Vector(vs[0])) if k >= 4 else None,
            }
        )

    # Show the actual acquisition camera, separately from the illustration camera.
    m = data["camera"]["world_to_camera"]
    transform = Matrix(
        [[float(Fraction(x)) for x in row] for row in m] + [[0, 0, 0, 1]]
    )
    inverse = transform.inverted()
    center = inverse @ Vector((0, 0, 0))
    dark = material("Camera", "27364A")
    amber = material("Projection guides", "CF9D55")
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    body = bpy.context.object
    body.name = "Acquisition camera"
    body.rotation_euler = inverse.to_euler()
    body.scale = (0.4, 0.28, 0.26)
    body.data.materials.append(dark)
    frame = [
        inverse @ Vector((x, y, 1))
        for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    ]
    line("Normalized image plane", frame, amber, 0.012)
    for p in frame[:4]:
        line("Camera ray", [center, p], amber, 0.006)
    line(
        "Optical axis",
        [inverse @ Vector((0, 0, 1)), inverse @ Vector((0, 0, 5))],
        amber,
        0.007,
    )
    labels.append(
        {
            "name": "CAMERA",
            "point": list(center + Vector((0, 0, -0.75))),
            "detail": "Perspective / no backface culling",
            "color": "27364A",
        }
    )

    bpy.ops.object.light_add(type="AREA", location=(8, -12, 18))
    light = bpy.context.object
    light.data.energy = 2500
    light.data.size = 18
    light.rotation_euler = (
        (Vector((10, 0, 0)) - light.location).to_track_quat("-Z", "Y").to_euler()
    )
    bpy.ops.object.light_add(type="AREA", location=(24, 12, 12))
    bpy.context.object.data.energy = 1800
    bpy.context.object.data.size = 15
    bpy.context.object.rotation_euler = (
        (Vector((15, 0, 0)) - bpy.context.object.location)
        .to_track_quat("-Z", "Y")
        .to_euler()
    )
    bpy.ops.object.camera_add(location=(15, -33, 30))
    camera = bpy.context.object
    target = Vector((max(8, 2.35 * len(files)), 0, 0))
    camera.location = target + Vector((0, -33, 30))
    camera.rotation_euler = (
        (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(24, 6.5 + 4.7 * len(files))
    scene.camera = camera
    bpy.context.view_layer.update()
    for label in labels:
        p = world_to_camera_view(scene, camera, Vector(label.pop("point")))
        label["screen"] = [
            p.x * scene.render.resolution_x,
            (1 - p.y) * scene.render.resolution_y,
        ]
        if label.get("fragment_point"):
            q = world_to_camera_view(scene, camera, Vector(label.pop("fragment_point")))
            label["fragment_screen"] = [
                q.x * scene.render.resolution_x,
                (1 - q.y) * scene.render.resolution_y,
            ]
    a.output.with_suffix(".labels.json").write_text(json.dumps(labels, indent=2))
    scene.render.filepath = str(a.output.resolve())
    bpy.ops.wm.save_as_mainfile(filepath=str(a.output.with_suffix(".blend").resolve()))
    bpy.ops.render.render(write_still=True)


def annotate(a):
    from PIL import Image, ImageDraw, ImageFont

    rgba = Image.open(a.output).convert("RGBA")
    bounds = rgba.getchannel("A").getbbox()
    crop_top = max(0, bounds[1] - 50) if bounds else 0
    crop_bottom = min(rgba.height, bounds[3] + 60) if bounds else rgba.height
    rgba = rgba.crop((0, crop_top, rgba.width, crop_bottom))
    image = Image.new("RGB", rgba.size, "white")
    image.paste(rgba, mask=rgba.getchannel("A"))
    canvas = Image.new("RGB", (image.width, image.height + 220), "white")
    canvas.paste(image, (0, 110))
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = lambda size: ImageFont.truetype(font_path, size)
    data = json.loads(a.scene.read_text())
    metrics = json.loads((a.layers / "metrics.json").read_text())
    draw.text((55, 24), "Exact geometric depth peeling", font=font(43), fill="#17263B")
    draw.text(
        (57, 79),
        f"{len(data['triangles']):,} input triangles  /  {len(metrics['passes'])} complete layers  /  same scale throughout",
        font=font(24),
        fill="#536174",
    )
    for label in json.loads(a.output.with_suffix(".labels.json").read_text()):
        x, y = label["screen"]
        y = min(y - crop_top + 110, image.height + 30)
        if "fragment_screen" in label:
            fx, fy = label["fragment_screen"]
            fy += 110 - crop_top
            draw.line([(x, y - 12), (fx, fy + 7)], fill="#BCC4CF", width=1)
            draw.ellipse((fx - 4, fy - 4, fx + 4, fy + 4), fill="#" + label["color"])
        draw.text(
            (x, y), label["name"], anchor="mt", font=font(23), fill="#" + label["color"]
        )
        draw.text(
            (x, y + 31), label["detail"], anchor="mt", font=font(17), fill="#536174"
        )
    draw.text(
        (55, canvas.height - 88),
        "Layers are lifted onto their original 3D surfaces, then translated for display. No geometry is simplified.",
        font=font(23),
        fill="#27364A",
    )
    draw.text(
        (55, canvas.height - 49),
        "Dots locate tiny later-rank fragments. The camera icon and image plane show the extraction viewpoint.",
        font=font(21),
        fill="#657185",
    )
    canvas.save(a.output)


def main():
    a = arguments()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.inside_blender:
        render(a)
    else:
        subprocess.run(
            [
                a.blender,
                "--background",
                "--factory-startup",
                "--python",
                str(Path(__file__).resolve()),
                "--",
                "--scene",
                str(a.scene.resolve()),
                "--layers",
                str(a.layers.resolve()),
                "--output",
                str(a.output.resolve()),
                "--inside-blender",
            ],
            check=True,
        )
        annotate(a)


if __name__ == "__main__":
    main()
