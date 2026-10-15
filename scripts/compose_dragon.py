#!/usr/bin/env python3
"""Add restrained labels and encode a looping GIF from the Blender PNG sequence."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/dragon"))
    p.add_argument("--fps", type=int, default=16)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    def font(size, heavy=False):
        return ImageFont.truetype(bold if heavy else regular, size)

    frames = []
    files = sorted(a.frames.glob("frame-*.png"))
    metrics = json.loads((a.output / "fast/metrics.json").read_text())
    n = len(metrics["passes"])
    for i, path in enumerate(files):
        image = Image.open(path).convert("RGB")
        w, h = image.size
        draw = ImageDraw.Draw(image)
        t = i / max(1, len(files) - 1)
        draw.rectangle((0, 0, w, 91), fill="#101923")
        draw.rectangle((0, h - 92, w, h), fill="#101923")
        draw.text(
            (32, 17),
            "DRAGON  /  GEOMETRIC DEPTH LAYERS",
            font=font(23, True),
            fill="#f0f4f8",
        )
        if t < 0.38:
            count = min(n, max(0, int((t - 0.025) / (0.32 / n)) + 1))
            title = f"01   STACK LAYERS   ·   {count:02d} / {n}"
            subtitle = "Acquisition camera view · semi-transparent surfaces and coverage outlines"
        elif t < 0.60:
            title = "02   ORBIT THE BILLBOARDS"
            subtitle = "Each layer is a flat plane · vertices retain their original camera rays"
        elif t < 0.80:
            title = "03   RECONSTRUCT THE SURFACE"
            subtitle = "Each vertex moves along its ray to its exported 3D position"
        else:
            title = "04   RETURN TO THE CAMERA"
            subtitle = "The original projection is preserved · loop restarts with an empty stack"
        draw.text((32, 54), title, font=font(17), fill="#8fcbe0")
        draw.text((32, h - 73), subtitle, font=font(16), fill="#d5dce5")
        draw.text(
            (32, h - 43),
            "721,510 input faces  ·  26 layers  ·  Approximate: exact-ray audit has mismatches  ·  XYZ RGB / Stanford",
            font=font(14),
            fill="#8f9dae",
        )
        draw.rounded_rectangle(
            (w - 186, h - 38, w - 30, h - 31), radius=3, fill="#293847"
        )
        draw.rounded_rectangle(
            (w - 186, h - 38, w - 186 + 156 * t, h - 31), radius=3, fill="#76c4df"
        )
        frames.append(image)
    if frames:
        # Shared palette reduces temporal palette flicker; keep all animation frames.
        thumb = Image.new("RGB", (275 * 8, 195 * 4))
        for j in range(32):
            idx = min(len(frames) - 1, int(j * (len(frames) - 1) / 31))
            v = frames[idx].resize((275, 195))
            thumb.paste(v, ((j % 8) * 275, (j // 8) * 195))
        palette = thumb.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        encoded = [
            f.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
            for f in frames
        ]
        encoded[0].save(
            a.output / "layers.gif",
            save_all=True,
            append_images=encoded[1:],
            duration=round(1000 / a.fps),
            loop=0,
            disposal=2,
            optimize=False,
        )
        frames[int(len(frames) * 0.58)].save(a.output / "billboard-stack.png")
        frames[int(len(frames) * 0.79)].save(a.output / "reconstructed.png")
    hero = Image.open(a.frames / "hero.png").convert("RGB")
    draw = ImageDraw.Draw(hero)
    w, h = hero.size
    draw.rectangle((0, 0, w, 142), fill="#101923")
    draw.rectangle((0, h - 112, w, h), fill="#101923")
    draw.text(
        (66, 28),
        "ONE DRAGON. 26 GEOMETRIC LAYERS.",
        font=font(42, True),
        fill="#f0f4f8",
    )
    draw.text(
        (68, 90),
        "721,510 input triangles  /  perspective camera  /  26 extracted layers",
        font=font(25),
        fill="#a8b9ca",
    )
    draw.text(
        (68, h - 87),
        "Original mesh + acquisition camera + extracted billboard stack",
        font=font(24, True),
        fill="#d5e1ee",
    )
    draw.text(
        (68, h - 49),
        "Approximate output · exact-ray audit has mismatches · 40% opacity · XYZ RGB / Stanford",
        font=font(20),
        fill="#8f9dae",
    )
    hero.save(a.output / "exploded.png")


if __name__ == "__main__":
    main()
