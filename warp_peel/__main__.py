"""JSON upload and explicit NPZ download around a device-resident extraction."""

import argparse
import json
from pathlib import Path

import numpy as np

from . import Capacities, DevicePeeler
from .io import download_mesh, load_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=0, help="0 means all layers")
    parser.add_argument("--pieces", type=int)
    parser.add_argument("--pool-scale", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    arrays, labels, _ = load_scene(args.scene, args.device)
    from dataclasses import replace

    cap = Capacities.for_mesh(arrays[1].size)
    cap = replace(
        cap,
        residual=cap.residual * args.pool_scale,
        polygons=cap.polygons * args.pool_scale,
        vertices=cap.vertices * args.pool_scale,
        triangles=cap.triangles * args.pool_scale,
        pieces_per_worker=args.pieces or cap.pieces_per_worker,
    )
    p = DevicePeeler(*arrays, capacities=cap, layers=args.layers)
    p.run()
    mesh = download_mesh(p, labels)
    info = mesh.pop("status")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **mesh)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "warp-peel-approximate-v1",
                "zero_based_layers": True,
                "exact_geometry": False,
                "scene": str(args.scene),
                "output": info,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(info))


if __name__ == "__main__":
    main()
