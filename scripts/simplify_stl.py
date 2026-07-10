"""Make a collision STL from a visual STL (decimate / convex-hull / convex-decompose).

A collision mesh only needs the rough shape, so we reduce the triangle count of the
visual mesh while keeping its surface. Three modes:

  decimate (default) — quadric-decimate to a fraction of faces (keeps concave detail)
  hull               — single convex hull (cheapest; fills concavities)
  coacd              — convex decomposition into pieces (faithful concave collision)

Examples:
  # keep 50% of the faces:
  python simplify_stl.py visual.stl collision.stl --ratio 0.5
  # target an exact face budget:
  python simplify_stl.py visual.stl collision.stl --faces 800
  # single convex hull:
  python simplify_stl.py visual.stl collision.stl --mode hull
  # convex decomposition (writes collision_0.stl, collision_1.stl, ...):
  python simplify_stl.py visual.stl collision.stl --mode coacd --threshold 0.05
"""

import argparse, os, numpy as np, trimesh

ap = argparse.ArgumentParser()
ap.add_argument("src", help="input visual mesh (.stl/.obj/.dae)")
ap.add_argument("dst", help="output collision .stl")
ap.add_argument("--mode", choices=["decimate", "hull", "coacd"], default="decimate")
ap.add_argument("--ratio", type=float, default=0.5, help="decimate: fraction of faces to keep")
ap.add_argument("--faces", type=int, default=None, help="decimate: exact target face count (overrides ratio)")
ap.add_argument("--threshold", type=float, default=0.05, help="coacd: concavity (lower=more pieces)")
ap.add_argument("--max_hulls", type=int, default=8, help="coacd: cap number of pieces")
args = ap.parse_args()

m = trimesh.load(args.src, process=False)
print(f"input: {len(m.faces)} faces, extents(m)={np.round(m.extents,4)}, watertight={m.is_watertight}")


def deviation(orig, parts):
    pts = orig.sample(4000)
    d = np.min(np.stack([trimesh.proximity.closest_point(p, pts)[1] for p in parts]), 0)
    return d.max() * 1000, d.mean() * 1000


if args.mode == "decimate":
    target = args.faces or max(50, int(len(m.faces) * args.ratio))
    out = m.simplify_quadric_decimation(face_count=target) if len(m.faces) > target else m
    mx, mn = deviation(m, [out])
    out.export(args.dst)
    print(f"decimated -> {len(out.faces)} faces | maxdev {mx:.2f}mm mean {mn:.3f}mm -> {args.dst}")

elif args.mode == "hull":
    out = m.convex_hull
    mx, mn = deviation(m, [out])
    out.export(args.dst)
    print(f"convex hull -> {len(out.faces)} faces | maxdev {mx:.2f}mm mean {mn:.3f}mm -> {args.dst}")

elif args.mode == "coacd":
    import coacd
    parts = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces),
                            threshold=args.threshold, max_convex_hull=args.max_hulls)
    pieces, base, ext = [], *os.path.splitext(args.dst)
    for i, (v, f) in enumerate(parts):
        pc = trimesh.Trimesh(v, f).convex_hull
        pc.export(f"{base}_{i}{ext}")
        pieces.append(pc)
    mx, mn = deviation(m, pieces)
    print(f"coacd -> {len(parts)} pieces ({sum(len(p.faces) for p in pieces)} faces) | "
          f"maxdev {mx:.2f}mm mean {mn:.3f}mm -> {base}_*{ext}")
