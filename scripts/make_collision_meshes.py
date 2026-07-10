"""Build high-res collision geometry for the hand and write a new URDF.

Strategy (chosen: high-res decimated meshes; convert later with convex_decomposition):
  • Every finger/thumb/fingertip + the palm keep their TRUE surface (incl. concavities),
    lightly quadric-decimated to ~RATIO of the original faces. -> meshes/.../<name>_simcol.stl
  • The forearm becomes a primitive BOX (it's the fixed base; exact shape irrelevant).
The new URDF keeps the full-res mesh for <visual> and points <collision> at the decimated
meshes (and the box for the forearm). Convert to USD with collider_type="convex_decomposition".

    python make_collision_meshes.py [--ratio 0.5] [--max_faces 6000]
"""

import argparse, os, re, numpy as np, trimesh

ROOT = "/home/ayush/Desktop/real_to_sim/roto/roto/assets/shadow_lite"
SRC_URDF = os.path.join(ROOT, "SHADOW_TOUCHLAB.urdf")
OUT_URDF = os.path.join(ROOT, "SHADOW_TOUCHLAB_simcol.urdf")
FOREARM_REL = "meshes/components/forearm/forearm_G1M5.stl"
# all mesh-collision links (decimated, surface preserved); forearm handled as a box
MESHES = [
    "meshes/touchlab/fingertip_v5_simple_m.stl",
    "meshes/components/palm/palm_G1M5_m.stl",
    "meshes/components/th_middle/th_middle_G1M5_m.stl",
    "meshes/components/f_proximal/f_proximal_G1M5_m.stl",
    "meshes/components/f_knuckle/f_knuckle_G1M5_m.stl",
    "meshes/components/f_middle/f_middle_G1M5_m.stl",
    "meshes/components/th_proximal/th_proximal_G1M5_m.stl",
]


def simcol(rel):
    b, e = os.path.splitext(rel)
    if b.endswith("_m"):
        b = b[:-2]
    return b + "_simcol" + e


def dev_mm(orig, simp, n=4000):
    pts = orig.sample(n)
    d = trimesh.proximity.closest_point(simp, pts)[1]
    return d.max() * 1000, d.mean() * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.5, help="keep this fraction of faces")
    ap.add_argument("--max_faces", type=int, default=6000, help="cap per-link collision faces")
    ap.add_argument("--min_faces", type=int, default=200)
    args = ap.parse_args()

    print(f"{'mesh':<40} {'orig':>6} {'simp':>6} {'maxdev':>8} {'meandev':>8}")
    print("-" * 80)
    for rel in MESHES:
        m = trimesh.load(os.path.join(ROOT, rel), process=False)
        tgt = int(np.clip(round(len(m.faces) * args.ratio), args.min_faces, args.max_faces))
        s = m.simplify_quadric_decimation(face_count=tgt) if len(m.faces) > tgt else m
        mx, mn = dev_mm(m, s)
        s.export(os.path.join(ROOT, simcol(rel)))
        print(f"{rel.split('/')[-1]:<40} {len(m.faces):>6} {len(s.faces):>6} {mx:>6.2f}mm {mn:>7.3f}mm")

    # --- write new URDF ------------------------------------------------------
    urdf = open(SRC_URDF).read()

    # forearm -> AABB box, computed in the LINK frame (apply the original collision
    # <origin> transform to the mesh first, since the forearm mesh is placed with a
    # 90° z-rotation). Then the box uses rpy 0 at the transformed AABB center.
    fm = trimesh.load(os.path.join(ROOT, FOREARM_REL), process=False)
    fa_link = re.search(r'<link name="rh_forearm".*?</link>', urdf, re.DOTALL).group(0)
    fa_col = re.search(r"<collision>.*?forearm_G1M5\.stl.*?</collision>", fa_link, re.DOTALL).group(0)
    o = re.search(r'<origin\s+rpy="([^"]*)"\s+xyz="([^"]*)"', fa_col)
    rpy = [float(x) for x in o.group(1).split()]
    xyz = [float(x) for x in o.group(2).split()]
    T = trimesh.transformations.euler_matrix(*rpy)
    T[:3, 3] = xyz
    fm.apply_transform(T)
    lo, hi = fm.bounds
    c = (lo + hi) / 2.0
    sz = (hi - lo)
    print(f"{'forearm -> BOX':<40} size={np.round(sz,4)} center={np.round(c,4)} (orig rpy={rpy})")

    # forearm collision -> box
    box_block = (f'    <collision>\n'
                 f'      <origin rpy="0 0 0" xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}"/>\n'
                 f'      <geometry>\n'
                 f'        <box size="{sz[0]:.6f} {sz[1]:.6f} {sz[2]:.6f}"/>\n'
                 f'      </geometry>\n'
                 f'    </collision>')
    urdf = re.sub(r"[ \t]*<collision>(?:(?!</collision>).)*?forearm_G1M5\.stl.*?</collision>",
                  box_block, urdf, count=1, flags=re.DOTALL)

    # swap remaining collision mesh filenames -> _simcol (collision blocks only)
    def swap(block):
        b = block.group(0)
        for rel in MESHES:
            b = b.replace(rel, simcol(rel))
        return b
    urdf = re.sub(r"<collision>.*?</collision>", swap, urdf, flags=re.DOTALL)

    open(OUT_URDF, "w").write(urdf)
    print(f"\nwrote {OUT_URDF}")
    print("  forearm collision = <box>; all other collisions = decimated _simcol mesh")
    print("  visuals untouched. Convert to USD with collider_type='convex_decomposition'.")

    # tidy: remove orphaned palm CoACD pieces from the previous approach
    for f in os.listdir(os.path.join(ROOT, "meshes/components/palm")):
        if re.match(r"palm_G1M5_ch\d+\.stl", f):
            os.remove(os.path.join(ROOT, "meshes/components/palm", f))


if __name__ == "__main__":
    main()
