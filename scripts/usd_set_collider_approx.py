"""Set the collision approximation on the hand's collider meshes in a USD.

The prebuilt SHADOW_TOUCHLAB.usd bakes every collider link as convexDecomposition
(many convex pieces) — very expensive with self-collision across thousands of envs.
The fingers/thumb/fingertips are convex, so a single convexHull is faithful AND much
cheaper. The palm is concave (cradles the balls), so it keeps convexDecomposition.

Usage (operates in place; back up first):
    python usd_set_collider_approx.py SHADOW_TOUCHLAB.usd
    python usd_set_collider_approx.py SHADOW_TOUCHLAB.usd --out SHADOW_TOUCHLAB_hull.usd
    python usd_set_collider_approx.py SHADOW_TOUCHLAB.usd --concave palm forearm
"""

import argparse
from pxr import Usd, UsdPhysics

parser = argparse.ArgumentParser(description="Set collider approximation in a USD.")
parser.add_argument("usd", type=str, help="Path to the USD to edit.")
parser.add_argument("--out", type=str, default=None, help="Write to this path instead of in place.")
parser.add_argument("--concave", type=str, nargs="+", default=["palm"],
                    help="Substrings of link names to KEEP as convexDecomposition (default: palm).")
parser.add_argument("--hull", type=str, default="convexHull",
                    help="Approximation for the convex links (default: convexHull).")
args = parser.parse_args()


def link_of(path: str) -> str:
    # /colliders/<link>/<mesh>/node_STL_BINARY_  -> <link>
    parts = path.strip("/").split("/")
    return parts[1] if len(parts) > 1 else parts[0]


stage = Usd.Stage.Open(args.usd, load=Usd.Stage.LoadAll)
changed, kept = [], []
for p in stage.Traverse():
    if "PhysicsMeshCollisionAPI" not in p.GetAppliedSchemas():
        continue
    path = str(p.GetPath())
    link = link_of(path)
    mca = UsdPhysics.MeshCollisionAPI(p)
    attr = mca.GetApproximationAttr()
    cur = attr.Get() if attr else None
    if any(s.lower() in link.lower() for s in args.concave):
        kept.append((link, str(cur)))
        continue
    attr.Set(args.hull)
    changed.append((link, f"{cur} -> {args.hull}"))

if args.out:
    stage.Export(args.out)
    where = args.out
else:
    stage.Save()
    where = args.usd

print(f"\nSet to {args.hull} ({len(changed)} links):")
for l, c in changed:
    print(f"  {l:>14}  {c}")
print(f"\nKept as-is ({len(kept)} links):")
for l, c in kept:
    print(f"  {l:>14}  {c}")
print(f"\nwrote: {where}")
