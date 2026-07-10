"""Headless Blender: decimate the hand's visual _m.stl meshes into collision _simcol.stl.

Blender's Decimate (Collapse) modifier is robust on non-watertight meshes and gives
clean topology. Input meshes are already metric (scale=1.0), so coordinates are
preserved 1:1 — the resulting STLs drop straight into SHADOW_TOUCHLAB_simcol.urdf.

Run (args after -- are: <rest_ratio> <fingertip_ratio>, both optional):
  /home/ayush/blender-4.2/blender --background --python scripts/blender_decimate.py -- 0.2 0.05
                                                                                       ^rest ^tips
"""

import bpy, sys, os

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
RATIO_REST = float(argv[0]) if len(argv) >= 1 else 0.2   # all non-fingertip meshes
RATIO_TIP  = float(argv[1]) if len(argv) >= 2 else 0.05  # fingertips (over-tessellated)

ROOT = "/home/ayush/Desktop/real_to_sim/roto/roto/assets/shadow_lite"
MESHES = [
    "meshes/touchlab/fingertip_v5_simple_m.stl",
    "meshes/components/palm/palm_G1M5_m.stl",
    "meshes/components/th_middle/th_middle_G1M5_m.stl",
    "meshes/components/f_proximal/f_proximal_G1M5_m.stl",
    "meshes/components/f_knuckle/f_knuckle_G1M5_m.stl",
    "meshes/components/f_middle/f_middle_G1M5_m.stl",
    "meshes/components/th_proximal/th_proximal_G1M5_m.stl",
]


def ratio_for(rel):
    return RATIO_TIP if "fingertip" in rel else RATIO_REST


def simcol(rel):
    b, e = os.path.splitext(rel)
    if b.endswith("_m"):
        b = b[:-2]
    return b + "_simcol" + e


def imp(path):
    try:
        bpy.ops.wm.stl_import(filepath=path)          # Blender 4.x
    except AttributeError:
        bpy.ops.import_mesh.stl(filepath=path)        # older


def exp(path):
    try:
        bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True)
    except (AttributeError, TypeError):
        bpy.ops.export_mesh.stl(filepath=path, use_selection=True)


bpy.ops.wm.read_factory_settings(use_empty=True)
print(f"\n[blender] ratios: rest={RATIO_REST}  fingertip={RATIO_TIP}\n")
for rel in MESHES:
    # clear scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    imp(os.path.join(ROOT, rel))
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    before = len(obj.data.polygons)

    mod = obj.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio_for(rel)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    after = len(obj.data.polygons)

    obj.select_set(True)
    dst = os.path.join(ROOT, simcol(rel))
    exp(dst)
    print(f"[blender] {rel.split('/')[-1]:<34} {before:>6} -> {after:>6} faces  -> {os.path.basename(dst)}")

print("\n[blender] done. forearm stays a <box> in the URDF (handled separately).")
