# Shadow Hand Lite — Coupling Work (2026-06-26)

Work on the FF/MF/RF coupled-finger (J1←J2) mimic mechanism, domain randomization,
diagnostics, and an interactive viewer. All coupling code lives in
`roto/tasks/roto_env.py` (`_asymmetric_backlash`), with config in
`roto/tasks/robots/shadowlite/shadowlite.py` and the settle phase in
`roto/tasks/baoding/baoding.py`.

---

## 1. The coupling mechanism (what it models)

The real hand drives J2 (PIP) and J1 (DIP) from **one motor** (`ffj0`, 0–180°) via a
tendon. In sim J1/J2 are separate joints, so we emulate the tendon. We work in the
**combined motor frame** `m ∈ [0°,180°]` (`m = 2·proxy`), where J2 owns 0–100° and J1
owns 100–180°.

### Stateful backlash law (per finger)
Per-episode latched **unlock angle `R ∈ [100°,140°]`**; J1 carried as state.

- **Curl (m rising):** J2 fills 0→100° first, then J1 fills 0→80°. J1 only moves once
  J2 saturates at 100°.
- **Uncurl (m falling):** J1 unwinds, hitting 0 at m=100°. J2 **unlocks early at R**
  (`j2 = m/R·100°` for m<R), so over `[100°, R]` **both J2 and J1 drop together**.
- **Reversal (uncurl, stop in (100°,R), curl back):** J1 **freezes** until m climbs
  back to R, then resumes from the frozen value up to 80° at m=180°. (`R=100°` →
  symmetric/no backlash; `R=140°` → max play.)

---

## 2. Bugs found & fixed (in `_asymmetric_backlash` / `_pre_physics_step`)

1. **`frac=1` strict gate was a silent no-op.** The earlier measured-J2 gate needed
   `meas_j2 > j2_upper`, which never happens, so J1 never fired. Fixed with a tolerance
   band (`couple_gate_j2_tol ≈ 2°`) so J1 fires when J2 is within tol of its limit. (The
   backlash model later superseded this gate when `couple_asymmetric_backward=True`.)

2. **J2 "bounced back to the 100° limit" on uncurl.** J2's branch was keyed on the
   **instantaneous** direction (`falling`). The slider only fires on movement, so most
   sim frames are "steady" (`Δm≈0`) → J2 reverted to `j2_fresh = clamp(m,0,100) = 100°`.
   **Fix:** key J2 (and the freeze-clear) on the **latched** direction (`couple_dir`),
   so steady frames keep the uncurl state. Verified offline: at m=124° J2 went from
   `[91,100,100,100]` → `[91,91,91,91]`.

3. **Settle phase corrupted coupling state.** During the ball-settle window the hand is
   held in its catch pose, but the backlash state kept advancing from the policy proxy.
   **Fix:** snapshot the coupling buffers and restore them for settling envs, so the
   state stays frozen-open and hands over cleanly when the policy takes over.

---

## 3. Domain randomization (per episode, on reset)

- **Unlock angle `R`** — sampled per finger in `couple_release_range_deg = (100°,140°)`,
  latched for the episode (`_sample_coupling_params`).
- **Hand mounting tilt** — `hand_tilt_range_deg = (0°,15°)`, nlerp between the upright
  (0°) and 15°-forward root quaternions, written via `write_root_pose_to_sim`
  (`_randomize_hand_tilt`). *Caveat: fixed-base root write — verify it re-tilts per env.*

---

## 4. Ball-settle phase (`baoding.py` + `roto_env.py`)

`settle_steps = 15`: after reset the hand holds its catch pose for N steps so the
dropped balls settle into the palm before the policy acts. Terminations are masked and
the coupling state is frozen during settle.

---

## 5. Geometry / collision investigation

- The **palm-collision** theory for the ~90° J2 stop was **wrong**. Measured in-sim:
  the limit is **finger–finger self-collision** (MF↔RF along their length, FF↔MF
  proximal), onset ≈48° of curl.
- At soft realistic gains (stiffness 1) the contact **blocks** J2 at ~90°; at the
  viewer's stiffness 20, fingers slide past and **all reach 100°**.
- FF/RF knuckle abduction (J4) is **already at its ±20° limit** in the default pose, so
  the viewer's `--spread_deg` widens the J4 limit (viewer-only, non-physical) to splay
  the fingers apart for a clean mimic view. (A real fix would be knuckle spacing /
  collision geometry — separate plant issue, not done.)

---

## 6. Scripts (in `scripts/`)

| Script | Purpose |
|---|---|
| `view_coupling_slider.py` | **Interactive** omni.ui viewer — master + per-finger curl sliders, R override, asymmetric on/off, live J2/J1 + FROZEN readout, `--spread_deg`, `--stiffness`. |
| `test_backlash_coupling.py` | Drives exact combined-`m` waypoints; validates curl / uncurl-overlap / reversal-freeze + R/tilt DR. |
| `test_backlash_robustness.py` | 12 adversarial cases + per-step invariants (NaN, bounds, consistency): R extremes, symmetric, deadband chatter, double reversal, multi-env reset, settle, tilt, etc. |
| `test_coupling_cases.py` | Earlier 7-case forward/uncurl PASS/FAIL (strict-gate era). |
| `diag_self_contact.py` | Widens contact sensor to all ff/mf/rf segments; reports which bodies collide & at what J2 angle; sweeps abduction. |

Offline logic replicas (scratchpad) confirmed the core scenarios **10/10** and
reproduced/verified the bounce fix.

---

## 7. Asset workflow change

`shadow_hand_lite.py` now loads a **prebuilt `SHADOW_TOUCHLAB.usd`** via `UsdFileCfg`
(instead of converting the URDF at runtime). Pushed alongside `SHADOW_TOUCHLAB.urdf`
and the finger/palm/thumb STL meshes.

---

## 8. Config flags (on `ShadowLiteEnvCfg`)

```python
couple_asymmetric_backward = True            # stateful backlash on (supersedes the gate)
couple_release_range_deg   = (100.0, 140.0)  # per-episode unlock R
couple_dir_deadband        = 0.002           # rad; below this, direction latches
hand_tilt_range_deg        = (0.0, 15.0)     # per-episode mounting tilt
# legacy strict gate (used when asymmetric is off):
couple_gate_j1_on_measured = True
couple_gate_lo_frac        = 1.0
couple_gate_j2_tol         = 0.035           # ~2°
settle_steps               = 15              # (baoding cfg)
```

---

## 9. Git

Commit **`762b121`** on local branch **`s2r`** (29 files: the 3 coupling `.py`,
`shadow_hand_lite.py`, `view_coupling_slider.py`, `SHADOW_TOUCHLAB.usd/.urdf`, 22 STLs).
Push it with:

```bash
git push origin s2r
```

(Auth wasn't available in the work environment; the commit is staged locally, one clean
fast-forward ahead of `origin/s2r`.)

---

## 10. Collision-mesh authoring (visual ↔ collision)

Separate from the coupling logic, a large effort went into giving the hand **faithful
collision geometry**. The stock URDF used hand-tuned primitive colliders
(boxes/cylinders/spheres) that only loosely approximate the detailed visual meshes, so the
goal was collision that actually matches what you see.

What we tried, and the dead-ends along the way:

- **Point each link's `<collision>` at its visual mesh.** The visuals ship as COLLADA
  (`.dae`) — a *scene* format — and Isaac's collision cooker silently produces **empty
  colliders for DAE**, so nothing showed except the links that already used primitives or an
  STL fingertip.
- **Convert the body meshes DAE → STL** (via a headless Blender pipeline, exporting in the
  raw DAE frame so the URDF `scale`/`origin` stay valid). This fixed the DAE problem but
  surfaced a **scale gotcha**: the meshes are authored in millimetres and referenced with
  `scale="0.001"`. Isaac renders that fine for *visuals*, but the **collision cook mishandles
  any non-1.0 scale**, so only the forearm (already metric, `scale=1.0`) cooked. We re-baked
  every mesh to **metric coordinates at `scale=1.0`**.
- **Simplify the colliders** (single convex hull / V-HACD-style decimation) to make the
  import lighter and avoid the per-link convex-decomposition step at cook time.

The recurring red herring was that **only the forearm collider ever appeared** — which looked
like a geometry bug but wasn't. The Isaac Sim **GUI URDF Importer + RTX viewport kept
crashing partway** through import on the 8 GB GPU (Hydra "error code 6"), so the first link in
the tree (forearm) survived and the rest dropped. A **headless conversion** (no RTX viewport)
proved the URDF and meshes were correct all along — it cooks **all 19 colliders**. The last
wrinkle was the converter's layered USD leaving the `/colliders` references unresolved, fixed
by **flattening to a single self-contained USD**.

**Where it landed:** we used the **same STL for both visual and collision** on every meshed
link (one mesh, identical `origin`/`scale`), and then applied **filtered collision pairing** —
disabling collisions between **adjacent / proximal links** (parent↔child along each finger).
Because the full-mesh colliders overlap at the joints *by design* (the visual meshes interlock
to look continuous), neighbouring links would otherwise self-collide and shove the fingers
apart; filtering those near-neighbour pairs keeps the faithful collision shapes while removing
the spurious push-apart, leaving real finger↔finger and finger↔object contact intact.

---

## 11. Open items / next

- **Retrain** after this — both the plant (backlash) and the episode (tilt + settle)
  changed.
- Confirm `write_root_pose_to_sim` actually re-tilts the fixed-base hand per env.
- Decide the **uncurl J2 profile**: keep "pinned until R" vs "open from the top"
  (open question raised in the viewer).
- Address finger–finger collision properly (knuckle spacing / collision geometry) if
  the soft-gain plant needs J2 to reach 100°.
