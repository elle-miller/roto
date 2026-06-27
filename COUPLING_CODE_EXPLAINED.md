# Coupling Code — Line-by-Line Explained (simple terms)

This explains the FF/MF/RF finger coupling (J1 driven by J2) and its domain
randomization, exactly as written in:

- `roto/tasks/roto_env.py` — the coupling logic
- `roto/tasks/robots/shadowlite/shadowlite.py` — config + per-episode reset
- `roto/tasks/baoding/baoding.py` — the ball-settle phase

---

## 0. The big picture (read this first)

**Hardware reality:** on the real Shadow Hand Lite, each finger's two last joints —
**J2** (middle joint, "PIP") and **J1** (fingertip joint, "DIP") — are pulled by **one
tendon driven by one motor** (`ffj0`, travels 0–180°). You cannot move J1 and J2
independently; the tendon ties them together.

**Sim problem:** in simulation J1 and J2 are *separate* joints. The policy outputs one
"curl" number per finger, and we must turn that one number into two joint targets (J2 and
J1) that behave like the real tendon.

**The frame we think in — the "combined motor" `m`:**
- `m` goes from **0° (finger open)** to **180° (finger fully curled)**.
- The first half belongs to J2: **m = 0→100° drives J2 = 0→100°**.
- The second half belongs to J1: **m = 100→180° drives J1 = 0→80°**.
- So J2 owns `[0°, 100°]` of `m`, and J1 owns `[100°, 180°]`. The split point is **100°**.

**The behavior we want (a "backlash" tendon):**
1. **Curling (m going up):** J2 fills up first to 100°, *then* J1 starts. (Tendon pulls the
   base joint before the tip.)
2. **Uncurling (m going down):** the tendon has slack, so J2 **unlocks early** at a random
   angle **R** (somewhere in 100–140°) and starts opening *while* J1 is still uncurling —
   both move toward 0 together. J1 reaches 0 at m=100°.
3. **Reversal:** if you uncurl partway (stop between 100° and R) and then curl again, J1
   **stays frozen** until the motor climbs back up to R, then continues. (Tendon slack must
   be taken up again before the tip moves.)

`R` is the **only randomized knob** of the coupling — drawn fresh each episode.

---

## 1. Setup in `__init__` (roto_env.py, ~line 144–180)

This runs **once** when the environment is built. It reads config and creates the memory
("state") buffers the coupling needs.

```python
self.coupled_dependent_indices = [... for d in cfg.coupled_joint_map.keys()]    # J1 joints: FFJ1, MFJ1, RFJ1
self.coupled_driver_indices    = [... for d in cfg.coupled_joint_map.values()]  # J2 joints: FFJ2, MFJ2, RFJ2
```
- `coupled_joint_map` is a dict `{FFJ1: FFJ2, MFJ1: MFJ2, RFJ1: RFJ2}`. These two lines
  find the **column numbers** of the J1 (dependent) and J2 (driver) joints in the robot's
  joint array, so later we can read/write them by index. "driver" = J2 (leads),
  "dependent" = J1 (follows).

```python
self.couple_asymmetric_backward = getattr(cfg, "couple_asymmetric_backward", False)
```
- The master on/off switch. When **True**, the new backlash model runs. When False, the
  old "measured-J2 gate" runs instead (legacy). Config sets this **True**.

```python
_rel_lo, _rel_hi = getattr(cfg, "couple_release_range_deg", (100.0, 140.0))
self.couple_release_lo = _rel_lo * math.pi / 180.0   # 100° -> radians
self.couple_release_hi = _rel_hi * math.pi / 180.0   # 140° -> radians
```
- The **range for the random unlock angle R**, converted from degrees to radians (the sim
  works in radians). So R will be sampled between 100° and 140°.

```python
self.couple_dir_deadband = getattr(cfg, "couple_dir_deadband", 0.002)  # rad
```
- A tiny threshold (~0.1°). If the combined motor `m` changes by less than this between
  steps, we treat the finger as "not moving" and **keep the previous direction**. Prevents
  jitter from flipping curl/uncurl back and forth.

```python
self._couple_j2_top  = self.robot_joint_pos_upper_limits[self.coupled_driver_indices].clone()    # ~100° (1.745 rad)
self._couple_j1_span = self.robot_joint_pos_upper_limits[self.coupled_dependent_indices].clone()  # ~80°  (1.396 rad)
self._couple_m_top   = self._couple_j2_top + self._couple_j1_span                                 # ~180° (3.14 rad)
```
- Reads the joints' upper limits from the robot model. `j2_top` = how far J2 can go (100°),
  `j1_span` = how far J1 can go (80°), and `m_top` = the two added = 180° (the top of the
  combined motor). All are 3-element vectors (one per finger).

```python
self.couple_release     = torch.full((N, 3), self.couple_release_hi, ...)  # R per finger, latched per episode
self.prev_m             = torch.zeros((N, 3), ...)                          # last step's m (to detect direction)
self.couple_dir         = torch.ones((N, 3), ...)                           # +1 = curling, -1 = uncurling
self.j1_state           = torch.zeros((N, 3), ...)                          # J1's current angle (carried over)
self.couple_frozen_flag = torch.zeros((N, 3), dtype=bool, ...)             # is J1 frozen (mid-reversal)?
self.couple_frozen_val  = torch.zeros((N, 3), ...)                          # the angle J1 froze at
```
- These are the **memory** of the coupling — `(N envs × 3 fingers)` each. Because the
  backlash depends on history (direction, where J1 was, whether it's frozen), we must
  remember these between steps. This is what makes the coupling **stateful**.

---

## 2. Per-step entry: `_pre_physics_step` (line 261)

Called every control step with the policy's `actions`.

```python
self.joint_pos_cmd[:, self.control_dof_indices] = scale(self.actions, lower, upper)
```
- Converts each action from the policy's `[-1, +1]` range into an actual joint-angle target
  in `[lower, upper]`. For the J2 driver joints this becomes the **"curl proxy"** — the raw
  curl request before coupling.

```python
sc = getattr(self, "settle_counter", None)
settling = (sc > 0) if sc is not None else None
snap = None
if settling is not None and self.couple_asymmetric_backward and settling.any():
    snap = (self.prev_m.clone(), self.couple_dir.clone(), self.j1_state.clone(),
            self.couple_frozen_flag.clone(), self.couple_frozen_val.clone())
```
- During the **settle phase** (balls dropping into the palm at episode start), the hand is
  held still and the policy's action is ignored. But the coupling math below still runs, so
  it would corrupt the state. So we **take a snapshot** (a copy) of all 5 coupling buffers
  *before* running the coupling, to restore them afterward. `snap` stays `None` outside the
  settle phase.

```python
self._handle_coupled_joints()
```
- Runs the actual coupling (next section). This writes the J2 and J1 targets and advances
  the state buffers.

```python
if settling is not None and settling.any():
    self.joint_pos_cmd[settling] = self.robot.data.default_joint_pos[settling]  # hold default pose
    if snap is not None:
        self.prev_m[settling] = snap[0][settling]    # ... restore all 5 buffers
        ...
    self.settle_counter[settling] -= 1               # count down the settle window
```
- For envs still settling: overwrite the joint targets with the **default catch pose**
  (ignore the policy), **restore** the coupling state from the snapshot (so it stays
  frozen-open), and **decrement** the settle counter. When it reaches 0, the policy takes
  over with a clean coupling state.

---

## 3. The forward split: `_handle_coupled_joints` (line 301)

This turns the one curl proxy into a *first guess* of J2 and J1 (the simple, non-backlash
split), then hands off to the backlash model.

```python
proxy   = self.joint_pos_cmd[:, self.coupled_driver_indices]   # (N,3) the curl request
j2_upper = self.robot_joint_pos_upper_limits[self.coupled_driver_indices]      # ~100°
j1_upper = self.robot_joint_pos_upper_limits[self.coupled_dependent_indices]   # ~80°
theta = self.coupling_theta                                    # the split point in proxy-space
```
- Reads the curl proxy and the joint limits. `theta` is where, in proxy units, J2 finishes
  and J1 begins.

```python
j2_cmd = torch.clamp(proxy * (j2_upper / theta), 0, j2_upper)
```
- **J2 first guess:** for small proxy (below theta), J2 ramps up proportionally; above
  theta it's clamped at its max (100°). So J2 fills first.

```python
j1_cmd = torch.clamp((proxy - theta) / (j2_upper - theta) * j1_upper, 0, j1_upper)
```
- **J1 first guess:** stays 0 until proxy passes theta, then ramps 0→80°. So J1 only starts
  after J2 is done. (This is the simple "sequential" split.)

```python
if self.couple_asymmetric_backward:
    j2_cmd, j1_cmd = self._asymmetric_backlash(j2_cmd, j1_cmd, j2_upper, j1_upper)
elif self.couple_gate_j1_on_measured:
    ...   # legacy gate, only runs if backlash is OFF
```
- Since backlash is **on**, we pass the first-guess J2/J1 into `_asymmetric_backlash`, which
  re-computes them properly with history/direction. The `elif` is the old method, skipped.

```python
self.joint_pos_cmd[:, self.coupled_driver_indices]    = j2_cmd   # write final J2 target
self.joint_pos_cmd[:, self.coupled_dependent_indices] = j1_cmd   # write final J1 target
```
- Writes the final J2 and J1 angle targets into the command array that the simulator
  applies to the motors.

---

## 4. The heart: `_asymmetric_backlash` (line 356)

This is where the real tendon behavior lives. Inputs are the first-guess `j2_fwd`,
`j1_fwd`. Output is the corrected `(j2, j1)`. Everything is `(N envs × 3 fingers)`.

```python
j2_top, j1_span, m_top = self._couple_j2_top, self._couple_j1_span, self._couple_m_top  # 100°, 80°, 180°
R   = self.couple_release      # the random unlock angle, per finger, fixed this episode
db  = self.couple_dir_deadband # the "is it moving?" threshold
eps = 1e-4                      # tiny number to avoid divide-by-zero / exact-equality issues
```

```python
m = j2_fwd + j1_fwd            # rebuild the combined motor position (0..180°) from the guess
```
- **Key step:** adding the first-guess J2 and J1 reconstructs the single combined motor
  value `m`. From here on we reason purely in `m`.

### 4a. Which way is the finger going?

```python
delta   = m - self.prev_m      # how much m changed since last step
rising  = delta >  db          # curling (m increased meaningfully)
falling = delta < -db          # uncurling (m decreased meaningfully)
new_dir = where(rising, +1, where(falling, -1, self.couple_dir))
```
- Compares `m` to last step. If it grew → **curling (+1)**; shrank → **uncurling (−1)**; if
  it barely moved (within deadband) → **keep the old direction** (`self.couple_dir`).
  Keeping the old direction on "steady" frames is critical (see the bug note in 4e).

### 4b. The "demand" curve for J1

```python
l = torch.clamp(m - j2_top, 0, j1_span)     # = clamp(m - 100°, 0, 80°)
```
- `l` is **where J1 "wants" to be** for the current `m`, ignoring backlash: 0 below m=100°,
  then rising 1-for-1 up to 80° at m=180°. Used as J1's target when *not* frozen.

### 4c. Decide if J1 should freeze (reversal handling)

```python
frozen = self.couple_frozen_flag
fval   = self.couple_frozen_val
flip_up = (self.couple_dir < 0) & (new_dir > 0)                 # just switched uncurl -> curl
enter   = flip_up & (self.j1_state < j1_span - eps) & (m < R)   # ...mid-curl (J1 not full) and below R
frozen  = frozen | enter                                        # turn freeze ON if entering
fval    = torch.where(enter, self.j1_state, fval)               # remember the angle J1 froze at
```
- **`flip_up`**: true on the exact step the finger reverses from uncurling to curling.
- **`enter`**: we only freeze if that reversal happens while J1 is partway curled
  (`j1_state < 80°`) and the motor is still below the unlock point (`m < R`). That's the
  "stopped inside the slack zone and started closing again" case.
- If `enter`, we set the freeze flag and **record where J1 is** (`fval`) — it will hold
  there.

```python
uncurling = new_dir < 0
frozen  = frozen & ~uncurling     # any uncurl cancels the freeze
```
- If the finger is uncurling, J1 is **not** frozen (it should track back down). Note this
  uses the **latched** `new_dir`, not the instantaneous `falling`, so a "steady" frame
  (slider held still) keeps the freeze state correct instead of flickering.

### 4d. Compute J1

```python
denom  = torch.clamp(m_top - R, min=eps)                # width of the resume ramp (180° - R)
resume = fval + (m - R) / denom * (j1_span - fval)      # ramp from frozen value up to 80°
resume = torch.clamp(resume, fval, j1_span)             # never below the frozen value, never above 80°
j1_frozen_branch = torch.where(m >= R, resume, fval)    # below R: hold at fval; at/above R: resume
```
- This is the **reversal resume**. While frozen: if the motor is still below R, J1 stays
  pinned at `fval` (frozen). Once the motor climbs back to R and beyond, J1 smoothly
  continues from `fval` up toward 80° at m=180°. `denom` is clamped so we never divide by
  zero (e.g., if R were 180°).

```python
j1 = torch.where(frozen, j1_frozen_branch, l)           # frozen -> hold/resume; else -> follow demand
frozen = frozen & ~(j1 >= l - eps)                      # once J1 catches the demand, unfreeze
```
- J1 is either the frozen/resume value (if frozen) or the plain demand `l` (if free). The
  last line **clears the freeze** once the resuming J1 has caught back up to the demand
  curve — after that it's just normal again.

### 4e. Compute J2

```python
j2_down  = torch.clamp(m / R * j2_top, 0, j2_top)   # uncurl curve: J2 = 100° at m=R, lower below R
j2_fresh = torch.clamp(m, 0, j2_top)                # curl curve: J2 = m up to 100°, then capped
j2 = torch.where(uncurling | frozen, j2_down, j2_fresh)
```
- **`j2_fresh`** (used while curling): J2 simply follows `m` up to 100° and saturates. This
  is the "fill J2 first" behavior.
- **`j2_down`** (used while uncurling or frozen): J2 = 100° exactly at m=R, and **drops
  below 100° as m falls below R** — i.e. J2 **unlocks early at R**. During a frozen re-curl
  it also climbs back to 100° as m returns to R.
- **Why `uncurling | frozen` and not `falling`:** the slider only sends events when you
  move it, so most frames are "steady" (`delta≈0`). If we keyed on the instantaneous
  `falling`, those steady frames would fall back to `j2_fresh` and snap J2 back to 100° —
  the **"J2 bounces back to the limit and won't uncurl"** bug. Using the **latched**
  `uncurling` keeps J2 on the unlock curve through steady frames. (Fixed.)

### 4f. Save state for next step

```python
self.couple_dir         = new_dir
self.prev_m             = m
self.j1_state           = j1
self.couple_frozen_flag = frozen
self.couple_frozen_val  = fval
return j2, j1
```
- Stores everything so the next step can detect direction, resume freezes, etc. Returns the
  corrected J2/J1 targets.

---

## 5. Per-episode reset: `_sample_coupling_params` (line 430)

Called on every reset (from `ShadowLiteEnv._reset_idx`) for the envs being reset.

```python
self.couple_release[env_ids] = sample_uniform(self.couple_release_lo, self.couple_release_hi, (n, k), device)
```
- **The domain randomization of the coupling:** draw a fresh **unlock angle R** for each
  finger of each resetting env, uniformly between 100° and 140°. Different envs get
  different tendon "slack" — that's the variety the policy trains against.

```python
self.prev_m[env_ids]             = 0.0     # start at open hand
self.couple_dir[env_ids]         = 1.0     # assume curling to begin
self.j1_state[env_ids]           = 0.0     # J1 starts at 0
self.couple_frozen_flag[env_ids] = False   # not frozen
self.couple_frozen_val[env_ids]  = 0.0
```
- Clears all the history so each episode starts from a clean, open hand.

---

## 6. Domain randomization summary

There are two DR knobs that were built; **only the unlock angle is on**:

| Knob | Where | Status | Effect |
|---|---|---|---|
| **Unlock angle R** | `couple_release_range_deg = (100°, 140°)`, sampled in `_sample_coupling_params` | **ON** | Each episode/finger gets different tendon slack: how early J2 unlocks on release and how big the reversal "freeze zone" is. |
| **Hand mounting tilt** | `hand_tilt_range_deg`, applied in `_randomize_hand_tilt` | **OFF** (set to `(15°, 15°)`) | Would randomize the forward tilt of the whole hand per episode. Disabled — hand stays fixed at 15°. |

### Hand-tilt code (`shadowlite.py`, `_randomize_hand_tilt`)
```python
lo, hi = self.cfg.hand_tilt_range_deg
if lo == hi:
    return                       # no DR: keep the fixed init tilt, don't touch the root pose
```
- Because the range is `(15, 15)`, `lo == hi`, so the function **returns immediately** and
  the hand simply keeps the fixed 15° tilt from its initial pose. (If you set e.g.
  `(0, 15)`, it would instead interpolate between the upright and 15° orientations each
  episode and write the new root pose.)

---

## 7. The settle phase (`baoding.py`)

Not the coupling itself, but it interacts with it.

```python
settle_steps: int = 15                                  # config: hold for ~15 steps
self.settle_counter[env_ids] = getattr(self.cfg, "settle_steps", 0)   # set on reset
```
- After a reset, the balls drop from above. For the first 15 steps the hand is held in its
  catch pose (policy ignored) so the balls settle into the palm. During this window the
  coupling state is **frozen via the snapshot/restore** in `_pre_physics_step` (section 2),
  so the policy's pre-settle curl requests don't corrupt the tendon state.

```python
physics_termination = physics_termination & (settling == 0)
```
- While settling, episode-ending conditions are suppressed so a not-yet-landed ball can't
  trigger an instant reset.

---

## 8. Worked example (R = 136°)

Driving the combined motor `m` and reading out (J2, J1):

| Phase | m (°) | J2 (°) | J1 (°) | what's happening |
|---|---|---|---|---|
| Curl | 50 | 50 | 0 | J2 filling, J1 waits |
| Curl | 100 | 100 | 0 | J2 full, J1 about to start |
| Curl | 140 | 100 | 40 | J1 now rising |
| Curl | 180 | 100 | 80 | fully curled |
| Uncurl | 160 | 100 | 60 | J2 still 100 (above R), J1 retracting |
| Uncurl | 136 (=R) | 100 | 36 | J2 at the unlock point |
| Uncurl | 120 | 88 | 20 | **both** dropping (overlap zone) |
| Uncurl | 100 | 74 | 0 | J1 hits 0, J2 still opening |
| **Reverse → curl** from 120 | 120→136 | 88→100 | **20 (frozen)** | J1 holds until motor reaches R |
| Curl | 140 | 100 | 25 | past R → J1 resumes from 20 |
| Curl | 180 | 100 | 80 | back to full |

This matches: J2 leads on curl; J2 unlocks early at R on uncurl (overlap with J1); and on a
reversal J1 freezes until the motor returns to R.

---

## 9. One-line mental model

> The policy gives one curl number → we turn it into a combined motor angle `m` (0–180°) →
> J2 owns 0–100°, J1 owns 100–180° → on the way **up** J2 fills first then J1; on the way
> **down** J2 unlocks early at a random angle **R** so J2 and J1 open together; if you
> **reverse** inside the slack zone, J1 freezes until `m` climbs back to **R**. `R` is
> re-rolled every episode (the coupling's domain randomization); the hand tilt DR is off.
