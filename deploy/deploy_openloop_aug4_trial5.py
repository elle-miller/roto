#!/usr/bin/env python
"""Open-loop hardware replay of a recorded policy rollout, with the full
warmup / tactile-calibration flow of ``deploy_warmup_trial15_zerotac.py``.

This is that script with Phase D swapped: instead of running the policy
closed-loop, it publishes a pre-recorded 60 s joint trajectory. Everything
around it -- the empty-motion warmup, the p99.5 threshold fit, the hysteresis
state machines, the keyboard drop/place marks, the npz schema -- is the same
code, so an open-loop run and a closed-loop run are directly comparable.

No torch, no checkpoint, no observation stack: the policy already ran, in sim,
and its output is baked into REPLAY_FILE. That is the point of the comparison --
whatever the hand does here is what the trajectory alone produces, with no
feedback correcting it. Tactile is still read, calibrated and logged every
step (it just does not feed anything), so you can put hardware taxels next to
the sim taxels recorded in the same npz under the identical motion.

Flow, as shipped (WARMUP_REPEATS = 0):
  0) Move to DEFAULT_POSE (the sim reset / cup pose) over DEFAULT_POSE_MOVE_S,
     so the run always starts from the same configuration rather than from
     wherever the last script left the hand. Only prompted motion in the run.
  C) Prompt to place balls, move to trajectory start pose q[0].
  D) Open-loop replay REPLAY_REPEATS times, logging q / cmd / raw FSR (12) /
     BioTac PDC (5) / tac every step -> REPLAY_NPZ.
     Keyboard (non-blocking, no Enter):  d = balls dropped,  p = balls placed.
     Ctrl-C ends early and still saves.

Set WARMUP_REPEATS >= 1 to prepend the calibration phases (hand EMPTY):
  A) Replay the first WARMUP_SECONDS of the trajectory WARMUP_REPEATS times.
  B) Fit per-channel envelope thresholds from that empty motion:
        hi = p99.5(empty) + MARGIN_ABS
        lo = median + MARGIN_LO  (narrow pads)
        lo = hi - HYST_BAND_WIDE (wide empty envelopes)
     Hysteresis: ON if raw > hi, OFF if raw < lo.  Save WARMUP_NPZ.

WHAT IS RECORDED, EITHER WAY: raw FSR ADC (12 ch) and BioTac PDC (5 ch) are
logged every step regardless of warmup — that is the actual tactile record and
nothing is lost by skipping calibration. What the warmup adds is the *binary*
tac vector: without fitted thresholds it reads all-zero (deliberately, see
Phase A in main) and is not contact data. Threshold the raw columns offline
from the same npz if you want binary contact after a no-warmup run.

Generating REPLAY_FILE (on the sim box, not the control laptop):
    python scripts/record_openloop.py \
        --checkpoint best_agent_aug4_slew_fsr_no_noise_trial5.pt \
        --agent_cfg rl_only_pt_padtac_bt \
        --record_steps 3600 --seed 42 --headless \
        --out openloop_aug4_trial5_60s_seed42.npz

record_openloop.py runs ONE reset-free rollout, so the file holds a continuous
trajectory -- unlike play.py, which resets every 600 steps and would splice a
teleport into anything longer than 10 s.

SAFETY -- read before the first run on a new hand:
  * Every published target is clipped to PUB_LOWER/PUB_UPPER (J2/J1 capped at
    90 deg) exactly as in the closed-loop script.
  * The recorded trajectory reaches the sim joint-velocity limits. Shipped at
    REPLAY_SPEED_FRAC = None (each frame published straight through). On an
    unfamiliar hand, set it to 0.3 for one watched pass first — but note that
    governor does not slow playback down, it makes the hand lag and low-pass
    the motion, so it is a safety check only and its data is not usable.
  * The hand is commanded blind for 60 s per repeat. Keep a hand on the e-stop:
    if a ball jams, nothing in this loop will notice or back off.

Does NOT modify deploy_warmup_trial15_zerotac.py.

Laptop files needed next to this script (or absolute paths below):
  openloop_aug4_trial5_60s_seed42.npz
  fsr_pad_map.py
"""

from __future__ import print_function

import sys
import select
import rospy
import numpy as np

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sr_robot_msgs.msg import BiotacAll

import serial
import threading

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None

from fsr_pad_map import FSR_CHANNELS, FSR_NAMES

# Ball in-play markers (keyboard during policy loop)
EVENT_DROP = 0
EVENT_PLACE = 1

# ============================================================
# PATHS / PROTOCOL
# ============================================================
# Recorded open-loop trajectory (from scripts/record_openloop.py). Must carry
# q (T,16) + joints. Its sim tactile (tac) and commands (cmd) are copied into
# the output npz so hardware and sim can be compared under the same motion.
REPLAY_FILE = "openloop_aug4_trial5_60s_seed42.npz"

# --- Empty-hand tactile warmup (OFF by default) ---------------------------
# 0 = skip Phases A+B entirely and go straight to the replay.
#
# Nothing consumes tactile in open loop, so the calibration buys only a
# nicer logged binary vector -- raw FSR + BioTac ADC are logged either way and
# can be thresholded offline from the same npz. Not worth minutes of empty-hand
# motion up front. Set to >=1 only when you specifically want the binary tac
# stream live during the run, and then set WARMUP_SECONDS too: a full pass is
# the whole 60 s trajectory, and the pads see every pose they will ever see
# within the first couple of Baoding cycles (~1.1 s each).
WARMUP_REPEATS = 0
# Use only the first N seconds of the trajectory per warmup pass (None = all of
# it). ~5 s covers several full cycles, which is the entire pose envelope.
WARMUP_SECONDS = 5.0

REPLAY_REPEATS = 1   # with-balls open-loop passes (1 x 60 s trajectory = 60 s)

WARMUP_NPZ = "hw_warmup_openloop_aug4_trial5_fsr.npz"
REPLAY_NPZ = "hw_openloop_aug4_trial5.npz"

CONTROL_HZ = 60

# --- Replay rate governor -------------------------------------------------
# The recorded trajectory already contains the sim command slew it was recorded
# under (cmd_speed_frac in the npz; 0.6 for the shipped file), so this is a
# SECOND, hardware-side limiter on top of that -- not the same knob.
#   None : publish each recorded frame as-is (full recorded speed).
#   0.3  : conservative bring-up on an unfamiliar hand. START HERE.
#   1.0  : rated joint-velocity cap only.
# This does NOT time-scale the trajectory. One frame is still published per
# 1/60 s tick either way, so a pass always takes the recorded 60 s; the governor
# only caps how far the target may move per tick. Below ~1.0 the target simply
# falls behind and never catches up, which low-passes the motion — smaller,
# sluggish, wrong ball timing. It is a "does this explode" check, not a run.
# Set None for any pass whose data you intend to use.
REPLAY_SPEED_FRAC = None

# Same governor for the empty-hand warmup passes (None = publish as-is).
# Warmup only ever runs with an empty hand, so this is the safer default.
WARMUP_SPEED_FRAC = None

# Seconds held at q[0] before each replay pass begins.
SETTLE_S = 2.0

# --- Phase 0: default (cup) pose ------------------------------------------
# Move to a known pose before anything else, so the run always starts from the
# same hand configuration no matter where the previous script left it. Without
# this the first command is a jump from an arbitrary pose straight to q[0].
GO_TO_DEFAULT_POSE = True
DEFAULT_POSE_MOVE_S = 4.0   # slow; raise if the hand still snaps into it

# ShadowLiteEnvCfg init_state — the sim reset/cup pose, in PUBLISH_JOINTS order.
# Same values as go_to_pose.py (verified there against the reset steps of
# sim_policy_log_seed42.npz, where joint_pos_cmd == default_joint_pos).
# FFJ1/MFJ1/RFJ1 and THJ1 are 0 in sim (distals straight) and the deploy
# coupling keeps them near 0, so starting there avoids a jump into the replay.
DEFAULT_POSE = np.array([
    -0.349,  # FFJ4
     0.0,    # MFJ4
    -0.349,  # RFJ4
     0.4,    # THJ5
     0.65,   # FFJ3
     0.65,   # MFJ3
     0.65,   # RFJ3
     0.5,    # THJ4
     0.87,   # FFJ2
     0.87,   # MFJ2
     0.87,   # RFJ2
     0.0,    # FFJ1
     0.0,    # MFJ1
     0.0,    # RFJ1
     0.35,   # THJ2
     0.0,    # THJ1
], dtype=np.float32)


# --- Temporal tactile hold (logging only here; mirrors sim tactile smoothing) ---
# Nothing consumes tactile in open loop, but keeping the same debounce as the
# closed-loop script means the logged tac vector is the SAME signal the policy
# would have seen, which is the whole point of the comparison.
USE_TACTILE_HOLD = False
TACTILE_K_ON = 3   # consecutive ON steps before the channel reads 1 (~50 ms @ 60 Hz)
TACTILE_K_OFF = 1  # consecutive OFF steps before release

# No policy is running, so there is nothing to ablate: read_tactile() keeps the
# flag only so the shared sensor code stays identical to the closed-loop script.
ZERO_TACTILE = False

# Empty-motion envelope thresholds (not K*std — std blows up on spike pads):
#   hi = percentile(warmup, P_HI) + MARGIN_ABS
#   lo = percentile(warmup, P_LO) + MARGIN_LO   (narrow pads)
#   lo = hi - HYST_BAND_WIDE                    (wide empty envelopes)
P_HI = 99.5
P_LO = 50.0
MARGIN_ABS = 5.0   # ADC counts above empty p99.5 -> ON
MARGIN_LO = 2.0    # ADC counts above empty median -> release (narrow pads)
WIDE_ENVELOPE_ADC = 15.0
HYST_BAND_WIDE = 3.0
STD_FLOOR_BT = 1.0  # BioTac still uses a small noise floor for PDC

# ============================================================
# DIMENSIONS (no model here — kept so the shared sensor code matches)
# ============================================================
NUM_J = 13
NUM_TACTILE = 24


# ============================================================
# JOINTS / COUPLING (same as deploy_policy_new)
# ============================================================
POLICY_JOINTS = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",
    "rh_THJ2", "rh_THJ1",
]
JOINT_LOWER = np.array(
    [-0.3491, -0.3491, -0.3491, -1.0472,
     -0.2618, -0.2618, -0.2618, 0.0,
     0.0, 0.0, 0.0, -0.6981, -0.2618],
    dtype=np.float32,
)
JOINT_UPPER = np.array(
    [0.3491, 0.3491, 0.3491, 1.0472,
     1.5708, 1.5708, 1.5708, 1.2217,
     1.7450, 1.7450, 1.7450, 0.6981, 1.5708],
    dtype=np.float32,
)
JOINT_VEL_LIMIT = np.array(
    [2.0, 2.0, 2.0, 4.0,
     2.0, 2.0, 2.0, 4.0,
     2.0, 2.0, 2.0, 2.0, 4.0],
    dtype=np.float32,
)

COUPLING_THETA = 0.785
J2_UPPER_SIM = 1.7450
J1_UPPER_SIM = 1.3960
GATE_J2_TOL = 0.035
CURL_J2_IDX = [8, 9, 10]

PUBLISH_JOINTS = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",
    "rh_FFJ1", "rh_MFJ1", "rh_RFJ1",
    "rh_THJ2", "rh_THJ1",
]
CTRL_NONCOUPLED = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]
PUB_NONCOUPLED = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15]
PUB_J2_SLOTS = [8, 9, 10]
PUB_J1_SLOTS = [11, 12, 13]

PUB_LOWER = np.array(
    [-0.3491, -0.3491, -0.3491, -1.0472,
     -0.2618, -0.2618, -0.2618, 0.0,
     0.0, 0.0, 0.0,
     0.0, 0.0, 0.0,
     -0.6981, -0.2618],
    dtype=np.float32,
)
PUB_UPPER = np.array(
    [0.3491, 0.3491, 0.3491, 1.0472,
     1.5708, 1.5708, 1.5708, 1.2217,
     1.5708, 1.5708, 1.5708,
     1.5708, 1.5708, 1.5708,
     0.6981, 1.5708],
    dtype=np.float32,
)
PUB_VEL_LIMIT = np.array(
    [2.0, 2.0, 2.0, 4.0,
     2.0, 2.0, 2.0, 4.0,
     2.0, 2.0, 2.0,
     2.0, 2.0, 2.0,
     2.0, 4.0],
    dtype=np.float32,
)

# Mux -> sim ch map: fsr_pad_map.py (wire-checked; matches shadow_padtac.usd).
# Force these mux indices silent in the policy tactile vector (always 0).
N_FSR = 12
FSR_MUTE_MUX = []
BIOTAC_IDX = [0, 1, 2, 4]
BIOTAC_CH = [15, 16, 17, 22]
BIOTAC_NAMES = ["BT_ffdist", "BT_mfdist", "BT_rfdist", "BT_thdist"]
N_BIOTAC = len(BIOTAC_IDX)
USE_BIOTAC = True

SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200


class KeyboardBallEvents(object):
    """Non-blocking single-key reader for ball drop/place marks (cbreak mode).

    Keys (no Enter): d=drop, p=place. Restores terminal settings on close().
    If stdin is not a TTY (or termios missing), poll() is a no-op.
    """

    def __init__(self):
        self._fd = None
        self._old = None
        if termios is None or tty is None:
            return
        try:
            if not sys.stdin.isatty():
                return
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            self._fd = None
            self._old = None

    def poll(self):
        """Return list of lowercased single-char keys available this call."""
        keys = []
        if self._fd is None:
            return keys
        try:
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.0)
                if not ready:
                    break
                ch = sys.stdin.read(1)
                if not ch:
                    break
                keys.append(ch.lower())
        except Exception:
            pass
        return keys

    def close(self):
        if self._fd is not None and self._old is not None and termios is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass
        self._fd = None
        self._old = None


# ============================================================
# GLOBAL STATE
# ============================================================
current_joint_pos = np.zeros(NUM_J, dtype=np.float32)
current_joint_vel = np.zeros(NUM_J, dtype=np.float32)
current_joint_pos16 = np.zeros(16, dtype=np.float32)
current_joint_vel16 = np.zeros(16, dtype=np.float32)
joint_ready = False
biotac_ready = False


latest_fsr = np.zeros(N_FSR, dtype=np.float32)
fsr_lock = threading.Lock()
latest_biotac_pdc = np.full(5, np.nan, dtype=np.float32)
bt_lock = threading.Lock()

fsr_baseline = np.zeros(N_FSR, dtype=np.float32)
fsr_noise = np.ones(N_FSR, dtype=np.float32)
fsr_hi = np.zeros(N_FSR, dtype=np.float32)
fsr_lo = np.zeros(N_FSR, dtype=np.float32)
bt_baseline = np.zeros(N_BIOTAC, dtype=np.float32)
bt_noise = np.ones(N_BIOTAC, dtype=np.float32)
bt_hi = np.zeros(N_BIOTAC, dtype=np.float32)
bt_lo = np.zeros(N_BIOTAC, dtype=np.float32)

fsr_state = np.zeros(N_FSR, dtype=bool)
bt_state = np.zeros(N_BIOTAC, dtype=bool)

_hold_ct_max = max(int(TACTILE_K_ON), int(TACTILE_K_OFF))
hold_on_ct = np.zeros(NUM_TACTILE, dtype=np.int16)
hold_off_ct = np.zeros(NUM_TACTILE, dtype=np.int16)
hold_state = np.zeros(NUM_TACTILE, dtype=np.float32)

# Last REAL (pre-zeroing) tactile vector produced by read_tactile(). Always the
# true post-hysteresis / post-hold signal, even when ZERO_TACTILE feeds the
# policy zeros — logged as tac_real so contact is still visible in the npz.
last_tac_real = np.zeros(NUM_TACTILE, dtype=np.float32)



def joint_callback(msg):
    global joint_ready, current_joint_pos, current_joint_vel
    global current_joint_pos16, current_joint_vel16
    idx = {n: i for i, n in enumerate(msg.name)}
    try:
        for i, j in enumerate(POLICY_JOINTS):
            current_joint_pos[i] = msg.position[idx[j]]
            current_joint_vel[i] = msg.velocity[idx[j]]
        for i, j in enumerate(PUBLISH_JOINTS):
            current_joint_pos16[i] = msg.position[idx[j]]
            current_joint_vel16[i] = msg.velocity[idx[j]]
        joint_ready = True
    except KeyError as e:
        rospy.logwarn_throttle(5.0, "Missing joint in /joint_states: %s" % e)


def serial_reader():
    global latest_fsr
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1.0)
    ser.reset_input_buffer()
    while not rospy.is_shutdown():
        line = ser.readline().decode(errors="ignore").strip()
        try:
            vals = np.array([float(x) for x in line.split(",")], dtype=np.float32)
        except ValueError:
            continue
        if vals.shape[0] == N_FSR:
            with fsr_lock:
                latest_fsr = vals


def biotac_cb(msg):
    global latest_biotac_pdc, biotac_ready
    with bt_lock:
        for i, t in enumerate(msg.tactiles):
            latest_biotac_pdc[i] = float(t.pdc)
    biotac_ready = True


def publish_target(pub, target, duration):
    msg = JointTrajectory()
    msg.joint_names = PUBLISH_JOINTS
    pt = JointTrajectoryPoint()
    pt.positions = target.tolist()
    pt.time_from_start = rospy.Duration(duration)
    msg.points.append(pt)
    pub.publish(msg)


def _lo_from_envelope(baseline, phi, hi):
    """Release threshold: tight band under hi for wide empty envelopes."""
    envelope = (phi - baseline).astype(np.float32)
    lo_narrow = (baseline + MARGIN_LO).astype(np.float32)
    lo_wide = (hi - HYST_BAND_WIDE).astype(np.float32)
    wide = envelope > WIDE_ENVELOPE_ADC
    lo = np.where(wide, lo_wide, lo_narrow).astype(np.float32)
    lo = np.minimum(lo, hi - 1.0).astype(np.float32)
    return lo, wide, envelope


def fit_thresholds_from_warmup(fsr_arr, bt_arr):
    """fsr_arr (N,12), bt_arr (N,5) with NaNs possible on BT.

    FSR: hi = p99.5(empty) + MARGIN_ABS.
    lo = median + MARGIN_LO on narrow pads; hi - HYST_BAND_WIDE on wide pads.
    BioTac PDC: same idea on valid samples.
    """
    global fsr_baseline, fsr_noise, fsr_hi, fsr_lo
    global bt_baseline, bt_noise, bt_hi, bt_lo

    fsr_baseline = np.percentile(fsr_arr, P_LO, axis=0).astype(np.float32)
    fsr_phi = np.percentile(fsr_arr, P_HI, axis=0).astype(np.float32)
    fsr_noise = np.maximum(fsr_arr.std(axis=0), 1e-6).astype(np.float32)  # logged only
    fsr_hi = (fsr_phi + MARGIN_ABS).astype(np.float32)
    fsr_lo, fsr_wide, fsr_env = _lo_from_envelope(fsr_baseline, fsr_phi, fsr_hi)

    bt_sel = bt_arr[:, BIOTAC_IDX]
    bt_baseline = np.zeros(N_BIOTAC, dtype=np.float32)
    bt_noise = np.ones(N_BIOTAC, dtype=np.float32)
    bt_hi = np.zeros(N_BIOTAC, dtype=np.float32)
    bt_lo = np.zeros(N_BIOTAC, dtype=np.float32)
    for k in range(N_BIOTAC):
        col = bt_sel[:, k]
        valid = col[np.isfinite(col) & (col > 0)]
        if valid.size < 10:
            rospy.logwarn("BioTac tip idx %d: few valid warmup samples (%d)", BIOTAC_IDX[k], valid.size)
            bt_baseline[k] = 0.0
            bt_noise[k] = STD_FLOOR_BT
            bt_hi[k] = MARGIN_ABS
            bt_lo[k] = 0.0
        else:
            bt_baseline[k] = float(np.percentile(valid, P_LO))
            bt_noise[k] = max(float(valid.std()), STD_FLOOR_BT)
            phi = float(np.percentile(valid, P_HI))
            bt_hi[k] = float(phi + MARGIN_ABS)
            lo_arr, _, _ = _lo_from_envelope(
                np.array([bt_baseline[k]], dtype=np.float32),
                np.array([phi], dtype=np.float32),
                np.array([bt_hi[k]], dtype=np.float32),
            )
            bt_lo[k] = float(lo_arr[0])

    rospy.loginfo(
        "FSR thresh mode=p%.1f+%.1f / lo=med+%.1f or hi-%.1f if env>%.1f",
        P_HI, MARGIN_ABS, MARGIN_LO, HYST_BAND_WIDE, WIDE_ENVELOPE_ADC,
    )
    rospy.loginfo("FSR baseline(med)=%s", fsr_baseline)
    rospy.loginfo("FSR empty_envelope(p99.5-med)=%s", fsr_env)
    rospy.loginfo("FSR wide_pads=%s (%s)", np.where(fsr_wide)[0].tolist(),
                  [FSR_NAMES[i] for i in np.where(fsr_wide)[0]])
    rospy.loginfo("FSR hi=%s", fsr_hi)
    rospy.loginfo("FSR lo=%s", fsr_lo)
    rospy.loginfo("FSR_MUTE_MUX=%s (%s)", FSR_MUTE_MUX,
                  [FSR_NAMES[i] for i in FSR_MUTE_MUX])
    rospy.loginfo("BioTac baseline=%s hi=%s", bt_baseline, bt_hi)


def apply_fsr_mute(fsr_on):
    """Zero muted mux channels (e.g. broken C8/mfprox)."""
    out = fsr_on.astype(np.float32).copy()
    for mux_i in FSR_MUTE_MUX:
        out[mux_i] = 0.0
    return out


def reset_tactile_hold():
    """Clear hold latch/counters (call when starting a new policy segment)."""
    global hold_on_ct, hold_off_ct, hold_state
    hold_on_ct[:] = 0
    hold_off_ct[:] = 0
    hold_state[:] = 0.0


def apply_tactile_hold(t):
    """Debounce the 24-d binary tactile vector in time. Returns strict 0.0/1.0.

    Mirrors ShadowLitePadTacEnv._apply_tactile_smoothing / deploy_policy_new.
    """
    global hold_on_ct, hold_off_ct, hold_state
    raw = t > 0.5

    hold_on_ct = np.where(raw, hold_on_ct + 1, 0).astype(np.int16)
    hold_off_ct = np.where(raw, 0, hold_off_ct + 1).astype(np.int16)
    np.clip(hold_on_ct, None, _hold_ct_max, out=hold_on_ct)
    np.clip(hold_off_ct, None, _hold_ct_max, out=hold_off_ct)

    latched = hold_state > 0.5
    turn_on = ~latched & (hold_on_ct >= int(TACTILE_K_ON))
    turn_off = latched & (hold_off_ct >= int(TACTILE_K_OFF))

    hold_state = np.where(
        turn_on, 1.0, np.where(turn_off, 0.0, hold_state)
    ).astype(np.float32)
    return hold_state.copy()


def read_tactile():
    """Return the 24-d tactile vector the POLICY should consume.

    The real sensor-derived vector (after FSR/BioTac hysteresis, mute, and the
    optional temporal hold) is always computed and stashed in ``last_tac_real``
    so the hysteresis / hold state machines stay live and the log keeps ground
    truth. When ZERO_TACTILE is set, the policy is handed all zeros instead —
    applied LAST, mirroring sim's zero_tactile in _get_tactile().
    """
    global fsr_state, bt_state, last_tac_real
    with fsr_lock:
        fsr_vals = latest_fsr.copy()
    fsr_state = np.where(
        fsr_vals > fsr_hi,
        True,
        np.where(fsr_vals < fsr_lo, False, fsr_state),
    )
    with bt_lock:
        bt_vals = np.array([latest_biotac_pdc[i] for i in BIOTAC_IDX], dtype=np.float32)
    for k in range(N_BIOTAC):
        v = bt_vals[k]
        if not np.isfinite(v) or v < 0:
            continue
        if v > bt_hi[k]:
            bt_state[k] = True
        elif v < bt_lo[k]:
            bt_state[k] = False
    t = np.zeros(NUM_TACTILE, dtype=np.float32)
    fsr_on = apply_fsr_mute(fsr_state.astype(np.float32))
    t[FSR_CHANNELS] = np.maximum(t[FSR_CHANNELS], fsr_on)
    if USE_BIOTAC:
        for k, ch in enumerate(BIOTAC_CH):
            t[ch] = max(t[ch], float(bt_state[k]))
    if USE_TACTILE_HOLD and (int(TACTILE_K_ON) > 1 or int(TACTILE_K_OFF) > 1):
        t = apply_tactile_hold(t)

    last_tac_real = t.copy()
    if ZERO_TACTILE:
        return np.zeros(NUM_TACTILE, dtype=np.float32)
    return t


def slew_pub16(desired, prev_pub, speed_frac):
    """Rate-limit a 16-d publish target (warmup replay). Policy path untouched."""
    desired = np.clip(desired, PUB_LOWER, PUB_UPPER).astype(np.float32)
    if prev_pub is None or speed_frac is None:
        return desired
    max_delta = PUB_VEL_LIMIT * float(speed_frac) / CONTROL_HZ
    return (prev_pub + np.clip(desired - prev_pub, -max_delta, max_delta)).astype(
        np.float32
    )


def run_q_replay(pub, rec_q, episode_id, log, speed_frac=None, prev_pub=None):
    """One open-loop replay of sim achieved q. Appends to log dicts.

    If ``speed_frac`` is set, slews from ``prev_pub`` toward each clipped q
    (same delta cap as policy publish). Returns the last published 16-d cmd
    so multi-episode warmup can continue slewing smoothly.
    """
    rate = rospy.Rate(CONTROL_HZ)
    for q_sim in rec_q:
        if rospy.is_shutdown():
            break
        desired = np.clip(q_sim, PUB_LOWER, PUB_UPPER)
        target = slew_pub16(desired, prev_pub, speed_frac)
        prev_pub = target
        publish_target(pub, target, 1.0 / CONTROL_HZ)

        log["t"].append(rospy.get_time())
        log["episode"].append(episode_id)
        log["cmd"].append(target.copy())
        log["q"].append(current_joint_pos16.copy())
        if "speed_frac" in log:
            log["speed_frac"].append(-1.0 if speed_frac is None else float(speed_frac))
        with fsr_lock:
            log["fsr"].append(latest_fsr.copy())
        with bt_lock:
            log["biotac_pdc"].append(latest_biotac_pdc.copy())
        rate.sleep()
    return prev_pub



def go_to_default_pose(pub):
    """Phase 0: move to DEFAULT_POSE and settle, from wherever the hand is now.

    Clipped to the publish limits and forced to respect the J1 <= J2 coupling
    (same guard as go_to_pose.py) so the distals can never be commanded past
    their driving joint. Uses a long time_from_start so the controller
    interpolates instead of stepping.
    """
    pose = np.clip(DEFAULT_POSE, PUB_LOWER, PUB_UPPER).astype(np.float32)
    for j1, j2 in zip(PUB_J1_SLOTS, PUB_J2_SLOTS):
        pose[j1] = min(pose[j1], pose[j2])
    rospy.loginfo(
        "Phase 0: moving to default (cup) pose over %.1f s ...", DEFAULT_POSE_MOVE_S
    )
    publish_target(pub, pose, DEFAULT_POSE_MOVE_S)
    rospy.sleep(DEFAULT_POSE_MOVE_S + 1.0)
    err = np.abs(current_joint_pos16 - pose)
    rospy.loginfo(
        "At default pose. max |q - pose| = %.4f rad (%.2f deg, joint %s)",
        float(err.max()), float(np.degrees(err.max())),
        PUBLISH_JOINTS[int(err.argmax())],
    )
    return pose


def run_openloop_segment(pub, rec_q, repeat_id, rec, kb, events, speed_frac, prev_pub):
    """One with-balls open-loop pass over the recorded trajectory.

    Publishes every frame at CONTROL_HZ and logs hardware state + tactile each
    step. Mirrors the closed-loop policy loop step-for-step; the only difference
    is where the target comes from. Returns (prev_pub, in_play, aborted).
    """
    rate = rospy.Rate(CONTROL_HZ)
    in_play = rec["_in_play"]
    for frame, q_sim in enumerate(rec_q):
        if rospy.is_shutdown():
            return prev_pub, in_play, True
        now = rospy.get_time()

        for ch in kb.poll():
            if ch == "d":
                if in_play:
                    in_play = False
                    events["t"].append(now)
                    events["step"].append(len(rec["t"]))
                    events["type"].append(EVENT_DROP)
                    rospy.loginfo("EVENT drop @ repeat=%d frame=%d (out of play)",
                                  repeat_id, frame)
                else:
                    rospy.logwarn("drop ignored (already out of play)")
            elif ch == "p":
                if not in_play:
                    in_play = True
                    events["t"].append(now)
                    events["step"].append(len(rec["t"]))
                    events["type"].append(EVENT_PLACE)
                    rospy.loginfo("EVENT place @ repeat=%d frame=%d (back in play)",
                                  repeat_id, frame)
                else:
                    rospy.logwarn("place ignored (already in play)")

        desired = np.clip(q_sim, PUB_LOWER, PUB_UPPER)
        target = slew_pub16(desired, prev_pub, speed_frac)
        prev_pub = target
        publish_target(pub, target, 1.0 / CONTROL_HZ)

        # Tactile is read (and its hysteresis advanced) exactly as in the policy
        # loop, so the logged signal is what the policy WOULD have consumed.
        tac = read_tactile()

        rec["t"].append(now)
        rec["repeat"].append(repeat_id)
        rec["frame"].append(frame)
        rec["q"].append(current_joint_pos.copy())
        rec["q16"].append(current_joint_pos16.copy())
        rec["qd16"].append(current_joint_vel16.copy())
        rec["cmd"].append(target.copy())
        rec["q_ref16"].append(np.asarray(q_sim, dtype=np.float32).copy())
        rec["tac"].append(tac.copy())
        rec["tac_real"].append(last_tac_real.copy())
        with fsr_lock:
            rec["fsr"].append(latest_fsr.copy())
        with bt_lock:
            rec["biotac_pdc"].append(latest_biotac_pdc.copy())
        rec["speed_frac"].append(-1.0 if speed_frac is None else float(speed_frac))
        rec["in_play"].append(1 if in_play else 0)

        rate.sleep()

    rec["_in_play"] = in_play
    return prev_pub, in_play, False


def run_openloop(pub, rec_q, start_q, dur, thresholds_fitted, sim_tac, sim_cmd):
    """Phase D: with-balls open-loop replay of rec_q, logged to REPLAY_NPZ.

    Shared by both entry paths (warmup and no-warmup), so the recorded data is
    identical either way. ``thresholds_fitted`` only annotates the npz: when the
    warmup was skipped the binary ``tac`` column is all-zero and the raw ``fsr`` /
    ``biotac_pdc`` columns are the real tactile record.
    """
    global fsr_state, bt_state

    # ---------- Phase D: with-balls open-loop replay ----------
    fsr_state[:] = False
    bt_state[:] = False
    reset_tactile_hold()
    if USE_TACTILE_HOLD:
        rospy.loginfo("Tactile HOLD ON  K_ON=%d K_OFF=%d", int(TACTILE_K_ON), int(TACTILE_K_OFF))
    else:
        rospy.loginfo("Tactile HOLD OFF (raw hysteresis binary logged)")

    rospy.sleep(1.0)
    kb = KeyboardBallEvents()
    events = {"t": [], "step": [], "type": []}
    rec = {
        "t": [], "repeat": [], "frame": [], "q": [], "q16": [], "qd16": [],
        "cmd": [], "q_ref16": [], "tac": [], "tac_real": [], "fsr": [],
        "biotac_pdc": [], "speed_frac": [], "in_play": [],
        "_in_play": True,   # popped before saving
    }
    rospy.loginfo(
        "Open-loop replay @ %d Hz | %d x %.1f s | governor=%s | tactile logged, not consumed",
        CONTROL_HZ, REPLAY_REPEATS, dur,
        "None (as recorded)" if REPLAY_SPEED_FRAC is None else ("%.3f" % REPLAY_SPEED_FRAC),
    )
    rospy.loginfo(
        "Keyboard marks (no Enter): [d]=balls DROPPED  [p]=balls PLACED again  "
        "(Ctrl-C ends & saves)."
    )

    prev_pub = start_q.copy()
    try:
        for rep in range(REPLAY_REPEATS):
            if rospy.is_shutdown():
                break
            if rep > 0:
                # Ease back to q[0] between passes: the trajectory end pose is not
                # its start pose, so jumping straight in would be a step command.
                rospy.loginfo("Returning to q[0] before pass %d/%d ...", rep + 1, REPLAY_REPEATS)
                publish_target(pub, start_q, 2.0)
                rospy.sleep(2.0 + SETTLE_S)
                prev_pub = start_q.copy()
            rospy.loginfo("Open-loop pass %d/%d (%.1f s)", rep + 1, REPLAY_REPEATS, dur)
            prev_pub, _, aborted = run_openloop_segment(
                pub, rec_q, rep, rec, kb, events, REPLAY_SPEED_FRAC, prev_pub
            )
            if aborted:
                break
    finally:
        kb.close()
        rec.pop("_in_play", None)
        if rec["t"]:
            t0 = float(rec["t"][0])
            event_t_rel = np.array([float(x) - t0 for x in events["t"]], dtype=np.float64)
            extra = {}
            if sim_tac is not None:
                extra["sim_tac"] = sim_tac
            if sim_cmd is not None:
                extra["sim_cmd"] = sim_cmd
            np.savez(
                REPLAY_NPZ,
                **{k: np.array(v) for k, v in rec.items()},
                sim_q=rec_q,
                joints=PUBLISH_JOINTS,
                joints13=POLICY_JOINTS,
                replay_file=REPLAY_FILE,
                replay_repeats=np.int32(REPLAY_REPEATS),
                traj_steps=np.int32(len(rec_q)),
                control_hz=np.int32(CONTROL_HZ),
                replay_speed_frac=np.float32(
                    -1.0 if REPLAY_SPEED_FRAC is None else float(REPLAY_SPEED_FRAC)
                ),
                warmup_file=WARMUP_NPZ if thresholds_fitted else "",
                open_loop=np.bool_(True),
                thresholds_fitted=np.bool_(thresholds_fitted),
                fsr_baseline=fsr_baseline,
                fsr_hi=fsr_hi,
                fsr_lo=fsr_lo,
                bt_baseline=bt_baseline,
                bt_hi=bt_hi,
                bt_lo=bt_lo,
                use_tactile_hold=np.bool_(USE_TACTILE_HOLD),
                tactile_k_on=np.int32(TACTILE_K_ON),
                tactile_k_off=np.int32(TACTILE_K_OFF),
                event_t=np.array(events["t"], dtype=np.float64),
                event_t_rel=event_t_rel,
                event_step=np.array(events["step"], dtype=np.int32),
                event_type=np.array(events["type"], dtype=np.int32),
                event_drop_code=np.int32(EVENT_DROP),
                event_place_code=np.int32(EVENT_PLACE),
                **extra
            )
            n_drop = int(sum(1 for x in events["type"] if x == EVENT_DROP))
            n_place = int(sum(1 for x in events["type"] if x == EVENT_PLACE))
            tac_arr = np.array(rec["tac"], dtype=np.float32)
            q16 = np.array(rec["q16"], dtype=np.float32)
            ref = np.array(rec["q_ref16"], dtype=np.float32)
            track = np.abs(q16 - np.clip(ref, PUB_LOWER, PUB_UPPER))
            rospy.loginfo(
                "saved %s: %d steps (%.1f s) | passes=%d | events drop=%d place=%d | "
                "in_play steps=%d | hw tactile ON%%=%.1f%% | mean|q-ref|=%.4f rad "
                "(max %.4f, worst joint %s)",
                REPLAY_NPZ, len(rec["t"]), len(rec["t"]) / float(CONTROL_HZ),
                len(set(rec["repeat"])), n_drop, n_place, int(sum(rec["in_play"])),
                100.0 * tac_arr.mean() if tac_arr.size else 0.0,
                float(track.mean()), float(track.max()),
                PUBLISH_JOINTS[int(track.mean(axis=0).argmax())],
            )


def main():
    global fsr_state, bt_state

    rospy.init_node("deploy_openloop_aug4_trial5")
    pub = rospy.Publisher("/rh_trajectory_controller/command", JointTrajectory, queue_size=1)
    rospy.Subscriber("/joint_states", JointState, joint_callback)
    rospy.Subscriber("/rh/tactile", BiotacAll, biotac_cb, queue_size=1)

    rospy.loginfo("Waiting for /joint_states ...")
    while not joint_ready and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("Joint states OK.")

    rospy.loginfo("Waiting for /rh/tactile ...")
    while not biotac_ready and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("BioTac OK.")

    th = threading.Thread(target=serial_reader)
    th.daemon = True
    th.start()
    rospy.sleep(0.5)

    # ----- load the recorded open-loop trajectory -----
    data = np.load(REPLAY_FILE, allow_pickle=True)
    rec_q = data["q"].astype(np.float32)
    assert rec_q.ndim == 2 and rec_q.shape[1] == 16, rec_q.shape
    if "joints" in data.files:
        assert list(data["joints"]) == PUBLISH_JOINTS, (list(data["joints"]), PUBLISH_JOINTS)
    sim_tac = data["tac"].astype(np.float32) if "tac" in data.files else None
    sim_cmd = data["cmd"].astype(np.float32) if "cmd" in data.files else None

    # A recording spliced across an env reset would teleport the hand mid-run.
    # record_openloop.py cannot produce one, but a hand-edited or play.py file can.
    step_gap = np.abs(np.diff(rec_q, axis=0))
    worst = float(step_gap.max()) if len(rec_q) > 1 else 0.0
    if worst > 0.15:
        rospy.logerr(
            "REFUSING TO RUN: %s jumps %.3f rad (%.1f deg) in one 1/%d s frame at "
            "index %d. That is a reset discontinuity, not a trajectory — it would "
            "be published straight at the hand. Re-record with record_openloop.py.",
            REPLAY_FILE, worst, np.degrees(worst), CONTROL_HZ,
            int(np.unravel_index(step_gap.argmax(), step_gap.shape)[0]),
        )
        return

    dur = len(rec_q) / float(CONTROL_HZ)
    rospy.loginfo(
        "Loaded %s  T=%d (%.1f s)  max frame step=%.4f rad (%.2f rad/s)",
        REPLAY_FILE, len(rec_q), dur, worst, worst * CONTROL_HZ,
    )
    for k in ("checkpoint", "cmd_speed_frac", "ball_mass_g", "seed", "drop_step"):
        if k in data.files:
            rospy.loginfo("  recorded %s = %s", k, data[k])
    # Warmup may run on a prefix of the trajectory: the pads sweep their whole
    # pose envelope within a couple of Baoding cycles, so a full pass is waste.
    warm_q = rec_q
    if WARMUP_SECONDS is not None:
        warm_q = rec_q[: max(1, int(WARMUP_SECONDS * CONTROL_HZ))]
    warm_dur = len(warm_q) / float(CONTROL_HZ)

    if WARMUP_REPEATS > 0:
        rospy.loginfo(
            "Plan: warmup %d x %.1f s (EMPTY, sf=%s) then replay %d x %.1f s "
            "(WITH BALLS, sf=%s) = %.1f s of commanded motion",
            WARMUP_REPEATS, warm_dur, WARMUP_SPEED_FRAC, REPLAY_REPEATS, dur,
            REPLAY_SPEED_FRAC, WARMUP_REPEATS * warm_dur + REPLAY_REPEATS * dur,
        )
    else:
        rospy.loginfo(
            "Plan: NO warmup (WARMUP_REPEATS=0) — straight to replay %d x %.1f s "
            "(WITH BALLS, sf=%s) = %.1f s of commanded motion",
            REPLAY_REPEATS, dur, REPLAY_SPEED_FRAC, REPLAY_REPEATS * dur,
        )
    if REPLAY_SPEED_FRAC is not None:
        rospy.logwarn(
            "REPLAY_SPEED_FRAC=%.2f — the hand will LAG the trajectory, so ball "
            "timing will not match sim. Set it to None for the real comparison.",
            REPLAY_SPEED_FRAC,
        )

    # ---------- Phase 0: default pose ----------
    # First motion of the run, and the only one that starts from an unknown
    # configuration — hence the prompt and the slow move. Everything after this
    # begins from DEFAULT_POSE, whatever the previous script left behind.
    if GO_TO_DEFAULT_POSE:
        input("HAND EMPTY. Press Enter to move to the DEFAULT (cup) pose...")
        go_to_default_pose(pub)

    # ---------- Phase A: empty-hand warmup (skipped when WARMUP_REPEATS == 0) ----------
    if WARMUP_REPEATS <= 0:
        # No empty-motion envelope means no thresholds. Leaving them at zero would
        # make every raw reading (hundreds of ADC counts) exceed hi and latch the
        # whole binary vector ON — contact-looking garbage in the log. Park them at
        # +inf instead so tac reads an honest all-zero, and record the fact in the
        # npz. Raw FSR + BioTac PDC are still logged every step, so the binary
        # signal can be reconstructed offline from this same file.
        fsr_hi[:] = np.inf
        fsr_lo[:] = np.inf
        bt_hi[:] = np.inf
        bt_lo[:] = np.inf
        rospy.logwarn(
            "WARMUP SKIPPED — tactile thresholds NOT fitted. Raw FSR (12) and "
            "BioTac PDC (5) are still logged every step; the binary 'tac' vector "
            "will read all-zero and is NOT contact data. Threshold it offline from "
            "rec['fsr'] / rec['biotac_pdc'], or set WARMUP_REPEATS >= 1."
        )
        input("PLACE BALLS in cup. Press Enter to move to q[0] and start OPEN-LOOP replay...")
        start_q = np.clip(rec_q[0], PUB_LOWER, PUB_UPPER)
        publish_target(pub, start_q, 2.0)
        rospy.sleep(3.0)
        run_openloop(pub, rec_q, start_q, dur, thresholds_fitted=False,
                     sim_tac=sim_tac, sim_cmd=sim_cmd)
        return

    input(
        "HAND EMPTY — cup pose ready. Press Enter to start %d x %.1f s warmup replay..."
        % (WARMUP_REPEATS, warm_dur)
    )

    first = np.clip(warm_q[0], PUB_LOWER, PUB_UPPER)
    publish_target(pub, first, 2.0)
    rospy.sleep(3.0)

    warmup = {
        "t": [], "episode": [], "cmd": [], "q": [],
        "fsr": [], "biotac_pdc": [], "speed_frac": [],
    }
    prev_pub = first.copy()
    for ep in range(WARMUP_REPEATS):
        rospy.loginfo("Warmup replay %d/%d (%.1f s)", ep + 1, WARMUP_REPEATS, warm_dur)
        prev_pub = run_q_replay(
            pub, warm_q, ep, warmup, speed_frac=WARMUP_SPEED_FRAC, prev_pub=prev_pub
        )
        rospy.sleep(0.3)

    # ---------- Phase B: fit tactile thresholds from the empty motion ----------
    fsr_arr = np.asarray(warmup["fsr"], dtype=np.float32)
    bt_arr = np.asarray(warmup["biotac_pdc"], dtype=np.float32)
    fit_thresholds_from_warmup(fsr_arr, bt_arr)

    # Causal hysteresis replay over the warmup, to report empty-motion ON rates.
    fsr_state[:] = False
    bt_state[:] = False
    tac_warmup = []
    for i in range(len(fsr_arr)):
        fsr_vals = fsr_arr[i]
        fsr_state[:] = np.where(
            fsr_vals > fsr_hi, True, np.where(fsr_vals < fsr_lo, False, fsr_state)
        )
        bt_row = bt_arr[i, BIOTAC_IDX]
        for k in range(N_BIOTAC):
            v = bt_row[k]
            if not np.isfinite(v) or v < 0:
                continue
            if v > bt_hi[k]:
                bt_state[k] = True
            elif v < bt_lo[k]:
                bt_state[k] = False
        tvec = np.zeros(NUM_TACTILE, dtype=np.float32)
        fsr_on = apply_fsr_mute(fsr_state.astype(np.float32))
        tvec[FSR_CHANNELS] = fsr_on
        if USE_BIOTAC:
            for k, ch in enumerate(BIOTAC_CH):
                tvec[ch] = float(bt_state[k])
        tac_warmup.append(tvec.copy())
    tac_warmup = np.asarray(tac_warmup, dtype=np.float32)

    np.savez(
        WARMUP_NPZ,
        t=np.asarray(warmup["t"]),
        episode=np.asarray(warmup["episode"], dtype=np.int32),
        cmd=np.asarray(warmup["cmd"], dtype=np.float32),
        q=np.asarray(warmup["q"], dtype=np.float32),
        speed_frac_replay=np.asarray(warmup["speed_frac"], dtype=np.float64),
        fsr=fsr_arr,
        biotac_pdc=bt_arr,
        tac_bin=tac_warmup,
        fsr_names=np.array(FSR_NAMES),
        biotac_names=np.array(BIOTAC_NAMES),
        fsr_channels=np.array(FSR_CHANNELS, dtype=np.int32),
        biotac_channels=np.array(BIOTAC_CH, dtype=np.int32),
        biotac_idx=np.array(BIOTAC_IDX, dtype=np.int32),
        fsr_baseline=fsr_baseline,
        fsr_noise=fsr_noise,
        fsr_hi=fsr_hi,
        fsr_lo=fsr_lo,
        bt_baseline=bt_baseline,
        bt_noise=bt_noise,
        bt_hi=bt_hi,
        bt_lo=bt_lo,
        P_HI=np.float32(P_HI),
        P_LO=np.float32(P_LO),
        MARGIN_ABS=np.float32(MARGIN_ABS),
        MARGIN_LO=np.float32(MARGIN_LO),
        WIDE_ENVELOPE_ADC=np.float32(WIDE_ENVELOPE_ADC),
        HYST_BAND_WIDE=np.float32(HYST_BAND_WIDE),
        thresh_mode=np.array("p99_5_plus_margin_hysteresis_wide_tight"),
        fsr_mute_mux=np.array(FSR_MUTE_MUX, dtype=np.int32),
        joints=PUBLISH_JOINTS,
        replay_file=REPLAY_FILE,
        warmup_repeats=np.int32(WARMUP_REPEATS),
        warmup_speed_frac=np.float32(
            -1.0 if WARMUP_SPEED_FRAC is None else float(WARMUP_SPEED_FRAC)
        ),
    )
    rospy.loginfo("Saved warmup sensors -> %s  (%d steps)", WARMUP_NPZ, len(fsr_arr))
    rospy.loginfo("Empty-motion binary ON%% (should be low):")
    for i, name in enumerate(FSR_NAMES):
        rospy.loginfo("  %s  ON=%.1f%%  med=%.1f hi=%.1f",
                      name, 100.0 * tac_warmup[:, FSR_CHANNELS[i]].mean(),
                      fsr_baseline[i], fsr_hi[i])
    for k, name in enumerate(BIOTAC_NAMES):
        rospy.loginfo("  %s  ON=%.1f%%  med=%.1f hi=%.1f",
                      name, 100.0 * tac_warmup[:, BIOTAC_CH[k]].mean(),
                      bt_baseline[k], bt_hi[k])

    # ---------- Phase C: back to q[0], place the balls ----------
    start_q = np.clip(rec_q[0], PUB_LOWER, PUB_UPPER)
    rospy.loginfo("Warmup done. Moving to trajectory start q[0] from %s ...", REPLAY_FILE)
    publish_target(pub, start_q, 2.0)
    rospy.sleep(3.0)

    input("At q[0]. PLACE BALLS in cup, then press Enter to start OPEN-LOOP replay...")

    run_openloop(pub, rec_q, start_q, dur, thresholds_fitted=True,
                 sim_tac=sim_tac, sim_cmd=sim_cmd)



if __name__ == "__main__":
    main()
