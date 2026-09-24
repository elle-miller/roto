#!/usr/bin/env python
"""Trial 15 deploy with empty-motion tactile warmup (Gate B style).

Flow (HAND EMPTY until Phase C):
  A) Replay sim achieved-q  WARMUP_REPEATS times @ 60 Hz
     (optional SPEED_FRAC slew on replay only — see USE_SPEED_FRAC_ON_Q_REPLAY).
     Log raw FSR (12) + BioTac PDC (5) every step.
  B) Fit per-channel thresholds from empty-motion envelope:
        hi = p99.5(raw) + 5 ADC
        lo = median + 2 (narrow pads) or hi - HYST_BAND_WIDE (wide envelopes)
     Hysteresis ON/OFF. mfprox live (FSR_MUTE_MUX empty after pad replace).
     Save hw_warmup_trial15_fsr.npz for offline plots / sim comparison.
  C) Move to sim rollout start pose q[0], prompt to place balls in the cup.
  D) Closed-loop policy with REAL tactile + SPEED_FRAC governor
     (fixed or optional 0.8→0.5 ramp — see USE_SPEED_FRAC_RAMP).
     Optional temporal tactile hold (USE_TACTILE_HOLD) for FD+smooth ckpts.
     Optional tactile ABLATION (ZERO_TACTILE) — see below.
     Save hw_policy_log_trial15_warmupcal.npz.
     Keyboard (non-blocking, during policy loop — no Enter needed):
       d = balls dropped / out of play
       p = balls placed again / back in play
     Events + per-step in_play are stored in the policy npz for shaded plots.

TACTILE ABLATION (ZERO_TACTILE):
  Set ZERO_TACTILE = True to feed the policy an all-zero 24-d tactile vector
  while the hand still runs closed-loop on real proprioception. This is the
  hardware twin of the sim `--zero_tactile` ablation: it answers "does this
  policy actually rely on tactile, or is it running proprioception-only?".

  What it does / does not touch:
    * Zeroing happens at the END of read_tactile(), AFTER hysteresis and the
      optional temporal hold — exactly matching the sim ordering, where
      zero_tactile is applied last in _get_tactile(). The FSR/BioTac hysteresis
      and hold state machines keep running on the real signal, so the log stays
      meaningful.
    * Phase A/B (warmup + threshold fitting) is UNAFFECTED: it reads the raw
      ADC arrays directly and never calls read_tactile(). Thresholds are still
      fitted on the real empty-motion envelope.
    * Raw sensors are still logged every step (rec["fsr"], rec["biotac_pdc"]),
      and the real post-hysteresis binary vector is logged as rec["tac_real"],
      so you can show "contact WAS happening, the policy just never saw it".
      rec["tac"] is what the policy actually received (all zeros when ablating).
    * Observation dimensionality is unchanged (304 = 4x52 prop + 4x24 tactile);
      the 96 tactile dims are simply zero.

  ZERO_TACTILE = False (default) is a no-op: control behaviour is identical to
  the un-ablated script, with only the extra tac_real log field added.

Encoder check: on load, rospy logs missing/unexpected keys. You want
  encoder missing=[] unexpected=[]
If any net.* weights are missing, the encoder is partial — do not deploy.

Does NOT modify deploy_policy_new.py.

Laptop files needed next to this script (or absolute paths below):
  sim_policy_log_trial15_seed42.npz
  best_agent_legacy_padtac_bt_scratch_trial15.pt  (or under /home/user/experiments/)
"""

from __future__ import print_function

import sys
import select
import rospy
import torch
import torch.nn as nn
import numpy as np
from collections import deque

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
REPLAY_Q_FILE = "sim_policy_log_trial15_seed42.npz"  # must have q (T,16) + joints
CHECKPOINT = "/home/user/experiments/best_agent_nomass_dr.pt"

WARMUP_REPEATS = 2
WARMUP_NPZ = "hw_warmup_trial15_fsr.npz"
POLICY_NPZ = "hw_policy_log_fd_nosmoothing_trial2.npz"

CONTROL_HZ = 60
scaled_err = 1
scaled_vel = 1.0
scaled_pos = 1.0

# --- SPEED_FRAC governor (closed-loop policy only) ---
# False: fixed SPEED_FRAC for the whole run (0.5 = T15 recipe; 1.0 = full rate
#        cap; None = no slew / bang-bang publish).
# True:  hold SPEED_FRAC_START, then linear ramp to SPEED_FRAC_END, then hold.
USE_SPEED_FRAC_RAMP = False
SPEED_FRAC = 0.6  # used only when USE_SPEED_FRAC_RAMP is False

SPEED_FRAC_START = 0.8
SPEED_FRAC_END = 0.5
RAMP_HOLD_START_S = 0.0   # seconds at START before ramp begins
RAMP_DURATION_S = 10.0    # seconds of linear START → END
RAMP_START_STEP = int(RAMP_HOLD_START_S * CONTROL_HZ)
RAMP_END_STEP = RAMP_START_STEP + int(RAMP_DURATION_S * CONTROL_HZ)

# --- Open-loop q-replay slew (warmup only; does NOT affect policy loop) ---
# False (default): publish each sim q target immediately @ CONTROL_HZ (legacy).
# True: rate-limit toward each q with the same PUB_VEL_LIMIT * frac / Hz clip
#       used in action_to_publish. Useful to probe plant tracking under slew
#       without changing closed-loop deploy.
USE_SPEED_FRAC_ON_Q_REPLAY = False
# None → reuse SPEED_FRAC; set a float (e.g. 0.3 / 0.5 / 1.0) to override for
# replay only. Ignored when USE_SPEED_FRAC_ON_Q_REPLAY is False.
# None with SPEED_FRAC=None → unlimited (same as flag off).
REPLAY_SPEED_FRAC = None

# --- pos_err ablation (default off) ---
# False (default): last_command = unslewed sim joint_pos_cmd (raw_cmd / j2_cmd).
#   With SPEED_FRAC, plant tracks slewed pub but obs sees raw−q → artificial lag.
# True: last_command = 13-d equivalent of published pub_target (after slew).
#   pos_err ≈ 0 on stiff HW; tests whether bang-bang / velocity dominates over error.
USE_PUBLISHED_CMD_FOR_POS_ERR = True

# --- Temporal tactile hold (policy obs only; mirrors sim tactile_cfg.smoothing) ---
# False (default): raw hysteresis binary → policy (T5 / scale20 / no-smooth ckpts).
# True: debounce 24-d tac after FSR/BioTac hysteresis — MUST match training yaml
#   (FD+smooth: k_on=3, k_off=1). Same logic as deploy_policy_new.apply_tactile_hold.
# Warmup threshold fitting still uses raw ADC; hold applies only in read_tactile().
USE_TACTILE_HOLD = False
TACTILE_K_ON = 3   # consecutive ON steps before policy sees 1 (~50 ms @ 60 Hz)
TACTILE_K_OFF = 1  # consecutive OFF steps before release

# --- Tactile ABLATION: feed the policy an all-zero tactile vector ---
# False (default): policy gets the real (hysteresis / hold) binary tactile.
# True: policy gets zeros for all 24 channels. Applied at the END of
#   read_tactile(), after hysteresis + hold, mirroring sim's zero_tactile which
#   is checked last in _get_tactile(). Warmup / threshold fitting is unaffected
#   (it never calls read_tactile()), raw FSR + BioTac are still logged, and the
#   real binary vector is logged as tac_real for side-by-side comparison.
#   Use this to test whether the policy actually depends on touch.
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
# Legacy K*std (unused for FSR now; kept for optional BT fallback / docs):
# K_HI, K_LO = 5.0, 2.0

# ============================================================
# MODEL
# ============================================================
OBS_DIM = 304
PROP_DIM = 52
NUM_J = 13
NUM_TACTILE = 24
OBS_STACK = 4


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 1024), nn.LayerNorm(1024), nn.ELU(),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.ELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ELU(),
        )

    def forward(self, x):
        return self.net(x)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, NUM_J),
        )

    def forward(self, z):
        return self.policy_net(z)


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


def unscale(x, lo, hi):
    return (2.0 * x - hi - lo) / (hi - lo)


def scale(a, lo, hi):
    return 0.5 * (a + 1.0) * (hi - lo) + lo


def speed_frac_at_step(step):
    """Return SPEED_FRAC for this policy tick (float or None).

    Fixed mode: always SPEED_FRAC.
    Ramp mode: hold START → linear to END over [RAMP_START_STEP, RAMP_END_STEP) → hold END.
    """
    if not USE_SPEED_FRAC_RAMP:
        return None if SPEED_FRAC is None else float(SPEED_FRAC)
    if step < RAMP_START_STEP:
        return float(SPEED_FRAC_START)
    if step >= RAMP_END_STEP:
        return float(SPEED_FRAC_END)
    span = float(RAMP_END_STEP - RAMP_START_STEP)
    alpha = (step - RAMP_START_STEP) / span if span > 0 else 1.0
    return float(SPEED_FRAC_START + alpha * (SPEED_FRAC_END - SPEED_FRAC_START))


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


def action_to_publish(action, meas_j2, prev_pub, speed_frac):
    raw_cmd = scale(action, JOINT_LOWER, JOINT_UPPER)
    proxy = raw_cmd[CURL_J2_IDX]
    j2_cmd = np.clip(proxy * (J2_UPPER_SIM / COUPLING_THETA), 0.0, J2_UPPER_SIM)
    j1_cmd = np.clip(
        (proxy - COUPLING_THETA) / (J2_UPPER_SIM - COUPLING_THETA) * J1_UPPER_SIM,
        0.0, J1_UPPER_SIM,
    )
    opens_at = J2_UPPER_SIM - GATE_J2_TOL
    gate = np.clip((np.asarray(meas_j2, dtype=np.float32) - opens_at) / GATE_J2_TOL, 0.0, 1.0)
    j1_cmd = j1_cmd * gate

    pub = np.empty(16, dtype=np.float32)
    pub[PUB_NONCOUPLED] = raw_cmd[CTRL_NONCOUPLED]
    pub[PUB_J2_SLOTS] = j2_cmd
    pub[PUB_J1_SLOTS] = j1_cmd
    pub = np.clip(pub, PUB_LOWER, PUB_UPPER)

    if prev_pub is not None and speed_frac is not None:
        max_delta = PUB_VEL_LIMIT * float(speed_frac) / CONTROL_HZ
        pub = prev_pub + np.clip(pub - prev_pub, -max_delta, max_delta)

    return pub.astype(np.float32), raw_cmd, j2_cmd


def pub16_to_cmd13(pub):
    """Map published 16-d joint targets back to 13-d policy cmd_error space."""
    cmd = np.empty(NUM_J, dtype=np.float32)
    cmd[CTRL_NONCOUPLED] = pub[PUB_NONCOUPLED]
    cmd[CURL_J2_IDX] = pub[PUB_J2_SLOTS]
    return cmd


def verify_encoder_load(load_result, checkpoint_path):
    """Log whether the encoder checkpoint loaded completely (not partial).

    On the control laptop, after ``rospy.init_node``, look for::

        [INFO] Encoder load OK: all weights present ...

    If you see ``ENCODER INCOMPLETE`` or missing ``net.*`` keys, stop — you would
    be running a random or partial encoder.  Policy head uses strict=True separately.
    """
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    weight_missing = [k for k in missing if k.startswith("net.")]
    if weight_missing:
        rospy.logerr(
            "ENCODER INCOMPLETE — missing weight keys: %s  (checkpoint: %s)",
            weight_missing, checkpoint_path,
        )
    elif missing:
        rospy.logwarn(
            "Encoder missing_keys (non-weight; verify harmless): %s", missing,
        )
    else:
        rospy.loginfo(
            "Encoder load OK: all weights present (not partial). checkpoint=%s",
            checkpoint_path,
        )
    if unexpected:
        rospy.logwarn("Encoder unexpected_keys: %s", unexpected)
    return not weight_missing


# ============================================================
# GLOBAL STATE
# ============================================================
current_joint_pos = np.zeros(NUM_J, dtype=np.float32)
current_joint_vel = np.zeros(NUM_J, dtype=np.float32)
current_joint_pos16 = np.zeros(16, dtype=np.float32)
current_joint_vel16 = np.zeros(16, dtype=np.float32)
last_action = np.zeros(NUM_J, dtype=np.float32)
last_command = np.zeros(NUM_J, dtype=np.float32)
joint_ready = False
biotac_ready = False

prop_buffer = deque(maxlen=OBS_STACK)
tactile_buffer = deque(maxlen=OBS_STACK)

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


def build_prop():
    pos_norm = unscale(current_joint_pos, JOINT_LOWER, JOINT_UPPER)
    vel_norm = current_joint_vel / JOINT_VEL_LIMIT
    error = last_command - current_joint_pos
    return np.concatenate([pos_norm*scaled_pos, vel_norm*scaled_vel, error*scaled_err, last_action]).astype(np.float32)


def _replay_speed_frac():
    """SPEED_FRAC used for open-loop q replay only (or None = no slew)."""
    if not USE_SPEED_FRAC_ON_Q_REPLAY:
        return None
    if REPLAY_SPEED_FRAC is not None:
        return float(REPLAY_SPEED_FRAC)
    return None if SPEED_FRAC is None else float(SPEED_FRAC)


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


def main():
    global last_action, last_command, fsr_state, bt_state

    rospy.init_node("deploy_warmup_trial15")
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

    # ----- load Trial-15 q replay -----
    data = np.load(REPLAY_Q_FILE, allow_pickle=True)
    rec_q = data["q"].astype(np.float32)
    assert rec_q.ndim == 2 and rec_q.shape[1] == 16, rec_q.shape
    if "joints" in data.files:
        assert list(data["joints"]) == PUBLISH_JOINTS, (list(data["joints"]), PUBLISH_JOINTS)
    rospy.loginfo("Loaded %s  T=%d for warmup x%d", REPLAY_Q_FILE, len(rec_q), WARMUP_REPEATS)
    replay_sf = _replay_speed_frac()
    if replay_sf is None:
        rospy.loginfo("Q-replay slew: OFF (publish each sim q immediately)")
    else:
        rospy.loginfo(
            "Q-replay slew: ON  SPEED_FRAC=%.3f  (warmup only; policy loop unchanged)",
            replay_sf,
        )

    input(
        "HAND EMPTY — cup pose ready. Press Enter to start %d x q-replay warmup..."
        % WARMUP_REPEATS
    )

    # Move to first frame, settle
    first = np.clip(rec_q[0], PUB_LOWER, PUB_UPPER)
    publish_target(pub, first, 2.0)
    rospy.sleep(3.0)

    warmup = {
        "t": [], "episode": [], "cmd": [], "q": [],
        "fsr": [], "biotac_pdc": [], "speed_frac": [],
    }
    prev_pub = first.copy()
    for ep in range(WARMUP_REPEATS):
        rospy.loginfo("Warmup replay %d/%d", ep + 1, WARMUP_REPEATS)
        prev_pub = run_q_replay(
            pub, rec_q, ep, warmup, speed_frac=replay_sf, prev_pub=prev_pub
        )
        # brief pause between repeats (hold last pose)
        rospy.sleep(0.3)

    fsr_arr = np.asarray(warmup["fsr"], dtype=np.float32)
    bt_arr = np.asarray(warmup["biotac_pdc"], dtype=np.float32)
    fit_thresholds_from_warmup(fsr_arr, bt_arr)

    # Approximate binary ON rates under fitted thresholds (causal hysteresis replay)
    fsr_state[:] = False
    bt_state[:] = False
    tac_warmup = []
    for i in range(len(fsr_arr)):
        # feed hysteresis as if streaming
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
        replay_file=REPLAY_Q_FILE,
        warmup_repeats=np.int32(WARMUP_REPEATS),
        use_speed_frac_on_q_replay=np.bool_(USE_SPEED_FRAC_ON_Q_REPLAY),
        replay_speed_frac=np.float32(
            -1.0 if replay_sf is None else float(replay_sf)
        ),
        speed_frac=np.float32(
            -1.0 if USE_SPEED_FRAC_RAMP else (-1.0 if SPEED_FRAC is None else SPEED_FRAC)
        ),
        use_speed_frac_ramp=np.bool_(USE_SPEED_FRAC_RAMP),
        speed_frac_start=np.float32(SPEED_FRAC_START),
        speed_frac_end=np.float32(SPEED_FRAC_END),
        ramp_start_step=np.int32(RAMP_START_STEP),
        ramp_end_step=np.int32(RAMP_END_STEP),
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

    # Move to sim rollout START (q[0]), not end-of-replay pose — matches closed-loop IC.
    policy_start_q = np.clip(rec_q[0], PUB_LOWER, PUB_UPPER)
    rospy.loginfo(
        "Warmup done. Moving to sim q[0] (policy start pose) from %s ...",
        REPLAY_Q_FILE,
    )
    publish_target(pub, policy_start_q, 2.0)
    rospy.sleep(3.0)

    input(
        "At sim q[0]. PLACE BALLS in cup, then press Enter to start POLICY..."
    )

    # ----- load policy -----
    rospy.loginfo("Loading checkpoint: %s", CHECKPOINT)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    encoder, policy = Encoder(), Policy()
    e_res = encoder.load_state_dict(ckpt["encoder"], strict=False)
    verify_encoder_load(e_res, CHECKPOINT)
    rospy.loginfo("encoder missing=%s unexpected=%s", e_res.missing_keys, e_res.unexpected_keys)
    policy.load_state_dict(
        {k: v for k, v in ckpt["policy"].items() if k != "log_std_parameter"},
        strict=True,
    )
    encoder.eval()
    policy.eval()

    fsr_state[:] = False
    bt_state[:] = False
    reset_tactile_hold()
    last_command[:] = current_joint_pos
    last_action[:] = 0.0
    if USE_TACTILE_HOLD:
        rospy.loginfo(
            "Tactile HOLD ON  K_ON=%d K_OFF=%d (match FD+smooth training yaml)",
            int(TACTILE_K_ON), int(TACTILE_K_OFF),
        )
    else:
        rospy.loginfo("Tactile HOLD OFF (raw hysteresis binary → policy)")
    if ZERO_TACTILE:
        rospy.logwarn(
            "*** TACTILE ABLATION ACTIVE: ZERO_TACTILE=True — policy receives an "
            "all-zero 24-d tactile vector. Real sensors still logged as "
            "fsr / biotac_pdc / tac_real. Proprioception is UNCHANGED. ***"
        )
    else:
        rospy.loginfo("Tactile ABLATION OFF (policy receives real tactile)")
    p0, t0 = build_prop(), read_tactile()

    prop_buffer.clear()
    tactile_buffer.clear()
    for _ in range(OBS_STACK):
        prop_buffer.append(p0.copy())
        tactile_buffer.append(t0.copy())

    rospy.sleep(1.0)
    rate = rospy.Rate(CONTROL_HZ)
    prev_pub = None
    step = 0
    # After "PLACE BALLS" Enter, assume in play until 'd'
    in_play = True
    kb = KeyboardBallEvents()
    events = {"t": [], "step": [], "type": []}  # type: EVENT_DROP / EVENT_PLACE
    rec = {
        "t": [], "q": [], "cmd": [], "tac": [], "tac_real": [], "fsr": [],
        "biotac_pdc": [], "act": [], "speed_frac": [], "in_play": [],
    }
    if USE_SPEED_FRAC_RAMP:
        rospy.loginfo(
            "Policy loop @ %d Hz | motion-calibrated tactile | SPEED_FRAC RAMP: "
            "%.2f for %.1fs → linear to %.2f by %.1fs | pos_err_from_pub=%s | zero_tactile=%s",
            CONTROL_HZ,
            SPEED_FRAC_START, RAMP_HOLD_START_S,
            SPEED_FRAC_END, RAMP_HOLD_START_S + RAMP_DURATION_S,
            USE_PUBLISHED_CMD_FOR_POS_ERR, ZERO_TACTILE,
        )
    else:
        rospy.loginfo(
            "Policy loop @ %d Hz | motion-calibrated tactile | governor=%s | "
            "pos_err_from_pub=%s | zero_tactile=%s",
            CONTROL_HZ,
            "None (unlimited)" if SPEED_FRAC is None else ("%.3f" % float(SPEED_FRAC)),
            USE_PUBLISHED_CMD_FOR_POS_ERR, ZERO_TACTILE,
        )
    rospy.loginfo(
        "Keyboard marks (no Enter): [d]=balls DROPPED  [p]=balls PLACED again  "
        "(Ctrl-C ends & saves). Starting in_play=%s",
        in_play,
    )

    try:
        while not rospy.is_shutdown():
            now = rospy.get_time()
            for ch in kb.poll():
                if ch == "d":
                    if in_play:
                        in_play = False
                        events["t"].append(now)
                        events["step"].append(step)
                        events["type"].append(EVENT_DROP)
                        rospy.loginfo(
                            "EVENT drop @ step=%d t=%.2fs (out of play)",
                            step, now,
                        )
                    else:
                        rospy.logwarn("drop ignored (already out of play)")
                elif ch == "p":
                    if not in_play:
                        in_play = True
                        events["t"].append(now)
                        events["step"].append(step)
                        events["type"].append(EVENT_PLACE)
                        rospy.loginfo(
                            "EVENT place @ step=%d t=%.2fs (back in play)",
                            step, now,
                        )
                    else:
                        rospy.logwarn("place ignored (already in play)")

            prop_buffer.append(build_prop())
            tactile_buffer.append(read_tactile())

            obs = np.concatenate(list(prop_buffer) + list(tactile_buffer))
            assert obs.shape[0] == OBS_DIM, obs.shape
            obs_t = torch.from_numpy(obs).unsqueeze(0)

            with torch.no_grad():
                action = policy(encoder(obs_t)).numpy()[0].astype(np.float32)

            sf = speed_frac_at_step(step)
            if USE_SPEED_FRAC_RAMP and step in (0, RAMP_START_STEP, RAMP_END_STEP):
                rospy.loginfo(
                    "SPEED_FRAC @ step=%d (t=%.1fs): %s",
                    step, step / float(CONTROL_HZ),
                    "None" if sf is None else ("%.3f" % float(sf)),
                )

            pub_target, raw_cmd, j2_cmd = action_to_publish(
                action, current_joint_pos[CURL_J2_IDX], prev_pub, sf
            )
            prev_pub = pub_target
            publish_target(pub, pub_target, 1.0 / CONTROL_HZ)

            rec["t"].append(now)
            rec["q"].append(current_joint_pos.copy())
            rec["cmd"].append(pub_target.copy())
            rec["tac"].append(tactile_buffer[-1].copy())
            rec["tac_real"].append(last_tac_real.copy())
            with fsr_lock:
                rec["fsr"].append(latest_fsr.copy())
            with bt_lock:
                rec["biotac_pdc"].append(latest_biotac_pdc.copy())
            rec["act"].append(action.copy())
            rec["speed_frac"].append(-1.0 if sf is None else float(sf))
            rec["in_play"].append(1 if in_play else 0)

            last_action[:] = action
            if USE_PUBLISHED_CMD_FOR_POS_ERR:
                # Ablation: cmd_error tracks what the plant actually received (slewed).
                last_command[:] = pub16_to_cmd13(pub_target)
            else:
                last_command[:] = raw_cmd
                last_command[CURL_J2_IDX] = j2_cmd
            step += 1
            rate.sleep()
    finally:
        kb.close()
        if rec["t"]:
            t0 = float(rec["t"][0])
            event_t_rel = np.array(
                [float(x) - t0 for x in events["t"]], dtype=np.float64
            )
            np.savez(
                POLICY_NPZ,
                **{k: np.array(v) for k, v in rec.items()},
                fsr_baseline=fsr_baseline,
                fsr_hi=fsr_hi,
                fsr_lo=fsr_lo,
                bt_baseline=bt_baseline,
                bt_hi=bt_hi,
                bt_lo=bt_lo,
                warmup_file=WARMUP_NPZ,
                use_published_cmd_for_pos_err=USE_PUBLISHED_CMD_FOR_POS_ERR,
                use_tactile_hold=np.bool_(USE_TACTILE_HOLD),
                tactile_k_on=np.int32(TACTILE_K_ON),
                tactile_k_off=np.int32(TACTILE_K_OFF),
                zero_tactile=np.bool_(ZERO_TACTILE),
                use_speed_frac_ramp=np.bool_(USE_SPEED_FRAC_RAMP),
                speed_frac_fixed=np.float32(
                    -1.0 if SPEED_FRAC is None else float(SPEED_FRAC)
                ),
                speed_frac_start=np.float32(SPEED_FRAC_START),
                speed_frac_end=np.float32(SPEED_FRAC_END),
                ramp_start_step=np.int32(RAMP_START_STEP),
                ramp_end_step=np.int32(RAMP_END_STEP),
                event_t=np.array(events["t"], dtype=np.float64),
                event_t_rel=event_t_rel,
                event_step=np.array(events["step"], dtype=np.int32),
                event_type=np.array(events["type"], dtype=np.int32),
                event_drop_code=np.int32(EVENT_DROP),
                event_place_code=np.int32(EVENT_PLACE),
            )
            n_drop = int(sum(1 for x in events["type"] if x == EVENT_DROP))
            n_place = int(sum(1 for x in events["type"] if x == EVENT_PLACE))
            tac_real_arr = np.array(rec["tac_real"], dtype=np.float32)
            rospy.loginfo(
                "saved %s: %d steps | events drop=%d place=%d | in_play steps=%d | "
                "zero_tactile=%s | real tactile ON%%=%.1f%% (what the policy %s)",
                POLICY_NPZ,
                len(rec["t"]),
                n_drop,
                n_place,
                int(sum(rec["in_play"])),
                ZERO_TACTILE,
                100.0 * tac_real_arr.mean() if tac_real_arr.size else 0.0,
                "did NOT see" if ZERO_TACTILE else "saw",
            )


if __name__ == "__main__":
    main()
