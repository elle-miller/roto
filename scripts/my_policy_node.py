#!/usr/bin/env python3

"""
run_RL_shadowlite.py
====================
ROS node that runs a pre-trained RL policy on the Shadow Hand Lite.
 
Task   : Peace sign gesture
Robot  : Shadow Hand Lite (13 actuated controllers)
Policy : GaussianPolicy trained in IsaacLab with 13 actions
 
Key facts about the hardware:
  - /joint_states publishes 16 joints (includes J1 which mirrors J2 physically)
  - Only 13 ROS position controllers exist (ffj0 controls FFJ1+FFJ2 together)
  - Policy was trained with FFJ2/MFJ2/RFJ2 as the controlling joint (J1 mimics J2)
  - Scale mismatch: policy outputs 0 to pi/2 per coupled joint,
    but real J0 range is 0 to pi → multiply by 2.0 when publishing
 
Author : (your name)
"""
 
# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
# Standard Python libraries — nothing ROS-specific here
import os
import sys
sys.path.insert(0, "/root/repos/multimodal_rl")
import numpy as np
from threading import Lock
import rospy
rospy.loginfo(sys.executable)          # protects shared data between threads
from copy import deepcopy
 
# PyTorch — runs the neural network policy
import torch
 
# ROS Python client library — makes this script a ROS node
import rospy
 
# ROS message types — define the shape of data sent/received over topics
from std_msgs.msg import Float64              # single float — used per-joint command
from sensor_msgs.msg import JointState        # joint name + position + velocity + effort
 
# Neural network classes from your training codebase
from multimodal_rl.rl.policy_value import GaussianPolicy
from multimodal_rl.models.encoder import Encoder

from collections import deque           # for obs stack circular buffer
from std_msgs.msg import Float64MultiArray
 
import argparse
import select
import sys
import threading
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_OK = True
except Exception:
    _MATPLOTLIB_OK = False
 
 
# =============================================================================
# SECTION 2: DEVICE + DTYPE
# =============================================================================
# All torch tensors will be float32 throughout — must match training dtype
device = "cpu"
torch.set_default_dtype(torch.float32)
 
 
# =============================================================================
# SECTION 3: HYPERPARAMETERS — tune these without touching core logic
# =============================================================================
 
# How many times per second the policy runs and sends a command.
# Must match the Hz the policy was trained at in simulation.
RL_HZ = 60
 
# Low-pass filter on actions — prevents jerky motion.
# New command = 1% new + 99% old. Small = very smooth but slow to respond.
ACTION_TAU = 0.01
 
# Velocity scaling factor applied to all VEL_LIMITS.
# 0.1 = run at 10% of max hardware velocity — safe for initial testing.
OVERRIDE_VEL_SCALE = 0.1
 
# How many policy steps to run per episode (10 Hz × 100 steps = 10 seconds).
EPISODE_TIMESTEPS = 600

# Speed fraction: 1.0 = full speed, 0.5 = half speed, 0.best_agent25 = quarter speed.
# Overridden by --speed command-line argument.
SPEED = 1.0
 
# Path to the saved neural network weights from IsaacLab training.
CHECKPOINT_PATH = os.path.join(
    "/root/shadow_docker_ws/src/my_shadow_control/scripts",
    "best_agent_new_coup_prop_only.pt"
)



# =============================================================================
# OBS STACK + TACTILE CONFIG
# =============================================================================

# Number of consecutive observations the policy sees at once.
# Must match obs_stack in the training agent yaml (confirmed: 4).
OBS_STACK = 4

# Toggle tactile on/off without changing observation structure.
# Set False to run without Touch Lab sensor connected.
USE_TACTILE = False

# Minimum contact threshold (floor) applied to baseline-subtracted deviation.
# The live threshold is max(TACTILE_NOISE_SIGMA * measured_std, TACTILE_THRESHOLD_FLOOR).
TACTILE_BINARY          = False
TACTILE_THRESHOLD_FLOOR = np.array([0.02, 0.02, 0.02, 0.02], dtype=np.float32)
TACTILE_NOISE_SIGMA     = 1.5  # contact = deviation > 2× noise std at rest

# EMA smoothing factor for tactile sums (applied before thresholding).
# 0.1 = ~10-frame lag at 60 Hz — more smoothing reduces false triggers.
TACTILE_EMA_ALPHA = 0.1


# Number of tactile sensor values from /shadow_touchlab_translator/calibrated.
# ⚠️ VERIFY: run `rostopic echo /shadow_touchlab_translator/calibrated -n 1` and count values.
NUM_TACTILE = 4   # placeholder — update with real count
ff = 16
mf = 16
rf = 16
lf = 16
th = 16

PALM_LINK = 'rh_palm'    # fixed base of the hand
FF_TIP    = 'rh_fftip'   # first finger tip
MF_TIP    = 'rh_mftip'   # middle finger tip
RF_TIP    = 'rh_rftip'   # ring finger tip
TH_TIP    = 'rh_thtip'   # thumb tip

SUBSCRIBER_JOINT_ORDER = [
    "rh_FFJ1",   # index 0  — coupled to FFJ2 (J0 actuator)
    "rh_FFJ2",   # index 1  — policy controlling joint for FF coupling
    "rh_FFJ3",   # index 2
    "rh_FFJ4",   # index 3
    "rh_MFJ1",   # index 4  — coupled to MFJ2
    "rh_MFJ2",   # index 5  — policy controlling joint for MF coupling
    "rh_MFJ3",   # index 6
    "rh_MFJ4",   # index 7
    "rh_RFJ1",   # index 8  — coupled to RFJ2
    "rh_RFJ2",   # index 9  — policy controlling joint for RF coupling
    "rh_RFJ3",   # index 10
    "rh_RFJ4",   # index 11
    "rh_THJ1",   # index 12
    "rh_THJ2",   # index 13
    "rh_THJ4",   # index 14  — NOTE: THJ3 is absent (fixed joint)
    "rh_THJ5",   # index 15
]

# --- Policy ordering ---
# The order the neural network expects its 13 inputs/outputs.
# Confirmed from IsaacLab training config (inspect_shadowlite.py).
# J1 joints removed — policy uses J2 as the coupled representative.
POLICY_JOINT_ORDER = [
    "rh_FFJ4",   # policy index 0
    "rh_MFJ4",   # policy index 1
    "rh_RFJ4",   # policy index 2
    "rh_THJ5",   # policy index 3
    "rh_FFJ3",   # policy index 4
    "rh_MFJ3",   # policy index 5
    "rh_RFJ3",   # policy index 6
    "rh_THJ4",   # policy index 7
    "rh_FFJ2",   # policy index 8  — represents ffj0 (J2 is the controller in sim)
    "rh_MFJ2",   # policy index 9  — represents mfj0
    "rh_RFJ2",   # policy index 10 — represents rfj0
    "rh_THJ2",   # policy index 11
    "rh_THJ1",   # policy index 12
]

# --- Reshuffle map ---
# Maps: subscriber_index → policy_index
# For coupled joints, we use J2 (the controlling joint in sim), not J1 (the mimic).
# Example: FFJ2 is at subscriber index 1, and goes to policy index 8.
INDEX_RESHUFFLE_MAP = {
    3:  0,   # FFJ4: sub[3]  → policy[0]
    7:  1,   # MFJ4: sub[7]  → policy[1]
    11: 2,   # RFJ4: sub[11] → policy[2]
    15: 3,   # THJ5: sub[15] → policy[3]
    2:  4,   # FFJ3: sub[2]  → policy[4]
    6:  5,   # MFJ3: sub[6]  → policy[5]
    10: 6,   # RFJ3: sub[10] → policy[6]
    14: 7,   # THJ4: sub[14] → policy[7]
    1:  8,   # FFJ2: sub[1]  → policy[8]  (J2 = controller, J1 = mimic)
    5:  9,   # MFJ2: sub[5]  → policy[9]
    9:  10,  # RFJ2: sub[9]  → policy[10]
    13: 11,  # THJ2: sub[13] → policy[11]
    12: 12,  # THJ1: sub[12] → policy[12]
}



 
# =============================================================================
# SECTION 6: JOINT LIMITS — in policy order (13 values each)
# =============================================================================
# These must EXACTLY match what was used during training in IsaacLab.
# Used for: normalising observations (network input) and scaling actions (network output).
# Units: radians.
 
LOWER_LIMITS = np.array([
    -0.3491,   # FFJ4
    -0.3491,   # MFJ4
    -0.3491,   # RFJ4
    -1.0472,   # THJ5
    -0.2618,   # FFJ3
    -0.2618,   # MFJ3
    -0.2618,   # RFJ3
     0.0,      # THJ4
     0.0,      # FFJ2 (J0 representative — policy sees 0 to pi/2)
     0.0,      # MFJ2
     0.0,      # RFJ2
    -0.6981,   # THJ2
    -0.2618,   # THJ1
])
 
UPPER_LIMITS = np.array([
    0.3491,    # FFJ4
    0.3491,    # MFJ4
    0.3491,    # RFJ4
    1.0472,    # THJ5
    1.5708,    # FFJ3
    1.5708,    # MFJ3
    1.5708,    # RFJ3
    1.2217,    # THJ4
    1.745,    # FFJ2 (0 to pi/2 in policy space)
    1.745,    # MFJ2
    1.745,    # RFJ2
    0.6981,    # THJ2
    1.5708,    # THJ1
])
 
# For normalising observations — must match URDF limits exactly (sim used these)
VEL_LIMITS_NORM = np.array([
    2.0,   # FFJ4
    2.0,   # MFJ4
    2.0,   # RFJ4
    4.0,   # THJ5
    2.0,   # FFJ3
    2.0,   # MFJ3
    2.0,   # RFJ3
    4.0,   # THJ4
    2.0,   # FFJ2
    2.0,   # MFJ2
    2.0,   # RFJ2
    2.0,   # THJ2
    4.0,   # THJ1
])

# For hardware safety only — not used for normalisation
OVERRIDE_VEL_SCALE = 0.1   # keep this, use in controller if needed
 
# Safe default pose — all joints at zero = fully open hand.
# Change this if you want a different starting position.
#DEFAULT_JOINT_POS = np.zeros(13)
# Default (ball-holding) pose from simulation
DEFAULT_JOINT_POS = np.array([
    -0.349,  # rh_FFJ4
     0.0,    # rh_MFJ4
    -0.349,  # rh_RFJ4
     0.4,    # rh_THJ5
     0.65,   # rh_FFJ3
     0.65,   # rh_MFJ3
     0.65,   # rh_RFJ3
     0.5,    # rh_THJ4
     0.87,   # rh_FFJ2
     0.87,   # rh_MFJ2
     0.87,   # rh_RFJ2
     0.35,   # rh_THJ2
     0.0,    # rh_THJ1
], dtype=np.float32)


# =============================================================================
# COUPLED-FINGER FORWARD MODEL  (for the obs joint_pos_error ONLY)
# =============================================================================
# In sim, the J2 "command" the policy sees in joint_pos_error is NOT the raw
# scaled action — it is the coupling+backlash output `j2_cmd` produced by
# RotoEnv._handle_coupled_joints (theta split) + _asymmetric_backlash.
# We replicate that here so the coupled channels (FFJ2/MFJ2/RFJ2 = policy idx
# 8,9,10) of the error term match training. This drives ONLY the observation;
# the motor command is still ffj0 = 2*proxy (the hardware tendon does the real
# backlash). Constants come from the URDF / shadowlite cfg:
COUPLED_POLICY_IDX = np.array([8, 9, 10])                       # FFJ2, MFJ2, RFJ2
_THETA    = 0.8727                                              # coupling_theta (rad) = 50 deg
_J2_TOP   = np.array([1.745, 1.745, 1.745], dtype=np.float32)   # FFJ2 upper limit
_J1_SPAN  = np.array([1.396, 1.396, 1.396], dtype=np.float32)   # FFJ1 upper limit
# R = fixed per-finger backlash unlock angle. MUST match shadowlite.couple_release_deg
# (and the real hardware backlash). Only affects uncurl; curl is R-independent.
_R_RAD    = np.array([125.0, 140.0, 100.0], dtype=np.float32) * np.pi / 180.0 #FF/MF/RF
_DEADBAND = 0.002


class FingerCoupling:
    """Stateful numpy port of the sim J1<-J2 coupling. Given the policy proxy for
    the 3 coupled fingers, returns the J2 *command* (j2_cmd) the obs error expects.
    Carries per-finger state (prev_m, direction, j1, freeze) like the sim."""

    def __init__(self, theta, j2_top, j1_span, R_rad, deadband=0.002):
        self.theta = theta
        self.j2_top = j2_top
        self.j1_span = j1_span
        self.m_top = j2_top + j1_span
        self.R = R_rad
        self.db = deadband
        self.reset()

    def reset(self):
        z = np.zeros(3, dtype=np.float32)
        self.prev_m = z.copy()
        self.dir = np.ones(3, dtype=np.float32)   # +1 curl / -1 uncurl
        self.j1 = z.copy()
        self.frozen = np.zeros(3, dtype=bool)
        self.fval = z.copy()

    def j2_cmd(self, proxy):
        """proxy: (3,) scaled policy command for FFJ2/MFJ2/RFJ2 in [0, j2_top].
        Returns j2_cmd (3,) and advances the per-finger state once."""
        proxy = np.asarray(proxy, dtype=np.float32)
        eps = 1e-4
        # --- theta split (sim _handle_coupled_joints) -> j2_fwd, j1_fwd ---
        j2_fwd = np.clip(proxy * (self.j2_top / self.theta), 0.0, self.j2_top)
        j1_fwd = np.clip((proxy - self.theta) / (self.j2_top - self.theta) * self.j1_span,
                         0.0, self.j1_span)
        # --- backlash (sim _asymmetric_backlash) ---
        m = j2_fwd + j1_fwd
        delta = m - self.prev_m
        rising = delta > self.db
        falling = delta < -self.db
        new_dir = np.where(rising, 1.0, np.where(falling, -1.0, self.dir))
        l = np.clip(m - self.j2_top, 0.0, self.j1_span)
        flip_up = (self.dir < 0) & (new_dir > 0)
        enter = flip_up & (self.j1 < self.j1_span - eps) & (m < self.R)
        frozen = self.frozen | enter
        fval = np.where(enter, self.j1, self.fval)
        uncurling = new_dir < 0
        frozen = frozen & ~uncurling
        denom = np.clip(self.m_top - self.R, eps, None)
        resume = np.clip(fval + (m - self.R) / denom * (self.j1_span - fval), fval, self.j1_span)
        j1_frozen = np.where(m >= self.R, resume, fval)
        j1 = np.where(frozen, j1_frozen, l)
        frozen = frozen & ~(j1 >= l - eps)
        j2_down = np.clip(m / self.R * self.j2_top, 0.0, self.j2_top)   # unlock at R (uncurl)
        j2_fresh = np.clip(m, 0.0, self.j2_top)                          # saturate at 100° (curl)
        j2 = np.where(uncurling | frozen, j2_down, j2_fresh)
        # advance state
        self.dir = new_dir.astype(np.float32)
        self.prev_m = m.astype(np.float32)
        self.j1 = j1.astype(np.float32)
        self.frozen = frozen
        self.fval = fval.astype(np.float32)
        return j2.astype(np.float32)


_coupling = FingerCoupling(_THETA, _J2_TOP, _J1_SPAN, _R_RAD, _DEADBAND)

 
# =============================================================================
# SECTION 7: SHARED GLOBAL STATE
# =============================================================================
# These variables are written by the callback thread and read by the main thread.
# They must be protected by a Lock to prevent race conditions.
 
joint_pos      = None   # raw joint positions in policy order (13 values)
joint_pos_norm = None   # normalised to [-1, 1]
joint_vel_norm = None   # normalised velocity


# Raw tactile readings — updated by tactile_callback
tactile_raw            = None   # binarized (0/1), fed to policy
tactile_sum_raw        = None   # baseline-subtracted EMA deviation, shown in dashboard
tactile_ema            = None   # EMA of raw finger sums (internal smoothing state)
tactile_baseline       = None   # per-finger resting baseline, set before each episode
tactile_threshold_live = None   # per-finger threshold = max(k*std, floor), set at calibration

# Circular buffers holding the last OBS_STACK observations.
# deque(maxlen=N) automatically drops the oldest when you append a new one.
# Initialised with zeros — policy sees "no movement, no contact" at episode start.
_prop_buffer    = deque(maxlen=OBS_STACK)
_tactile_buffer = deque(maxlen=OBS_STACK)

def _init_obs_buffers():
    """Fill both buffers with zeros so we never read from an empty deque."""
    for _ in range(OBS_STACK):
        _prop_buffer.append(np.zeros(52))              # 52 = one prop frame
        _tactile_buffer.append(np.zeros(NUM_TACTILE))  # one tactile frame
 
# The Lock acts like a traffic warden — only one thread reads/writes at a time.
data_lock = Lock()


def reshuffle_data(data_list, index_mapping_dict):
    """
    Reorders a flat list from subscriber order to policy order.
 
    Why needed: /joint_states publishes in hardware order (FFJ1, FFJ2, FFJ3...)
    but the policy expects a completely different order (FFJ4, MFJ4, RFJ4...).
 
    Args:
        data_list         : list of 16 floats (joint positions or velocities)
        index_mapping_dict: dict mapping {subscriber_index: policy_index}
 
    Returns:
        list of 13 floats in policy order, or None where no mapping exists
    """
    if not data_list or not index_mapping_dict:
        return list(data_list)
 
    max_new_index = max(index_mapping_dict.values())
    reshuffled = [None] * (max_new_index + 1)
 
    for old_idx, new_idx in index_mapping_dict.items():
        if 0 <= old_idx < len(data_list):
            reshuffled[new_idx] = data_list[old_idx]
        else:
            rospy.logwarn("reshuffle_data: index {} out of bounds".format(old_idx))
 
    return reshuffled

def normalise(x, lower, upper):
    """
    Maps joint values from [lower, upper] → [-1, 1].
    Neural networks learn best when all inputs are on the same scale.
    At lower limit → -1.0. At upper limit → +1.0. At midpoint → 0.0.
    """
    return (2.0 * x - upper - lower) / (upper - lower)

def scale(x, lower, upper):
    """
    Inverse of normalise — maps policy output from [-1, 1] → [lower, upper].
    Policy outputs tanh-bounded values in [-1, 1].
    This converts them back to real joint angle space (radians).
    """
    return 0.5 * (x + 1.0) * (upper - lower) + lower

# =============================================================================
# SECTION 9: CALLBACK FUNCTION
# =============================================================================
 
def prop_callback(data):
    """
    Called automatically by ROS every time a new JointState message arrives
    on /joint_states (100 Hz).
 
    This runs in a BACKGROUND THREAD managed by rospy — not your main loop.
    It silently updates the global joint_pos_norm and joint_vel_norm variables
    so the main loop always has fresh data to read.
 
    The Lock ensures the main loop never reads half-updated data.
 
    Args:
        data : sensor_msgs/JointState
               data.name     → list of joint name strings
               data.position → list of joint positions (radians)
               data.velocity → list of joint velocities (rad/s)
    """
    global joint_pos, joint_pos_norm, joint_vel_norm
    with data_lock:
        joint_pos = np.array(
            reshuffle_data(list(data.position), INDEX_RESHUFFLE_MAP)
        )
        joint_vel = np.array(
            reshuffle_data(list(data.velocity), INDEX_RESHUFFLE_MAP)
        )
        joint_pos_norm = normalise(joint_pos, LOWER_LIMITS, UPPER_LIMITS)

        # FIX: use raw URDF vel limits for normalisation, not scaled ones
        joint_vel_norm = normalise(joint_vel, -VEL_LIMITS_NORM, VEL_LIMITS_NORM)


def tactile_callback(msg):
    global tactile_raw, tactile_sum_raw, tactile_ema

    raw = np.array(list(msg.data), dtype=np.float32)

    if len(raw) == 240:
        raw = raw[2::3]      # triplet-packed → 80 values
    elif len(raw) != 80:
        rospy.logwarn_throttle(
            5.0,
            "Unexpected tactile length: {}".format(len(raw))
        )
        return

    # Sum 16 taxels per finger, 4 real fingers (skip lf at 48-63)
    finger_sums = np.array([
        raw[0:16].sum(),    # ff
        raw[16:32].sum(),   # mf
        raw[32:48].sum(),   # rf
        raw[64:80].sum(),   # th
    ], dtype=np.float32)

    with data_lock:
        # EMA smoothing
        if tactile_ema is None:
            tactile_ema = finger_sums.copy()
        else:
            tactile_ema = TACTILE_EMA_ALPHA * finger_sums + (1.0 - TACTILE_EMA_ALPHA) * tactile_ema

        # Baseline subtraction — deviation from resting state
        baseline = tactile_baseline if tactile_baseline is not None else np.zeros(NUM_TACTILE, dtype=np.float32)
        deviation = tactile_ema - baseline

        thresh = tactile_threshold_live if tactile_threshold_live is not None                  else TACTILE_THRESHOLD_FLOOR
        if TACTILE_BINARY:
            finger_binary = (deviation > thresh).astype(np.float32)
        else:
            finger_binary = deviation

        tactile_raw     = finger_binary   # binarized → policy
        tactile_sum_raw = deviation       # deviation from baseline → dashboard
 
def get_proprioception(cur_targets_radians, prev_actions_raw):
    """
    Builds the stacked observation vector the policy expects.

    Structure per frame (52 values):
        normalised_joint_pos  (13)
        normalised_joint_vel  (13)
        joint_pos_error       (13)  ← cmd - actual, in radians
        prev_actions_raw      (13)  ← raw policy output from last step [-1, 1]

    With OBS_STACK=4, total prop = 52 × 4 = 208 values.
    With tactile stacked: total tactile = NUM_TACTILE × 4 values.

    The buffer always has OBS_STACK frames. At each step:
        1. Build this frame from current sensor data
        2. Append to buffer (oldest auto-dropped by deque)
        3. Concatenate all frames → stacked obs

    Args:
        cur_targets_radians : 13-element array, current command in radians
        prev_actions_raw    : 13-element array, last policy output in [-1, 1]
    
    Returns:
        prop_stacked    : torch.Tensor shape (52 * OBS_STACK,)
        tactile_stacked : torch.Tensor shape (NUM_TACTILE * OBS_STACK,)
                          or zeros if USE_TACTILE is False
    """
    # --- Read sensor data under lock ---
    with data_lock:
        pos_norm   = joint_pos_norm.copy()
        vel_norm   = joint_vel_norm.copy()
        actual_pos = joint_pos.copy()
        tac        = tactile_raw.copy() if (USE_TACTILE and tactile_raw is not None) \
                     else np.zeros(NUM_TACTILE)

    # --- Build current prop frame ---
    joint_pos_error = cur_targets_radians - actual_pos   # cmd - actual (radians)

    current_prop_frame = np.concatenate([
        pos_norm,           # 13: where joints are
        vel_norm,           # 13: how fast moving
        joint_pos_error,    # 13: tracking error
        prev_actions_raw,   # 13: what policy output last step
    ])   # shape: (52,)

    # --- Push into circular buffers ---
    _prop_buffer.append(current_prop_frame)
    _tactile_buffer.append(tac)

    # --- Stack all OBS_STACK frames ---
    # deque preserves order: oldest first, newest last
    # np.concatenate flattens them into one long vector
    prop_stacked    = np.concatenate(list(_prop_buffer))       # (52 × OBS_STACK,)
    tactile_stacked = np.concatenate(list(_tactile_buffer))    # (NUM_TACTILE × OBS_STACK,)

    return (
        torch.tensor(prop_stacked,    dtype=torch.float32),
        torch.tensor(tactile_stacked, dtype=torch.float32),
    )

def create_hand_publishers():
    """
    Creates one ROS publisher per Shadow Hand controller.
 
    Unlike Torobo (one JointTrajectory topic for all joints),
    Shadow Hand needs a separate Float64 publisher per joint controller.
 
    The 13 controller names are confirmed from:
      rostopic list | grep position_controller
 
    Returns:
        dict mapping controller_name → rospy.Publisher
    """
    # These 13 names match exactly what rostopic list showed
    controller_names = [
        "ffj0",   # controls FFJ1 + FFJ2 together (range 0 to pi)
        "ffj3",
        "ffj4",
        "mfj0",   # controls MFJ1 + MFJ2 together
        "mfj3",
        "mfj4",
        "rfj0",   # controls RFJ1 + RFJ2 together
        "rfj3",
        "rfj4",
        "thj1",
        "thj2",
        "thj4",
        "thj5",
    ]
 
    publishers = {}
    for name in controller_names:
        topic = "/sh_rh_{}_position_controller/command".format(name)
        pub = rospy.Publisher(topic, Float64, queue_size=1)
        publishers[name] = pub
        rospy.loginfo("Created publisher: {}".format(topic))
 
    return publishers


def check_publishers_connected(publishers, timeout_secs=5.0):
    """
    Verifies that publishers have subscribers (i.e., controllers are listening).
    Waits up to timeout_secs for at least one subscriber per publisher.
    
    Args:
        publishers   : dict from create_hand_publishers()
        timeout_secs : max time to wait for connections
    
    Returns:
        True if all publishers have at least one subscriber, False otherwise
    """
    start_time = rospy.get_time()
    
    while rospy.get_time() - start_time < timeout_secs:
        all_connected = True
        for name, pub in publishers.items():
            if pub.get_num_connections() == 0:
                all_connected = False
                break
        
        if all_connected:
            rospy.loginfo("✓ All publishers connected to controllers!")
            return True
        
        rospy.loginfo("Waiting for controllers to connect... ({} subscribers seen)".format(
            sum(pub.get_num_connections() for pub in publishers.values())
        ))
        rospy.sleep(0.5)
    
    rospy.logwarn("⚠ Timeout waiting for publishers to connect.")
    rospy.logwarn("Controller connections:")
    for name, pub in publishers.items():
        rospy.logwarn("  {}: {} subscriber(s)".format(name, pub.get_num_connections()))
    
    return False
 

def publish_to_hand(publishers, actions_policy_order):
    """
    Sends joint position commands to all 13 Shadow Hand controllers.
 
    Key: J0 SCALING FIX
    In simulation, FFJ2 controlled the curl (0 to pi/2) and FFJ1 was a mimic.
    On real hardware, ffj0 = FFJ1 + FFJ2 combined (range 0 to pi).
    So: ffj0_command = 2.0 × policy_action_for_FFJ2
 
    This doubles the coupled joint outputs to account for the mimic being
    physically realised as a summed actuator on the real hand.
 
    Args:
        publishers          : dict from create_hand_publishers()
        actions_policy_order: 13-element numpy array in POLICY_JOINT_ORDER
    """
    # Map policy outputs to controller commands
    # Policy order: FFJ4(0), MFJ4(1), RFJ4(2), THJ5(3),
    #               FFJ3(4), MFJ3(5), RFJ3(6), THJ4(7),
    #               FFJ2(8), MFJ2(9), RFJ2(10), THJ2(11), THJ1(12)
    commands = {
        # Independent joints — direct 1:1 mapping
        "ffj4": float(actions_policy_order[0]),
        "mfj4": float(actions_policy_order[1]),
        "rfj4": float(actions_policy_order[2]),
        "thj5": float(actions_policy_order[3]),
        "ffj3": float(actions_policy_order[4]),
        "mfj3": float(actions_policy_order[5]),
        "rfj3": float(actions_policy_order[6]),
        "thj4": float(actions_policy_order[7]),
        "thj2": float(actions_policy_order[11]),
        "thj1": float(actions_policy_order[12]),
 
        # Coupled joints — multiply by 2.0 to account for mimic
        # Policy thinks: FFJ2 = x (FFJ1 mirrors it silently in sim)
        # Real hardware: ffj0 = FFJ1 + FFJ2 = x + x = 2x
        "ffj0": 2.0 * float(actions_policy_order[8]),
        "mfj0": 2.0 * float(actions_policy_order[9]),
        "rfj0": 2.0 * float(actions_policy_order[10]),
    }
 
    # Publish each command as a Float64 message
    for name, value in commands.items():
        msg = Float64()
        msg.data = value
        publishers[name].publish(msg)
 
 
# def publish_default_pose(publishers, duration_secs=3.0, rate_hz=10):
#     """
#     Sends all joints to the open-hand default position, repeatedly.
#     ROS messages are fire-and-forget, so we must republish many times to
#     ensure the hardware controller actually receives the command.
    
#     Args:
#         publishers   : dict from create_hand_publishers()
#         duration_secs: how long to keep publishing (default 3 seconds)
#         rate_hz      : how often to publish per second (default 10 Hz)
#     """
#     rospy.loginfo("Moving to default open-hand pose (publishing for {} seconds)...".format(duration_secs))
    
#     # Calculate how many times to publish
#     num_publishes = int(duration_secs * rate_hz)
#     publish_rate = rospy.Rate(hz=rate_hz)
    
#     for i in range(num_publishes):
#         for name in publishers:
#             msg = Float64()
#             msg.data = 0.0   # zero = fully open for all joints
#             publishers[name].publish(msg)
        
#         publish_rate.sleep()
        
#         if i % 10 == 0:
#             rospy.loginfo("  Publishing default pose... ({}/{})".format(i, num_publishes))
    
#     rospy.loginfo("Default pose publishing complete.")

def wait_for_ball_placement(duration_s=4.0):
    """
    Hold default pose and count down so the user can place the ball.
    """
    rospy.loginfo(
        "\n{}\n"
        "  PLACE THE BALL NOW  —  holding default pose for {:.0f}s\n"
        "{}".format("="*60, duration_s, "="*60)
    )

    for remaining in range(int(duration_s), 0, -1):
        if rospy.is_shutdown():
            break

        rospy.loginfo("  Starting in {}s ...".format(remaining))
        rospy.sleep(1.0)

    rospy.loginfo("Starting policy!")


def publish_default_pose(publishers, duration_secs=3.0):
    """
    Move all joints to the Baoding-ball holding pose.
    """
    rospy.loginfo(
        "Moving to default (ball-holding) pose for {}s ...".format(duration_secs)
    )

    rate = rospy.Rate(10)
    n = int(duration_secs * 10)

    for _ in range(n):
        publish_to_hand(publishers, DEFAULT_JOINT_POS)
        rate.sleep()

    rospy.loginfo("Default pose reached.")

# =============================================================================
# LIVE TACTILE DASHBOARD
# =============================================================================

class LiveTactilePlot:
    """Bar-chart popup showing continuous fingertip sum values with threshold lines."""

    _LABELS = ["FF", "MF", "RF", "TH"]
    _COLORS = ["steelblue", "seagreen", "tomato", "darkorange"]

    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        x = np.arange(4)
        self.bars = self.ax.bar(x, np.zeros(4), color=self._COLORS)
        thresh = tactile_threshold_live if tactile_threshold_live is not None \
                 else TACTILE_THRESHOLD_FLOOR
        self._thresh_lines = [
            self.ax.hlines(thresh[i], i - 0.4, i + 0.4,
                           colors="red", linestyles="dashed", linewidth=2)
            for i in range(4) if thresh[i] > 0
        ]
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(self._LABELS, fontsize=12)
        self.ax.set_ylim(0, max(float(thresh.max()) * 3, 1.0))
        self.ax.set_ylabel("Deviation from baseline")
        self.ax.set_title("Fingertip Tactile  —  red = contact threshold")
        self.fig.tight_layout()
        plt.pause(0.05)

    def update(self, finger_sums, step, total):
        try:
            display = np.maximum(finger_sums, 0.0)   # clip negatives — shown in terminal logs
            for bar, val in zip(self.bars, display):
                bar.set_height(float(val))
            thresh = tactile_threshold_live if tactile_threshold_live is not None \
                     else TACTILE_THRESHOLD_FLOOR
            peak = max(float(display.max()), float(thresh.max()), 0.1)
            self.ax.set_ylim(0, peak * 1.4)
            self.ax.set_title(
                "Fingertip Tactile  |  step {}/{}  (red = threshold)".format(step, total)
            )
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            pass

    def close(self):
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass
        try:
            plt.close('all')   # destroys Tk root so it can't steal focus
        except Exception:
            pass
        try:
            # Force Tk to finish processing any remaining events
            self.fig.canvas.get_tk_widget().destroy()
        except Exception:
            pass


# =============================================================================
# SECTION 13: MAIN POLICY LOOP
# =============================================================================


_stop_event    = threading.Event()
_episode_active = threading.Event()   # watcher only reads stdin during an episode

def _stdin_watcher():
    """Background thread: polls stdin only while an episode is active.
    Outside episodes _episode_active is clear so we never compete with input()."""
    while True:
        _episode_active.wait()          # sleep until episode starts
        ready = select.select([sys.stdin], [], [], 0.05)[0]
        if not ready:
            continue
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if line.strip().lower() == "s":
            _stop_event.set()

_stdin_thread = threading.Thread(target=_stdin_watcher, daemon=True)
_stdin_thread.start()

def calibrate_tactile_baseline(n_frames=90, rate_hz=30):
    """Sample tactile EMA at rest: compute per-finger mean (baseline) and std (noise).
    Live contact threshold = max(TACTILE_NOISE_SIGMA * std, TACTILE_THRESHOLD_FLOOR).
    """
    global tactile_baseline, tactile_threshold_live
    rospy.loginfo("Calibrating tactile baseline ({} frames at rest)...".format(n_frames))
    samples = []
    rate = rospy.Rate(rate_hz)
    for _ in range(n_frames):
        with data_lock:
            if tactile_ema is not None:
                samples.append(tactile_ema.copy())
        rate.sleep()
    if not samples:
        rospy.logwarn("No tactile data during calibration — using zero baseline.")
        baseline  = np.zeros(NUM_TACTILE, dtype=np.float32)
        threshold = TACTILE_THRESHOLD_FLOOR.copy()
    else:
        arr       = np.array(samples, dtype=np.float32)          # (N, 4)
        baseline  = arr.mean(axis=0)
        noise_std = arr.std(axis=0)
        threshold = np.maximum(TACTILE_NOISE_SIGMA * noise_std,
                               TACTILE_THRESHOLD_FLOOR).astype(np.float32)
    rospy.loginfo("Tactile baseline  (FF MF RF TH): {}".format(np.round(baseline, 3).tolist()))
    rospy.loginfo("Tactile threshold (FF MF RF TH): {}".format(np.round(threshold, 3).tolist()))
    with data_lock:
        tactile_baseline       = baseline
        tactile_threshold_live = threshold


def rl_policy_loop(speed: float = 1.0, no_plot: bool = False):
    """
    Main function. Does everything in this order:
      1. Load neural network weights
      2. Initialise ROS node
      3. Create publishers (one per controller)
      4. Subscribe to joint states
      5. Wait for first sensor data
      6. Loop: move to default → run episode → repeat
    """
 
    torch.set_default_dtype(torch.float32)
 
    # ------------------------------------------------------------------
    # 13.1 DEFINE OBSERVATION AND ACTION SPACES
    # ------------------------------------------------------------------
    # num_prop = 13 joints × 4 (pos, vel, cur_target, prev_target) = 52
    num_prop         = 52 * OBS_STACK                          # 208 with stack=4
    num_tactile      = NUM_TACTILE * OBS_STACK if USE_TACTILE else 0
    num_actions      = 13

    observation_space = {
        "prop": np.zeros(num_prop),
    }
    if USE_TACTILE:
        observation_space["tactile"] = np.zeros(num_tactile)
    action_space = np.zeros(num_actions)
 
    # ------------------------------------------------------------------
    # 13.2 BUILD NETWORK ARCHITECTURE
    # ------------------------------------------------------------------
    # These configs must EXACTLY match what was used during IsaacLab training.
    # The encoder compresses the observation into a latent vector z.
    # The policy takes z and outputs joint position targets.
 
    encoder_cfg = {
    "encoder": {
        "method":             "early",   # no vision, so early fusion = simple concatenation
        "layernorm":          True,
        "state_preprocessor": None,
        "hiddens":            [1024, 512, 256],
        "activations":        ["elu", "elu", "elu"],
        #"latent_state_dim":   64,        # only used for intermediate, but avoids KeyError
    },
    
}
 
    policy_cfg = {
        "clip_log_std":   True,
        "initial_log_std": 0,
        "min_log_std":    -20.0,
        "max_log_std":     2.0,
        "hiddens":        [128, 64],
        "activations":    ["elu", "elu", "identity"]  # identity = no squashing on output
    }
 
    encoder = Encoder(
        observation_space,
	action_space,
	{},
        encoder_cfg,
        device=device
    )
    policy = GaussianPolicy(
        z_dim=encoder.num_outputs,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        **policy_cfg
    )
 
    rospy.loginfo("Encoder architecture:\n{}".format(encoder))
    rospy.loginfo("Policy architecture:\n{}".format(policy))
 
    # ------------------------------------------------------------------
    # 13.3 LOAD TRAINED WEIGHTS
    # ------------------------------------------------------------------
    # torch.load reads the .pt checkpoint file saved by IsaacLab trainer.
    # map_location="cpu" ensures it works even without a GPU.
    rospy.loginfo("Loading checkpoint from: {}".format(CHECKPOINT_PATH))
    modules = torch.load(CHECKPOINT_PATH, map_location=device)
 
    if isinstance(modules, dict):
        rospy.loginfo("Checkpoint keys: {}".format(list(modules.keys())))
 
    encoder.load_state_dict(modules["encoder"])
    encoder = encoder.to(device)
    encoder.eval()   # disables dropout etc — important for inference
 
    policy.load_state_dict(modules["policy"])
    policy.eval()
 
    rospy.loginfo("Checkpoint loaded successfully.")
 
    # ------------------------------------------------------------------
    # 13.4 INITIALISE ROS NODE
    # ------------------------------------------------------------------
    # This MUST come before any Publisher or Subscriber creation.
    # It registers this Python process with the ROS Master as a named node.
    # anonymous=True appends a random suffix so you can run multiple copies.
    rospy.init_node('rl_policy_node', anonymous=True)
 
    # rospy.Rate controls loop timing — sleep() will pause until next 10Hz tick
    effective_hz = max(1.0, RL_HZ * speed)
    episode_steps = int(round(EPISODE_TIMESTEPS / speed))
    rate = rospy.Rate(hz=effective_hz)
    rospy.loginfo("Speed={:.2f}  effective_hz={:.1f}  episode_steps={}".format(
        speed, effective_hz, episode_steps))
 
    # ------------------------------------------------------------------
    # 13.5 CREATE PUBLISHERS — one per Shadow Hand controller
    # ------------------------------------------------------------------
    publishers = create_hand_publishers()
 
    # Give publishers time to connect to subscribers (controllers) on the other end
    # This is critical — without sufficient time, early messages are dropped.
    rospy.loginfo("Waiting for publishers to fully connect to controllers (3 seconds)...")
    for i in range(3):
        rospy.sleep(1.0)
        rospy.loginfo("  ... waiting ({}/3)".format(i + 1))
    
    # Verify connection before proceeding
    if not check_publishers_connected(publishers, timeout_secs=10.0):
        rospy.logwarn("Publishers may not be fully connected. Proceeding anyway...")
    
    rospy.loginfo("Publishers ready. Sleeping another 2 seconds before first command...")
    rospy.sleep(2.0)
 
    # ------------------------------------------------------------------
    # 13.6 CREATE SUBSCRIBER — reads joint state from the real hand
    # ------------------------------------------------------------------
    # rospy.Subscriber(topic, message_type, callback_function)
    # Every time a JointState message arrives on /joint_states,
    # ROS calls prop_callback in a background thread automatically.
    rospy.Subscriber("/joint_states", JointState, prop_callback)
    rospy.loginfo("Subscribed to /joint_states")

    # Tactile subscriber — only register if USE_TACTILE is enabled
    if USE_TACTILE:
        rospy.Subscriber(
            "/shadow_touchlab_translator/calibrated_flat",   
            Float64MultiArray,                 
            tactile_callback
        )
        rospy.loginfo("Subscribed to /shadow_touchlab_translator/calibrated_flat")
    else:
        rospy.loginfo("Tactile disabled — skipping touchlab subscriber")


    # Initialise obs buffers with zeros before episode starts
    _init_obs_buffers()
    rospy.loginfo("Obs buffers initialised ({} frames × {} prop + {} tactile)".format(
        OBS_STACK, 52, NUM_TACTILE if USE_TACTILE else 0
    ))
    
    # ------------------------------------------------------------------
    # 13.7 WAIT FOR FIRST SENSOR DATA
    # ------------------------------------------------------------------
    # The subscriber is registered but the first message hasn't arrived yet.
    # Trying to build an observation from None would crash immediately.
    # This loop waits until prop_callback has run at least once.
    rospy.loginfo("Waiting for first joint state message...")
    while not rospy.is_shutdown():
        with data_lock:
            prop_ready    = joint_pos_norm is not None
            tactile_ready = (tactile_raw is not None) if USE_TACTILE else True
            if prop_ready and tactile_ready:
                break
        rospy.loginfo("Waiting for sensor data... prop={} tactile={}".format(
            prop_ready, tactile_ready if USE_TACTILE else "N/A"
        ))
        rate.sleep()
 
    rospy.loginfo("First joint state received. Ready to run policy.")
 
    # ------------------------------------------------------------------
    # 13.8 INITIALISE TARGETS
    # ------------------------------------------------------------------
    # Both cur and prev start at the default open pose.
    # These are in POLICY space (radians, not normalised).
    cur_targets  = deepcopy(DEFAULT_JOINT_POS)
    prev_targets = deepcopy(DEFAULT_JOINT_POS)
    
    # Initialise action history for observation building
    prev_actions_raw = np.zeros(13)   # raw policy output in [-1, 1] space
 
    # ------------------------------------------------------------------
    # 13.9 MAIN EPISODE LOOP
    # ------------------------------------------------------------------
    while not rospy.is_shutdown():
 
        # --- User confirmation (matches run_simgap_hardware pattern) ---
        user_input = input("\nPress y to run policy, n to exit: ")
        if user_input.strip().lower() != "y":
            rospy.loginfo("Exiting.")
            break

        # --- Move to ball-holding pose, then count down for ball placement ---
        rospy.loginfo("="*60)
        rospy.loginfo("EPISODE START: Moving to default (ball-holding) pose")
        rospy.loginfo("="*60)
        publish_default_pose(publishers, duration_secs=3.0)
        if USE_TACTILE:
            calibrate_tactile_baseline()
        wait_for_ball_placement(duration_s=5.0)

        # --- Open live tactile dashboard ---
        tac_plot = None
        if USE_TACTILE and _MATPLOTLIB_OK and not no_plot:
            tac_plot = LiveTactilePlot()
        elif USE_TACTILE and not no_plot:
            rospy.logwarn("matplotlib unavailable — tactile dashboard disabled")

        rospy.loginfo("Running episode for {} steps at {:.1f} Hz...".format(episode_steps, effective_hz))
        rospy.loginfo("  Press 's' + Enter at any time to stop the episode early.")

        # Reset the coupled-finger forward model and seed the obs command for J2.
        # cur_targets_obs mirrors cur_targets but with FFJ2/MFJ2/RFJ2 replaced by the
        # backlash-mapped j2_cmd (what the policy's joint_pos_error saw in sim).
        _coupling.reset()
        cur_targets_obs = cur_targets.copy()
        cur_targets_obs[COUPLED_POLICY_IDX] = _coupling.j2_cmd(cur_targets[COUPLED_POLICY_IDX])

        # --- Run one episode ---
        _episode_active.set()    # allow watcher thread to poll stdin
        _stop_episode = False
        for t in range(episode_steps):
 
            # Step 1: Read latest sensor data (written by callback thread)
            # The lock ensures we get a consistent snapshot, not a half-update.
            with data_lock:
                pos_norm = joint_pos_norm.copy()
                vel_norm = joint_vel_norm.copy()
 
            # Step 2: Build observation vector (cur_targets_obs has j2_cmd in the coupled slots)
            prop_tensor, tactile_tensor = get_proprioception(cur_targets_obs, prev_actions_raw)
            obs = {"prop": prop_tensor}
            if USE_TACTILE:
                obs["tactile"] = tactile_tensor
 
            # Step 3: Run neural network — encoder compresses obs, policy outputs action
            with torch.no_grad():   # no_grad = don't compute gradients (saves memory)
                z = encoder(obs)#.T
                # deterministic=True uses the mean action, not a random sample
                actions = policy.act(z, deterministic=True)[0][0]
                actions = actions.detach().cpu().numpy()
 
            # Step 4: Scale from [-1, 1] back to joint angle space (radians)
            cur_targets = scale(actions, LOWER_LIMITS, UPPER_LIMITS)
 
            # Step 5: Smooth — blend new target with previous target
            # This prevents sudden jumps even if the policy output changes sharply.
            cur_targets = (
                ACTION_TAU * cur_targets
                + (1.0 - ACTION_TAU) * prev_targets
            )
 
            # Step 6: Hard safety clip — cannot exceed URDF joint limits
            cur_targets = np.clip(cur_targets, LOWER_LIMITS, UPPER_LIMITS)
 
            # Step 7: Send to real hardware
            # This is where the J0 × 2 scaling happens inside publish_to_hand.
            publish_to_hand(publishers, cur_targets)
 
            # Step 8: Sleep until next 10Hz tick
            rate.sleep()

            # Step 9: Check for mid-episode stop request
            if _stop_event.is_set():
                _stop_event.clear()
                rospy.loginfo("Stop requested — ending episode at step {}/{}.".format(t, episode_steps))
                _stop_episode = True
                break
 
            # Step 10: Remember what we just commanded
            prev_targets = cur_targets.copy()
            prev_actions_raw = actions.copy()

            # Advance the coupling model from the published proxy and rebuild the obs
            # command: coupled slots = backlash-mapped j2_cmd (matches sim's
            # joint_pos_cmd[FFJ2]); all other joints unchanged. Used next step's obs.
            cur_targets_obs = cur_targets.copy()
            cur_targets_obs[COUPLED_POLICY_IDX] = _coupling.j2_cmd(cur_targets[COUPLED_POLICY_IDX])
 
            if t % 6 == 0:   # ~10 Hz refresh at 60 Hz policy
                with data_lock:
                    tac_vals = tactile_sum_raw.copy() if tactile_sum_raw is not None \
                               else None
                if tac_plot is not None and tac_vals is not None:
                    tac_plot.update(tac_vals, t, episode_steps)
                rospy.loginfo("  Step {:3d}/{}  tactile={}".format(
                    t, episode_steps,
                    "NO DATA" if tac_vals is None else np.round(tac_vals, 2).tolist()
                ))

        _episode_active.clear()  # stop watcher thread from touching stdin
        rospy.loginfo("Episode complete.")

        if tac_plot is not None:
            tac_plot.close()

        # --- Return to default pose after episode (matches run_simgap_hardware) ---
        publish_default_pose(publishers, duration_secs=2.0)

        with data_lock:
            final_pos = joint_pos.copy() if joint_pos is not None else None

            if final_pos is not None:
                rospy.loginfo("=" * 60)
                rospy.loginfo("FINAL JOINT POSITIONS (actual hardware readings):")
                rospy.loginfo("=" * 60)
                for i, name in enumerate(POLICY_JOINT_ORDER):
                    rospy.loginfo("  {:12s} : {:.4f} rad  ({:.2f} deg)".format(
                        name, final_pos[i], np.degrees(final_pos[i])
                    ))
                rospy.loginfo("=" * 60)
            else:
                rospy.logwarn("No joint position data available to report.")
 
        # --- Ask whether to run again ---
        again = input("Run again? (y/n): ")
        if again.strip().lower() != "y":
            break
 
    rospy.loginfo("RL Policy Node shutting down.")
 
 
# =============================================================================
# SECTION 14: ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument('--speed', type=float, default=SPEED,
                     help='Replay speed fraction (default: 1.0, e.g. 0.5 = half speed)')
    _ap.add_argument('--no-plot', action='store_true',
                     help='Disable live tactile graph (avoids Tkinter focus issues)')
    _args, _ros_args = _ap.parse_known_args()
    try:
        rl_policy_loop(speed=_args.speed, no_plot=_args.no_plot)
    except rospy.ROSInterruptException:
        # Raised when Ctrl+C is pressed — clean shutdown
        rospy.loginfo("Interrupted. Shutting down.")
    except Exception as e:
        rospy.logerr("Unexpected error: {}".format(e))
        raise