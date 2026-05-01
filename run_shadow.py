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
import numpy as np
from threading import Lock
import sys
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
 
import warnings
warnings.filterwarnings("ignore")
 
 
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
RL_HZ = 10
 
# Low-pass filter on actions — prevents jerky motion.
# New command = 1% new + 99% old. Small = very smooth but slow to respond.
ACTION_TAU = 0.01
 
# Velocity scaling factor applied to all VEL_LIMITS.
# 0.1 = run at 10% of max hardware velocity — safe for initial testing.
OVERRIDE_VEL_SCALE = 0.1
 
# How many policy steps to run per episode (10 Hz × 100 steps = 10 seconds).
EPISODE_TIMESTEPS = 100
 
# Path to the saved neural network weights from IsaacLab training.
CHECKPOINT_PATH = os.path.join(
    "/root/shadow_docker_ws/src/my_shadow_control/scripts",
    "awesome.pt"
)


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
    1.5708,    # FFJ2 (0 to pi/2 in policy space)
    1.5708,    # MFJ2
    1.5708,    # RFJ2
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
DEFAULT_JOINT_POS = np.zeros(13)
 
# =============================================================================
# SECTION 7: SHARED GLOBAL STATE
# =============================================================================
# These variables are written by the callback thread and read by the main thread.
# They must be protected by a Lock to prevent race conditions.
 
joint_pos      = None   # raw joint positions in policy order (13 values)
joint_pos_norm = None   # normalised to [-1, 1]
joint_vel_norm = None   # normalised velocity
 
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
 
def get_proprioception(cur_targets_radians, prev_actions_raw):
    """
    Builds the 52-element observation vector matching sim exactly.

    Args:
        cur_targets_radians : current joint position command in radians (13 values)
                              Used to compute position error vs actual joint pos.
        prev_actions_raw    : previous policy output (13 values)
                              This is self.actions in the sim — raw network output
                              BEFORE scaling to radians.

    Observation structure (must match sim exactly):
        [normalised_joint_pos (13),    ← where joints actually are, scaled to [-1,1]
         normalised_joint_vel (13),    ← how fast joints are moving, scaled to [-1,1]
         joint_pos_error (13),         ← command - actual, in radians (NOT normalised)
         prev_actions_raw (13)]        ← last policy output
    """
    with data_lock:
        pos_norm = joint_pos_norm.copy()   # already normalised in callback
        vel_norm = joint_vel_norm.copy()   # already normalised in callback
        actual_pos = joint_pos.copy()      # raw radians

    # joint_pos_error = commanded position - actual position (radians)
    # Matches sim: self.joint_pos_error = self.joint_pos_cmd - self.joint_pos
    joint_pos_error = cur_targets_radians - actual_pos

    prop = torch.cat((
        torch.tensor(pos_norm,          dtype=torch.float32),   # 13
        torch.tensor(vel_norm,          dtype=torch.float32),   # 13
        torch.tensor(joint_pos_error,   dtype=torch.float32),   # 13
        torch.tensor(prev_actions_raw,  dtype=torch.float32),   # 13
    ))
    return prop   # shape: (52,)

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
 
 
def publish_default_pose(publishers, duration_secs=3.0, rate_hz=10):
    """
    Sends all joints to the open-hand default position, repeatedly.
    ROS messages are fire-and-forget, so we must republish many times to
    ensure the hardware controller actually receives the command.
    
    Args:
        publishers   : dict from create_hand_publishers()
        duration_secs: how long to keep publishing (default 3 seconds)
        rate_hz      : how often to publish per second (default 10 Hz)
    """
    rospy.loginfo("Moving to default open-hand pose (publishing for {} seconds)...".format(duration_secs))
    
    # Calculate how many times to publish
    num_publishes = int(duration_secs * rate_hz)
    publish_rate = rospy.Rate(hz=rate_hz)
    
    for i in range(num_publishes):
        for name in publishers:
            msg = Float64()
            msg.data = 0.0   # zero = fully open for all joints
            publishers[name].publish(msg)
        
        publish_rate.sleep()
        
        if i % 10 == 0:
            rospy.loginfo("  Publishing default pose... ({}/{})".format(i, num_publishes))
    
    rospy.loginfo("Default pose publishing complete.")


# =============================================================================
# SECTION 13: MAIN POLICY LOOP
# =============================================================================

def rl_policy_loop():
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
    num_prop    = 52
    num_actions = 13   # one per ROS controller
 
    # These dicts tell the encoder and policy the shape of their inputs/outputs
    observation_space = {
        "prop": np.zeros(num_prop),
    }
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
    rospy.init_node('rl_policy_node', anonymous