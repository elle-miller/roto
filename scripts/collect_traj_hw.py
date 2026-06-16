#!/usr/bin/env python3
"""Collect per-joint sinusoidal trajectories on the Shadow Hand Lite hardware.

For each of the 13 policy joints, commands a sinusoid while holding all other
joints at zero, and records commanded vs actual position from /joint_states.

Run inside the shadow docker (where ROS + hand controllers are live):
    python collect_traj_hw.py
    python collect_traj_hw.py --output_dir /tmp/trajectories/hw
    python collect_traj_hw.py --start_joint 4       # resume from joint index 4
    python collect_traj_hw.py --joint_idx 8         # test only joint 8 (FFJ2)
"""

import argparse
import os
import sys
import numpy as np
from threading import Lock

import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

# ---------------------------------------------------------------------------
# CLI args (parsed before rospy.init_node)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Collect per-joint sinusoidal trajectories on hardware.")
parser.add_argument("--output_dir",     type=str,   default="../trajectories/hw")
parser.add_argument("--freq",           type=float, default=0.5,  help="Sinusoid frequency (Hz)")
parser.add_argument("--cycles",         type=float, default=4.0,  help="Number of sinusoid cycles per joint")
parser.add_argument("--amplitude_frac", type=float, default=0.8,  help="Amplitude as fraction of half-range")
parser.add_argument("--settle_secs",    type=float, default=3.0,  help="Zero-hold seconds before each sinusoid")
parser.add_argument("--start_joint",    type=int,   default=0,    help="Start from this policy joint index")
parser.add_argument("--joint_idx",      type=int,   default=None, help="Test only this policy joint index (0-12)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Constants — must match run_shadow.py exactly
# ---------------------------------------------------------------------------

RL_HZ = 60

SUBSCRIBER_JOINT_ORDER = [
    "rh_FFJ1",  # 0  — coupled to FFJ2
    "rh_FFJ2",  # 1
    "rh_FFJ3",  # 2
    "rh_FFJ4",  # 3
    "rh_MFJ1",  # 4  — coupled to MFJ2
    "rh_MFJ2",  # 5
    "rh_MFJ3",  # 6
    "rh_MFJ4",  # 7
    "rh_RFJ1",  # 8  — coupled to RFJ2
    "rh_RFJ2",  # 9
    "rh_RFJ3",  # 10
    "rh_RFJ4",  # 11
    "rh_THJ1",  # 12
    "rh_THJ2",  # 13
    "rh_THJ4",  # 14
    "rh_THJ5",  # 15
]

POLICY_JOINT_ORDER = [
    "rh_FFJ4",   # 0
    "rh_MFJ4",   # 1
    "rh_RFJ4",   # 2
    "rh_THJ5",   # 3
    "rh_FFJ3",   # 4
    "rh_MFJ3",   # 5
    "rh_RFJ3",   # 6
    "rh_THJ4",   # 7
    "rh_FFJ2",   # 8  — coupled (subscriber idx 1 → policy idx 8)
    "rh_MFJ2",   # 9
    "rh_RFJ2",   # 10
    "rh_THJ2",   # 11
    "rh_THJ1",   # 12
]

# Maps subscriber_index → policy_index (J1 joints skipped — use J2 as representative)
INDEX_RESHUFFLE_MAP = {
    3:  0,   # FFJ4
    7:  1,   # MFJ4
    11: 2,   # RFJ4
    15: 3,   # THJ5
    2:  4,   # FFJ3
    6:  5,   # MFJ3
    10: 6,   # RFJ3
    14: 7,   # THJ4
    1:  8,   # FFJ2 (J2 = controller, J1 = mimic)
    5:  9,   # MFJ2
    9:  10,  # RFJ2
    13: 11,  # THJ2
    12: 12,  # THJ1
}

LOWER_LIMITS = np.array([
    -0.3491,  # FFJ4
    -0.3491,  # MFJ4
    -0.3491,  # RFJ4
    -1.0472,  # THJ5
    -0.2618,  # FFJ3
    -0.2618,  # MFJ3
    -0.2618,  # RFJ3
     0.0,     # THJ4
     0.0,     # FFJ2 proxy
     0.0,     # MFJ2 proxy
     0.0,     # RFJ2 proxy
    -0.6981,  # THJ2
    -0.2618,  # THJ1
], dtype=np.float32)

UPPER_LIMITS = np.array([
    0.3491,   # FFJ4
    0.3491,   # MFJ4
    0.3491,   # RFJ4
    1.0472,   # THJ5
    1.5708,   # FFJ3
    1.5708,   # MFJ3
    1.5708,   # RFJ3
    1.2217,   # THJ4
    1.5708,   # FFJ2 proxy
    1.5708,   # MFJ2 proxy
    1.5708,   # RFJ2 proxy
    0.6981,   # THJ2
    1.5708,   # THJ1
], dtype=np.float32)

# ---------------------------------------------------------------------------
# Shared sensor state (written by callback, read by main loop)
# ---------------------------------------------------------------------------

joint_pos = None   # 13-element array in policy order (radians)
joint_vel = None   # 13-element array in policy order (rad/s)
data_lock  = Lock()


def reshuffle_data(data_list, index_mapping_dict):
    if not data_list or not index_mapping_dict:
        return list(data_list)
    max_new = max(index_mapping_dict.values())
    out = [0.0] * (max_new + 1)
    for old, new in index_mapping_dict.items():
        if 0 <= old < len(data_list):
            out[new] = data_list[old]
    return out


def prop_callback(data):
    global joint_pos, joint_vel
    with data_lock:
        joint_pos = np.array(reshuffle_data(list(data.position), INDEX_RESHUFFLE_MAP),
                             dtype=np.float32)
        joint_vel = np.array(reshuffle_data(list(data.velocity), INDEX_RESHUFFLE_MAP),
                             dtype=np.float32)


# ---------------------------------------------------------------------------
# Publishing helpers
# ---------------------------------------------------------------------------

def create_hand_publishers():
    controller_names = [
        "ffj0", "ffj3", "ffj4",
        "mfj0", "mfj3", "mfj4",
        "rfj0", "rfj3", "rfj4",
        "thj1", "thj2", "thj4", "thj5",
    ]
    publishers = {}
    for name in controller_names:
        topic = f"/sh_rh_{name}_position_controller/command"
        publishers[name] = rospy.Publisher(topic, Float64, queue_size=1)
    return publishers


def publish_joint_cmd(publishers, cmd_policy_order):
    """Send position commands (radians) to all 13 controllers.

    cmd_policy_order: 13-element array in POLICY_JOINT_ORDER.
    Coupled joints (ffj0/mfj0/rfj0) are scaled by 2.0 to account for the
    hardware combining FFJ1+FFJ2 into a single actuator range of [0, pi].
    """
    commands = {
        "ffj4": float(cmd_policy_order[0]),
        "mfj4": float(cmd_policy_order[1]),
        "rfj4": float(cmd_policy_order[2]),
        "thj5": float(cmd_policy_order[3]),
        "ffj3": float(cmd_policy_order[4]),
        "mfj3": float(cmd_policy_order[5]),
        "rfj3": float(cmd_policy_order[6]),
        "thj4": float(cmd_policy_order[7]),
        "thj2": float(cmd_policy_order[11]),
        "thj1": float(cmd_policy_order[12]),
        # Coupled: ffj0 spans [0, pi] = J1 + J2 combined → 2 × policy proxy
        "ffj0": 2.0 * float(cmd_policy_order[8]),
        "mfj0": 2.0 * float(cmd_policy_order[9]),
        "rfj0": 2.0 * float(cmd_policy_order[10]),
    }
    for name, val in commands.items():
        msg = Float64()
        msg.data = val
        publishers[name].publish(msg)


def hold_zero(publishers, duration_secs, rate_hz=10):
    """Repeatedly publish zero to all joints for duration_secs."""
    rate = rospy.Rate(rate_hz)
    n = int(duration_secs * rate_hz)
    zero_cmd = np.zeros(13, dtype=np.float32)
    for _ in range(n):
        publish_joint_cmd(publishers, zero_cmd)
        rate.sleep()


def wait_for_publisher_connections(publishers, timeout=10.0):
    deadline = rospy.get_time() + timeout
    while rospy.get_time() < deadline:
        if all(p.get_num_connections() > 0 for p in publishers.values()):
            rospy.loginfo("All publishers connected.")
            return True
        rospy.sleep(0.5)
    rospy.logwarn("Publisher connection timeout — proceeding anyway.")
    return False


# ---------------------------------------------------------------------------
# Trajectory collection for one joint
# ---------------------------------------------------------------------------

def collect_joint_trajectory(publishers, joint_idx, output_dir):
    joint_name = POLICY_JOINT_ORDER[joint_idx]
    lower = float(LOWER_LIMITS[joint_idx])
    upper = float(UPPER_LIMITS[joint_idx])
    center = (upper + lower) / 2.0
    amp    = args.amplitude_frac * (upper - lower) / 2.0

    duration  = args.cycles / args.freq
    n_steps   = int(duration * RL_HZ)

    rospy.loginfo(
        f"Joint {joint_idx} ({joint_name}): "
        f"lower={lower:.3f}  upper={upper:.3f}  center={center:.3f}  amp={amp:.3f}  "
        f"duration={duration:.1f}s  steps={n_steps}"
    )

    # Settle at zero before sinusoid
    rospy.loginfo(f"  Settling at zero for {args.settle_secs}s ...")
    hold_zero(publishers, args.settle_secs, rate_hz=RL_HZ)

    # Run sinusoid
    ts, cmds, actual_pos_buf, actual_vel_buf = [], [], [], []
    rate = rospy.Rate(RL_HZ)

    for step in range(n_steps):
        t = step / RL_HZ
        proxy = center + amp * np.sin(2.0 * np.pi * args.freq * t)
        proxy = float(np.clip(proxy, lower, upper))

        cmd = np.zeros(13, dtype=np.float32)
        cmd[joint_idx] = proxy
        publish_joint_cmd(publishers, cmd)

        with data_lock:
            pos = joint_pos.copy() if joint_pos is not None else np.zeros(13, np.float32)
            vel = joint_vel.copy() if joint_vel is not None else np.zeros(13, np.float32)

        ts.append(t)
        cmds.append(proxy)
        actual_pos_buf.append(pos)
        actual_vel_buf.append(vel)

        rate.sleep()

    # Return to zero
    rospy.loginfo("  Returning to zero ...")
    hold_zero(publishers, 2.0, rate_hz=RL_HZ)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"joint_{joint_idx:02d}_{joint_name}.npz")
    np.savez(
        fname,
        t=np.array(ts, dtype=np.float32),
        cmd=np.array(cmds, dtype=np.float32),
        actual_pos=np.array(actual_pos_buf, dtype=np.float32),
        actual_vel=np.array(actual_vel_buf, dtype=np.float32),
        joint_name=np.array(joint_name),
        joint_idx=np.array(joint_idx),
        lower=np.array(lower, dtype=np.float32),
        upper=np.array(upper, dtype=np.float32),
    )
    rospy.loginfo(f"  → Saved {fname}  ({len(ts)} steps @ {RL_HZ} Hz)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = os.path.abspath(args.output_dir)

    rospy.init_node("traj_collector", anonymous=True)

    publishers = create_hand_publishers()

    rospy.loginfo("Waiting 3s for publishers to connect to controllers ...")
    rospy.sleep(3.0)
    wait_for_publisher_connections(publishers, timeout=10.0)

    rospy.Subscriber("/joint_states", JointState, prop_callback)
    rospy.loginfo("Subscribed to /joint_states. Waiting for first sensor data ...")

    poll_rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        with data_lock:
            if joint_pos is not None:
                break
        poll_rate.sleep()
    rospy.loginfo("Sensor data ready.")

    # Move to safe open-hand pose
    rospy.loginfo("Moving to zero pose (5s) ...")
    hold_zero(publishers, 5.0, rate_hz=10)

    joints_to_test = (
        [args.joint_idx] if args.joint_idx is not None
        else list(range(args.start_joint, 13))
    )

    for ji in joints_to_test:
        joint_name = POLICY_JOINT_ORDER[ji]
        print(f"\n{'='*60}")
        print(f"  Next: joint {ji:2d}  {joint_name}")
        print(f"  freq={args.freq}Hz  cycles={args.cycles}  amp_frac={args.amplitude_frac}")
        print(f"{'='*60}")

        resp = input("  [Enter] start  |  [s] skip  |  [q] quit: ").strip().lower()
        if resp == "q":
            rospy.loginfo("User quit.")
            break
        if resp == "s":
            rospy.loginfo(f"Skipping joint {ji}.")
            continue

        collect_joint_trajectory(publishers, ji, output_dir)

    rospy.loginfo("Collection complete. Returning to zero.")
    hold_zero(publishers, 5.0, rate_hz=10)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted.")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
        raise
