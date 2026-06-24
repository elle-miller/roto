#!/usr/bin/env python3
"""Replay a sim-recorded policy command trajectory open-loop on the hardware.

Loads a recording produced by record_policy.py (which captured the policy's
commanded + actual joint trajectory in simulation), reconstructs the per-step
13-joint command in policy order, and *replays that exact command sequence* on
the real Shadow Hand Lite — no policy in the loop. It records the hardware's
actual joint response so you can compare sim-actual vs hw-actual against the
identical command (see plot_traj_compare.py).

Coupled fingers (FF/MF/RF) are commanded via the J2 "curl proxy": the proxy for
a finger is (J2_cmd + J1_cmd)/2 from the sim recording, and publish_joint_cmd
doubles it into the single ffj0/mfj0/rfj0 actuator (range [0, pi]). This matches
how the policy is deployed live (run_shadow.py / my_policy_node.py).

Two replay modes (--replay):
    command  (default) — send the policy command (joint_pos_cmd). Compares how
                         sim and hardware each respond to the SAME command.
    sim_pos            — send the sim's achieved joint positions (joint_pos).
                         Compares how well the hardware can follow the motion
                         the sim actually produced.

Run inside the shadow docker (ROS + hand controllers live):
    python replay_traj_hw.py --input policy_recording.npz                      # command
    python replay_traj_hw.py --input policy_recording.npz --replay sim_pos     # sim positions
    python replay_traj_hw.py --input policy_recording.npz --settle_secs 2.0
"""

import argparse
import os

import numpy as np
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from threading import Lock

# ---------------------------------------------------------------------------
# Constants — must match the sim env / run_shadow.py
# ---------------------------------------------------------------------------

RL_HZ = 60

# Policy/control joint order (identical to ShadowLiteCfg.control_joint_names).
POLICY_JOINT_ORDER = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",   # 0,1,2,3
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",   # 4,5,6,7
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",              # 8,9,10  ← coupled J2 drivers
    "rh_THJ2", "rh_THJ1",                          # 11,12
]

# Coupled finger: J2 driver -> its J1 mimic. Proxy = (J2 + J1) / 2.
COUPLED_DEP = {"rh_FFJ2": "rh_FFJ1", "rh_MFJ2": "rh_MFJ1", "rh_RFJ2": "rh_RFJ1"}

# Joint limits in policy order (proxy units). Match collect_traj_hw.py.
LOWER_LIMITS = np.array([
    -0.3491, -0.3491, -0.3491, -1.0472,   # FFJ4 MFJ4 RFJ4 THJ5
    -0.2618, -0.2618, -0.2618,  0.0,       # FFJ3 MFJ3 RFJ3 THJ4
     0.0,     0.0,     0.0,                 # FFJ2 MFJ2 RFJ2 (proxy)
    -0.6981, -0.2618,                       # THJ2 THJ1
], dtype=np.float32)

UPPER_LIMITS = np.array([
    0.3491, 0.3491, 0.3491, 1.0472,
    1.5708, 1.5708, 1.5708, 1.2217,
    1.5708, 1.5708, 1.5708,                # FFJ2 MFJ2 RFJ2 (proxy)
    0.6981, 1.5708,
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Proxy reconstruction (sim 16-DOF recording -> 13 policy-order proxy)
# ---------------------------------------------------------------------------

def to_proxy13(data16, actuated_names):
    """Map a [T, 16] actuated-order array to [T, 13] policy-order proxy space.

    For coupled fingers the proxy is the mean of the J2 driver and its J1 mimic
    (so 2*proxy = J2 + J1 = the combined curl the hardware actuator produces).
    """
    actuated_names = [str(n) for n in actuated_names]
    col = {n: i for i, n in enumerate(actuated_names)}
    out = []
    for jn in POLICY_JOINT_ORDER:
        if jn in COUPLED_DEP:
            out.append(0.5 * (data16[:, col[jn]] + data16[:, col[COUPLED_DEP[jn]]]))
        else:
            out.append(data16[:, col[jn]])
    return np.stack(out, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Shared sensor state
# ---------------------------------------------------------------------------

latest_pos = {}   # joint_name -> position (rad), updated by callback
data_lock = Lock()


def joint_state_callback(msg):
    global latest_pos
    with data_lock:
        latest_pos = dict(zip(msg.name, msg.position))


def read_proxy_actual():
    """Assemble the 13 policy-order actual positions (proxy space) from /joint_states.

    Returns zeros for any joint not yet seen. Coupled fingers report the mean of
    their measured J1 and J2 angles, matching the proxy command space.
    """
    with data_lock:
        pos = dict(latest_pos)
    out = np.zeros(len(POLICY_JOINT_ORDER), dtype=np.float32)
    for i, jn in enumerate(POLICY_JOINT_ORDER):
        if jn in COUPLED_DEP:
            j2 = pos.get(jn)
            j1 = pos.get(COUPLED_DEP[jn])
            if j2 is not None and j1 is not None:
                out[i] = 0.5 * (j1 + j2)
            elif j2 is not None:
                out[i] = j2
        else:
            v = pos.get(jn)
            if v is not None:
                out[i] = v
    return out


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------

def create_hand_publishers():
    controller_names = [
        "ffj0", "ffj3", "ffj4",
        "mfj0", "mfj3", "mfj4",
        "rfj0", "rfj3", "rfj4",
        "thj1", "thj2", "thj4", "thj5",
    ]
    return {
        name: rospy.Publisher(f"/sh_rh_{name}_position_controller/command", Float64, queue_size=1)
        for name in controller_names
    }


def publish_joint_cmd(publishers, cmd_policy_order):
    """Send 13 policy-order position commands (radians) to the controllers.

    Coupled drivers (ffj0/mfj0/rfj0) span [0, pi] = J1 + J2 combined, so the
    proxy is doubled — identical to collect_traj_hw.py / run_shadow.py.
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
        "ffj0": 2.0 * float(cmd_policy_order[8]),
        "mfj0": 2.0 * float(cmd_policy_order[9]),
        "rfj0": 2.0 * float(cmd_policy_order[10]),
    }
    for name, val in commands.items():
        msg = Float64()
        msg.data = val
        publishers[name].publish(msg)


def ramp_to(publishers, target, secs):
    """Smoothly interpolate from the current measured pose to `target` over `secs`."""
    target = np.clip(target, LOWER_LIMITS, UPPER_LIMITS)
    start = read_proxy_actual()
    n = max(1, int(secs * RL_HZ))
    rate = rospy.Rate(RL_HZ)
    for k in range(1, n + 1):
        alpha = k / n
        publish_joint_cmd(publishers, (1.0 - alpha) * start + alpha * target)
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Replay a sim trajectory on hardware.")
    parser.add_argument("--input", required=True, help="policy_recording.npz from record_policy.py")
    parser.add_argument("--output", default=None, help="Output npz (default: traj_hw_<ts>.npz next to input).")
    parser.add_argument("--replay", choices=["command", "sim_pos"], default="command",
                        help="What to send to the hardware: the policy 'command' (joint_pos_cmd) "
                             "or the sim's achieved 'sim_pos' (joint_pos). Default: command.")
    parser.add_argument("--settle_secs", type=float, default=3.0,
                        help="Ramp time to the first reference of each episode.")
    args = parser.parse_args()

    # --- load + reconstruct the reference trajectory ------------------------
    data = np.load(args.input, allow_pickle=True)
    key = "joint_pos_cmd" if args.replay == "command" else "joint_pos"
    ref16 = data[key]                                   # [T, 16] actuated order
    actuated_names = list(data["actuated_names"])
    control_names = [str(n) for n in data["control_names"]]
    ep_ends = [int(e) for e in data["episode_ends"]] if "episode_ends" in data.files else []
    rl_dt = float(data["rl_dt"]) if "rl_dt" in data.files else 1.0 / RL_HZ

    if control_names != POLICY_JOINT_ORDER:
        rospy.logwarn("control_names in recording differ from POLICY_JOINT_ORDER; "
                      "reconstruction maps by name so order is handled, but verify the source.")

    ref13 = to_proxy13(ref16, actuated_names)           # [T, 13] policy order, proxy space
    ref13 = np.clip(ref13, LOWER_LIMITS, UPPER_LIMITS)
    T = ref13.shape[0]
    rospy.loginfo("Replay source: %s ('%s')", args.replay, key)

    # episode segment boundaries: [start, end) for each contiguous replay block
    ends = sorted(e + 1 for e in ep_ends if 0 <= e < T)
    if not ends or ends[-1] != T:
        ends.append(T)
    segments = []
    s = 0
    for e in ends:
        if e > s:
            segments.append((s, e))
        s = e

    # --- ROS setup ----------------------------------------------------------
    rospy.init_node("traj_replay", anonymous=True)
    publishers = create_hand_publishers()
    rospy.loginfo("Waiting 3s for publishers to connect ...")
    rospy.sleep(3.0)
    wait_for_publisher_connections(publishers, timeout=10.0)

    rospy.Subscriber("/joint_states", JointState, joint_state_callback)
    rospy.loginfo("Subscribed to /joint_states. Waiting for first sensor data ...")
    poll = rospy.Rate(10)
    while not rospy.is_shutdown():
        with data_lock:
            if latest_pos:
                break
        poll.sleep()
    rospy.loginfo("Sensor data ready.")

    rospy.loginfo("Loaded trajectory: %d steps, %d episode(s), %.1f Hz (%.2fs total)",
                  T, len(segments), 1.0 / rl_dt, T * rl_dt)
    resp = input("Press y to replay on hardware, anything else to abort: ")
    if resp.strip().lower() != "y":
        rospy.loginfo("Aborted.")
        return

    # --- replay -------------------------------------------------------------
    hw_actual = np.zeros((T, len(POLICY_JOINT_ORDER)), dtype=np.float32)
    rate = rospy.Rate(RL_HZ)

    for seg_i, (start, end) in enumerate(segments):
        rospy.loginfo("Episode %d/%d: ramping to first reference (%.1fs) ...",
                      seg_i + 1, len(segments), args.settle_secs)
        ramp_to(publishers, ref13[start], args.settle_secs)

        rospy.loginfo("Replaying steps %d..%d ...", start, end - 1)
        for t in range(start, end):
            publish_joint_cmd(publishers, ref13[t])
            hw_actual[t] = read_proxy_actual()
            rate.sleep()
            if (t - start) % 30 == 0:
                rospy.loginfo("  step %d/%d", t - start, end - start)

    # hold final pose
    publish_joint_cmd(publishers, ref13[-1])
    rospy.loginfo("Replay complete.")

    # --- save ---------------------------------------------------------------
    out_path = args.output
    if out_path is None:
        import time
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.input)),
                                f"traj_hw_{args.replay}_{ts}.npz")
    np.savez_compressed(
        out_path,
        replayed13=ref13,                               # [T, 13] reference sent to hw (proxy space)
        hw_actual13=hw_actual,                          # [T, 13] hardware actual (proxy space)
        replay_source=args.replay,                      # "command" or "sim_pos"
        control_names=np.array(POLICY_JOINT_ORDER),
        episode_ends=np.array(ep_ends, dtype=np.int32),
        rl_dt=np.float32(rl_dt),
        source="hw",
    )
    err = np.abs(hw_actual - ref13).mean(axis=0)
    rospy.loginfo("Saved %d steps -> %s", T, out_path)
    rospy.loginfo("Mean |hw_actual - %s| per joint (rad):", args.replay)
    for jn, e in zip(POLICY_JOINT_ORDER, err):
        rospy.loginfo("  %-10s %.4f", jn, e)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted.")
