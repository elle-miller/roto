Here’s your README with the **policy joint order added** (and nothing else changed):

---

# Shadow Hand Lite — ROS Interface Notes

> Legend: ✅ Confirmed on hardware | ⚠️ Needs verifying | ❓ Unknown | 🚨 Critical issue

---

## 1. Environment

| Item               | Value             |
| ------------------ | ----------------- |
| ROS version        | ⚠️ verify         |
| Python version     | ⚠️ verify         |
| Shadow Hand driver | `sr_robot_launch` |
| Workspace path     | ⚠️ verify         |
| How to launch      | ⚠️ verify         |

---

## 2. Key Topics

### Joint State — Read ✅

| Field        | Value                    |
| ------------ | ------------------------ |
| Topic        | `/joint_states`          |
| Message type | `sensor_msgs/JointState` |
| Rate         | 100 Hz                   |
| Joint count  | 16                       |

### Joint Command — Write ✅

| Field                   | Value                                        |
| ----------------------- | -------------------------------------------- |
| Topic pattern           | `/sh_rh_<joint>_position_controller/command` |
| Message type            | `std_msgs/Float64`                           |
| Commandable controllers | **13** (not 16 — see Section 4)              |

### Tactile ❓

| Field     | Value                                                                                |
| --------- | ------------------------------------------------------------------------------------ |
| Topic     | ❓ not found yet                                                                      |
| Source    | Touch Lab sensor (external — driver may not be running)                              |
| Next step | Run full `rostopic list`, or `rosnode list` to check if Touch Lab driver is launched |

### Palm Extras (IMU) ✅

| Field        | Value                                                  |
| ------------ | ------------------------------------------------------ |
| Topic        | `/rh/palm_extras`                                      |
| Message type | `std_msgs/Float64MultiArray`                           |
| Rate         | 84 Hz                                                  |
| Data [0:3]   | Accelerometer → confirmed `[280.0, 12784.0, -12188.0]` |
| Data [3:6]   | Gyrometer → confirmed `[-521.0, 110.0, 188.0]`         |
| Data [6:10]  | Analog inputs → confirmed `[11.0, 12.0, 6.0, 6.0]`     |

IMU confirmed present on this hand. Both accelerometer and gyrometer are active.

---

## 3. Joint State Ordering — Confirmed ✅

Real output from `rostopic echo /joint_states`:

```
name:
  - rh_FFJ1    # index 0
  - rh_FFJ2    # index 1
  - rh_FFJ3    # index 2
  - rh_FFJ4    # index 3
  - rh_MFJ1    # index 4
  - rh_MFJ2    # index 5
  - rh_MFJ3    # index 6
  - rh_MFJ4    # index 7
  - rh_RFJ1    # index 8
  - rh_RFJ2    # index 9
  - rh_RFJ3    # index 10
  - rh_RFJ4    # index 11
  - rh_THJ1    # index 12
  - rh_THJ2    # index 13
  - rh_THJ4    # index 14  ← NOTE: THJ3 is absent (its a fixed joint)
  - rh_THJ5    # index 15
```

No little finger. No WRJ. THJ3 absent. Total: **16 joints readable**.

---

## 3.1 Policy Joint Ordering — From Policy Side 🚨

Policy outputs actions in the following order:

```
0  rh_FFJ4
1  rh_MFJ4
2  rh_RFJ4
3  rh_THJ5
4  rh_FFJ3
5  rh_MFJ3
6  rh_RFJ3
7  rh_THJ4
8  rh_FFJ2
9  rh_MFJ2
10 rh_RFJ2
11 rh_THJ2
12 rh_FFJ1
13 rh_MFJ1
14 rh_RFJ1
15 rh_THJ1
```

> ⚠️ This ordering **does NOT match** `/joint_states` ordering.
> A reshuffle map is required before sending actions to the controller.

---

## 4. 🚨 J0 Coupling — Critical Issue - Needs Verification

This is the most important hardware constraint to understand.

### What the position controller list shows ✅

```
/sh_rh_ffj0_position_controller/command   ← ffj0 controls FFJ1 + FFJ2 together
/sh_rh_ffj3_position_controller/command
/sh_rh_ffj4_position_controller/command
/sh_rh_mfj0_position_controller/command   ← mfj0 controls MFJ1 + MFJ2 together
/sh_rh_mfj3_position_controller/command
/sh_rh_mfj4_position_controller/command
/sh_rh_rfj0_position_controller/command   ← rfj0 controls RFJ1 + RFJ2 together
/sh_rh_rfj3_position_controller/command
/sh_rh_rfj4_position_controller/command
/sh_rh_thj1_position_controller/command
/sh_rh_thj2_position_controller/command
/sh_rh_thj4_position_controller/command
/sh_rh_thj5_position_controller/command
```

**Total commandable: 13 controllers.**

### The Mismatch

|                                      | Count  |
| ------------------------------------ | ------ |
| Policy outputs (actions)             | 16     |
| Joints readable from `/joint_states` | 16     |
| Joints commandable via controllers   | **13** |

FFJ1, FFJ2 → commanded together via `ffj0`
MFJ1, MFJ2 → commanded together via `mfj0`
RFJ1, RFJ2 → commanded together via `rfj0`

There is NO `ffj1`, `ffj2`, `mfj1`, `mfj2`, `rfj1`, `rfj2` controller.

### What J0 Means Physically

`J0` is the sum of `J1 + J2`. When you command `ffj0 = 1.5`, the hardware distributes that across FFJ1 and FFJ2. You cannot command them independently.

### Mapping Strategy — Policy Actions → Controllers 🚨

```python
# Policy outputs 16 values in this order (subscriber ordering):
# [FFJ1, FFJ2, FFJ3, FFJ4, MFJ1, MFJ2, MFJ3, MFJ4, RFJ1, RFJ2, RFJ3, RFJ4, THJ1, THJ2, THJ4, THJ5]


# To command the 13 controllers, collapse J1+J2 pairs:
def actions_to_controller_commands(actions):
    return {
        "ffj0": actions[0] + actions[1],  # FFJ1 + FFJ2 → ffj0
        "ffj3": actions[2],
        "ffj4": actions[3],
        "mfj0": actions[4] + actions[5],  # MFJ1 + MFJ2 → mfj0
        "mfj3": actions[6],
        "mfj4": actions[7],
        "rfj0": actions[8] + actions[9],  # RFJ1 + RFJ2 → rfj0
        "rfj3": actions[10],
        "rfj4": actions[11],
        "thj1": actions[12],
        "thj2": actions[13],
        "thj4": actions[14],
        "thj5": actions[15],
    }
```

> ⚠️ This is a best-effort mapping. If the policy was trained treating J1 and J2 as
> independent, the real behaviour may differ. Verify with whoever trained the policy (das me).

---

## 5. Joint Reference Table

| Subscriber Index | Joint   | Controller Topic | Lower   | Upper  |
| ---------------- | ------- | ---------------- | ------- | ------ |
| 0                | rh_FFJ1 | `ffj0` (shared)  | 0.0     | 1.5708 |
| 1                | rh_FFJ2 | `ffj0` (shared)  | 0.0     | 1.5708 |
| 2                | rh_FFJ3 | `ffj3`           | -0.2618 | 1.5708 |
| 3                | rh_FFJ4 | `ffj4`           | -0.3491 | 0.3491 |
| 4                | rh_MFJ1 | `mfj0` (shared)  | 0.0     | 1.5708 |
| 5                | rh_MFJ2 | `mfj0` (shared)  | 0.0     | 1.5708 |
| 6                | rh_MFJ3 | `mfj3`           | -0.2618 | 1.5708 |
| 7                | rh_MFJ4 | `mfj4`           | -0.3491 | 0.3491 |
| 8                | rh_RFJ1 | `rfj0` (shared)  | 0.0     | 1.5708 |
| 9                | rh_RFJ2 | `rfj0` (shared)  | 0.0     | 1.5708 |
| 10               | rh_RFJ3 | `rfj3`           | -0.2618 | 1.5708 |
| 11               | rh_RFJ4 | `rfj4`           | -0.3491 | 0.3491 |
| 12               | rh_THJ1 | `thj1`           | -0.2618 | 1.5708 |
| 13               | rh_THJ2 | `thj2`           | -0.6981 | 0.6981 |
| 14               | rh_THJ4 | `thj4`           | 0.0     | 1.2217 |
| 15               | rh_THJ5 | `thj5`           | -1.0472 | 1.0472 |

---

## 6. TF Frames

| Frame      | Description       | Verified |
| ---------- | ----------------- | -------- |
| `rh_palm`  | Base frame        | ⚠️       |
| `rh_fftip` | First finger tip  | ⚠️       |
| `rh_mftip` | Middle finger tip | ⚠️       |
| `rh_rftip` | Ring finger tip   | ⚠️       |
| `rh_thtip` | Thumb tip         | ⚠️       |

---

## 7. Code Templates

### Subscriber

```python
from sensor_msgs.msg import JointState
from threading import Lock
import numpy as np

joint_pos = None
joint_pos_norm = None
joint_vel_norm = None
data_lock = Lock()

# Subscriber ordering (confirmed from hardware)
SUBSCRIBER_JOINT_ORDER = [
    "rh_FFJ1", "rh_FFJ2", "rh_FFJ3", "rh_FFJ4",
    "rh_MFJ1", "rh_MFJ2", "rh_MFJ3", "rh_MFJ4",
    "rh_RFJ1", "rh_RFJ2", "rh_RFJ3", "rh_RFJ4",
    "rh_THJ1", "rh_THJ2", "rh_THJ4", "rh_THJ5",
]

def prop_callback(data):
    global joint_pos, joint_pos_norm, joint_vel_norm
    with data_lock:
        # data.position is already in SUBSCRIBER_JOINT_ORDER
        # apply reshuffle if policy order differs
        raw_pos = np.array(list(data.position))
        raw_vel = np.array(list(data.velocity))
        joint_pos = reshuffle_data(raw_pos, index_reshuffle_map)
        joint_vel = reshuffle_data(raw_vel, index_reshuffle_map)
        joint_pos_norm = normalise(joint_pos, LOWER_LIMITS, UPPER_LIMITS)
        joint_vel_norm = normalise(joint_vel, -VEL_LIMITS, VEL_LIMITS)

rospy.Subscriber("/joint_states", JointState, prop_callback)
```

### Publisher — One Per Controller

```python
from std_msgs.msg import Float64

# Confirmed controller names from hardware
CONTROLLER_NAMES = [
    "ffj0", "ffj3", "ffj4",
    "mfj0", "mfj3", "mfj4",
    "rfj0", "rfj3", "rfj4",
    "thj1", "thj2", "thj4", "thj5",
]

def create_publishers():
    publishers = {}
    for name in CONTROLLER_NAMES:
        topic = f"/sh_rh_{name}_position_controller/command"
        publishers[name] = rospy.Publisher(topic, Float64, queue_size=1)
    return publishers

def publish_joint_positions(publishers, actions):
    """
    actions: 16-element array in subscriber order
    [FFJ1, FFJ2, FFJ3, FFJ4, MFJ1, MFJ2, MFJ3, MFJ4,
     RFJ1, RFJ2, RFJ3, RFJ4, THJ1, THJ2, THJ4, THJ5]
    """
    commands = {
        "ffj0": float(actions[0] + actions[1]),  # J1+J2 coupled
        "ffj3": float(actions[2]),
        "ffj4": float(actions[3]),
        "mfj0": float(actions[4] + actions[5]),  # J1+J2 coupled
        "mfj3": float(actions[6]),
        "mfj4": float(actions[7]),
        "rfj0": float(actions[8] + actions[9]),  # J1+J2 coupled
        "rfj3": float(actions[10]),
        "rfj4": float(actions[11]),
        "thj1": float(actions[12]),
        "thj2": float(actions[13]),
        "thj4": float(actions[14]),
        "thj5": float(actions[15]),
    }
    for name, value in commands.items():
        msg = Float64()
        msg.data = value
        publishers[name].publish(msg)
```

---

## 8. Useful Terminal Commands

```bash
# Joint state (confirmed working)
rostopic echo /joint_states

# All position controller topics (confirmed working)
rostopic list | grep position_controller

# Find tactile topic

# Check IMU
rostopic echo /rh/palm_extras

# Check calibration
rostopic echo /calibrated

# TF frames
rosrun tf tf_echo rh_palm rh_fftip

# Test single joint command manually
rostopic pub /sh_rh_ffj3_position_controller/command std_msgs/Float64 "data: 0.5"
```

---

## 9. Verification Checklist

* [x] `/joint_states` topic confirmed
* [x] Joint subscriber ordering confirmed (16 joints)
* [x] Position controller topic names confirmed (13 controllers)
* [x] J0 coupling confirmed (FFJ0, MFJ0, RFJ0)
* [x] `rostopic echo /calibrated` → returns `True` ✅
* [x] `rostopic echo /rh/palm_extras` → IMU confirmed present ✅
* [ ] `rostopic list | grep -i tactile` → Touch Lab topic not found yet
* [x] `rosrun tf tf_echo rh_palm rh_fftip` → confirm TF frames
* [ ] Confirm policy joint ordering from IsaacLab training config
* [ ] Build and verify reshuffle map
* [ ] Test single joint via `rostopic pub` before running policy
* [ ] Confirm `num_prop` matches trained policy input size

---

## 10. Open Questions

| Question                                                                    | Status                |
| --------------------------------------------------------------------------- | --------------------- |
| Was the policy trained with J0 (13 actions) or J1+J2 separate (16 actions)? | ❓ Ask policy trainer  |
| What is the tactile sensor topic name?                                      | ❓ Run `rostopic list` |

| What is the policy joint ordering from IsaacLab? | ❓ Check training config |

---
