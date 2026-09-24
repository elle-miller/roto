#!/usr/bin/env python
"""Gate C / notac deploy: SIM / ZERO / REAL tactile.  [EXPERIMENTAL: curl-joint pos_err amplification]

Built on deploy_policy_simtactile_diag.py (same live pos_err diagnostics), with ONE
additional change: pos_err for the 3 coupled curl joints (FFJ2/MFJ2/RFJ2, CURL_J2_IDX)
is scaled by CURL_AMP instead of the uniform scaled_error used for the other 10 joints.

*** THIS CHANGES WHAT THE POLICY SEES (unlike the pure-diagnostic copy). ***

Why: hw ball/no-ball logs showed a real, per-seed-consistent ball-related signal in
curl pos_err (FFJ2*, RFJ2*), but it's small (std ~0.89) next to several OTHER joints'
pos_err (RFJ3 std ~6.9, MFJ3 ~3-5, THJ1 large) which are dominated by slew-gap noise
(SPEED_FRAC=0.35 rate-limiter vs unslewed last_command -- same root cause as before,
untouched here). Those noisy, mostly-uninformative channels sit in the same 304-d
input to the encoder's first nn.Linear (before any LayerNorm), so the curl signal is
diluted relative to them. Amplifying only the curl elements increases their relative
weight in that raw input -- it does NOT change SNR within the curl channel itself
(signal and noise there both get scaled by CURL_AMP together).

CAVEAT: sim's joint_pos_error has NO per-joint scaling (roto_env.py applies it
uniformly). This has no training-time justification -- it is an empirical experiment,
not a fidelity fix like the earlier scaled_error/slew-gap discussion. Start with a
small CURL_AMP (2-3x), watch actual grasp/hold behavior on hardware (not just whether
curl_bias got numerically bigger -- that's tautological under this change), and back
off if behavior degrades.

Edit PROTOCOL flags at the top of this file, then:
 python deploy_policy_simtactile_curlamp.py


No CLI overrides — change the constants below for each run.
"""


from __future__ import print_function


import torch
import torch.nn as nn
import numpy as np
from collections import deque


import threading


# ============================================================
# PROTOCOL — edit these for each run (no CLI)
# ============================================================
# "zero" | "sim" | "real"
TACTILE_MODE = "zero"


SIM_LOG_FILE = "sim_policy_log_trial15_seed42.npz"
CHECKPOINT = "/home/user/experiments/best_agent_legacy_mass_diff_best.pt"
# zero-tac / mass-diff (304-d, trained with zero tactile):
# CHECKPOINT = "/home/user/experiments/best_agent_mass_diff_best.pt"


POS_ERR_MLP_PATH = "pos_error_mlp.pt"


# Log path — edit freely (not auto-tied to mode)
LOG_NPZ = "hw_policy_log_prop_only_curlamp.npz"


CONTROL_HZ = 60


# Plant: rate-limit published joint targets (None = no slew).
SPEED_FRAC = 0.65


# --- pos_err source ---
# False: last_command - q  (raw unslewed desire by default)
# True:  last_command from slewed publish (pos_err ~0). Ignored if USE_MLP_POS_ERR.
USE_PUBLISHED_CMD_FOR_POS_ERR = False


# True: soft-sim pos_err from pos_error_mlp.pt (plant slew unchanged)
USE_MLP_POS_ERR = False


scaled_error = 0.4
# Obs: multiply MLP pos_err by this after predict (independent of SPEED_FRAC).
#   None -> *1.0 (raw MLP prediction)
#   0.5  -> half the predicted error, etc.
# Only used when USE_MLP_POS_ERR is True.
POS_ERR_SPEED_FRAC = 5.0


# --- EXPERIMENTAL: per-joint pos_err amplification (curl joints only) ---
# Applied ONLY to the non-MLP path (build_prop). CURL_J2_IDX joints get CURL_AMP;
# every other joint keeps the uniform `scaled_error`. Start small (2-3x) and watch
# hardware behavior, not just the printed curl_bias number.
CURL_AMP = 0.4 #damping is 1 on all joints and stiffness is 0.22 on all joints except the mfj4,ffj4,rhj4-0.3
#stiffness on mfj2,rfj3,ffj3 - 1


# --- DIAG: live pos_err instrumentation (does not affect control) ---
DIAG_PRINT_EVERY = 30       # ~2 Hz at 60 Hz control
DIAG_CURL_BIAS_WARN = 0.05  # rad; flag persistent positive bias on curl joints


# ============================================================
# MODEL
# ============================================================
OBS_DIM = 304
PROP_DIM = 52
NUM_J = 13
NUM_TACTILE = 24
OBS_STACK = 4
MLP_HIST = 4
MLP_IN_DIM = 169




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




class PosErrorMLP(nn.Module):
   """Soft-sim joint_pos_error from pos_error_mlp.pt (169 -> 13)."""


   def __init__(self, ckpt):
       super(PosErrorMLP, self).__init__()
       assert int(ckpt["input_dim"]) == MLP_IN_DIM
       assert int(ckpt["target_dim"]) == NUM_J
       self.net = nn.Sequential(
           nn.Linear(MLP_IN_DIM, 256), nn.LayerNorm(256), nn.ELU(),
           nn.Linear(256, 256), nn.LayerNorm(256), nn.ELU(),
           nn.Linear(256, 128), nn.LayerNorm(128),
       )
       self.head = nn.Linear(128, NUM_J)
       self.load_state_dict(ckpt["state_dict"], strict=True)
       self.register_buffer(
           "x_mean", torch.tensor(np.asarray(ckpt["x_mean"]), dtype=torch.float32)
       )
       self.register_buffer(
           "x_std", torch.tensor(np.asarray(ckpt["x_std"]), dtype=torch.float32)
       )
       self.register_buffer(
           "y_mean", torch.tensor(np.asarray(ckpt["y_mean"]), dtype=torch.float32)
       )
       self.register_buffer(
           "y_std", torch.tensor(np.asarray(ckpt["y_std"]), dtype=torch.float32)
       )
       self.control_names = list(ckpt["control_names"])


   def forward(self, x):
       if x.dim() == 1:
           x = x.unsqueeze(0)
       xn = (x - self.x_mean) / (self.x_std + 1e-8)
       y = self.head(self.net(xn))
       return y * self.y_std + self.y_mean




# ============================================================
# JOINTS / COUPLING (same as deploy_warmup_trial15)
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


# Per-joint pos_err scale vector: uniform scaled_error everywhere except CURL_J2_IDX,
# which get CURL_AMP. Built once at import time.
SCALE_VEC = np.full(NUM_J, float(scaled_error), dtype=np.float32)
SCALE_VEC[CURL_J2_IDX] = float(CURL_AMP)


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


# Real-tactile path only (MODE=real). Prefer fsr_pad_map when available.
try:
   from fsr_pad_map import FSR_CHANNELS
except ImportError:
   FSR_CHANNELS = [10, 7, 4, 9, 5, 13, 2, 3, 8, 18, 12, 11]


N_FSR = 12
BIOTAC_IDX = [0, 1, 2, 4]
BIOTAC_CH = [15, 16, 17, 22]
N_BIOTAC = len(BIOTAC_IDX)
USE_BIOTAC = True
K_HI, K_LO = 5.0, 2.0  # static calibrate for MODE=real only


SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200




def unscale(x, lo, hi):
   return (2.0 * x - hi - lo) / (hi - lo)




def scale(a, lo, hi):
   return 0.5 * (a + 1.0) * (hi - lo) + lo




def action_to_publish(action, meas_j2, prev_pub):
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


   if prev_pub is not None and SPEED_FRAC is not None:
       max_delta = PUB_VEL_LIMIT * SPEED_FRAC / CONTROL_HZ
       pub = prev_pub + np.clip(pub - prev_pub, -max_delta, max_delta)


   return pub.astype(np.float32), raw_cmd, j2_cmd




def pub16_to_cmd13(pub):
   cmd = np.empty(NUM_J, dtype=np.float32)
   cmd[CTRL_NONCOUPLED] = pub[PUB_NONCOUPLED]
   cmd[CURL_J2_IDX] = pub[PUB_J2_SLOTS]
   return cmd




def _torch_load(path):
   try:
       return torch.load(path, map_location="cpu", weights_only=False)
   except TypeError:
       return torch.load(path, map_location="cpu")




def load_pos_err_mlp(path):
   ckpt = _torch_load(path)
   mlp = PosErrorMLP(ckpt)
   mlp.eval()
   if mlp.control_names != POLICY_JOINTS:
       raise RuntimeError(
           "MLP control_names %s != POLICY_JOINTS %s"
           % (mlp.control_names, POLICY_JOINTS)
       )
   return mlp




def predict_pos_err(mlp, pos_hist, vel_hist, act4, cur_action):
   x = np.concatenate(
       [
           np.concatenate(list(pos_hist), axis=0),
           np.concatenate(list(vel_hist), axis=0),
           np.concatenate(list(act4), axis=0),
           cur_action.astype(np.float32),
       ]
   ).astype(np.float32)
   with torch.no_grad():
       y = mlp(torch.from_numpy(x)).numpy()[0].astype(np.float32)
   return y




def mlp_pos_err_scale():
   """Independent scale for MLP obs error (not plant slew)."""
   if POS_ERR_SPEED_FRAC is None:
       return 1.0
   return float(POS_ERR_SPEED_FRAC)




def verify_encoder_load(load_result, checkpoint_path, logfn):
   missing = list(load_result.missing_keys)
   unexpected = list(load_result.unexpected_keys)
   weight_missing = [k for k in missing if k.startswith("net.")]
   if weight_missing:
       logfn(
           "ENCODER INCOMPLETE — missing weight keys: %s  (checkpoint: %s)"
           % (weight_missing, checkpoint_path)
       )
   elif missing:
       logfn("Encoder missing_keys (non-weight): %s" % missing)
   else:
       logfn(
           "Encoder load OK: all weights present (not partial). checkpoint=%s"
           % checkpoint_path
       )
   if unexpected:
       logfn("Encoder unexpected_keys: %s" % unexpected)
   return not weight_missing




def publish_target(pub, target, duration):
   from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
   import rospy


   msg = JointTrajectory()
   msg.joint_names = PUBLISH_JOINTS
   pt = JointTrajectoryPoint()
   pt.positions = target.tolist()
   pt.time_from_start = rospy.Duration(duration)
   msg.points.append(pt)
   pub.publish(msg)




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
pos_hist = deque(maxlen=MLP_HIST)
vel_hist = deque(maxlen=MLP_HIST)
act4 = deque(maxlen=MLP_HIST)
pos_err_mlp = None


latest_fsr = np.zeros(N_FSR, dtype=np.float32)
fsr_lock = threading.Lock()
latest_biotac_pdc = np.full(5, np.nan, dtype=np.float32)
bt_lock = threading.Lock()


fsr_baseline = np.zeros(N_FSR, dtype=np.float32)
fsr_noise = np.ones(N_FSR, dtype=np.float32)
bt_baseline = np.zeros(N_BIOTAC, dtype=np.float32)
bt_noise = np.ones(N_BIOTAC, dtype=np.float32)
fsr_state = np.zeros(N_FSR, dtype=bool)
bt_state = np.zeros(N_BIOTAC, dtype=bool)


sim_tac = None
sim_tac_i = 0
start_q16 = None




def joint_callback(msg):
   global joint_ready, current_joint_pos, current_joint_vel
   global current_joint_pos16, current_joint_vel16
   import rospy


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
   import rospy
   import serial


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




def calibrate_fsr(seconds=2.0):
   global fsr_baseline, fsr_noise
   import rospy


   samples = []
   t_end = rospy.get_time() + seconds
   while rospy.get_time() < t_end and not rospy.is_shutdown():
       with fsr_lock:
           samples.append(latest_fsr.copy())
       rospy.sleep(0.01)
   s = np.array(samples)
   fsr_baseline = s.mean(0)
   fsr_noise = s.std(0) + 1e-6
   rospy.loginfo("FSR baseline=%s noise=%s", fsr_baseline, fsr_noise)




def calibrate_biotac(seconds=2.0):
   global bt_baseline, bt_noise
   import rospy


   samples = []
   t_end = rospy.get_time() + seconds
   while rospy.get_time() < t_end and not rospy.is_shutdown():
       with bt_lock:
           row = np.array([latest_biotac_pdc[i] for i in BIOTAC_IDX], dtype=np.float32)
       if np.all(np.isfinite(row)) and np.all(row > 0):
           samples.append(row)
       rospy.sleep(0.01)
   if not samples:
       rospy.logwarn("BioTac calibrate: no valid samples")
       return
   s = np.array(samples)
   bt_baseline = s.mean(0)
   bt_noise = s.std(0) + 1e-6
   rospy.loginfo("BioTac baseline=%s noise=%s", bt_baseline, bt_noise)




def read_tactile_real():
   """Live FSR+BioTac binary (MODE=real only; static K*sigma calibrate)."""
   global fsr_state, bt_state
   with fsr_lock:
       fsr_vals = latest_fsr.copy()
   fsr_hi = fsr_baseline + K_HI * fsr_noise
   fsr_lo = fsr_baseline + K_LO * fsr_noise
   fsr_state = np.where(
       fsr_vals > fsr_hi,
       True,
       np.where(fsr_vals < fsr_lo, False, fsr_state),
   )
   with bt_lock:
       bt_vals = np.array([latest_biotac_pdc[i] for i in BIOTAC_IDX], dtype=np.float32)
   bt_hi = bt_baseline + K_HI * bt_noise
   bt_lo = bt_baseline + K_LO * bt_noise
   for k in range(N_BIOTAC):
       v = bt_vals[k]
       if not np.isfinite(v) or v < 0:
           continue
       if v > bt_hi[k]:
           bt_state[k] = True
       elif v < bt_lo[k]:
           bt_state[k] = False
   t = np.zeros(NUM_TACTILE, dtype=np.float32)
   t[FSR_CHANNELS] = np.maximum(t[FSR_CHANNELS], fsr_state.astype(np.float32))
   if USE_BIOTAC:
       for k, ch in enumerate(BIOTAC_CH):
           t[ch] = max(t[ch], float(bt_state[k]))
   return t




def get_tactile():
   """Obs tactile: zero / sim playback / live."""
   global sim_tac_i
   if TACTILE_MODE == "zero":
       return np.zeros(NUM_TACTILE, dtype=np.float32)
   if TACTILE_MODE == "sim":
       assert sim_tac is not None
       t = sim_tac[min(sim_tac_i, len(sim_tac) - 1)].copy()
       sim_tac_i += 1
       return t
   return read_tactile_real()




def build_prop():
   pos_norm = unscale(current_joint_pos, JOINT_LOWER, JOINT_UPPER)
   vel_norm = current_joint_vel / JOINT_VEL_LIMIT
   error = last_command - current_joint_pos
   print('poserr: %f', error)
   # EXPERIMENTAL: per-joint scale (uniform scaled_error, CURL_AMP on curl joints)
   # instead of the single scalar scaled_error used in the diag/baseline scripts.
   return np.concatenate([pos_norm, vel_norm, SCALE_VEC*error, last_action]).astype(np.float32)




def build_prop_mlp():
   """MLP pos_err; optional * POS_ERR_SPEED_FRAC. Plant slew is separate.

   NOTE: curl amplification is NOT applied here -- these runs use USE_MLP_POS_ERR=False.
   """
   pos_norm = unscale(current_joint_pos, JOINT_LOWER, JOINT_UPPER)
   vel_norm = current_joint_vel / JOINT_VEL_LIMIT
   if len(pos_hist) == MLP_HIST:
       pos_hist.append(pos_norm.copy())
       vel_hist.append(vel_norm.copy())
   else:
       while len(pos_hist) < MLP_HIST:
           pos_hist.append(pos_norm.copy())
           vel_hist.append(vel_norm.copy())
   while len(act4) < MLP_HIST:
       act4.append(np.zeros(NUM_J, dtype=np.float32))
   pred_raw = predict_pos_err(pos_err_mlp, pos_hist, vel_hist, act4, last_action)
   scale = mlp_pos_err_scale()
   pred_err = (pred_raw * scale).astype(np.float32)
   prop = np.concatenate([pos_norm, vel_norm, pred_err, last_action]).astype(np.float32)
   return prop, pred_err, pred_raw




# ============================================================
# DIAG: pos_err decomposition (read-only, does not affect control)
# ============================================================
def diag_pos_err(pub_target, q_now):
   """Decompose this step's pos_err. Mirrors build_prop()'s SCALE_VEC application so
   obs_err here matches the LITERAL value fed to the policy under curl amplification.

   raw_err      = last_command[t-1] - q[t]        (what build_prop() used, pre-scale)
   tracking_err = published_cmd13[t] - q[t]        (sim-style error: no slew gap)
   slew_gap     = raw_err - tracking_err           (== last_command[t-1] - published[t])
   obs_err      = SCALE_VEC * raw_err              (the literal value fed to the policy)

   Called AFTER action_to_publish() but BEFORE last_command is updated for this step,
   so `last_command` here is still last_command[t-1].
   """
   pub13 = pub16_to_cmd13(pub_target)
   raw_err = last_command - q_now
   tracking_err = pub13 - q_now
   slew_gap = raw_err - tracking_err
   obs_err = SCALE_VEC * raw_err
   return raw_err.astype(np.float32), tracking_err.astype(np.float32), \
       slew_gap.astype(np.float32), obs_err.astype(np.float32)




def diag_print(step_i, raw_err, tracking_err, slew_gap, obs_err):
   if step_i % DIAG_PRINT_EVERY != 0:
       return
   l2_raw = float(np.linalg.norm(raw_err))
   l2_track = float(np.linalg.norm(tracking_err))
   l2_slew = float(np.linalg.norm(slew_gap))
   curl_bias = float(np.mean(obs_err[CURL_J2_IDX]))
   flag = (
       "  <-- persistent positive curl bias: policy thinks it hasn't reached target"
       if curl_bias > DIAG_CURL_BIAS_WARN else ""
   )
   print(
       "[DIAG t=%d] |raw_err|=%.3f |tracking_err|=%.3f(sim-like) |slew_gap|=%.3f "
       "obs_err(base x%s, curl x%s) curl_bias=%+.3f rad%s"
       % (step_i, l2_raw, l2_track, l2_slew, scaled_error, CURL_AMP, curl_bias, flag)
   )




def main():
   global last_action, last_command, sim_tac, sim_tac_i, start_q16, pos_err_mlp


   if TACTILE_MODE not in ("zero", "sim", "real"):
       raise ValueError("TACTILE_MODE must be 'zero', 'sim', or 'real', got %r" % TACTILE_MODE)


   import rospy
   from sensor_msgs.msg import JointState
   from trajectory_msgs.msg import JointTrajectory
   from sr_robot_msgs.msg import BiotacAll


   rospy.init_node("deploy_policy_simtactile_curlamp")
   pub = rospy.Publisher("/rh_trajectory_controller/command", JointTrajectory, queue_size=1)
   rospy.Subscriber("/joint_states", JointState, joint_callback)


   rospy.loginfo("Waiting for /joint_states ...")
   while not joint_ready and not rospy.is_shutdown():
       rospy.sleep(0.1)
   rospy.loginfo("Joint states OK.")


   rospy.Subscriber("/rh/tactile", BiotacAll, biotac_cb, queue_size=1)
   th = threading.Thread(target=serial_reader)
   th.daemon = True
   th.start()
   rospy.sleep(0.5)


   data = np.load(SIM_LOG_FILE, allow_pickle=True)
   rec_q = data["q"].astype(np.float32)
   assert rec_q.ndim == 2 and rec_q.shape[1] == 16, rec_q.shape
   if "joints" in data.files:
       assert list(data["joints"]) == PUBLISH_JOINTS, list(data["joints"])
   start_q16 = np.clip(rec_q[0], PUB_LOWER, PUB_UPPER)
   rospy.loginfo("Loaded %s  T=%d  (will start policy at q[0])", SIM_LOG_FILE, len(rec_q))


   if TACTILE_MODE == "sim":
       sim_tac = data["tac"].astype(np.float32)
       assert sim_tac.ndim == 2 and sim_tac.shape[1] == NUM_TACTILE, sim_tac.shape
       sim_tac_i = 0
       rospy.loginfo("SIM TACTILE playback T=%d", len(sim_tac))
   else:
       rospy.loginfo("TACTILE_MODE=%s (no sim-tac playback)", TACTILE_MODE)


   if TACTILE_MODE == "real":
       rospy.loginfo("Waiting for /rh/tactile ...")
       while not biotac_ready and not rospy.is_shutdown():
           rospy.sleep(0.1)
       rospy.loginfo("Static calibrate FSR+BioTac (hand empty)...")
       calibrate_fsr(2.0)
       calibrate_biotac(2.0)
       rospy.logwarn(
           "MODE=real uses static K*sigma calibrate. For warmup-envelope real tac, "
           "use deploy_warmup_trial15.py instead."
       )


   rospy.loginfo("Moving to sim q[0] ...")
   publish_target(pub, start_q16, 2.0)
   rospy.sleep(3.0)


   input(
       "At sim q[0]. PLACE BALLS, then Enter to start POLICY "
       "(TACTILE_MODE=%s, plant SPEED_FRAC=%s, MLP=%s, POS_ERR_SPEED_FRAC=%s, CURL_AMP=%s)..."
       % (TACTILE_MODE, SPEED_FRAC, USE_MLP_POS_ERR, POS_ERR_SPEED_FRAC, CURL_AMP)
   )


   rospy.loginfo("Loading checkpoint: %s", CHECKPOINT)
   ckpt = _torch_load(CHECKPOINT)
   encoder, policy = Encoder(), Policy()
   e_res = encoder.load_state_dict(ckpt["encoder"], strict=False)
   verify_encoder_load(e_res, CHECKPOINT, rospy.loginfo)
   rospy.loginfo("encoder missing=%s unexpected=%s", e_res.missing_keys, e_res.unexpected_keys)
   w0 = ckpt["encoder"]["net.0.weight"]
   if tuple(w0.shape) != (1024, OBS_DIM):
       raise RuntimeError(
           "Checkpoint encoder in_features=%s, expected %s. Wrong file?"
           % (w0.shape[1], OBS_DIM)
       )
   policy.load_state_dict(
       {k: v for k, v in ckpt["policy"].items() if k != "log_std_parameter"},
       strict=True,
   )
   encoder.eval()
   policy.eval()


   if USE_MLP_POS_ERR:
       rospy.loginfo(
           "pos_err mode=MLP  path=%s  POS_ERR_SPEED_FRAC=%s (obs x%s)",
           POS_ERR_MLP_PATH, POS_ERR_SPEED_FRAC, mlp_pos_err_scale(),
       )
       pos_err_mlp = load_pos_err_mlp(POS_ERR_MLP_PATH)
   else:
       rospy.loginfo(
           "pos_err mode=%s  SCALE_VEC=%s (base=%s, curl idx %s = %s)",
           "published" if USE_PUBLISHED_CMD_FOR_POS_ERR else "raw last_command",
           SCALE_VEC.tolist(), scaled_error, CURL_J2_IDX, CURL_AMP,
       )
       rospy.loginfo(
           "DIAG active: live pos_err decomposition every %d steps. "
           "curl_bias warn > %.3f rad",
           DIAG_PRINT_EVERY, DIAG_CURL_BIAS_WARN,
       )


   fsr_state[:] = False
   bt_state[:] = False
   last_command[:] = current_joint_pos
   last_action[:] = 0.0
   pos_hist.clear()
   vel_hist.clear()
   act4.clear()


   if USE_MLP_POS_ERR:
       p0, _, _ = build_prop_mlp()
   else:
       p0 = build_prop()
   t0 = get_tactile()
   prop_buffer.clear()
   tactile_buffer.clear()
   for _ in range(OBS_STACK):
       prop_buffer.append(p0.copy())
       tactile_buffer.append(t0.copy())


   rospy.sleep(1.0)
   rate = rospy.Rate(CONTROL_HZ)
   prev_pub = None
   step_i = 0
   rec = {
       "t": [], "q": [], "cmd": [], "tac": [], "fsr": [], "biotac_pdc": [],
       "act": [], "pred_pos_err": [], "pred_pos_err_raw": [],
       # DIAG: extra fields for offline analysis (see scratchpad/diagnose_pos_err.py)
       "diag_raw_err": [], "diag_tracking_err": [], "diag_slew_gap": [], "diag_obs_err": [],
   }
   rospy.loginfo(
       "Closed-loop @ %d Hz | MODE=%s | plant SF=%s | MLP=%s | "
       "POS_ERR_SPEED_FRAC=%s (x%s) | pub_pos_err=%s | CURL_AMP=%s | log=%s",
       CONTROL_HZ, TACTILE_MODE, SPEED_FRAC, USE_MLP_POS_ERR,
       POS_ERR_SPEED_FRAC, mlp_pos_err_scale(),
       USE_PUBLISHED_CMD_FOR_POS_ERR, CURL_AMP, LOG_NPZ,
   )


   try:
       while not rospy.is_shutdown():
           if USE_MLP_POS_ERR:
               prop, pred_err, pred_raw = build_prop_mlp()
           else:
               prop = build_prop()
               pred_err = pred_raw = None


           prop_buffer.append(prop)
           tactile_buffer.append(get_tactile())
           obs = np.concatenate(list(prop_buffer) + list(tactile_buffer))
           assert obs.shape[0] == OBS_DIM, obs.shape
           obs_t = torch.from_numpy(obs).unsqueeze(0)


           with torch.no_grad():
               action = policy(encoder(obs_t)).numpy()[0].astype(np.float32)


           # DIAG: snapshot q before last_command is updated below, for the decomposition.
           q_now = current_joint_pos.copy()


           pub_target, raw_cmd, j2_cmd = action_to_publish(
               action, current_joint_pos[CURL_J2_IDX], prev_pub
           )
           prev_pub = pub_target
           publish_target(pub, pub_target, 1.0 / CONTROL_HZ)


           # DIAG: only meaningful for the raw-last_command pos_err path (not MLP).
           if not USE_MLP_POS_ERR:
               d_raw, d_track, d_slew, d_obs = diag_pos_err(pub_target, q_now)
               diag_print(step_i, d_raw, d_track, d_slew, d_obs)
               rec["diag_raw_err"].append(d_raw)
               rec["diag_tracking_err"].append(d_track)
               rec["diag_slew_gap"].append(d_slew)
               rec["diag_obs_err"].append(d_obs)


           rec["t"].append(rospy.get_time())
           rec["q"].append(current_joint_pos.copy())
           rec["cmd"].append(pub_target.copy())
           rec["tac"].append(tactile_buffer[-1].copy())
           with fsr_lock:
               rec["fsr"].append(latest_fsr.copy())
           with bt_lock:
               rec["biotac_pdc"].append(latest_biotac_pdc.copy())
           rec["act"].append(action.copy())
           if pred_err is not None:
               rec["pred_pos_err"].append(pred_err.copy())
               rec["pred_pos_err_raw"].append(pred_raw.copy())


           if USE_MLP_POS_ERR:
               act4.append(last_action.copy())
           last_action[:] = action
           if not USE_MLP_POS_ERR:
               if USE_PUBLISHED_CMD_FOR_POS_ERR:
                   last_command[:] = pub16_to_cmd13(pub_target)
               else:
                   last_command[:] = raw_cmd
                   last_command[CURL_J2_IDX] = j2_cmd
           else:
               # keep last_command updated for logging consistency only
               last_command[:] = raw_cmd
               last_command[CURL_J2_IDX] = j2_cmd
           step_i += 1
           rate.sleep()
   finally:
       if rec["t"]:
           payload = dict((k, np.array(v)) for k, v in rec.items() if v)
           payload.update(
               tactile_mode=np.array(TACTILE_MODE),
               speed_frac=np.float32(SPEED_FRAC if SPEED_FRAC is not None else -1.0),
               use_published_cmd_for_pos_err=USE_PUBLISHED_CMD_FOR_POS_ERR,
               use_mlp_pos_err=bool(USE_MLP_POS_ERR),
               pos_err_speed_frac=np.float32(
                   POS_ERR_SPEED_FRAC if POS_ERR_SPEED_FRAC is not None else -1.0
               ),
               mlp_pos_err_scale=np.float32(mlp_pos_err_scale()),
               scaled_error=np.float32(scaled_error),
               curl_amp=np.float32(CURL_AMP),
               scale_vec=SCALE_VEC,
               control_hz=np.int32(CONTROL_HZ),
               sim_log_file=SIM_LOG_FILE,
               checkpoint=CHECKPOINT,
           )
           if USE_MLP_POS_ERR:
               payload["pos_err_mlp"] = POS_ERR_MLP_PATH
           np.savez(LOG_NPZ, **payload)
           rospy.loginfo("saved %s: %d steps", LOG_NPZ, len(rec["t"]))




if __name__ == "__main__":
   main()
