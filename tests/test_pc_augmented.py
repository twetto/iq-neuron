#!/usr/bin/env python3
"""
Two-population Predictive Coding: Nesterov acceleration + spiking neurons.

Part 1: NumPy Nesterov vs gradient descent on the RAW system.
  Shows O(sqrt(kappa)) convergence — the two-population / augmented
  matrix approach works.  Equivalent to predictive coding with
  look-ahead: predict -> compute error -> correct.

Part 2: Spiking PC circuit on the QR-whitened system.
  Error neurons (eps+/eps-) SPIKE at rates proportional to |error|.
  Spikes propagate through the connection matrix (encoding A^T) to
  sub-threshold value neurons.  This is biologically plausible PC
  with actual spike-based gradient transmission.

  Why QR-whitened: IQIF integer quantization error (+-0.5 per step)
  gets amplified by 1/sigma_min ~ 32x on the raw system, drowning
  the small singular value components.  QR whitening (kappa=1)
  eliminates this amplification.
"""

import os
import sys
import tempfile
import numpy as np
import cv2

# ── Helpers ─────────────────────────────────────────────────────────────

def skew(t):
    return np.array([[ 0,    -t[2],  t[1]],
                     [ t[2],  0,    -t[0]],
                     [-t[1],  t[0],  0   ]], dtype=np.float64)

def generate_test_data(n_points=20, noise_sigma=0.5, seed=42):
    rng = np.random.RandomState(seed)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    angle = 0.15
    R = np.array([[ np.cos(angle), 0, np.sin(angle)],
                  [ 0,             1, 0             ],
                  [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float64)
    t = np.array([1.0, 0.2, 0.1], dtype=np.float64)
    t = t / np.linalg.norm(t)
    E = skew(t) @ R
    F_true = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    F_true = F_true / np.linalg.norm(F_true, 'fro')
    pts3d = rng.randn(n_points, 3) * 3 + np.array([0, 0, 8])
    P1 = K @ np.eye(3, 4)
    pts1_h = (P1 @ np.hstack([pts3d, np.ones((n_points, 1))]).T).T
    pts1 = pts1_h[:, :2] / pts1_h[:, 2:3]
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    pts2_h = (P2 @ np.hstack([pts3d, np.ones((n_points, 1))]).T).T
    pts2 = pts2_h[:, :2] / pts2_h[:, 2:3]
    pts1 += rng.randn(n_points, 2) * noise_sigma
    pts2 += rng.randn(n_points, 2) * noise_sigma
    return pts1, pts2, F_true

def hartley_normalize(pts):
    mean = pts.mean(axis=0)
    centered = pts - mean
    mean_dist = np.mean(np.sqrt(np.sum(centered**2, axis=1)))
    scale = np.sqrt(2) / (mean_dist + 1e-12)
    T = np.array([[scale, 0, -scale*mean[0]],
                  [0, scale, -scale*mean[1]],
                  [0, 0, 1]], dtype=np.float64)
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T

def build_A_matrix(pts1, pts2):
    n = len(pts1)
    A = np.zeros((n, 9), dtype=np.float64)
    for i in range(n):
        u, v = pts1[i]
        up, vp = pts2[i]
        A[i] = [up*u, up*v, up, vp*u, vp*v, vp, u, v, 1]
    return A

def enforce_rank2(F):
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    return U @ np.diag(S) @ Vt

def mean_algebraic_error(F, pts1, pts2):
    n = len(pts1)
    err = sum(abs(np.array([pts2[i,0], pts2[i,1], 1.0]) @ F @ np.array([pts1[i,0], pts1[i,1], 1.0])) for i in range(n))
    return err / n

def f_matrix_angular_distance(F1, F2):
    f1 = F1.flatten() / np.linalg.norm(F1, 'fro')
    f2 = F2.flatten() / np.linalg.norm(F2, 'fro')
    return np.degrees(np.arccos(min(abs(np.dot(f1, f2)), 1.0)))

def f_to_F(f8, T1, T2):
    f_full = np.append(f8, 1.0)
    F = T2.T @ enforce_rank2(f_full.reshape(3, 3)) @ T1
    return F / np.linalg.norm(F, 'fro')

# ══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("Two-Population PC: Nesterov + Spiking Neurons")
print("=" * 65)

N_POINTS = 20
pts1_raw, pts2_raw, F_true = generate_test_data(n_points=N_POINTS, noise_sigma=0.5)
pts1_n, T1 = hartley_normalize(pts1_raw)
pts2_n, T2 = hartley_normalize(pts2_raw)
A = build_A_matrix(pts1_n, pts2_n)
A_red = A[:, :8]
b = -A[:, 8]

# ── SVD and conditioning ───────────────────────────────────────────────
_, S_vals, _ = np.linalg.svd(A_red, full_matrices=False)
sigma_max, sigma_min = S_vals[0], S_vals[-1]
kappa_A = sigma_max / sigma_min
kappa_ATA = kappa_A**2

# QR whitening (same as test_pc_8point.py)
Q, R_qr = np.linalg.qr(A_red, mode='reduced')
R_inv = np.linalg.inv(R_qr)
A_w = A_red @ R_inv  # = Q, kappa = 1

f_ls, _, _, _ = np.linalg.lstsq(A_red, b, rcond=None)
F_ls = f_to_F(f_ls, T1, T2)
alg_err_ls = mean_algebraic_error(F_ls, pts1_raw, pts2_raw)
ang_ls_gt = f_matrix_angular_distance(F_ls, F_true)

print(f"\n  kappa(A) = {kappa_A:.0f},  kappa(A^T A) = {kappa_ATA:.0f}")
print(f"  LS reference: alg_err = {alg_err_ls:.4e},  ang_to_GT = {ang_ls_gt:.2f} deg")

# ═══════════════════════════════════════════════════════════════════════
# PART 1: NumPy GD vs Nesterov (raw system, 5000 steps)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Part 1: Nesterov Acceleration (NumPy, raw system)")
print("=" * 65)

N_STEPS = 5000
CHECKPOINTS = [0, 100, 500, 1000, 2000, 4999]

# -- Gradient Descent --
lr_gd = 0.9 * 2.0 / sigma_max**2
f_gd = np.zeros(8)
conv_gd = []
for step in range(N_STEPS):
    f_gd += lr_gd * A_red.T @ (b - A_red @ f_gd)
    if step in CHECKPOINTS:
        conv_gd.append((step, f_matrix_angular_distance(f_to_F(f_gd, T1, T2), F_ls)))

print(f"\n  Gradient Descent (lr = {lr_gd:.6f}):")
for step, ang in conv_gd:
    print(f"    step {step:5d}: ang_to_LS = {ang:.4f} deg")

# -- Nesterov --
L = sigma_max**2
step_size = 1.0 / L
alpha_nest = (kappa_A - 1.0) / (kappa_A + 1.0)

f_nest = np.zeros(8)
f_nest_prev = np.zeros(8)
conv_nest = []
for step in range(N_STEPS):
    y = f_nest + alpha_nest * (f_nest - f_nest_prev)
    f_nest_prev = f_nest.copy()
    f_nest = y + step_size * A_red.T @ (b - A_red @ y)
    if step in CHECKPOINTS:
        conv_nest.append((step, f_matrix_angular_distance(f_to_F(f_nest, T1, T2), F_ls)))

F_nest = f_to_F(f_nest, T1, T2)
ang_nest_ls = f_matrix_angular_distance(F_nest, F_ls)
ang_nest_gt = f_matrix_angular_distance(F_nest, F_true)
alg_err_nest = mean_algebraic_error(F_nest, pts1_raw, pts2_raw)

print(f"\n  Nesterov (alpha = {alpha_nest:.6f}, step = {step_size:.6f}):")
for step, ang in conv_nest:
    print(f"    step {step:5d}: ang_to_LS = {ang:.4f} deg")

speedup = conv_gd[-1][1] / max(ang_nest_ls, 0.001)
print(f"\n  After 5000 steps:  GD = {conv_gd[-1][1]:.2f} deg,  Nesterov = {ang_nest_ls:.2f} deg")
print(f"  Nesterov converges {speedup:.0f}x faster (O(sqrt(kappa)) vs O(kappa)).")

# ═══════════════════════════════════════════════════════════════════════
# PART 2: Spiking PC Circuit (QR-whitened system)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Part 2: Spiking PC Circuit (QR-whitened system)")
print("=" * 65)

from iqif import iqnet

# Architecture (all 48 neurons share the same LIF-integrator parameters):
#   eps+  neurons [ 0..19]:  fire at rate ~ max(0,  eps_i)
#   eps-  neurons [20..39]:  fire at rate ~ max(0, -eps_i)
#   val   neurons [40..47]:  one per g component; potential integrates
#                            the gradient signal. Signed, so val_j
#                            spikes only while g_j drifts positive;
#                            for g_j < 0 the neuron is silent but the
#                            shadow-potential readout stays correct.
#
# Readout:  g_j = shadow(val_j) / V_SCALE_S    (see read_g())
#
# Connections use SIGNED integer weights — iqnet allows negative weights
# (see iq_neuron.cpp:92, add_input just does accumulator += weight). Only
# the firing RATE has to be non-negative, which is why eps is split into
# eps+ / eps-. Two connections per (i, j):
#
#     eps+_i -> val_j  with weight +round(A_w[i,j] * W_SCALE)
#     eps-_i -> val_j  with weight -round(A_w[i,j] * W_SCALE)
#
# Net drive on val_j per step = sum_i A_w[i,j] * (eps+_rate_i - eps-_rate_i)
#                             = sum_i A_w[i,j] * eps_i = (A_w^T eps)_j.

N_ERR_POS  = N_POINTS
N_ERR_NEG  = N_POINTS
N_ERR_TOT  = N_ERR_POS + N_ERR_NEG
N_VAL      = 8
N_TOTAL    = N_ERR_TOT + N_VAL
VAL_OFFSET = N_ERR_TOT   # 40

W_SCALE = 20
# Shared LIF parameters. shift_a=15 makes x>>15 round to zero for
# |x|<32768, i.e. no restoring force — neurons act as pure integrators.
# Each spike resets x by QUANTUM (= VMAX - reset); shadow-pot adds back
# that lost quantum so the integrator's cumulative input is recovered.
NEURON_VMAX   = 5000
NEURON_RESET  = 4500
NEURON_QUANTUM = NEURON_VMAX - NEURON_RESET
NEURON_VMIN   = -1000000    # wide enough that clamping never bites val

tmpdir = tempfile.mkdtemp(prefix="iqtest_spike_")
par_path = os.path.join(tmpdir, "params.txt")
con_path = os.path.join(tmpdir, "conn.txt")

# Unified neuron params for BOTH error and value neurons.
# Note: the 'threshold' column only feeds Izhikevich f_min math; the
# real spike gate is VMAX (set below via set_vmax).
with open(par_path, "w") as pf:
    for i in range(N_TOTAL):
        pf.write(f"{i} 0 {NEURON_VMAX} {NEURON_RESET} 15 1 0\n")

# tau_conn=8: accumulator decays ~1/8 per step, so a sign flip in eps
# propagates to val within a few steps.
with open(con_path, "w") as cf:
    tau_conn = 8
    n_conn = 0
    for i in range(N_POINTS):
        for j in range(N_VAL):
            w = int(round(A_w[i, j] * W_SCALE))
            if w == 0:
                continue
            cf.write(f"{i}          {VAL_OFFSET + j} {+w} {tau_conn}\n")  # eps+
            cf.write(f"{N_POINTS+i} {VAL_OFFSET + j} {-w} {tau_conn}\n")  # eps-
            n_conn += 2
    if n_conn == 0:
        cf.write("0 0 0 32\n")

net = iqnet(par_path, con_path)
print(f"\n  Neurons: {N_ERR_TOT} error + {N_VAL} value (all spiking) = {N_TOTAL}")
print(f"  Connections: {n_conn} (signed A_w * W_SCALE={W_SCALE})")
print(f"  VMAX = {NEURON_VMAX} (spike gate), reset = {NEURON_RESET}")

# VMAX defaults to 255 in iq_neuron.h; override for all neurons so both
# error and value populations share the same spike threshold. VMIN is
# only relevant for val neurons (error pot stays positive under bias).
for i in range(N_TOTAL):
    net.set_vmax(i, NEURON_VMAX)
    net.set_vmin(i, NEURON_VMIN)

# Error neuron bias scale. With shift_a=15 (no leak) firing rate
# converges to bias / (VMAX - reset) = bias / 500. RATE_SCALE=50 gives
# rate 0.1/step at |eps|=1, comparable to the old 120-threshold config.
RATE_SCALE = 50.0

N_SPIKE_STEPS = 3000
SPIKE_CHECKS = [0, 100, 200, 500, 1000, 1500, 2000, 2999]

# V_SCALE_S: shadow-potential units per g-unit. Higher = lower effective
# learning rate = more stable. Target max shadow ~20000 at convergence
# gives good integer precision without runaway spiking.
g_ls = R_qr @ f_ls
V_SCALE_S = 20000.0 / np.max(np.abs(g_ls))
print(f"  V_SCALE_S = {V_SCALE_S:.1f} (target max shadow ~20000)")

# Random g_init off the g_ls ray, so eps_init has mixed signs and BOTH
# eps+ and eps- populations fire. Seed 11 gives a +12/-8 split.
g_init = np.random.RandomState(11).randn(8) * np.max(np.abs(g_ls)) * 0.8

# Desired shadow at t=0 is g_init * V_SCALE_S. The raw membrane
# potential must satisfy VMIN <= raw < VMAX; any positive excess is
# stashed in val_offset (as if the neuron had already spiked that many
# times). Negative init fits directly since VMIN is very low.
shadow_init = g_init * V_SCALE_S
raw_init    = np.minimum(shadow_init, NEURON_VMAX - 1)
val_offset  = shadow_init - raw_init   # non-negative

for j in range(N_VAL):
    net.set_potential(VAL_OFFSET + j, int(round(raw_init[j])))

eps_init = b - A_w @ g_init
print(f"  g_init random, eps_init: {(eps_init > 0).sum()} positive, "
      f"{(eps_init < 0).sum()} negative")

# iqnet's spike_count() resets the counter on each read, so accumulate
# into a Python array to get cumulative totals.
val_spikes_cum = np.zeros(N_VAL, dtype=np.float64)

def read_g():
    """Reconstruct g from each val neuron's shadow potential.

    shadow_j = raw_pot_j + cum_spikes_j * NEURON_QUANTUM + val_offset_j
             ≈ what the unclamped integrator would hold without resets.
    The per-spike quantum (NEURON_QUANTUM = VMAX - reset) is exactly
    what each spike discards from the membrane, so adding it back
    recovers the total integrated input. val_offset_j absorbs any init
    shadow beyond VMAX-1 that couldn't be loaded into the raw membrane.
    """
    val_spikes_cum[:] += [net.spike_count(VAL_OFFSET + j) for j in range(N_VAL)]
    pot = np.array([net.potential(VAL_OFFSET + j) for j in range(N_VAL)],
                   dtype=np.float64)
    shadow = pot + val_spikes_cum * NEURON_QUANTUM + val_offset
    return shadow / V_SCALE_S

conv_spike = []

for step in range(N_SPIKE_STEPS):
    g_scaled = read_g()

    # Compute error in whitened space: eps = b - A_w * g
    eps_vals = b - A_w @ g_scaled

    # Drive error neurons
    for i in range(N_POINTS):
        bias_pos = max(0, int(round(eps_vals[i] * RATE_SCALE)))
        bias_neg = max(0, int(round(-eps_vals[i] * RATE_SCALE)))
        net.set_biascurrent(i, bias_pos)
        net.set_biascurrent(N_POINTS + i, bias_neg)

    net.send_synapse()

    if step in SPIKE_CHECKS:
        f_check = R_inv @ g_scaled
        try:
            ang = f_matrix_angular_distance(f_to_F(f_check, T1, T2), F_ls)
        except Exception:
            ang = float('nan')
        conv_spike.append((step, ang))

# Final results
g_final = read_g()
f_spike_final = R_inv @ g_final

total_sp_ep = sum(net.spike_count(i) for i in range(N_ERR_POS))
total_sp_en = sum(net.spike_count(N_POINTS + i) for i in range(N_ERR_NEG))
# Val spikes already accumulated inside read_g during the loop; fold in
# any residual spikes emitted by the final update_state after the last read.
val_spikes_cum[:] += [net.spike_count(VAL_OFFSET + j) for j in range(N_VAL)]
total_sp_val = int(val_spikes_cum.sum())

try:
    F_spike = f_to_F(f_spike_final, T1, T2)
    ang_spike_ls = f_matrix_angular_distance(F_spike, F_ls)
    ang_spike_gt = f_matrix_angular_distance(F_spike, F_true)
    alg_err_spike = mean_algebraic_error(F_spike, pts1_raw, pts2_raw)
except Exception:
    ang_spike_ls = ang_spike_gt = alg_err_spike = float('nan')

print(f"\n  Convergence:")
for step, ang in conv_spike:
    print(f"    step {step:5d}: ang_to_LS = {ang:.4f} deg")

print(f"\n  Final: alg_err = {alg_err_spike:.4e}, ang_to_GT = {ang_spike_gt:.2f} deg")
print(f"  Error spikes: eps+ = {total_sp_ep}, eps- = {total_sp_en}")
print(f"  Value spikes: {total_sp_val} (across {N_VAL} neurons)")

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Summary")
print("=" * 65)
print(f"  {'Method':<45s} {'Alg.Err':>10s} {'->GT':>8s} {'->LS':>8s}")
print(f"  {'-'*45} {'-'*10} {'-'*8} {'-'*8}")
print(f"  {'NumPy LS (direct)':<45s} {alg_err_ls:>10.4e} {ang_ls_gt:>7.2f}d {'ref':>8s}")

F_gd = f_to_F(f_gd, T1, T2)
alg_err_gd = mean_algebraic_error(F_gd, pts1_raw, pts2_raw)
ang_gd_gt = f_matrix_angular_distance(F_gd, F_true)
print(f"  {'NumPy GD (raw, 5000 steps)':<45s} {alg_err_gd:>10.4e} {ang_gd_gt:>7.2f}d {conv_gd[-1][1]:>7.2f}d")
print(f"  {'NumPy Nesterov (raw, 5000 steps)':<45s} {alg_err_nest:>10.4e} {ang_nest_gt:>7.2f}d {ang_nest_ls:>7.2f}d")
print(f"  {'IQIF Spiking PC (whitened, 3000 steps)':<45s} {alg_err_spike:>10.4e} {ang_spike_gt:>7.2f}d {ang_spike_ls:>7.2f}d")

print()
total_sp_err = total_sp_ep + total_sp_en
if ang_spike_ls < 5.0:
    print(f"  PASS  Spiking PC converged to LS (< 5 deg)")
    print(f"        {total_sp_err} error spikes, {total_sp_val} value spikes")
elif ang_spike_ls < 15.0:
    print(f"  OK    Spiking PC approaching LS ({ang_spike_ls:.2f} deg)")
else:
    print(f"  WARN  Spiking PC ang_to_LS = {ang_spike_ls:.2f} deg")

print(f"\n  Part 1: Nesterov = predict + error + correct = predictive coding.")
print(f"          Converges in O(sqrt(kappa)) = {int(kappa_A)} vs O(kappa) = {kappa_ATA:.0f}.")
print(f"  Part 2: All 48 neurons share the same LIF-integrator params.")
print(f"          Signed weights carry eps_i from eps+/- into val_j,")
print(f"          and val_j's shadow potential encodes signed g_j.")
