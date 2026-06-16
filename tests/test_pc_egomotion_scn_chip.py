#!/usr/bin/env python3
"""
6-DoF egomotion spike-coding network running on the ACTUAL IQIF chip (iqnet).

This is the on-hardware validation of the continuous-LIF SCN prototype
(test_pc_egomotion_scn.py) and its numpy 8-bit emulation (test_pc_egomotion_
scn_int.py).  Instead of emulating the integer membrane in numpy, we build the
network as a real iqnet and let the C++ IQIF neurons run it.

SCN recap:  min 1/2||U - A_w g||^2,  readout g = D r,  Phi = A_w D, lateral
inhibition Omega = Phi^T Phi, uniform threshold T = 1/2 D_scale^2 (unit-norm
decoder columns).  Chip map (membrane x in 0..255, fires at 255, x -= 255-reset):

    x = s*V + 127 ,  s = 255/(2T) = 255/D_scale^2     (V=T -> x=255, V=0 -> x=127)
    reset = 0                       -> self-drop 255 = s*Omega_ii (the diagonal)
    recurrent  j->i (i!=j):  w = -round(255 * cos_ij) = -round(s*Omega_ij)
    bias_i = round(s * dt * lam * (Phi^T U)_i)        constant per-tick drive
    leak-free neurons; readout = D * (leaky-filtered spikes), host-side decode.
"""

import os
import tempfile
import numpy as np


def motion_field_rows(x, y, Z):
    A = np.array([[-1.0, 0.0, x], [0.0, -1.0, y]])
    B = np.array([[x*y, -(1.0 + x*x), y], [1.0 + y*y, -x*y, -x]])
    return np.hstack([A / Z, B])


def generate_scene(n_points=40, noise_sigma=2e-3, seed=7):
    rng = np.random.RandomState(seed)
    m_gt = np.array([0.30, 0.05, 0.12, 0.02, -0.05, 0.015])
    xs = rng.uniform(-0.70, 0.70, n_points)
    ys = rng.uniform(-0.50, 0.50, n_points)
    Zs = rng.uniform(2.0, 10.0, n_points)
    G = np.vstack([motion_field_rows(x, y, Z) for x, y, Z in zip(xs, ys, Zs)])
    U = G @ m_gt + rng.randn(2 * n_points) * noise_sigma
    return G, U, m_gt


def motion_errors(m_est, m_ref):
    v_est, w_est = m_est[:3], m_est[3:]
    v_ref, w_ref = m_ref[:3], m_ref[3:]
    cos = np.dot(v_est, v_ref) / (np.linalg.norm(v_est) * np.linalg.norm(v_ref) + 1e-12)
    v_ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    v_mag = abs(np.linalg.norm(v_est) - np.linalg.norm(v_ref))
    return v_ang, v_mag, np.linalg.norm(w_est - w_ref)


print("=" * 70)
print("Spike-coding-network egomotion on the IQIF chip (iqnet)")
print("=" * 70)

G, U, m_gt = generate_scene(n_points=40)
N_VAL = 6
Q, R_qr = np.linalg.qr(G, mode="reduced")
R_inv = np.linalg.inv(R_qr)
A_w = Q
m_ls, _, _, _ = np.linalg.lstsq(G, U, rcond=None)
g_ls = R_qr @ m_ls

rng = np.random.RandomState(1)
rand_dirs = rng.randn(N_VAL, 48)
rand_dirs /= np.linalg.norm(rand_dirs, axis=0, keepdims=True)
D_dirs = np.hstack([np.eye(N_VAL), -np.eye(N_VAL), rand_dirs])     # unit columns
N = D_dirs.shape[1]
D_SCALE = 0.02 * np.linalg.norm(g_ls)
D = D_dirs * D_SCALE
Phi = A_w @ D
fU = Phi.T @ U
COS = D_dirs.T @ D_dirs
T = 0.5 * D_SCALE ** 2

VMAX, VMIN, RESET = 255, 0, 0
X0 = VMAX // 2                                        # operating point (V=0) = rest
s = VMAX / (2.0 * T)                                  # voltage -> membrane counts
# IQIF intrinsic damping toward rest (= the SCN -lam*V term): rest=X0,
# threshold=VMAX, shift small enough that f = (x-rest)>>shift actually pulls.
SHIFT = int(os.environ.get("SHIFT", "6"))             # smaller => stronger leak
NOISE = int(os.environ.get("NOISE", "8"))             # desynchronize the network
LAM = float(os.environ.get("LAM", "2.5"))
DT = 1e-3
N_STEPS = int(os.environ.get("N_STEPS", "40000"))
DRIVE = s * DT * LAM

bias = np.round(DRIVE * fU).astype(int)               # integer per-tick drive
W = np.round(VMAX * COS).astype(int)                  # integer recurrent (counts)
np.fill_diagonal(W, 0)                                # diagonal = reset (self-drop)
print(f"\n  N={N} neurons,  D_scale={D_SCALE:.5f}")
print(f"  bias range [{bias.min()},{bias.max()}]   |recurrent| max {np.abs(W).max()}")

from iqif import iqnet

tmpdir = tempfile.mkdtemp(prefix="iq_scn_")
par_path = os.path.join(tmpdir, "p.txt")
con_path = os.path.join(tmpdir, "c.txt")
with open(par_path, "w") as pf:                       # QIF: stable rest=X0, fire at 255
    for i in range(N):
        pf.write(f"{i} {X0} {VMAX} {RESET} {SHIFT} {SHIFT} {NOISE}\n")
n_conn = 0
with open(con_path, "w") as cf:
    for j in range(N):                                # pre = j (spiking neuron)
        for i in range(N):                            # post = i
            if i == j:
                continue
            w = -int(W[i, j])                         # lateral inhibition -Omega_ij
            if w != 0:
                cf.write(f"{j} {i} {w} 1\n")
                n_conn += 1
    if n_conn == 0:
        cf.write("0 0 0 1\n")
print(f"  recurrent connections: {n_conn}")

net = iqnet(par_path, con_path)
for i in range(N):
    net.set_vmax(i, VMAX)
    net.set_vmin(i, VMIN)
    net.set_surrogate_tau(i, 1)                       # no synaptic lingering (1-tick pulse)
    net.set_biascurrent(i, int(bias[i]))
    net.set_potential(i, int(X0))

# ── Run the chip ────────────────────────────────────────────────────────────
r = np.zeros(N)
g_hist = np.zeros((N_STEPS, N_VAL))
rate = np.zeros(N)
CHECKS = [0, 500, 2000, 5000, 10000, 20000, N_STEPS - 1]
conv = []
for t in range(N_STEPS):
    net.send_synapse()
    counts = net.get_all_spike_counts()[:N].astype(float)
    rate += counts
    r += counts - LAM * DT * r                        # leaky-filtered spikes
    g_hist[t] = D @ r
    if t in CHECKS:
        conv.append((t,) + motion_errors(R_inv @ g_hist[t], m_ls))

TAIL = N_STEPS // 2
g_chip = g_hist[TAIL:].mean(0)
m_chip = R_inv @ g_chip

print("\n  Convergence (chip readout vs LS):")
print(f"    {'tick':>6s} {'v_dir(deg)':>12s} {'|v|err':>10s} {'w_err':>10s}")
for tt, va, vm, we in conv:
    print(f"    {tt:6d} {va:12.4f} {vm:10.4f} {we:10.4f}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
va_ls, vm_ls, we_ls = motion_errors(m_chip, m_ls)
va_gt, vm_gt, we_gt = motion_errors(m_chip, m_gt)
n_active = int((rate > 0).sum())
print(f"  chip readout vs LS:  v_dir = {va_ls:.4f} deg,  |v|err = {vm_ls:.4f},  w_err = {we_ls:.4f}")
print(f"  chip readout vs GT:  v_dir = {va_gt:.4f} deg,  |v|err = {vm_gt:.4f},  w_err = {we_gt:.4f}")
print(f"  ||g_chip - g_ls|| = {np.linalg.norm(g_chip - g_ls):.5f}  (rel {np.linalg.norm(g_chip-g_ls)/np.linalg.norm(g_ls)*100:.1f}%)")
print(f"  active neurons: {n_active}/{N}   total spikes: {int(rate.sum())}   "
      f"mean rate: {rate.sum()/(N*N_STEPS):.3f} /tick/neuron")
print()
if va_ls < 3.0 and we_ls < 0.03:
    print("  PASS  the IQIF chip's SCN readout reproduced the LS egomotion")
elif va_ls < 8.0:
    print(f"  OK    chip readout closing onto LS (v_dir = {va_ls:.2f} deg)")
else:
    print(f"  WARN  chip v_dir = {va_ls:.2f} deg")

# ── Plot ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ticks = np.arange(N_STEPS)
LAB = ["v1", "v2", "v3", "w1", "w2", "w3"]
C = plt.cm.tab10(np.arange(N_VAL))
m_hist = (R_inv @ g_hist.T).T
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Spike-coding-network egomotion on the IQIF chip", fontweight="bold")
for j in range(N_VAL):
    ax[0].plot(ticks, m_hist[:, j], color=C[j], lw=1.0, label=LAB[j])
    ax[0].axhline(m_ls[j], color=C[j], ls=":", lw=0.8, alpha=0.7)
ax[0].set_title("chip readout m_hat(t)  (dotted = LS)"); ax[0].legend(ncol=3, fontsize=8)
ax[0].set_xlabel("chip tick")
err_deg = np.array([motion_errors(m_hist[t], m_ls)[0] for t in range(N_STEPS)])
ax[1].semilogy(ticks, err_deg + 1e-6, color="tab:purple")
ax[1].set_title(f"translation-dir error vs LS (deg)  -> {va_ls:.2f} deg")
ax[1].set_xlabel("chip tick")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "pc_egomotion_scn_chip_plots.png")
fig.savefig(out, dpi=120)
print(f"  Plot written to {out}")
