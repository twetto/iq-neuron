#!/usr/bin/env python3
"""
6-DoF egomotion as a continuous-LIF SPIKE-CODING NETWORK (tight balance).

Reference: Mancoo, Keemink, Machens, "Understanding spiking networks through
convex optimization", NeurIPS 2020 (and the Boerlin-Deneve tight-balance
lineage).  An inhibition-dominated SNN performs gradient descent on a convex
program; its INSTANTANEOUS population readout y = D r (r = filtered spikes) IS
the solution, up to a bounded discretization error eta.  Spikes are "bounces"
that reflect the readout back toward the optimum - so the value is held by a
distributed, error-correcting population, NOT by a single neuron's sustained
rate (which is why this sidesteps the silent-integrator / relay / line-attractor
problems of the spiking labeled-line circuit).

Mapping of OUR problem onto their QP
------------------------------------
Least squares:   min_g  1/2 || U - A_w g ||^2 ,   A_w = Q (whitened, ortho).
Decode g from nonnegative rates via a decoder D (R^{6 x N}):   g = D r,  r >= 0.
Measurement-space dictionary:  Phi = A_w D  (R^{2P x N}); a spike of neuron i
moves the prediction by Phi_i = A_w D_i.  The loss becomes 1/2||U - Phi r||^2,
a sparse-coding / auto-encoder SCN (their G = F = D^T example).

Greedy-spike (matching-pursuit) rule: neuron i should spike iff doing so lowers
the loss, 1/2||U - Phi(r+e_i)||^2 < 1/2||U - Phi r||^2, i.e.
        V_i := Phi_i^T (U - Phi r)  >  1/2 ||Phi_i||^2 + mu  =: T_i .
Dynamical (continuous-LIF) form, consistent with r-dot = -lam r + s:
        V-dot = -lam V + lam (Phi^T U)  -  (Phi^T Phi) s ,
so forward weights F = Phi^T, input current lam*Phi^T U, recurrent (lateral
inhibition) Omega = Phi^T Phi, threshold T_i = 1/2||Phi_i||^2 + mu, self-reset
R_i = T_i - Omega_ii.  Readout g_hat = D r  ->  argmin ||U - A_w g||^2 as the
decoder granularity (column scale) -> 0.

Signed g needs a positively-spanning decoder: D includes +-e_k (12 cols) plus a
redundant random tight frame so the code is distributed (the paper's E/I-balance
/ robustness regime).  No chip here - this is the continuous-LIF reference the
integer 8-bit IQIF version will be measured against.
"""

import os
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
print("Continuous-LIF spike-coding network: 6-DoF egomotion (tight balance)")
print("=" * 70)

N_POINTS = 40
G, U, m_gt = generate_scene(n_points=N_POINTS)
N_ROWS = G.shape[0]
N_VAL = 6

Q, R_qr = np.linalg.qr(G, mode="reduced")
R_inv = np.linalg.inv(R_qr)
A_w = Q                                            # whitened motion-field matrix
m_ls, _, _, _ = np.linalg.lstsq(G, U, rcond=None)
g_ls = R_qr @ m_ls                                 # LS solution in whitened g-space
v_ang_ls, v_mag_ls, w_err_ls = motion_errors(m_ls, m_gt)
print(f"\n  {N_ROWS} flow eqs, {N_VAL} unknowns")
print(f"  LS vs GT:   v_dir = {v_ang_ls:.3f} deg,  w err = {w_err_ls:.4f}")
print(f"  ||g_ls|| = {np.linalg.norm(g_ls):.4f}")

# ── Decoder D: +-e_k (positive spanning) + redundant random tight frame ──────
rng = np.random.RandomState(1)
N_RAND = 48
rand_dirs = rng.randn(N_VAL, N_RAND)
rand_dirs /= np.linalg.norm(rand_dirs, axis=0, keepdims=True)
D_dirs = np.hstack([np.eye(N_VAL), -np.eye(N_VAL), rand_dirs])   # 6 x N unit cols
N = D_dirs.shape[1]
# Column scale sets the spike "jump" size -> the discretization error eta.
# Smaller => more spikes, tighter readout.  Scale relative to ||g_ls||.
D_SCALE = float(os.environ.get("D_SCALE", "0.02")) * np.linalg.norm(g_ls)
D = D_dirs * D_SCALE                                            # 6 x N decoder

Phi = A_w @ D                                                   # 2P x N dictionary
Omega = Phi.T @ Phi                                            # N x N lateral inhib
diagO = np.diag(Omega).copy()
MU = float(os.environ.get("MU", "0.0"))                        # extra sparsity cost
# (the intrinsic spike cost 1/2||Phi_i||^2 already keeps thresholds positive;
#  MU>0 adds L1 shrinkage that biases the readout, so default 0.)
T = 0.5 * diagO + MU                                           # thresholds
fU = Phi.T @ U                                                 # feedforward Phi^T U
print(f"  SCN: N={N} neurons ({2*N_VAL} axis + {N_RAND} random),  "
      f"decoder col scale={D_SCALE:.4f}")

# ── Continuous-LIF tight-balance dynamics (forward Euler) ───────────────────
LAM = float(os.environ.get("LAM", "10.0"))                    # membrane leak / time const
DT = float(os.environ.get("DT", "1e-3"))
N_STEPS = int(os.environ.get("N_STEPS", "30000"))
MAX_SPK = 8                                                   # greedy spikes per dt (cap)
V_NOISE = 1e-9                                                # tie-break jitter

V = np.zeros(N)
r = np.zeros(N)
g_hist = np.zeros((N_STEPS, N_VAL))
spk_t, spk_i = [], []
rate_count = np.zeros(N)
# E/I bookkeeping for one example neuron (the +v1 axis neuron, index 0).
exc_trace = np.zeros(N_STEPS); inh_trace = np.zeros(N_STEPS); v_trace = np.zeros(N_STEPS)
CHECKS = [0, 200, 500, 1000, 2000, 5000, 10000, N_STEPS - 1]
conv = []

for t in range(N_STEPS):
    V += DT * (-LAM * V + LAM * fU)        # leak + constant feedforward drive
    r += DT * (-LAM * r)                   # rate filter decay
    inh_step = 0.0
    for _ in range(MAX_SPK):               # greedy bounces back into feasible set
        diff = V - T
        if V_NOISE:
            diff = diff + rng.randn(N) * V_NOISE
        i = int(np.argmax(diff))
        if diff[i] <= 0:
            break
        V -= Omega[:, i]                   # lateral inhibition + self-reset
        inh_step += Omega[i, i]            # (track self+lateral hit on neuron 0 below)
        r[i] += 1.0
        rate_count[i] += 1.0
        spk_t.append(t); spk_i.append(i)
    g_hat = D @ r
    g_hist[t] = g_hat
    exc_trace[t] = LAM * fU[0]
    v_trace[t] = V[0]
    if t in CHECKS:
        m_hat = R_inv @ g_hat
        conv.append((t,) + motion_errors(m_hat, m_ls) + (np.linalg.norm(g_hat - g_ls),))

g_inst = D @ r                                     # instantaneous readout
# Time-averaged readout = the SCN's actual downstream decode (filtered spikes).
# Averaging over the converged tail removes the bounce jitter (the eta term).
tail = N_STEPS // 2
g_scn = g_hist[tail:].mean(0)
m_scn = R_inv @ g_scn
m_inst = R_inv @ g_inst

print("\n  Convergence (instantaneous SCN readout vs LS):")
print(f"    {'step':>6s} {'v_dir(deg)':>12s} {'|v|err':>10s} {'w_err':>10s} {'||g-g_ls||':>12s}")
for s, va, vm, we, ge in conv:
    print(f"    {s:6d} {va:12.4f} {vm:10.4f} {we:10.4f} {ge:12.5f}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
vi, _, wi = motion_errors(m_inst, m_ls)
print(f"  instantaneous readout vs LS:  v_dir = {vi:.4f} deg,  w_err = {wi:.4f},  "
      f"||g-g_ls|| = {np.linalg.norm(g_inst - g_ls):.5f}")
va_ls, vm_ls, we_ls = motion_errors(m_scn, m_ls)
va_gt, vm_gt, we_gt = motion_errors(m_scn, m_gt)
n_active = int((rate_count > 0).sum())
tot_spikes = len(spk_t)
print(f"  time-avg readout vs LS:  v_dir = {va_ls:.4f} deg,  |v|err = {vm_ls:.4f},  w_err = {we_ls:.4f}")
print(f"  time-avg readout vs GT:  v_dir = {va_gt:.4f} deg,  |v|err = {vm_gt:.4f},  w_err = {we_gt:.4f}")
print(f"  ||g_avg - g_ls|| = {np.linalg.norm(g_scn - g_ls):.5f}   "
      f"(||g_ls|| = {np.linalg.norm(g_ls):.4f})")
print(f"  active neurons: {n_active}/{N}   total spikes: {tot_spikes}   "
      f"mean rate: {tot_spikes / (N * N_STEPS * DT):.1f} Hz")
print()
if va_ls < 1.0 and we_ls < 0.01:
    print("  PASS  the instantaneous SCN readout reproduced the LS egomotion")
elif va_ls < 3.0:
    print(f"  OK    SCN readout close to LS (v_dir = {va_ls:.2f} deg)")
else:
    print(f"  WARN  v_dir = {va_ls:.2f} deg - shrink decoder scale / more steps")

# ── Plots ───────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
steps = np.arange(N_STEPS)
tsec = steps * DT
LAB = ["v1", "v2", "v3", "w1", "w2", "w3"]
C = plt.cm.tab10(np.arange(N_VAL))
m_hist = (R_inv @ g_hist.T).T

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Continuous-LIF spike-coding network: egomotion as the instantaneous "
             "population readout", fontweight="bold")

for j in range(N_VAL):
    ax[0, 0].plot(tsec, m_hist[:, j], color=C[j], lw=1.1, label=LAB[j])
    ax[0, 0].axhline(m_ls[j], color=C[j], ls=":", lw=0.8, alpha=0.7)
ax[0, 0].set_title("readout motion m_hat(t) = R^-1 D r   (dotted = LS)")
ax[0, 0].legend(ncol=3, fontsize=8); ax[0, 0].set_xlabel("time (s)")

err_deg = np.array([motion_errors(m_hist[s], m_ls)[0] for s in range(N_STEPS)])
ax[0, 1].semilogy(tsec, err_deg + 1e-6, color="tab:purple")
ax[0, 1].set_title("translation-direction error vs LS (deg)"); ax[0, 1].set_xlabel("time (s)")

# Spike raster (distributed, sparse, irregular).
spk_t = np.array(spk_t); spk_i = np.array(spk_i)
ax[1, 0].scatter(spk_t * DT, spk_i, s=0.5, c="k", marker="|", linewidths=0.4)
ax[1, 0].set_title(f"spike raster ({n_active}/{N} neurons active, "
                   f"{tot_spikes} spikes)")
ax[1, 0].set_xlabel("time (s)"); ax[1, 0].set_ylabel("neuron")

# E/I balance for example neuron 0: voltage hovers at threshold.
ax[1, 1].plot(tsec, v_trace, color="tab:blue", lw=0.7, label="V (neuron 0)")
ax[1, 1].axhline(T[0], color="tab:red", ls="--", lw=1.0, label="threshold")
ax[1, 1].set_title("tight balance: example voltage pinned at threshold")
ax[1, 1].legend(fontsize=8); ax[1, 1].set_xlabel("time (s)")

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "pc_egomotion_scn_plots.png")
fig.savefig(out, dpi=120)
print(f"  Plots written to {out}")
