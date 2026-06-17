#!/usr/bin/env python3
"""
INHIBITION-DOMINATED spike-coding-network egomotion on the IQIF chip (iqnet).

test_pc_egomotion_scn_chip.py locked into mass firing (~0.75/tick) because the
+-e_k decoder pairs (cosine -1) get an EXCITATORY lateral weight +255, and under
the chip's synchronous one-tick-delayed update they ping-pong.  MKM-2020's
delay-robustness comes from an inhibition-dominated recurrent (G_ij>=0 -> all
lateral weights <=0 -> a delayed spike can only INHIBIT, never trigger a
follow-up spike).  Two levers, both togglable here:

  FRAME=rand : generic overcomplete random frame (NO exactly-anti-parallel
               pairs)         -> bounds the worst excitation well below threshold
  CLIP=1     : zero every excitatory lateral weight (keep only inhibitory)
               -> recurrent is PROVABLY all-inhibitory (the guarantee)

A random frame of N unit vectors in R^6 still positively spans R^6 (so r>=0 can
decode any signed g), but pairwise cosines are mild, so the residual excitation
is small; clipping removes it entirely.  Everything else matches the chip SCN:
QIF leak toward rest (rest=127, threshold=255, small shift), per-tick integer
bias, readout g = D * (leaky-filtered spikes).
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


FRAME = os.environ.get("FRAME", "rand")               # "rand" or "axis"
CLIP = os.environ.get("CLIP", "1") == "1"             # zero excitatory weights
N_RAND = int(os.environ.get("N_RAND", "72"))
SHIFT = int(os.environ.get("SHIFT", "6"))
NOISE = int(os.environ.get("NOISE", "8"))
LAM = float(os.environ.get("LAM", "2.0"))
N_STEPS = int(os.environ.get("N_STEPS", "40000"))

print("=" * 70)
print(f"Inhibition-dominated SCN egomotion on IQIF chip  "
      f"(FRAME={FRAME}, CLIP={CLIP})")
print("=" * 70)

G, U, m_gt = generate_scene(n_points=40)
N_VAL = 6
Q, R_qr = np.linalg.qr(G, mode="reduced")
R_inv = np.linalg.inv(R_qr)
A_w = Q
m_ls, _, _, _ = np.linalg.lstsq(G, U, rcond=None)
g_ls = R_qr @ m_ls

rng = np.random.RandomState(1)
if FRAME == "axis":
    rand_dirs = rng.randn(N_VAL, 48)
    rand_dirs /= np.linalg.norm(rand_dirs, axis=0, keepdims=True)
    D_dirs = np.hstack([np.eye(N_VAL), -np.eye(N_VAL), rand_dirs])
else:                                                  # generic overcomplete frame
    D_dirs = rng.randn(N_VAL, N_RAND)
    D_dirs /= np.linalg.norm(D_dirs, axis=0, keepdims=True)
N = D_dirs.shape[1]
D_SCALE = 0.02 * np.linalg.norm(g_ls)
D = D_dirs * D_SCALE
Phi = A_w @ D
fU = Phi.T @ U
COS = D_dirs.T @ D_dirs
np.fill_diagonal(COS, 1.0)
off = COS - np.eye(N)
print(f"\n  N={N} neurons,  worst pairwise cosine = {off.min():.3f}  "
      f"(axis pairs would be -1.0)")

VMAX, VMIN, RESET = 255, 0, 0
X0 = VMAX // 2
T = 0.5 * D_SCALE ** 2
s = VMAX / (2.0 * T)
DT = 1e-3
DRIVE = s * DT * LAM
bias = np.round(DRIVE * fU).astype(int)

# Recurrent weight (j->i, i!=j) = -round(255 * cos_ij).  Excitatory entries are
# the ones with cos_ij < 0 (weight > 0); CLIP zeros them -> all-inhibitory.
W = np.round(VMAX * COS).astype(float)                 # = round(s*Omega) in counts
np.fill_diagonal(W, 0.0)
weight = -W                                            # connection weight on spike
n_exc_before = int((weight > 0).sum())
if CLIP:
    weight = np.minimum(weight, 0.0)                   # keep only inhibitory (sign-clip)
# W_BUDGET: bound the worst incoming Sum|w| <= budget by a GLOBAL uniform scale
# (keeps both signs AND the relative structure -> the LS *direction* is
# preserved; only the readout magnitude inflates by 1/f, corrected below).
W_BUDGET = float(os.environ.get("W_BUDGET", "0"))
w_corr = 1.0
if W_BUDGET > 0:
    max_rs = np.abs(weight).sum(axis=1).max()
    if max_rs > W_BUDGET:
        f = W_BUDGET / max_rs
        weight *= f
        w_corr = f  # readout correction: g_true ≈ f · (D r)
    print(f"  W_BUDGET={W_BUDGET:.0f}: global scale f={w_corr:.4f}, "
          f"max incoming Sum|w| {max_rs:.0f} -> {np.abs(weight).sum(axis=1).max():.0f}")
# EXC_BUDGET: bound only the EXCITATORY (positive) incoming sum per neuron --
# excitation is what cascades; inhibition only pushes down, so keep it at full
# strength (preserves the competition that solves the LS). 0 = off.
EXC_BUDGET = float(os.environ.get("EXC_BUDGET", "0"))
if EXC_BUDGET > 0:
    for i in range(N):
        pos = weight[i, :] > 0
        es = weight[i, pos].sum()
        if es > EXC_BUDGET:
            weight[i, pos] *= EXC_BUDGET / es
    print(f"  EXC_BUDGET={EXC_BUDGET:.0f}: max excitatory incoming sum -> "
          f"{max((weight[i, weight[i,:]>0].sum()) for i in range(N)):.0f}")
weight = np.round(weight).astype(int)
n_exc_after = int((weight > 0).sum())
print(f"  excitatory lateral connections: {n_exc_before} -> {n_exc_after}")
print(f"  bias range [{bias.min()},{bias.max()}]   |weight| max {int(np.abs(weight).max())}")

from iqif import iqnet

tmpdir = tempfile.mkdtemp(prefix="iq_inh_")
par_path = os.path.join(tmpdir, "p.txt")
con_path = os.path.join(tmpdir, "c.txt")
with open(par_path, "w") as pf:                        # QIF: stable rest=X0, fire at 255
    for i in range(N):
        pf.write(f"{i} {X0} {VMAX} {RESET} {SHIFT} {SHIFT} {NOISE}\n")
n_conn = 0
with open(con_path, "w") as cf:
    for j in range(N):
        for i in range(N):
            if i == j:
                continue
            w = int(weight[i, j])
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
    net.set_surrogate_tau(i, 1)
    net.set_biascurrent(i, int(bias[i]))
    net.set_potential(i, int(X0))

r = np.zeros(N)
g_hist = np.zeros((N_STEPS, N_VAL))
rate = np.zeros(N)
spk_t, spk_i = [], []
for t in range(N_STEPS):
    net.send_synapse()
    counts = net.get_all_spike_counts()[:N].astype(float)
    rate += counts
    r += counts - LAM * DT * r
    g_hist[t] = D @ r
    fired = np.nonzero(counts)[0]
    if fired.size:
        spk_t.extend([t] * fired.size)
        spk_i.extend(fired.tolist())

TAIL = N_STEPS // 2
g_chip = g_hist[TAIL:].mean(0) * w_corr  # undo the global recurrent down-scale
m_chip = R_inv @ g_chip

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
va_ls, vm_ls, we_ls = motion_errors(m_chip, m_ls)
va_gt, vm_gt, we_gt = motion_errors(m_chip, m_gt)
rel = np.linalg.norm(g_chip - g_ls) / np.linalg.norm(g_ls)
print(f"  chip readout vs LS:  v_dir = {va_ls:.4f} deg,  w_err = {we_ls:.4f},  rel.eta = {rel*100:.1f}%")
print(f"  chip readout vs GT:  v_dir = {va_gt:.4f} deg,  w_err = {we_gt:.4f}")
print(f"  active neurons: {int((rate>0).sum())}/{N}   "
      f"mean rate: {rate.sum()/(N*N_STEPS):.3f} /tick/neuron")
print()
if va_ls < 3.0 and we_ls < 0.03:
    print("  PASS  inhibition-dominated SCN reproduced LS egomotion on the chip")
elif va_ls < 10.0:
    print(f"  OK    closing onto LS (v_dir = {va_ls:.2f} deg)")
else:
    print(f"  WARN  v_dir = {va_ls:.2f} deg  (mean rate {rate.sum()/(N*N_STEPS):.2f}/tick)")

# ── Plot: readout convergence + the sparse, irregular, all-inhibitory raster ─
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ticks = np.arange(N_STEPS)
LAB = ["v1", "v2", "v3", "w1", "w2", "w3"]
C = plt.cm.tab10(np.arange(N_VAL))
m_hist = (R_inv @ g_hist.T).T
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Inhibition-dominated SCN egomotion on IQIF chip "
             f"(FRAME={FRAME}, CLIP={CLIP}):  v_dir={va_ls:.2f} deg vs LS",
             fontweight="bold")
for j in range(N_VAL):
    ax[0].plot(ticks, m_hist[:, j], color=C[j], lw=1.0, label=LAB[j])
    ax[0].axhline(m_ls[j], color=C[j], ls=":", lw=0.8, alpha=0.7)
ax[0].set_title("chip readout m_hat(t)  (dotted = LS)"); ax[0].legend(ncol=3, fontsize=8)
ax[0].set_xlabel("chip tick")
ax[1].scatter(spk_t, spk_i, s=0.6, c="k", marker="|", linewidths=0.4)
ax[1].set_title(f"spike raster: sparse, irregular, all-inhibitory "
                f"({int((rate>0).sum())}/{N} active, {len(spk_t)} spikes)")
ax[1].set_xlabel("chip tick"); ax[1].set_ylabel("neuron"); ax[1].set_ylim(-1, N)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "pc_egomotion_scn_chip_inhib_plots.png")
fig.savefig(out, dpi=120)
print(f"  Plot written to {out}")
