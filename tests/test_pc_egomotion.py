#!/usr/bin/env python3
"""
Known-depth 6-DoF egomotion via spiking Predictive Coding (design notes §6).

Problem (continuous / differential epipolar, depth KNOWN, e.g. stereo):

    u(x) = (1/Z(x)) A(x) v + B(x) omega = M(x) m,      m = [v; omega] in R^6

    A(x) = [ -1   0   x ]        B(x) = [  xy   -(1+x^2)   y ]
           [  0  -1   y ]               [ 1+y^2   -xy     -x ]

where x = (x, y) are NORMALIZED (calibrated) image coords and u(x) is the
optical-flow (motion-field) vector at that pixel.  Stacking N pixels (2 rows
each) gives an honestly-inhomogeneous linear system:

    U = G m,     G in R^{2N x 6},  U in R^{2N}  (both known),  m in R^6

Because depth is known, absolute translation scale is observable: there is NO
||v||=1 constraint, no null space, no minor-component instability (contrast the
8-point case).  This is fixed-A / fixed-b least squares — exactly the inference-
only circuit proven in test_pc_augmented.py Part 2, with three swaps:

    value neurons  g (8)      -> m = (v1 v2 v3, w1 w2 w3)   (6)
    error neurons  eps (2*20) -> eps = (U - G m)            (2*2N rows)
    weights        A_w (20x8) -> G (2N x 6) from flow geom + 1/Z
    clamped input  b          -> observed flow U

This is the memoryless core; the IMU pre-integration predict layer (§6) wraps
it later.  Here we validate the solve itself against numpy lstsq and ground
truth on a synthetic scene.
"""

import os
import tempfile
import numpy as np

# ── Geometry ─────────────────────────────────────────────────────────────

def motion_field_rows(x, y, Z):
    """Return the 2x6 block M(x) = [ (1/Z)A(x) | B(x) ] for one pixel.

    Row 0 is the u-flow equation, row 1 the v-flow equation.
    Columns are [v1, v2, v3, w1, w2, w3].
    """
    A = np.array([[-1.0, 0.0, x],
                  [ 0.0, -1.0, y]])
    B = np.array([[ x*y, -(1.0 + x*x),  y],
                  [ 1.0 + y*y, -x*y,    -x]])
    return np.hstack([A / Z, B])  # 2x6


def generate_scene(n_points=40, noise_sigma=2e-3, seed=7):
    """Synthetic known-depth flow field.

    Pixels sampled over a realistic normalized FOV (~EuRoC), random metric
    depths, a small ground-truth 6-DoF velocity.  Returns G (2N x 6),
    U (2N,), and the ground-truth motion m_gt (6,).
    """
    rng = np.random.RandomState(seed)

    # Ground-truth instantaneous motion (velocity units; scale arbitrary but
    # fixed, since known depth makes it observable).
    v_gt = np.array([0.30, 0.05, 0.12])      # forward-ish translation
    w_gt = np.array([0.02, -0.05, 0.015])    # small rotation
    m_gt = np.concatenate([v_gt, w_gt])

    xs = rng.uniform(-0.70, 0.70, n_points)   # normalized image coords
    ys = rng.uniform(-0.50, 0.50, n_points)
    Zs = rng.uniform(2.0, 10.0, n_points)     # metric depth (stereo)

    rows = []
    for x, y, Z in zip(xs, ys, Zs):
        rows.append(motion_field_rows(x, y, Z))
    G = np.vstack(rows)                        # 2N x 6
    U = G @ m_gt
    U += rng.randn(U.shape[0]) * noise_sigma   # flow measurement noise
    return G, U, m_gt


def motion_errors(m_est, m_ref):
    """Report angular error on translation DIRECTION and rotation error."""
    v_est, w_est = m_est[:3], m_est[3:]
    v_ref, w_ref = m_ref[:3], m_ref[3:]
    cos = np.dot(v_est, v_ref) / (np.linalg.norm(v_est) * np.linalg.norm(v_ref) + 1e-12)
    v_ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    v_mag = abs(np.linalg.norm(v_est) - np.linalg.norm(v_ref))
    w_err = np.linalg.norm(w_est - w_ref)
    return v_ang, v_mag, w_err


# ══════════════════════════════════════════════════════════════════════════
print("=" * 68)
print("Known-depth 6-DoF egomotion: spiking Predictive Coding (Gm = U)")
print("=" * 68)

N_POINTS = 40
G, U, m_gt = generate_scene(n_points=N_POINTS, noise_sigma=2e-3)
N_ROWS = G.shape[0]          # = 2*N_POINTS
N_VAL = 6

# ── Conditioning + QR whitening (same recipe as test_pc_augmented.py) ──────
# The translation block scales as 1/Z while the rotation block is O(1), so the
# columns have very different magnitudes -> poor conditioning -> integer
# quantization noise (+-0.5/step) gets amplified by 1/sigma_min.  Whitening
# G = Q R, A_w = Q (kappa = 1) removes that amplification; we solve for
# g = R m and recover m = R_inv g.
_, S_vals, _ = np.linalg.svd(G, full_matrices=False)
kappa_G = S_vals[0] / S_vals[-1]

Q, R_qr = np.linalg.qr(G, mode='reduced')
R_inv = np.linalg.inv(R_qr)
A_w = G @ R_inv              # = Q, kappa = 1,  shape N_ROWS x 6

# Least-squares reference (the PC equilibrium x* = (G^T G)^-1 G^T U).
m_ls, _, _, _ = np.linalg.lstsq(G, U, rcond=None)
v_ang_ls, v_mag_ls, w_err_ls = motion_errors(m_ls, m_gt)

# Problem-scale normalization.  Flow is ~O(0.1) and motion ~O(0.3) here, an
# order of magnitude below the Hartley-normalized O(1) regime the spiking
# circuit is tuned for.  Without rescaling, V_SCALE_S blows up and the
# effective learning rate collapses to ~1e-5 (value neurons never fire).  The
# system is linear, so scale U -> s*U, solve, then divide the recovered motion
# by s.  Pick s so the whitened solution g = R_qr m has max magnitude ~2.
g_ls = R_qr @ m_ls
PROBLEM_SCALE = 2.0 / np.max(np.abs(g_ls))
U_s = U * PROBLEM_SCALE

print(f"\n  N points = {N_POINTS}  ->  {N_ROWS} flow equations, {N_VAL} unknowns")
print(f"  kappa(G) = {kappa_G:.1f}   (whitened kappa = 1)")
print(f"  problem scale s = {PROBLEM_SCALE:.2f} (lifts U into the O(1) circuit regime)")
print(f"  GT motion  v = {m_gt[:3]},  w = {m_gt[3:]}")
print(f"  LS  motion v = {m_ls[:3].round(4)},  w = {m_ls[3:].round(4)}")
print(f"  LS vs GT:  v_dir = {v_ang_ls:.3f} deg,  |v| err = {v_mag_ls:.4f},  "
      f"w err = {w_err_ls:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Spiking PC circuit  (mirrors test_pc_augmented.py Part 2)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("Spiking PC circuit (QR-whitened system, Gm=U)")
print("=" * 68)

from iqif import iqnet

# Architecture (all neurons share the same LIF-integrator params):
#   eps+ neurons [0 .. N_ROWS-1]            : fire ~ max(0,  eps_k)
#   eps- neurons [N_ROWS .. 2*N_ROWS-1]     : fire ~ max(0, -eps_k)
#   val  neurons [2*N_ROWS .. 2*N_ROWS+5]   : one per motion component m_j
#
# Two signed connections per (row k, val j):
#     eps+_k -> val_j  weight +round(A_w[k,j]*W_SCALE)
#     eps-_k -> val_j  weight -round(A_w[k,j]*W_SCALE)
# Net drive on val_j = sum_k A_w[k,j]*(eps+_rate - eps-_rate) = (A_w^T eps)_j.

N_ERR_TOT  = 2 * N_ROWS
VAL_OFFSET = N_ERR_TOT
N_TOTAL    = N_ERR_TOT + N_VAL

W_SCALE = 100
NEURON_VMAX    = 5000
NEURON_RESET   = 4500
NEURON_QUANTUM = NEURON_VMAX - NEURON_RESET
NEURON_VMIN    = -1000000

tmpdir = tempfile.mkdtemp(prefix="iqtest_egomotion_")
par_path = os.path.join(tmpdir, "params.txt")
con_path = os.path.join(tmpdir, "conn.txt")

with open(par_path, "w") as pf:
    for i in range(N_TOTAL):
        pf.write(f"{i} 0 {NEURON_VMAX} {NEURON_RESET} 15 1 0\n")

with open(con_path, "w") as cf:
    tau_conn = 8
    n_conn = 0
    for k in range(N_ROWS):
        for j in range(N_VAL):
            w = int(round(A_w[k, j] * W_SCALE))
            if w == 0:
                continue
            cf.write(f"{k}          {VAL_OFFSET + j} {+w} {tau_conn}\n")  # eps+
            cf.write(f"{N_ROWS + k} {VAL_OFFSET + j} {-w} {tau_conn}\n")  # eps-
            n_conn += 2
    if n_conn == 0:
        cf.write("0 0 0 32\n")

net = iqnet(par_path, con_path)
print(f"\n  Neurons: {N_ERR_TOT} error + {N_VAL} value = {N_TOTAL}")
print(f"  Connections: {n_conn} (signed A_w * W_SCALE={W_SCALE})")

for i in range(N_TOTAL):
    net.set_vmax(i, NEURON_VMAX)
    net.set_vmin(i, NEURON_VMIN)

RATE_SCALE = 50.0

# V_SCALE_S: shadow units per g-unit.  Target max shadow ~20000.  g_ls_s is
# the whitened solution of the SCALED system (max magnitude ~2 by construction).
g_ls_s = R_qr @ (m_ls * PROBLEM_SCALE)
V_SCALE_S = 20000.0 / np.max(np.abs(g_ls_s))
print(f"  V_SCALE_S = {V_SCALE_S:.1f}")

# Random g_init off the g_ls ray so both eps+/eps- populations fire.
g_init = np.random.RandomState(3).randn(N_VAL) * np.max(np.abs(g_ls_s)) * 0.8

shadow_init = g_init * V_SCALE_S
raw_init    = np.minimum(shadow_init, NEURON_VMAX - 1)
val_offset  = shadow_init - raw_init
for j in range(N_VAL):
    net.set_potential(VAL_OFFSET + j, int(round(raw_init[j])))

val_spikes_cum = np.zeros(N_VAL, dtype=np.float64)

def read_g():
    """Reconstruct scaled g from value-neuron shadow potentials.

    Uses the externally maintained val_spikes_cum (updated once per step from
    the all-neuron readout below), so spike counters are consumed exactly once.
    """
    pot = np.array([net.potential(VAL_OFFSET + j) for j in range(N_VAL)],
                   dtype=np.float64)
    shadow = pot + val_spikes_cum * NEURON_QUANTUM + val_offset
    return shadow / V_SCALE_S

N_SPIKE_STEPS = 6000
SPIKE_CHECKS = [0, 100, 200, 500, 1000, 2000, 3000, 4000, 5999]
conv = []

# Per-step recordings for the membrane-potential / spike-train plots.
m_hist       = np.zeros((N_SPIKE_STEPS, N_VAL))           # recovered motion
val_pot_hist = np.zeros((N_SPIKE_STEPS, N_VAL))           # value membrane pot
err_pot_hist = np.zeros((N_SPIKE_STEPS, N_ERR_TOT))       # error membrane pot
spike_hist   = np.zeros((N_SPIKE_STEPS, N_TOTAL), dtype=np.int16)  # raster

for step in range(N_SPIKE_STEPS):
    g_scaled = read_g()
    eps_vals = U_s - A_w @ g_scaled        # whitened-space error (scaled system)
    for k in range(N_ROWS):
        bias_pos = max(0, int(round(eps_vals[k] * RATE_SCALE)))
        bias_neg = max(0, int(round(-eps_vals[k] * RATE_SCALE)))
        net.set_biascurrent(k, bias_pos)
        net.set_biascurrent(N_ROWS + k, bias_neg)
    net.send_synapse()

    # Read all membrane potentials and spike counts ONCE, after the update.
    # spike_count() resets on read, so this is the per-step spike for each
    # neuron; fold value spikes into the cumulative used by read_g().
    for i in range(N_TOTAL):
        spike_hist[step, i] = net.spike_count(i)
    err_pot_hist[step] = [net.potential(i) for i in range(N_ERR_TOT)]
    val_pot_hist[step] = [net.potential(VAL_OFFSET + j) for j in range(N_VAL)]
    val_spikes_cum[:] += spike_hist[step, VAL_OFFSET:]

    m_hist[step] = (R_inv @ g_scaled) / PROBLEM_SCALE
    if step in SPIKE_CHECKS:
        v_ang, v_mag, w_err = motion_errors(m_hist[step], m_ls)
        conv.append((step, v_ang, v_mag, w_err))

g_final = read_g()
m_spike = (R_inv @ g_final) / PROBLEM_SCALE             # undo problem scale

total_sp_ep  = int(spike_hist[:, :N_ROWS].sum())
total_sp_en  = int(spike_hist[:, N_ROWS:N_ERR_TOT].sum())
total_sp_val = int(spike_hist[:, VAL_OFFSET:].sum())

print(f"\n  Convergence (motion vs LS):")
print(f"    {'step':>6s} {'v_dir(deg)':>12s} {'|v|err':>10s} {'w_err':>10s}")
for step, v_ang, v_mag, w_err in conv:
    print(f"    {step:6d} {v_ang:12.4f} {v_mag:10.4f} {w_err:10.4f}")

v_ang_ls_s, v_mag_ls_s, w_err_ls_s = motion_errors(m_spike, m_ls)   # vs LS
v_ang_gt_s, v_mag_gt_s, w_err_gt_s = motion_errors(m_spike, m_gt)   # vs GT

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("Summary")
print("=" * 68)
print(f"  IQIF motion  v = {m_spike[:3].round(4)},  w = {m_spike[3:].round(4)}")
print(f"\n  {'Comparison':<28s} {'v_dir(deg)':>12s} {'|v|err':>10s} {'w_err':>10s}")
print(f"  {'-'*28} {'-'*12} {'-'*10} {'-'*10}")
print(f"  {'LS  vs GT':<28s} {v_ang_ls:12.4f} {v_mag_ls:10.4f} {w_err_ls:10.4f}")
print(f"  {'IQIF vs GT':<28s} {v_ang_gt_s:12.4f} {v_mag_gt_s:10.4f} {w_err_gt_s:10.4f}")
print(f"  {'IQIF vs LS':<28s} {v_ang_ls_s:12.4f} {v_mag_ls_s:10.4f} {w_err_ls_s:10.4f}")
print(f"\n  Error spikes: eps+ = {total_sp_ep}, eps- = {total_sp_en}")
print(f"  Value spikes: {total_sp_val} (across {N_VAL} neurons)")

print()
if v_ang_ls_s < 2.0 and w_err_ls_s < 0.02:
    print(f"  PASS  Spiking PC matched LS (v_dir < 2 deg, w_err < 0.02)")
elif v_ang_ls_s < 5.0 and w_err_ls_s < 0.05:
    print(f"  OK    Spiking PC approaching LS "
          f"(v_dir = {v_ang_ls_s:.2f} deg, w_err = {w_err_ls_s:.4f})")
else:
    print(f"  WARN  Spiking PC vs LS: v_dir = {v_ang_ls_s:.2f} deg, "
          f"w_err = {w_err_ls_s:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Plots: membrane potentials + spike trains
# ═══════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt

steps = np.arange(N_SPIKE_STEPS)
LABELS = ["v1", "v2", "v3", "w1", "w2", "w3"]
COLORS = plt.cm.tab10(np.arange(N_VAL))

fig, ax = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle("Known-depth 6-DoF egomotion PC circuit — membrane potentials "
             "& spike trains", fontsize=13, fontweight="bold")

# (0,0) Motion estimate convergence vs LS reference.
for j in range(N_VAL):
    ax[0, 0].plot(steps, m_hist[:, j], color=COLORS[j], lw=1.2, label=LABELS[j])
    ax[0, 0].axhline(m_ls[j], color=COLORS[j], ls=":", lw=0.9, alpha=0.7)
ax[0, 0].set_title("Recovered motion m(t)  (dotted = numpy LS)")
ax[0, 0].set_xlabel("step"); ax[0, 0].set_ylabel("value")
ax[0, 0].legend(ncol=3, fontsize=8)

# (0,1) Value-neuron membrane potential (the sub-threshold integrators).
for j in range(N_VAL):
    ax[0, 1].plot(steps, val_pot_hist[:, j], color=COLORS[j], lw=1.0,
                  label=LABELS[j])
ax[0, 1].axhline(NEURON_VMAX, color="k", ls="--", lw=0.8, label="VMAX")
ax[0, 1].set_title("Value-neuron membrane potential (sub-threshold integrators)")
ax[0, 1].set_xlabel("step"); ax[0, 1].set_ylabel("potential")
ax[0, 1].legend(ncol=4, fontsize=7)

# (1,0) Error-neuron membrane potentials — a few sample neurons (sawtooth:
# integrate-to-threshold, reset). Pick the 4 most active eps+ and eps-.
act = spike_hist[:, :N_ERR_TOT].sum(axis=0)
top_pos = np.argsort(act[:N_ROWS])[-3:]
top_neg = N_ROWS + np.argsort(act[N_ROWS:N_ERR_TOT])[-3:]
for k in top_pos:
    ax[1, 0].plot(steps, err_pot_hist[:, k], lw=0.7, label=f"eps+ #{k}")
for k in top_neg:
    ax[1, 0].plot(steps, err_pot_hist[:, k], lw=0.7, ls="--",
                  label=f"eps- #{k - N_ROWS}")
ax[1, 0].axhline(NEURON_VMAX, color="k", ls="--", lw=0.8, label="VMAX")
ax[1, 0].set_title("Error-neuron membrane potential (integrate-and-fire sawtooth)")
ax[1, 0].set_xlabel("step"); ax[1, 0].set_ylabel("potential")
ax[1, 0].legend(ncol=2, fontsize=7)

# (1,1) Error-neuron spike raster (eps+ rows 0..N_ROWS-1, eps- above).
ev_steps, ev_neuron = np.nonzero(spike_hist[:, :N_ERR_TOT])
colors = np.where(ev_neuron < N_ROWS, "tab:blue", "tab:red")
ax[1, 1].scatter(ev_steps, ev_neuron, s=1.5, c=colors, marker="|", linewidths=0.6)
ax[1, 1].axhline(N_ROWS - 0.5, color="k", lw=0.6)
ax[1, 1].set_title(f"Error-neuron spike raster  (blue eps+ [0,{N_ROWS}), "
                   f"red eps- [{N_ROWS},{N_ERR_TOT}))")
ax[1, 1].set_xlabel("step"); ax[1, 1].set_ylabel("neuron index")
ax[1, 1].set_ylim(-1, N_ERR_TOT)

# (2,0) Population firing rate over time (boxcar-smoothed spikes/step).
win = 50
kern = np.ones(win) / win
rate_pos = np.convolve(spike_hist[:, :N_ROWS].sum(axis=1), kern, mode="same")
rate_neg = np.convolve(spike_hist[:, N_ROWS:N_ERR_TOT].sum(axis=1), kern, mode="same")
ax[2, 0].plot(steps, rate_pos, color="tab:blue", lw=1.0, label="eps+ total")
ax[2, 0].plot(steps, rate_neg, color="tab:red", lw=1.0, label="eps- total")
ax[2, 0].set_title(f"Population firing rate (spikes/step, {win}-step boxcar)")
ax[2, 0].set_xlabel("step"); ax[2, 0].set_ylabel("spikes / step")
ax[2, 0].legend(fontsize=8)

# (2,1) Whitened-space prediction error energy ||eps||^2 -> 0 at equilibrium.
energy = np.array([np.sum((U_s - A_w @ (R_qr @ (m_hist[s] * PROBLEM_SCALE)))**2)
                   for s in range(N_SPIKE_STEPS)])
ax[2, 1].semilogy(steps, energy + 1e-12, color="tab:green", lw=1.2)
ax[2, 1].set_title("PC free energy  F = ||U - G m||^2  (log scale)")
ax[2, 1].set_xlabel("step"); ax[2, 1].set_ylabel("energy")

fig.tight_layout(rect=[0, 0, 1, 0.97])
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "pc_egomotion_plots.png")
fig.savefig(out_png, dpi=120)
print(f"\n  Plots written to {out_png}")
