#!/usr/bin/env python3
"""
Meeting visuals for the inhibition-dominated spike-coding-network egomotion
circuit, on a real EuRoC frame. Produces, in the style of the closed-loop PC
plots:
  - pc_egomotion_scn_raster.png : per-neuron membrane heatmap + spike raster
  - pc_egomotion_scn_weights.png: the all-inhibitory lateral weight matrix

The circuit (mirrors iqif-vio scn_core): N decoder neurons over the whitened
motion space, constant bias = Phi^T U, all-inhibitory lateral recurrent
(weights = -round(VMAX*cos_ij)/tau, clipped <=0), exp-decay lateral synapse
(surrogate_tau), QIF rest=127.  The 6-DoF readout g = D * (full-frame spikes).

Usage:  OBS_CSV=obs_frame_150.csv python tests/plot_scn_viz.py
"""
import os
import tempfile
import numpy as np
from iqif import iqnet

VMAX, RESET, N_VAL, N, N_STEPS = 255, 0, 6, 72, 8000
TAU = int(os.environ.get("SCN_TAU", "4"))


def motion_field_rows(x, y, Z):
    A = np.array([[-1.0, 0.0, x], [0.0, -1.0, y]])
    B = np.array([[x*y, -(1.0 + x*x), y], [1.0 + y*y, -x*y, -x]])
    return np.hstack([A / Z, B])


OBS = os.environ.get("OBS_CSV", "obs_frame_150.csv")
if not os.path.isabs(OBS):
    OBS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), OBS)
rows = np.loadtxt(OBS, delimiter=",", skiprows=1)
G = np.vstack([motion_field_rows(x, y, z) for x, y, z, _, _ in rows])
U = np.empty(2 * len(rows)); U[0::2] = rows[:, 3]; U[1::2] = rows[:, 4]
Q, R_qr = np.linalg.qr(G, mode="reduced"); R_inv = np.linalg.inv(R_qr); A_w = Q
g_ls = Q.T @ U
print(f"  {len(rows)} obs from {OBS},  N={N} neurons, tau={TAU}")

rng = np.random.RandomState(1)
D_dirs = rng.randn(N_VAL, N); D_dirs /= np.linalg.norm(D_dirs, axis=0, keepdims=True)
d_scale = 0.02 * np.linalg.norm(g_ls); D = D_dirs * d_scale
Phi = A_w @ D; fU = Phi.T @ U; COS = D_dirs.T @ D_dirs
T = 0.5 * d_scale ** 2; s = VMAX / (2.0 * T)
LAM, DT, X0, SHIFT = 2.0, 1e-3, VMAX // 2, 6
bias = np.round(s * DT * LAM * fU).astype(int)
Wc = np.round(VMAX * COS).astype(float); np.fill_diagonal(Wc, 0.0)
weight = np.round(np.minimum(-Wc, 0.0) / TAU).astype(int)  # all-inhibitory, /tau

td = tempfile.mkdtemp(prefix="iq_viz_")
pp, cp = os.path.join(td, "p"), os.path.join(td, "c")
with open(pp, "w") as f:
    for i in range(N):
        f.write(f"{i} {X0} {VMAX} {RESET} {SHIFT} {SHIFT} 0\n")
with open(cp, "w") as f:
    nc = 0
    for j in range(N):
        for i in range(N):
            if i != j and weight[i, j] != 0:
                f.write(f"{j} {i} {int(weight[i,j])} 1\n"); nc += 1
    if nc == 0:
        f.write("0 0 0 1\n")
net = iqnet(pp, cp)
for i in range(N):
    net.set_vmax(i, VMAX); net.set_vmin(i, 0); net.set_surrogate_tau(i, TAU)
    net.set_biascurrent(i, int(bias[i])); net.set_potential(i, int(X0))

pot_hist = np.zeros((N_STEPS, N), dtype=np.int16)
spike_hist = np.zeros((N_STEPS, N), dtype=np.int16)
g_hist = np.zeros((N_STEPS, N_VAL))
r = np.zeros(N)
for t in range(N_STEPS):
    net.send_synapse()
    c = net.get_all_spike_counts()[:N]
    spike_hist[t] = c
    pot_hist[t] = [net.potential(i) for i in range(N)]
    r += c - LAM * DT * r
    g_hist[t] = D @ r
m_hist = (R_inv @ g_hist.T).T
m_ls = R_inv @ g_ls
rate = spike_hist.sum(0)
order = np.argsort(rate)[::-1]               # most-active neurons first (rows)
n_active = int((rate > 0).sum())
print(f"  active neurons: {n_active}/{N}   total spikes: {int(rate.sum())}")

# ── Figure 1: membrane heatmap + spike raster ───────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = ["v1", "v2", "v3", "w1", "w2", "w3"]
C = plt.cm.tab10(np.arange(N_VAL))
fig, ax = plt.subplots(3, 1, figsize=(15, 13), sharex=True,
                       gridspec_kw={"height_ratios": [3, 2, 2]})
fig.suptitle("Inhibition-dominated SCN egomotion on the IQIF chip (EuRoC frame): "
             "membrane, spikes, readout", fontweight="bold")
ds = max(1, N_STEPS // 1500)
im = ax[0].imshow(pot_hist[::ds][:, order].T, aspect="auto", origin="lower",
                  cmap="viridis", vmin=0, vmax=255,
                  extent=[0, N_STEPS, 0, N], interpolation="nearest")
ax[0].axhline(n_active, color="w", lw=1.0, ls="--")
ax[0].set_title(f"membrane potential (0..255), neurons sorted by activity "
                f"({n_active}/{N} active, below dashed line)")
ax[0].set_ylabel("neuron (sorted)")
fig.colorbar(im, ax=ax[0], fraction=0.02, label="potential")

inv = np.argsort(order)                      # map neuron -> sorted row
ev_t, ev_n = np.nonzero(spike_hist)
sr = inv[ev_n]
ax[1].scatter(ev_t, sr, s=8, c="tab:blue", marker="|", linewidths=0.7)
ax[1].set_ylim(-1, max(12, n_active + 2))    # zoom to the active band
ax[1].set_title(f"spike raster (sparse competitive code: only {n_active} neurons "
                f"win, ~{int(rate.sum())} spikes)")
ax[1].set_ylabel("neuron (sorted)")

for j in range(N_VAL):
    ax[2].plot(np.arange(N_STEPS), m_hist[:, j], color=C[j], lw=1.0, label=LAB[j])
    ax[2].axhline(m_ls[j], color=C[j], ls=":", lw=0.8, alpha=0.6)
ax[2].set_title("recovered egomotion m(t) = R⁻¹ D r  (dotted = least-squares)")
ax[2].legend(ncol=6, fontsize=8); ax[2].set_xlabel("chip tick")
ax[2].set_ylabel("m (v: m/s, w: rad/s)")
fig.tight_layout(rect=[0, 0, 1, 0.97])
out1 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "pc_egomotion_scn_raster.png")
fig.savefig(out1, dpi=120)
print(f"  wrote {out1}")

# ── Figure 2: all-inhibitory lateral weight matrix ──────────────────────────
from matplotlib.colors import SymLogNorm
W_ord = weight[np.ix_(order, order)].astype(float)
fig2, ax2 = plt.subplots(figsize=(9.5, 8.5))
fig2.suptitle("Inhibition-dominated SCN: lateral weight matrix W[post, pre] "
              "(all <= 0)", fontweight="bold")
wmax = max(1.0, np.abs(W_ord).max())
im2 = ax2.imshow(W_ord, cmap="RdBu_r", origin="upper",
                 norm=SymLogNorm(linthresh=1.0, vmin=-wmax, vmax=wmax),
                 interpolation="nearest")
fig2.colorbar(im2, ax=ax2, fraction=0.046, label="weight (all inhibitory, symlog)")
ax2.axhline(n_active - 0.5, color="0.3", lw=0.8, ls="--")
ax2.axvline(n_active - 0.5, color="0.3", lw=0.8, ls="--")
ax2.set_xlabel("presynaptic neuron (sorted by activity)")
ax2.set_ylabel("postsynaptic neuron (sorted by activity)")
ax2.set_title(f"every off-diagonal weight <= 0 -> a delayed spike can only "
              f"inhibit (delay-robust). diagonal = reset.\n"
              f"{n_active} active neurons (top-left block); "
              f"min weight {int(weight.min())}, exp-decay tau={TAU}", fontsize=9)
fig2.tight_layout(rect=[0, 0, 1, 0.96])
out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "pc_egomotion_scn_weights.png")
fig2.savefig(out2, dpi=120)
print(f"  wrote {out2}")

# ── Figure 3: all three matrices (input encoder, recurrent, readout) ────────
F = Phi.T[order]                                   # input F = Phi^T : U -> drive (N x 2P)
Wrec = weight[np.ix_(order, order)].astype(float)  # recurrent (N x N), all <= 0
Dd = D_dirs[:, order]                              # readout directions (6 x N)
fig3, ax3 = plt.subplots(3, 1, figsize=(13, 12))
fig3.suptitle("Inhibition-dominated SCN: the three weight matrices "
              "(neurons sorted by activity)", fontweight="bold")

fmax = np.abs(F).max()
i0 = ax3[0].imshow(F, aspect="equal", cmap="RdBu_r", vmin=-fmax, vmax=fmax,
                   interpolation="nearest")
ax3[0].set_title(f"1. INPUT  F = Φᵀ  [{F.shape[0]}×{F.shape[1]}]  — flow U → "
                 f"neuron drive (on-chip bias = F·U)", fontsize=10)
ax3[0].set_xlabel("flow component (2 per feature)"); ax3[0].set_ylabel("neuron")
fig3.colorbar(i0, ax=ax3[0], fraction=0.015, label="weight (signed)")

wmx = max(1.0, np.abs(Wrec).max())
i1 = ax3[1].imshow(Wrec, aspect="equal", cmap="RdBu_r",
                   norm=SymLogNorm(linthresh=1.0, vmin=-wmx, vmax=wmx),
                   interpolation="nearest")
ax3[1].set_title(f"2. RECURRENT  Ω = DᵀD  [{N}×{N}]  — on-chip lateral inhibition, "
                 f"all ≤ 0 (delay-robust)", fontsize=10)
ax3[1].set_xlabel("presynaptic neuron"); ax3[1].set_ylabel("postsynaptic neuron")
fig3.colorbar(i1, ax=ax3[1], fraction=0.015, label="weight (all inhibitory)")

dmax = np.abs(Dd).max()
i2 = ax3[2].imshow(Dd, aspect="equal", cmap="RdBu_r", vmin=-dmax, vmax=dmax,
                   interpolation="nearest")
ax3[2].set_title(f"3. READOUT  D  [{N_VAL}×{N}]  — spike rates → g  (m = R⁻¹ g)",
                 fontsize=10)
ax3[2].set_xlabel("neuron"); ax3[2].set_yticks(range(N_VAL))
ax3[2].set_yticklabels(["v1", "v2", "v3", "w1", "w2", "w3"])
fig3.colorbar(i2, ax=ax3[2], fraction=0.015, label="decoder weight (signed)")

fig3.tight_layout(rect=[0, 0, 1, 0.97])
out3 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "pc_egomotion_scn_matrices.png")
fig3.savefig(out3, dpi=120)
print(f"  wrote {out3}")
