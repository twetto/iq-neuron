#!/usr/bin/env python3
"""
Known-depth 6-DoF egomotion via PC - CLOSED-LOOP circuit (G as feedback weights).

The open-loop version (test_pc_egomotion_8bit.py) computes the prediction Gm on
the HOST each step (eps = U - A_w @ g) and injects it as the error bias. This
version closes the loop on-chip: a second weight matrix carries G as FEEDBACK
(value -> error), so the error neurons compute eps = U - Gm themselves. Both
matrix-vector products (Gm and G^T eps) now happen in the synaptic fabric; the
host does no matmul.

Three populations (all 8-bit, potential 0..255, leak-free):

  eps+/-   [tonic]       fire ~ max(0, +-eps_k),  eps = U - A_w g
  hold+/-  [integrator]  store g via shadow = membrane + count*quantum = INT(A_w^T eps)
  relay+/- [tonic]       fire ~ max(0, +-g_j); their spikes carry the prediction

Why a relay population: the hold neurons are integrate-and-hold units - at
equilibrium they are SILENT (they only spike while g changes), so they emit no
feedback. The prediction needs g as a persistent firing RATE. The relay neurons
provide that tonic rate (bias set from the held value - a local per-neuron read,
NOT a matmul); their spikes x G feedback weights = Gm at the error neurons.
(The alternative, a self-recurrent line-attractor hold neuron, drifts under
integer weights - see docs/pc-egomotion-circuit-derivation.md sec 6.)

Connections (whitened coords, A_w = Q):
  forward  A_w^T  (eps -> hold):   gradient path, hold integrates +-(A_w^T eps)_j
  feedback A_w    (relay -> eps):  prediction path, delivers -(A_w g)_k to eps+
Data U enters as a constant signed bias on the eps neurons. The error membrane
then integrates (U_k - p_k) and fires tonically ~ that, i.e. ~ eps_k.
"""

import os
import tempfile
import numpy as np

# ── Geometry (shared) ──────────────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Closed-loop 6-DoF egomotion PC (G as on-chip feedback weights)")
print("=" * 70)

N_POINTS = 40
G, U, m_gt = generate_scene(n_points=N_POINTS)
N_ROWS = G.shape[0]
N_VAL = 6

Q, R_qr = np.linalg.qr(G, mode='reduced')
R_inv = np.linalg.inv(R_qr)
A_w = G @ R_inv
m_ls, _, _, _ = np.linalg.lstsq(G, U, rcond=None)
g_ls = R_qr @ m_ls
PROBLEM_SCALE = 2.0 / np.max(np.abs(g_ls))
U_s = U * PROBLEM_SCALE
g_ls_s = g_ls * PROBLEM_SCALE
v_ang_ls, v_mag_ls, w_err_ls = motion_errors(m_ls, m_gt)
print(f"\n  {N_ROWS} flow eqs, {N_VAL} unknowns,  problem scale = {PROBLEM_SCALE:.2f}")
print(f"  LS vs GT:  v_dir = {v_ang_ls:.3f} deg,  w err = {w_err_ls:.4f}")

from iqif import iqnet

VMAX, VMIN, RESET = 255, 0, 0
QUANTUM = VMAX - RESET
RS = 50.0           # error/relay rate scale (bias units per scaled unit)
RS_RELAY = 50.0
S_SCALE = 2000.0
WF = QUANTUM        # feedback weight scale so (WF*RS_RELAY/QUANTUM) = RS  ->  delivers -p*RS
N_STEPS = 9000
CHECKS = [0, 200, 500, 1000, 2000, 3000, 5000, 7000, 8999]

# Forward weight scale bounded so per-step hold drive <= one quantum (8-bit).
col_l1_max = np.max(np.sum(np.abs(A_w), axis=0))
WB = max(1, int(QUANTUM / col_l1_max))
print(f"  WB(forward)={WB}  WF(feedback)={WF}  (max col 1-norm {col_l1_max:.2f})")

# Index layout
EP, EN = 0, N_ROWS                       # eps+ , eps-
HP = 2 * N_ROWS                          # hold+
HN = HP + N_VAL                          # hold-
RP = HN + N_VAL                          # relay+
RN = RP + N_VAL                          # relay-
N_TOTAL = RN + N_VAL

tmpdir = tempfile.mkdtemp(prefix="iq_cl_")
par_path = os.path.join(tmpdir, "p.txt")
con_path = os.path.join(tmpdir, "c.txt")
with open(par_path, "w") as pf:                 # leak-free: threshold=0, shift_b=15
    for i in range(N_TOTAL):
        pf.write(f"{i} 0 0 {RESET} 15 15 0\n")

with open(con_path, "w") as cf:
    n_conn = 0
    for k in range(N_ROWS):
        for j in range(N_VAL):
            wb = int(round(A_w[k, j] * WB))
            wf = int(round(A_w[k, j] * WF))
            if wb != 0:                          # forward A_w^T : eps -> hold
                cf.write(f"{EP + k} {HP + j} {+wb} 1\n")
                cf.write(f"{EN + k} {HP + j} {-wb} 1\n")
                cf.write(f"{EP + k} {HN + j} {-wb} 1\n")
                cf.write(f"{EN + k} {HN + j} {+wb} 1\n")
                n_conn += 4
            if wf != 0:                          # feedback A_w : relay -> eps (delivers -p)
                cf.write(f"{RP + j} {EP + k} {-wf} 1\n")
                cf.write(f"{RN + j} {EP + k} {+wf} 1\n")
                cf.write(f"{RP + j} {EN + k} {+wf} 1\n")
                cf.write(f"{RN + j} {EN + k} {-wf} 1\n")
                n_conn += 4
    if n_conn == 0:
        cf.write("0 0 0 1\n")

net = iqnet(par_path, con_path)
for i in range(N_TOTAL):
    net.set_vmax(i, VMAX)
    net.set_vmin(i, VMIN)
for i in range(2 * N_ROWS + 2 * N_VAL):          # eps + hold receive synapses: no lingering
    net.set_surrogate_tau(i, 1)

print(f"\n  Neurons: {2*N_ROWS} eps + {2*N_VAL} hold + {2*N_VAL} relay = {N_TOTAL}")
print(f"  Connections: {n_conn}")

# Data bias on the error neurons (constant: U is the clamped input).
for k in range(N_ROWS):
    net.set_biascurrent(EP + k, int(round( U_s[k] * RS)))
    net.set_biascurrent(EN + k, int(round(-U_s[k] * RS)))

# Init held estimate g_init, split across hold+/hold- shadows.
g_init = np.random.RandomState(3).randn(N_VAL) * np.max(np.abs(g_ls_s)) * 0.8
sp0 = np.maximum(0.0, g_init) * S_SCALE
sn0 = np.maximum(0.0, -g_init) * S_SCALE
cum_p = np.zeros(N_VAL); cum_n = np.zeros(N_VAL)
off_p = np.zeros(N_VAL); off_n = np.zeros(N_VAL)
for j in range(N_VAL):
    rp = min(int(sp0[j]), VMAX - 1); off_p[j] = sp0[j] - rp; net.set_potential(HP + j, rp)
    rn = min(int(sn0[j]), VMAX - 1); off_n[j] = sn0[j] - rn; net.set_potential(HN + j, rn)


def read_g():
    pp = np.array([net.potential(HP + j) for j in range(N_VAL)], float)
    pn = np.array([net.potential(HN + j) for j in range(N_VAL)], float)
    return ((pp + cum_p * QUANTUM + off_p) - (pn + cum_n * QUANTUM + off_n)) / S_SCALE


m_hist = np.zeros((N_STEPS, N_VAL))
eps_rate = np.zeros(N_STEPS); relay_rate = np.zeros(N_STEPS); hold_rate = np.zeros(N_STEPS)
pot_hist = np.zeros((N_STEPS, N_TOTAL), dtype=np.int16)   # per-neuron membrane potential
spike_hist = np.zeros((N_STEPS, N_TOTAL), dtype=np.int16)  # per-neuron spikes (raster)
conv = []
for step in range(N_STEPS):
    g = read_g()
    # Relay tonic readout: fire ~ held value (local per-neuron copy, NOT a matmul).
    for j in range(N_VAL):
        net.set_biascurrent(RP + j, max(0, int(round( g[j] * RS_RELAY))))
        net.set_biascurrent(RN + j, max(0, int(round(-g[j] * RS_RELAY))))
    net.send_synapse()
    counts = net.get_all_spike_counts()
    pot_hist[step] = [net.potential(i) for i in range(N_TOTAL)]
    spike_hist[step] = counts
    cum_p[:] += counts[HP:HN]
    cum_n[:] += counts[HN:RP]
    eps_rate[step]   = counts[EP:HP].sum()
    hold_rate[step]  = counts[HP:RP].sum()
    relay_rate[step] = counts[RP:N_TOTAL].sum()
    m_hist[step] = (R_inv @ g) / PROBLEM_SCALE
    if step in CHECKS:
        conv.append((step,) + motion_errors(m_hist[step], m_ls))

m_cl = (R_inv @ read_g()) / PROBLEM_SCALE

print("\n  Convergence (closed loop, motion vs LS):")
print(f"    {'step':>6s} {'v_dir(deg)':>12s} {'|v|err':>10s} {'w_err':>10s}")
for s, va, vm, we in conv:
    print(f"    {s:6d} {va:12.4f} {vm:10.4f} {we:10.4f}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
va_ls, vm_ls, we_ls = motion_errors(m_cl, m_ls)
va_gt, vm_gt, we_gt = motion_errors(m_cl, m_gt)
print(f"  closed-loop vs LS:  v_dir = {va_ls:.4f} deg,  |v|err = {vm_ls:.4f},  w_err = {we_ls:.4f}")
print(f"  closed-loop vs GT:  v_dir = {va_gt:.4f} deg,  |v|err = {vm_gt:.4f},  w_err = {we_gt:.4f}")
print(f"  (open-loop host-matmul reference reached ~0.13 deg vs LS)")
print()
if va_ls < 2.0 and we_ls < 0.02:
    print("  PASS  on-chip G-feedback prediction reproduced the LS solution")
elif va_ls < 5.0:
    print(f"  OK    closing onto LS (v_dir = {va_ls:.2f} deg)")
else:
    print(f"  WARN  closed loop v_dir = {va_ls:.2f} deg vs LS - check feedback gain / delay")

# ── Plot ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
steps = np.arange(N_STEPS)
LAB = ["v1", "v2", "v3", "w1", "w2", "w3"]
C = plt.cm.tab10(np.arange(N_VAL))
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Closed-loop egomotion PC: G computed as on-chip feedback", fontweight="bold")
for j in range(N_VAL):
    ax[0, 0].plot(steps, m_hist[:, j], color=C[j], lw=1.1, label=LAB[j])
    ax[0, 0].axhline(m_ls[j], color=C[j], ls=":", lw=0.8, alpha=0.7)
ax[0, 0].set_title("recovered motion m(t)  (dotted = LS)"); ax[0, 0].legend(ncol=3, fontsize=8)
ax[0, 0].set_xlabel("step")
k = 50
ax[0, 1].plot(steps, np.convolve(eps_rate, np.ones(k)/k, 'same'), label="eps (error)", color="tab:red")
ax[0, 1].plot(steps, np.convolve(relay_rate, np.ones(k)/k, 'same'), label="relay (prediction)", color="tab:blue")
ax[0, 1].plot(steps, np.convolve(hold_rate, np.ones(k)/k, 'same'), label="hold (gradient)", color="tab:green")
ax[0, 1].set_title("population firing rates (spikes/step)"); ax[0, 1].legend(fontsize=8); ax[0, 1].set_xlabel("step")
energy = np.array([np.sum((U_s - A_w @ (R_qr @ (m_hist[s]*PROBLEM_SCALE)))**2) for s in range(N_STEPS)])
ax[1, 0].semilogy(steps, energy + 1e-12, color="tab:green")
ax[1, 0].set_title("PC free energy ||U - G m||^2"); ax[1, 0].set_xlabel("step")
ax[1, 1].plot(steps, [motion_errors(m_hist[s], m_ls)[0] for s in range(N_STEPS)], color="tab:purple")
ax[1, 1].set_title("translation-direction error vs LS (deg)"); ax[1, 1].set_xlabel("step")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc_egomotion_closedloop_plots.png")
fig.savefig(out, dpi=120)
print(f"\n  Plots written to {out}")

# ── Per-neuron membrane potentials (heatmap) + spike raster ─────────────────
fig2, ax2 = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
fig2.suptitle("Closed-loop egomotion PC: per-neuron potentials (heatmap) and spike raster",
              fontweight="bold")

# Potential heatmap: every neuron's membrane over time (downsampled in time).
ds = max(1, N_STEPS // 1500)
im = ax2[0].imshow(pot_hist[::ds].T, aspect="auto", origin="lower",
                   cmap="viridis", vmin=0, vmax=255,
                   extent=[0, N_STEPS, 0, N_TOTAL], interpolation="nearest")
for b in (EN, HP, RP):
    ax2[0].axhline(b, color="w", lw=0.6)
ax2[0].set_title("membrane potential (0..255)   bands: eps+ | eps- | hold | relay")
ax2[0].set_ylabel("neuron index")
fig2.colorbar(im, ax=ax2[0], fraction=0.025, label="potential")

# Spike raster: dot at every (step, neuron) where the neuron fired.
ev_s, ev_n = np.nonzero(spike_hist)
col = np.where(ev_n < HP, "tab:red", np.where(ev_n < RP, "tab:green", "tab:blue"))
ax2[1].scatter(ev_s, ev_n, s=0.6, c=col, marker="|", linewidths=0.4)
for b in (EN, HP, RP):
    ax2[1].axhline(b - 0.5, color="k", lw=0.5)
ax2[1].set_ylim(-1, N_TOTAL)
ax2[1].set_title("spike raster   (red eps, green hold, blue relay)")
ax2[1].set_xlabel("step"); ax2[1].set_ylabel("neuron index")

fig2.tight_layout(rect=[0, 0, 1, 0.97])
out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "pc_egomotion_closedloop_raster.png")
fig2.savefig(out2, dpi=120)
print(f"  Potential heatmap + spike raster written to {out2}")

# ── Synaptic weight matrix (the on-chip fabric), block-labeled ───────────────
# Rebuild the full post x pre weight matrix exactly as written to the
# connection file, then annotate which population-to-population block is which.
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle

W = np.zeros((N_TOTAL, N_TOTAL))     # W[post, pre] = synaptic weight
for k in range(N_ROWS):
    for j in range(N_VAL):
        wb = int(round(A_w[k, j] * WB))
        wf = int(round(A_w[k, j] * WF))
        # forward A_w^T  (eps -> hold):    gradient path,  "error -> value"
        W[HP + j, EP + k] += wb
        W[HP + j, EN + k] += -wb
        W[HN + j, EP + k] += -wb
        W[HN + j, EN + k] += wb
        # feedback A_w   (relay -> eps):   prediction path, "value -> error"
        W[EP + k, RP + j] += -wf
        W[EP + k, RN + j] += wf
        W[EN + k, RP + j] += wf
        W[EN + k, RN + j] += -wf

# Population boundaries and (centre, label) for ticks.
bounds = [0, EN, HP, HN, RP, RN, N_TOTAL]
pop_lab = ["eps+", "eps-", "hold+", "hold-", "relay+", "relay-"]
centres = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(pop_lab))]

fig3, ax3 = plt.subplots(figsize=(11, 10))
fig3.suptitle("Closed-loop egomotion PC: synaptic weight matrix  W[post, pre]",
              fontweight="bold")
wmax = np.max(np.abs(W))
im3 = ax3.imshow(W, cmap="RdBu_r", origin="upper",
                 norm=SymLogNorm(linthresh=1.0, vmin=-wmax, vmax=wmax),
                 interpolation="nearest")
fig3.colorbar(im3, ax=ax3, fraction=0.046, label="weight (signed, symlog)")

# Block grid + group ticks.
for b in bounds[1:-1]:
    ax3.axhline(b - 0.5, color="0.4", lw=0.6)
    ax3.axvline(b - 0.5, color="0.4", lw=0.6)
ax3.set_xticks(centres); ax3.set_xticklabels(pop_lab, rotation=45, ha="right")
ax3.set_yticks(centres); ax3.set_yticklabels(pop_lab)
ax3.set_xlabel("presynaptic neuron  (source of spike)")
ax3.set_ylabel("postsynaptic neuron  (receives weight)")

# Outline + label the two active blocks.
def label_block(rlo, rhi, clo, chi, text, edge):
    ax3.add_patch(Rectangle((clo - 0.5, rlo - 0.5), chi - clo, rhi - rlo,
                            fill=False, edgecolor=edge, lw=2.0))
    ax3.annotate(text, xy=((clo + chi) / 2, (rlo + rhi) / 2),
                 ha="center", va="center", fontsize=10, fontweight="bold",
                 color=edge,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=edge, alpha=0.85))

# forward gradient block: post = hold, pre = eps
label_block(HP, RP, EP, HP, "A$_w^\\mathsf{T}$  (forward)\nerror $\\to$ value\ngradient", "tab:green")
# feedback prediction block: post = eps, pre = relay
label_block(EP, HP, RP, N_TOTAL, "A$_w$  (feedback)\nvalue $\\to$ error\nprediction", "tab:blue")

# The hold -> relay readout is NOT a synapse (host copies the held value into
# the relay bias each step); mark its block so the diagram is honest.
ax3.add_patch(Rectangle((HP - 0.5, RP - 0.5), RP - HP, N_TOTAL - RP,
                        fill=False, edgecolor="0.5", lw=1.2, ls="--"))
ax3.annotate("value $\\to$ relay\n(host copy,\nnot synaptic)",
             xy=((HP + RP) / 2, (RP + N_TOTAL) / 2), ha="center", va="center",
             fontsize=8, color="0.35",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.8))

fig3.tight_layout(rect=[0, 0, 1, 0.97])
out3 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "pc_egomotion_closedloop_weights.png")
fig3.savefig(out3, dpi=120)
print(f"  Weight matrix written to {out3}")
