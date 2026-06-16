#!/usr/bin/env python3
"""
Closed-loop 6-DoF egomotion PC with a FULLY SYNAPTIC value->relay path.

In test_pc_egomotion_closedloop.py the relay's tonic rate is loaded from the
held value by a HOST copy each step (set_biascurrent(relay, g*RS_RELAY)). That
copy is a per-neuron, one-to-one map - an *identity matrix* - but it lives on
the host because the hold integrator is SILENT at equilibrium (it only spikes
while g changes), so a plain identity synapse hold->relay would deliver nothing
once converged and the prediction feedback would collapse.

This variant makes value->relay a genuine synaptic identity block:

  relay -> relay   (W_SELF, leaky tau):  line attractor that SUSTAINS the tonic
                                         rate when hold falls silent.
  hold  -> relay   (W_LOAD, identity):   LOADS / corrects the relay toward g
                                         whenever hold fires.

The line attractor is only marginally stable on integer leak-free hardware
(W_SELF * tau ~ QUANTUM holds an arbitrary rate -> drifts). The bet is that the
PC prediction-error loop closes around the relay and REGULATES it: if the relay
drifts off g, eps reappears, hold fires, and the identity load re-corrects it.
No host write touches the relay after init - the whole circuit is synaptic.

Run:  python test_pc_egomotion_synaptic_relay.py
Knobs: TAU_RELAY, W_SELF, W_LOAD (env vars, optional).
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
print("Synaptic-relay egomotion PC (value->relay = identity line attractor)")
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
print(f"\n  {N_ROWS} flow eqs, {N_VAL} unknowns,  problem scale = {PROBLEM_SCALE:.2f}")

from iqif import iqnet

VMAX, VMIN, RESET = 255, 0, 0
QUANTUM = VMAX - RESET
RS = 50.0
RS_RELAY = 50.0
S_SCALE = 2000.0
WF = QUANTUM
N_STEPS = 9000
CHECKS = [0, 200, 500, 1000, 2000, 3000, 5000, 7000, 8999]

# Synaptic-relay knobs.  The line-attractor sustain condition is
# W_SELF * TAU_RELAY = (firing quantum) = VMAX - reset.  The membrane range is
# hardwired 0..255 (VMAX=255) and 255 is ODD, so no integer W_SELF * (power-of-
# two TAU) can balance it.  Fix: set the RELAY reset to 127 so its firing
# quantum is 255-127=128, a power of two -> W_SELF*TAU=128 balances exactly
# (e.g. 8*16).  Reset stays in 0..255 and the post-spike membrane lands at ~127
# (no VMIN clamp loss).
RESET_RELAY = int(os.environ.get("RESET_RELAY", "127"))
Q_RELAY = VMAX - RESET_RELAY                       # relay firing quantum (128)
TAU_RELAY = int(os.environ.get("TAU_RELAY", "16"))
W_SELF = int(os.environ.get("W_SELF", str(Q_RELAY // TAU_RELAY)))
W_LOAD = int(os.environ.get("W_LOAD", "40"))
print(f"  relay line attractor:  reset={RESET_RELAY}  Q_relay={Q_RELAY}  "
      f"TAU={TAU_RELAY}  W_SELF={W_SELF}  (W_SELF*TAU={W_SELF*TAU_RELAY}"
      f" vs Q_relay={Q_RELAY})  W_LOAD={W_LOAD}")

col_l1_max = np.max(np.sum(np.abs(A_w), axis=0))
WB = max(1, int(QUANTUM / col_l1_max))
print(f"  WB(forward)={WB}  WF(feedback)={WF}")

EP, EN = 0, N_ROWS
HP = 2 * N_ROWS
HN = HP + N_VAL
RP = HN + N_VAL
RN = RP + N_VAL
N_TOTAL = RN + N_VAL

tmpdir = tempfile.mkdtemp(prefix="iq_sr_")
par_path = os.path.join(tmpdir, "p.txt")
con_path = os.path.join(tmpdir, "c.txt")
with open(par_path, "w") as pf:                  # relay reset=127 -> quantum 128
    for i in range(N_TOTAL):
        rst = RESET_RELAY if i >= RP else RESET
        pf.write(f"{i} 0 0 {rst} 15 15 0\n")

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
            if wf != 0:                          # feedback A_w : relay -> eps
                cf.write(f"{RP + j} {EP + k} {-wf} 1\n")
                cf.write(f"{RN + j} {EP + k} {+wf} 1\n")
                cf.write(f"{RP + j} {EN + k} {+wf} 1\n")
                cf.write(f"{RN + j} {EN + k} {-wf} 1\n")
                n_conn += 4
    for j in range(N_VAL):                       # value -> relay (SYNAPTIC, identity)
        cf.write(f"{RP + j} {RP + j} {W_SELF} 1\n")   # self loop: sustain rate
        cf.write(f"{RN + j} {RN + j} {W_SELF} 1\n")
        cf.write(f"{HP + j} {RP + j} {W_LOAD} 1\n")   # load/correct from hold
        cf.write(f"{HN + j} {RN + j} {W_LOAD} 1\n")
        n_conn += 4

net = iqnet(par_path, con_path)
for i in range(N_TOTAL):
    net.set_vmax(i, VMAX)
    net.set_vmin(i, VMIN)
for i in range(2 * N_ROWS + 2 * N_VAL):          # eps + hold: no synaptic lingering
    net.set_surrogate_tau(i, 1)
for j in range(N_VAL):                           # relay: leaky -> line attractor
    net.set_surrogate_tau(RP + j, TAU_RELAY)
    net.set_surrogate_tau(RN + j, TAU_RELAY)

print(f"\n  Neurons: {2*N_ROWS} eps + {2*N_VAL} hold + {2*N_VAL} relay = {N_TOTAL}")
print(f"  Connections: {n_conn}")

for k in range(N_ROWS):
    net.set_biascurrent(EP + k, int(round( U_s[k] * RS)))
    net.set_biascurrent(EN + k, int(round(-U_s[k] * RS)))

g_init = np.random.RandomState(3).randn(N_VAL) * np.max(np.abs(g_ls_s)) * 0.8
sp0 = np.maximum(0.0, g_init) * S_SCALE
sn0 = np.maximum(0.0, -g_init) * S_SCALE
cum_p = np.zeros(N_VAL); cum_n = np.zeros(N_VAL)
off_p = np.zeros(N_VAL); off_n = np.zeros(N_VAL)
for j in range(N_VAL):
    rp = min(int(sp0[j]), VMAX - 1); off_p[j] = sp0[j] - rp; net.set_potential(HP + j, rp)
    rn = min(int(sn0[j]), VMAX - 1); off_n[j] = sn0[j] - rn; net.set_potential(HN + j, rn)


def read_g():                                    # host READ of hold (measurement only)
    pp = np.array([net.potential(HP + j) for j in range(N_VAL)], float)
    pn = np.array([net.potential(HN + j) for j in range(N_VAL)], float)
    return ((pp + cum_p * QUANTUM + off_p) - (pn + cum_n * QUANTUM + off_n)) / S_SCALE


m_hist = np.zeros((N_STEPS, N_VAL))
relay_rate_eff = np.zeros((N_STEPS, N_VAL))      # relay rate per value channel (signed)
eps_rate = np.zeros(N_STEPS); relay_rate = np.zeros(N_STEPS); hold_rate = np.zeros(N_STEPS)
conv = []
# NOTE: no host write to the relay anywhere in this loop - value->relay is synaptic.
W = 100                                          # window for relay rate estimate
rp_buf = np.zeros((W, N_VAL)); rn_buf = np.zeros((W, N_VAL))
for step in range(N_STEPS):
    net.send_synapse()
    counts = net.get_all_spike_counts()
    cum_p[:] += counts[HP:HN]
    cum_n[:] += counts[HN:RP]
    eps_rate[step]   = counts[EP:HP].sum()
    hold_rate[step]  = counts[HP:RP].sum()
    relay_rate[step] = counts[RP:N_TOTAL].sum()
    rp_buf[step % W] = counts[RP:RN]; rn_buf[step % W] = counts[RN:N_TOTAL]
    relay_rate_eff[step] = (rp_buf.mean(0) - rn_buf.mean(0)) / RS_RELAY * QUANTUM
    g = read_g()
    m_hist[step] = (R_inv @ g) / PROBLEM_SCALE
    if step in CHECKS:
        conv.append((step,) + motion_errors(m_hist[step], m_ls))

m_cl = (R_inv @ read_g()) / PROBLEM_SCALE

print("\n  Convergence (synaptic relay, motion vs LS):")
print(f"    {'step':>6s} {'v_dir(deg)':>12s} {'|v|err':>10s} {'w_err':>10s}")
for s, va, vm, we in conv:
    print(f"    {s:6d} {va:12.4f} {vm:10.4f} {we:10.4f}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
va_ls, vm_ls, we_ls = motion_errors(m_cl, m_ls)
va_gt, vm_gt, we_gt = motion_errors(m_cl, m_gt)
print(f"  synaptic-relay vs LS:  v_dir = {va_ls:.4f} deg,  |v|err = {vm_ls:.4f},  w_err = {we_ls:.4f}")
print(f"  synaptic-relay vs GT:  v_dir = {va_gt:.4f} deg,  |v|err = {vm_gt:.4f},  w_err = {we_gt:.4f}")
print(f"  (host-copy relay reference reached ~0.17 deg vs LS)")
print()
if va_ls < 3.0 and we_ls < 0.03:
    print("  PASS  fully-synaptic value->relay reproduced the LS solution")
elif va_ls < 8.0:
    print(f"  OK    closing onto LS (v_dir = {va_ls:.2f} deg) - relay regulated by loop")
else:
    print(f"  WARN  v_dir = {va_ls:.2f} deg - relay line attractor drifted off g")

# ── Plot: does the synaptic relay rate track the held value g? ──────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
steps = np.arange(N_STEPS)
LAB = ["v1", "v2", "v3", "w1", "w2", "w3"]
C = plt.cm.tab10(np.arange(N_VAL))
g_hist = np.array([R_qr @ (m_hist[s] * PROBLEM_SCALE) for s in range(N_STEPS)]) / PROBLEM_SCALE
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Synaptic value->relay: relay tonic rate tracks the held value g",
             fontweight="bold")
for j in range(N_VAL):
    ax[0, 0].plot(steps, m_hist[:, j], color=C[j], lw=1.0, label=LAB[j])
    ax[0, 0].axhline(m_ls[j], color=C[j], ls=":", lw=0.8, alpha=0.7)
ax[0, 0].set_title("recovered motion m(t)  (dotted = LS)"); ax[0, 0].legend(ncol=3, fontsize=8)
ax[0, 0].set_xlabel("step")
for j in range(N_VAL):
    ax[0, 1].plot(steps, g_hist[:, j], color=C[j], lw=1.4, alpha=0.9)
    ax[0, 1].plot(steps, relay_rate_eff[:, j], color=C[j], lw=0.8, ls="--")
ax[0, 1].set_title("held value g (solid) vs relay rate readout (dashed)")
ax[0, 1].set_xlabel("step")
kk = 50
ax[1, 0].plot(steps, np.convolve(eps_rate, np.ones(kk)/kk, 'same'), label="eps", color="tab:red")
ax[1, 0].plot(steps, np.convolve(relay_rate, np.ones(kk)/kk, 'same'), label="relay", color="tab:blue")
ax[1, 0].plot(steps, np.convolve(hold_rate, np.ones(kk)/kk, 'same'), label="hold", color="tab:green")
ax[1, 0].set_title("population firing rates"); ax[1, 0].legend(fontsize=8); ax[1, 0].set_xlabel("step")
ax[1, 1].plot(steps, [motion_errors(m_hist[s], m_ls)[0] for s in range(N_STEPS)], color="tab:purple")
ax[1, 1].set_title("translation-direction error vs LS (deg)"); ax[1, 1].set_xlabel("step")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "pc_egomotion_synaptic_relay_plots.png")
fig.savefig(out, dpi=120)
print(f"  Plots written to {out}")
