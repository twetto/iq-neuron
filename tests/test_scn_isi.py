#!/usr/bin/env python3
"""
Does exp-decay on the LATERAL INHIBITION regularize the SCN's firing (lower ISI
CV) so a short readout window suffices ("read after ISI stable")?

Lateral inhibition is delivered through the post-synaptic surrogate_tau decay:
  tau=1  -> instantaneous 1-tick pulse (sharp, irregular firing, our baseline)
  tau>1  -> exp-decay current (smooth inhibition -> hopefully more regular firing)
Weights are rescaled by 1/tau so the *effective* (time-integrated) inhibition
strength is held fixed; only its smoothness changes.  Sweep tau, track ISI CV,
within-frame dither at a fixed window, and accuracy.  OBS_CSV=...frame.csv.
"""
import os
import tempfile
import numpy as np
from iqif import iqnet


def motion_field_rows(x, y, Z):
    A = np.array([[-1.0, 0.0, x], [0.0, -1.0, y]])
    B = np.array([[x*y, -(1.0 + x*x), y], [1.0 + y*y, -x*y, -x]])
    return np.hstack([A / Z, B])


def v_dir_err(m_est, m_ref):
    a, b = m_est[:3], m_ref[:3]
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


OBS_CSV = os.environ.get("OBS_CSV", "")
rows = np.loadtxt(OBS_CSV, delimiter=",", skiprows=1)
G = np.vstack([motion_field_rows(x, y, z) for x, y, z, _, _ in rows])
U = np.empty(2 * len(rows)); U[0::2] = rows[:, 3]; U[1::2] = rows[:, 4]
Q, R_qr = np.linalg.qr(G, mode="reduced")
R_inv = np.linalg.inv(R_qr); A_w = Q
m_ls = np.linalg.lstsq(G, U, rcond=None)[0]; g_ls = R_qr @ m_ls
print(f"  {len(rows)} obs from {OBS_CSV}")

VMAX, RESET, N_VAL, N, N_STEPS = 255, 0, 6, 72, 15000
rng = np.random.RandomState(1)
D_dirs = rng.randn(N_VAL, N); D_dirs /= np.linalg.norm(D_dirs, axis=0, keepdims=True)
d_scale = 0.02 * np.linalg.norm(g_ls)
D = D_dirs * d_scale
Phi = A_w @ D; fU = Phi.T @ U; COS = D_dirs.T @ D_dirs
T_thr = 0.5 * d_scale ** 2; s = VMAX / (2.0 * T_thr)
LAM, DT, X0, SHIFT = 2.0, 1e-3, VMAX // 2, 6
bias = np.round(s * DT * LAM * fU).astype(int)
W_base = np.round(VMAX * COS).astype(float); np.fill_diagonal(W_base, 0.0)
weight_base = np.minimum(-W_base, 0.0)  # all-inhibitory


def run(tau):
    # rescale weights by 1/tau so effective (integrated) inhibition is constant
    w = np.round(weight_base / tau).astype(int)
    td = tempfile.mkdtemp(prefix="iq_tau_")
    pp, cp = os.path.join(td, "p"), os.path.join(td, "c")
    with open(pp, "w") as f:
        for i in range(N):
            f.write(f"{i} {X0} {VMAX} {RESET} {SHIFT} {SHIFT} 0\n")
    with open(cp, "w") as f:
        nc = 0
        for j in range(N):
            for i in range(N):
                if i != j and w[i, j] != 0:
                    f.write(f"{j} {i} {int(w[i,j])} 1\n"); nc += 1
        if nc == 0:
            f.write("0 0 0 1\n")
    net = iqnet(pp, cp)
    for i in range(N):
        net.set_vmax(i, VMAX); net.set_vmin(i, 0)
        net.set_surrogate_tau(i, int(tau))   # exp-decay on incoming (lateral) synapses
        net.set_biascurrent(i, int(bias[i])); net.set_potential(i, int(X0))
    sh = np.zeros((N_STEPS, N))
    for t in range(N_STEPS):
        net.send_synapse()
        sh[t] = net.get_all_spike_counts()[:N].astype(float)
    return sh


def analyze(sh, W=200):
    half = N_STEPS // 2
    totals = sh.sum(0)
    active = np.where(totals > 20)[0]
    cvs = []
    for i in active:
        ts = np.where(sh[half:, i] > 0)[0]
        if len(ts) >= 5:
            isi = np.diff(ts); cvs.append(isi.std() / (isi.mean() + 1e-9))
    cv = float(np.mean(cvs)) if cvs else float("nan")
    cs = np.cumsum(sh, axis=0)
    r = np.empty_like(sh)
    for t in range(N_STEPS):
        lo = max(0, t - W + 1)
        r[t] = (cs[t] - (cs[lo - 1] if lo > 0 else 0)) / (t - lo + 1)
    g = (r @ D.T)
    tail = g[int(0.6 * N_STEPS):]
    rel_dither = np.linalg.norm(tail.std(0)) / (np.linalg.norm(tail.mean(0)) + 1e-12)
    m_mean = R_inv @ tail.mean(0)
    return len(active), cv, rel_dither, v_dir_err(m_mean, m_ls)


print(f"\n  exp-decay on lateral inhibition (weights rescaled 1/tau), frame:")
print(f"  {'tau':>4s} {'active':>7s} {'ISI CV':>8s} {'dither@W200':>12s} {'v_dir(deg)':>11s}")
taus = [1, 2, 4, 8, 16, 32]
results = []
for tau in taus:
    na, cv, dith, vd = analyze(run(tau))
    results.append((tau, na, cv, dith, vd))
    print(f"  {tau:4d} {na:7d} {cv:8.2f} {dith*100:11.2f}% {vd:11.2f}")

# ── Plot CV and dither vs tau ───────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
arr = np.array(results, float)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Exp-decay on lateral inhibition: does smoother inhibition "
             "regularize SCN firing?", fontweight="bold")
ax[0].semilogx(arr[:, 0], arr[:, 2], "o-", color="tab:green", base=2)
ax[0].set_xlabel("lateral synaptic tau (ticks)"); ax[0].set_ylabel("ISI CV")
ax[0].set_title("firing regularity (lower = more clock-like)"); ax[0].grid(alpha=0.3)
ax[1].semilogx(arr[:, 0], arr[:, 3] * 100, "o-", color="tab:red", base=2, label="dither @ W=200")
ax[1].set_xlabel("lateral synaptic tau (ticks)")
ax[1].set_ylabel("within-frame dither @ W=200 (%)")
ax[1].set_title("readout dither at fixed window"); ax[1].grid(alpha=0.3)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scn_lateral_tau.png")
fig.savefig(out, dpi=120)
print(f"\n  Plot written to {out}")
