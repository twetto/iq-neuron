#!/usr/bin/env python3
"""
Within-frame value stability: PC (held integrator shadow) vs SCN (competitive
rate readout), on the SAME scene, same iqnet substrate. Logs the INSTANTANEOUS
recovered motion every step (no tail-averaging) and measures how much it dithers
once converged. Expectation: PC settles to a flat held line; SCN settles to a
noisy band of the same mean (tonic competition never stops fluctuating).
"""

import os
import tempfile
import numpy as np
from iqif import iqnet


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


OBS_CSV = os.environ.get("OBS_CSV", "")
if OBS_CSV:
    rows = np.loadtxt(OBS_CSV, delimiter=",", skiprows=1)  # x,y,z,ux,uy
    G = np.vstack([motion_field_rows(x, y, z) for x, y, z, _, _ in rows])
    U = np.empty(2 * len(rows))
    U[0::2] = rows[:, 3]
    U[1::2] = rows[:, 4]
    print(f"  loaded {len(rows)} real EuRoC observations from {OBS_CSV}")
else:
    G, U, _ = generate_scene(n_points=40)
N_ROWS = G.shape[0]
N_VAL = 6
Q, R_qr = np.linalg.qr(G, mode="reduced")
R_inv = np.linalg.inv(R_qr)
A_w = Q
m_ls, _, _, _ = np.linalg.lstsq(G, U, rcond=None)
g_ls = R_qr @ m_ls

VMAX, VMIN, RESET = 255, 0, 0
QUANTUM = VMAX - RESET


# ── PC: open-loop 8-bit push-pull (value held in integrator shadow) ─────────
def run_pc(n_steps=6000):
    RATE_SCALE, S_SCALE = 50.0, 2000.0
    problem_scale = 2.0 / np.max(np.abs(g_ls))
    U_s = U * problem_scale
    col_l1 = np.max(np.sum(np.abs(A_w), axis=0))
    wb = max(1, int(QUANTUM / col_l1))
    ep, en, vp, vn = 0, N_ROWS, 2 * N_ROWS, 2 * N_ROWS + N_VAL
    n_total = 2 * N_ROWS + 2 * N_VAL
    td = tempfile.mkdtemp(prefix="iq_pc_")
    pp, cp = os.path.join(td, "p"), os.path.join(td, "c")
    with open(pp, "w") as f:
        for i in range(n_total):
            f.write(f"{i} 0 0 {RESET} 15 15 0\n")
    nc = 0
    with open(cp, "w") as f:
        for k in range(N_ROWS):
            for j in range(N_VAL):
                w = int(round(A_w[k, j] * wb))
                if w == 0:
                    continue
                f.write(f"{ep+k} {vp+j} {+w} 1\n"); f.write(f"{en+k} {vp+j} {-w} 1\n")
                f.write(f"{ep+k} {vn+j} {-w} 1\n"); f.write(f"{en+k} {vn+j} {+w} 1\n")
                nc += 4
        if nc == 0:
            f.write("0 0 0 1\n")
    net = iqnet(pp, cp)
    for i in range(n_total):
        net.set_vmax(i, VMAX); net.set_vmin(i, VMIN)
    for j in range(2 * N_VAL):
        net.set_surrogate_tau(vp + j, 1)
    cum_p = np.zeros(N_VAL); cum_n = np.zeros(N_VAL)
    off_p = np.zeros(N_VAL); off_n = np.zeros(N_VAL)
    g_init = np.zeros(N_VAL)
    for j in range(N_VAL):
        sp = max(0.0, g_init[j]) * S_SCALE; sn = max(0.0, -g_init[j]) * S_SCALE
        rp = min(int(sp), VMAX - 1); rn = min(int(sn), VMAX - 1)
        off_p[j] = sp - rp; off_n[j] = sn - rn
        net.set_potential(vp + j, rp); net.set_potential(vn + j, rn)

    def read_g():
        pp_ = np.array([net.potential(vp + j) for j in range(N_VAL)], float)
        pn_ = np.array([net.potential(vn + j) for j in range(N_VAL)], float)
        return ((pp_ + cum_p * QUANTUM + off_p) - (pn_ + cum_n * QUANTUM + off_n)) / S_SCALE

    res_p = np.zeros(N_ROWS); res_n = np.zeros(N_ROWS)
    m_hist = np.zeros((n_steps, N_VAL))
    for s in range(n_steps):
        g = read_g()
        eps = U_s - A_w @ g
        for k in range(N_ROWS):
            tp = max(0.0, eps[k] * RATE_SCALE); tn = max(0.0, -eps[k] * RATE_SCALE)
            ap = tp + res_p[k]; bp = round(ap); res_p[k] = ap - bp
            an = tn + res_n[k]; bn = round(an); res_n[k] = an - bn
            net.set_biascurrent(ep + k, int(np.clip(bp, 0, VMAX)))
            net.set_biascurrent(en + k, int(np.clip(bn, 0, VMAX)))
        net.send_synapse()
        c = net.get_all_spike_counts()
        cum_p += c[vp:vn]; cum_n += c[vn:n_total]
        m_hist[s] = R_inv @ read_g() / problem_scale
    return m_hist


# ── SCN: inhibition-dominated (clipped), competitive rate readout ───────────
def run_scn(n_steps=15000, n_rand=72):
    rng = np.random.RandomState(1)
    D_dirs = rng.randn(N_VAL, n_rand); D_dirs /= np.linalg.norm(D_dirs, axis=0, keepdims=True)
    N = D_dirs.shape[1]
    d_scale = 0.02 * np.linalg.norm(g_ls)
    D = D_dirs * d_scale
    Phi = A_w @ D
    fU = Phi.T @ U
    COS = D_dirs.T @ D_dirs
    T = 0.5 * d_scale ** 2
    s = VMAX / (2.0 * T)
    LAM, DT, X0, SHIFT = 2.0, 1e-3, VMAX // 2, 6
    bias = np.round(s * DT * LAM * fU).astype(int)
    W = np.round(VMAX * COS).astype(int); np.fill_diagonal(W, 0)
    weight = np.minimum(-W, 0)  # all-inhibitory clip
    td = tempfile.mkdtemp(prefix="iq_scn_")
    pp, cp = os.path.join(td, "p"), os.path.join(td, "c")
    with open(pp, "w") as f:
        for i in range(N):
            f.write(f"{i} {X0} {VMAX} {RESET} {SHIFT} {SHIFT} 0\n")
    nc = 0
    with open(cp, "w") as f:
        for j in range(N):
            for i in range(N):
                if i != j and weight[i, j] != 0:
                    f.write(f"{j} {i} {int(weight[i,j])} 1\n"); nc += 1
        if nc == 0:
            f.write("0 0 0 1\n")
    net = iqnet(pp, cp)
    for i in range(N):
        net.set_vmax(i, VMAX); net.set_vmin(i, VMIN); net.set_surrogate_tau(i, 1)
        net.set_biascurrent(i, int(bias[i])); net.set_potential(i, int(X0))
    spike_hist = np.zeros((n_steps, N))
    for t in range(n_steps):
        net.send_synapse()
        spike_hist[t] = net.get_all_spike_counts()[:N].astype(float)
    return spike_hist, D


def scn_window_readout(spike_hist, D, W):
    """Boxcar window-average readout: r(t) = mean spikes over last W ticks
    (a difference of cumulative counts -> chip-friendly), m(t) = R^-1 D r(t)."""
    n_steps, N = spike_hist.shape
    cs = np.cumsum(spike_hist, axis=0)
    r = np.empty_like(spike_hist)
    for t in range(n_steps):
        lo = max(0, t - W + 1)
        r[t] = (cs[t] - (cs[lo - 1] if lo > 0 else 0)) / (t - lo + 1)
    g = r @ D.T            # (n_steps, N_VAL)
    return (R_inv @ g.T).T


def rel_dither(m, frac=0.4):
    """Scale-invariant within-frame dither: RMS tail std / tail-mean magnitude."""
    tail = m[int((1 - frac) * len(m)):]
    return np.linalg.norm(tail.std(axis=0)) / (np.linalg.norm(tail.mean(axis=0)) + 1e-12)


print("=" * 70)
print("SCN readout: window-average smoothing vs PC held shadow")
print("=" * 70)
m_pc = run_pc()
spike_hist, D = run_scn()
pc_dither = rel_dither(m_pc)
print(f"\n  PC (held shadow) relative within-frame dither = {pc_dither*100:.2f}%  (the target)")

# Sweep the boxcar window W (ticks). Lag = W/2 ticks.
WINDOWS = [1, 50, 200, 500, 1500, 4000, 8000]
print(f"\n  {'window':>7s} {'lag(ticks)':>11s} {'SCN dither':>11s} {'vs PC':>7s}")
dithers = []
m_examples = {}
for W in WINDOWS:
    m_w = scn_window_readout(spike_hist, D, W)
    d = rel_dither(m_w)
    dithers.append(d)
    m_examples[W] = m_w
    print(f"  {W:7d} {W//2:11d} {d*100:10.2f}% {d/max(pc_dither,1e-9):6.1f}x")

# ── Plots: dither-vs-window, and example trajectories at a few windows ──────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
LAB = ["v1", "v2", "v3", "w1", "w2", "w3"]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SCN window-average readout: dither vs window (and vs PC held shadow)",
             fontweight="bold")
ax[0].loglog(WINDOWS, dithers, "o-", color="tab:red", label="SCN window-avg")
ax[0].axhline(pc_dither, color="tab:blue", ls="--", label=f"PC shadow ({pc_dither:.4f})")
ax[0].set_xlabel("window W (ticks)   [lag = W/2]")
ax[0].set_ylabel("within-frame dither (RMS over DoF)")
ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
# v1 trajectory at short/medium/long window + PC
tscn = np.linspace(0, 1, len(spike_hist)); tpc = np.linspace(0, 1, len(m_pc))
for W, c in [(50, "tab:orange"), (500, "tab:red"), (4000, "tab:purple")]:
    ax[1].plot(tscn, m_examples[W][:, 0], color=c, lw=0.7, label=f"SCN W={W}")
ax[1].plot(tpc, m_pc[:, 0], color="tab:blue", lw=1.0, label="PC shadow")
ax[1].axhline(m_ls[0], color="k", ls=":", lw=0.8, label="LS")
ax[1].set_title("v1 readout vs window"); ax[1].set_xlabel("fraction of run")
ax[1].legend(fontsize=8)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "pc_vs_scn_window.png")
fig.savefig(out, dpi=120)
print(f"\n  Plot written to {out}")
