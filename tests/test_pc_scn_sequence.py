#!/usr/bin/env python3
"""
Frame-to-frame trajectory: LS (ideal) vs PC (held shadow) vs SCN (full-frame
spike count, tau=4 lateral exp-decay), over a real EuRoC sequence. With
full-frame counting the per-frame value is the converged solution, so the
remaining jitter is how far each solver's per-frame solution strays from the
(smooth, scene-driven) LS trajectory.  OBS_SEQ=obs_seq.csv (frame,x,y,z,ux,uy).
"""
import os
import tempfile
import numpy as np
from iqif import iqnet

VMAX, RESET, N_VAL = 255, 0, 6


def motion_field_rows(x, y, Z):
    A = np.array([[-1.0, 0.0, x], [0.0, -1.0, y]])
    B = np.array([[x*y, -(1.0 + x*x), y], [1.0 + y*y, -x*y, -x]])
    return np.hstack([A / Z, B])


def build(rows):
    G = np.vstack([motion_field_rows(x, y, z) for _, x, y, z, _, _ in rows])
    U = np.empty(2 * len(rows)); U[0::2] = rows[:, 4]; U[1::2] = rows[:, 5]
    return G, U


def run_pc(G, U, n_steps=1200):
    N_ROWS = G.shape[0]
    Q, R = np.linalg.qr(G, mode="reduced"); R_inv = np.linalg.inv(R); A_w = Q
    g_ls = Q.T @ U
    ps = 2.0 / np.max(np.abs(g_ls)); U_s = U * ps
    RATE, S = 50.0, 2000.0
    col = np.max(np.sum(np.abs(A_w), 0)); wb = max(1, int(VMAX / col))
    ep, en, vp, vn = 0, N_ROWS, 2 * N_ROWS, 2 * N_ROWS + N_VAL
    nt = 2 * N_ROWS + 2 * N_VAL
    td = tempfile.mkdtemp(); pp, cp = td + "/p", td + "/c"
    with open(pp, "w") as f:
        for i in range(nt):
            f.write(f"{i} 0 0 {RESET} 15 15 0\n")
    with open(cp, "w") as f:
        nc = 0
        for k in range(N_ROWS):
            for j in range(N_VAL):
                w = int(round(A_w[k, j] * wb))
                if w:
                    f.write(f"{ep+k} {vp+j} {w} 1\n"); f.write(f"{en+k} {vp+j} {-w} 1\n")
                    f.write(f"{ep+k} {vn+j} {-w} 1\n"); f.write(f"{en+k} {vn+j} {w} 1\n"); nc += 4
        if not nc:
            f.write("0 0 0 1\n")
    net = iqnet(pp, cp)
    for i in range(nt):
        net.set_vmax(i, VMAX); net.set_vmin(i, 0)
    for j in range(2 * N_VAL):
        net.set_surrogate_tau(vp + j, 1)
    cp_, cn_ = np.zeros(N_VAL), np.zeros(N_VAL)
    rp_, rn_ = np.zeros(N_ROWS), np.zeros(N_ROWS)

    def read_g():
        a = np.array([net.potential(vp + j) for j in range(N_VAL)], float)
        b = np.array([net.potential(vn + j) for j in range(N_VAL)], float)
        return ((a + cp_ * VMAX) - (b + cn_ * VMAX)) / S
    for _ in range(n_steps):
        g = read_g(); eps = U_s - A_w @ g
        tp = np.maximum(0.0, eps * RATE) + rp_; bp = np.round(tp); rp_ = tp - bp
        tn = np.maximum(0.0, -eps * RATE) + rn_; bn = np.round(tn); rn_ = tn - bn
        bpc = np.clip(bp, 0, VMAX).astype(int); bnc = np.clip(bn, 0, VMAX).astype(int)
        for k in range(N_ROWS):
            net.set_biascurrent(ep + k, int(bpc[k]))
            net.set_biascurrent(en + k, int(bnc[k]))
        net.send_synapse()
        c = net.get_all_spike_counts(); cp_ += c[vp:vn]; cn_ += c[vn:nt]
    return R_inv @ read_g() / ps


_FRAME = np.random.RandomState(1).randn(N_VAL, 72)
_FRAME /= np.linalg.norm(_FRAME, axis=0, keepdims=True)


def run_scn(G, U, n_steps=6000, tau=4):
    Q, R = np.linalg.qr(G, mode="reduced"); R_inv = np.linalg.inv(R); A_w = Q
    g_ls = Q.T @ U
    N = 72; d_scale = 0.02 * np.linalg.norm(g_ls); D = _FRAME * d_scale
    Phi = A_w @ D; fU = Phi.T @ U; COS = _FRAME.T @ _FRAME
    T = 0.5 * d_scale ** 2; s = VMAX / (2.0 * T)
    LAM, DT, X0, SH = 2.0, 1e-3, VMAX // 2, 6
    bias = np.round(s * DT * LAM * fU).astype(int)
    Wb = np.round(VMAX * COS).astype(float); np.fill_diagonal(Wb, 0.0)
    w = np.round(np.minimum(-Wb, 0.0) / tau).astype(int)  # all-inhibitory, /tau
    td = tempfile.mkdtemp(); pp, cp = td + "/p", td + "/c"
    with open(pp, "w") as f:
        for i in range(N):
            f.write(f"{i} {X0} {VMAX} {RESET} {SH} {SH} 0\n")
    with open(cp, "w") as f:
        nc = 0
        for j in range(N):
            for i in range(N):
                if i != j and w[i, j]:
                    f.write(f"{j} {i} {int(w[i,j])} 1\n"); nc += 1
        if not nc:
            f.write("0 0 0 1\n")
    net = iqnet(pp, cp)
    for i in range(N):
        net.set_vmax(i, VMAX); net.set_vmin(i, 0)
        net.set_surrogate_tau(i, tau)        # exp-decay lateral inhibition
        net.set_biascurrent(i, int(bias[i])); net.set_potential(i, int(X0))
    counts = np.zeros(N); warm = n_steps // 4
    for t in range(n_steps):
        net.send_synapse()
        c = net.get_all_spike_counts()[:N].astype(float)
        if t >= warm:                         # full-frame count over converged part
            counts += c
    rate = counts / (n_steps - warm)
    r = rate / (LAM * DT)                      # match the SCN readout calibration
    return R_inv @ (D @ r)


print("=" * 70)
print("Frame-to-frame: LS vs PC vs SCN(tau=4) over EuRoC sequence")
print("=" * 70)
data = np.loadtxt(os.environ["OBS_SEQ"], delimiter=",", skiprows=1)
frames = np.unique(data[:, 0]).astype(int)
m_ls, m_pc, m_scn = [], [], []
for fr in frames:
    rows = data[data[:, 0] == fr]
    if len(rows) < 8:
        continue
    G, U = build(rows)
    m_ls.append(np.linalg.lstsq(G, U, rcond=None)[0])
    m_pc.append(run_pc(G, U))
    m_scn.append(run_scn(G, U))
m_ls = np.array(m_ls); m_pc = np.array(m_pc); m_scn = np.array(m_scn)
print(f"  {len(m_ls)} frames solved")


def dev(m):  # RMS deviation from LS trajectory, per DoF, then overall
    return np.sqrt(((m - m_ls) ** 2).mean(0))


LAB = ["v1", "v2", "v3", "w1", "w2", "w3"]
print(f"\n  RMS deviation from LS trajectory (= frame-to-frame jitter added):")
print(f"    {'DoF':>4s} {'PC':>10s} {'SCN':>10s}")
for j in range(N_VAL):
    print(f"    {LAB[j]:>4s} {dev(m_pc)[j]:10.4f} {dev(m_scn)[j]:10.4f}")
print(f"\n  overall ||dev||:  PC = {np.linalg.norm(dev(m_pc)):.4f}   "
      f"SCN = {np.linalg.norm(dev(m_scn)):.4f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
fig.suptitle("EuRoC sequence: LS (black) vs PC (blue) vs SCN tau=4 (red) — "
             "full-frame count", fontweight="bold")
x = np.arange(len(m_ls))
for j in range(N_VAL):
    a = ax[j // 3, j % 3]
    a.plot(x, m_ls[:, j], color="k", lw=1.4, label="LS")
    a.plot(x, m_pc[:, j], color="tab:blue", lw=1.0, label="PC")
    a.plot(x, m_scn[:, j], color="tab:red", lw=1.0, label="SCN")
    a.set_title(f"{LAB[j]}  dev PC={dev(m_pc)[j]:.3f} SCN={dev(m_scn)[j]:.3f}", fontsize=9)
    a.set_xlabel("frame")
    if j == 0:
        a.legend(fontsize=8)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc_scn_sequence.png")
fig.savefig(out, dpi=120)
print(f"\n  Plot written to {out}")
