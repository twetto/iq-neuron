#!/usr/bin/env python3
"""
INTEGER 8-bit quantization of the continuous-LIF spike-coding-network egomotion
solver (test_pc_egomotion_scn.py).  Answers the one question the MKM-2020 paper
can't: does the integer IQIF substrate hold the "bounce geometry" tightly enough
for the population readout to still equal the least-squares egomotion?

The continuous SCN solves  min 1/2||U - A_w g||^2  with readout g = D r, lateral
inhibition Omega = Phi^T Phi (Phi = A_w D), threshold T_i = 1/2||Phi_i||^2.  All
decoder columns are unit-norm, so Phi_i has norm D_scale and the threshold is
UNIFORM: T = 1/2 D_scale^2.  That lets us map onto an integer membrane cleanly:

    M = c*V + VMAX/2 ,   c = VMAX/(2T) ,   VMAX = 2^B - 1     (fires at M >= VMAX)

Under this map the recurrent weights become, exactly,
    W[i,j] = round( c * Omega[i,j] ) = round( VMAX * (d_i . d_j) )            (!)
i.e. the bounce directions are just the (quantized) cosines between decoder
directions, scaled by the full membrane range.  The spike self-drop W[i,i] =
VMAX is the reset; off-diagonals are the lateral inhibition.  Per-step drive and
leak are integers too.  We sweep B and measure the readout error eta.

Everything that the chip computes (membrane, weights, drive, spikes) is integer;
only the downstream linear decode g = D r is real-valued (a readout layer).
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
print("Integer 8-bit spike-coding network: 6-DoF egomotion")
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
COS = D_dirs.T @ D_dirs                                            # cosine / bounce matrix
T = 0.5 * D_SCALE ** 2                                             # uniform threshold (V units)

LAM = 10.0
DT = 1e-3
N_STEPS = 30000
MAX_SPK = 400                   # relax fully each step (break when none > thr);
                                # tight balance keeps many neurons near threshold
TAIL = N_STEPS // 2


def run(bits, quantize=True, seed=0):
    """Faithful chip dynamics: synchronous, ONE spike per neuron per tick,
    recurrent delivered with a one-tick synaptic delay (like net.send_synapse),
    small membrane noise to desynchronize.  When quantize, the membrane M,
    weights W, bias and threshold are integers in [0, VMAX]; the only real-valued
    part is the downstream linear decode g = D r.  Map: M = c*V + VMAX/2,
    fires at M >= VMAX; W[i,j] = round(VMAX * cos_ij) (diag = VMAX = self-reset).
    Returns (m_readout, g_readout, n_active, total_spikes)."""
    rj = np.random.RandomState(seed)
    if quantize:
        VMAX = float(2 ** bits - 1)
        c = VMAX / (2.0 * T)
        W = np.round(VMAX * COS)                       # integer bounce geometry
        bias = np.round(c * DT * LAM * fU)             # integer per-step drive
        thr = VMAX
        rest = VMAX / 2.0
        noise = max(1.0, VMAX * 1e-3)                  # ~1 lsb tie-break noise
    else:
        c = 1.0 / (2.0 * T)                            # same V->count map, no rounding
        VMAX = c * (2.0 * T)                           # = 1.0  -> threshold at 1
        W = COS.copy()                                 # exact cosines (diag = 1)
        bias = c * DT * LAM * fU
        thr = VMAX
        rest = VMAX / 2.0
        noise = VMAX * 1e-9
    W_off = W - np.diag(np.diag(W))                    # lateral (off-diagonal) part
    M = np.full(N, rest)
    r = np.zeros(N)
    rec = np.zeros(N)                                  # recurrent input (1-tick delay)
    g_hist = np.zeros((N_STEPS, N_VAL))
    rate = np.zeros(N)
    for t in range(N_STEPS):
        M += bias + rec - (LAM * DT) * (M - rest)      # drive + delayed recurrent + leak
        M += rj.randn(N) * noise
        fired = M >= thr
        nf = int(fired.sum())
        if nf:
            M[fired] -= VMAX                           # self-reset (diagonal of W)
            rec = -(W_off[:, fired].sum(axis=1))       # lateral inhibition, next tick
            r[fired] += 1.0
            rate[fired] += 1.0
        else:
            rec = np.zeros(N)
        r += DT * (-LAM * r)
        g_hist[t] = D @ r
    g_out = g_hist[TAIL:].mean(0)
    return R_inv @ g_out, g_out, int((rate > 0).sum()), int(rate.sum())


# ── Sweep bit depth ─────────────────────────────────────────────────────────
print(f"\n  N={N} neurons,  D_scale={D_SCALE:.5f},  ||g_ls||={np.linalg.norm(g_ls):.4f}")
print(f"\n  {'bits':>5s} {'VMAX':>6s} {'v_dir(deg)':>11s} {'w_err':>9s} "
      f"{'||g-g_ls||':>11s} {'rel.eta':>8s} {'active':>7s} {'spikes':>8s}")
BITS = [4, 5, 6, 7, 8, 9, 10, 12]
rows = []
mf, gf, af, sf = run(0, quantize=False)
vf, _, wf = motion_errors(mf, m_ls)
ef = np.linalg.norm(gf - g_ls)
print(f"  {'float':>5s} {'-':>6s} {vf:11.4f} {wf:9.5f} {ef:11.5f} "
      f"{ef/np.linalg.norm(g_ls):8.4f} {af:7d} {sf:8d}")
for b in BITS:
    m_b, g_b, a_b, s_b = run(b, quantize=True)
    v_b, _, w_b = motion_errors(m_b, m_ls)
    e_b = np.linalg.norm(g_b - g_ls)
    rel = e_b / np.linalg.norm(g_ls)
    rows.append((b, 2 ** b - 1, v_b, w_b, e_b, rel, a_b, s_b))
    print(f"  {b:5d} {2**b-1:6d} {v_b:11.4f} {w_b:9.5f} {e_b:11.5f} "
          f"{rel:8.4f} {a_b:7d} {s_b:8d}")

b8 = next(r for r in rows if r[0] == 8)
print("\n" + "=" * 70)
print(f"  8-bit readout:  v_dir = {b8[2]:.3f} deg vs LS,  w_err = {b8[3]:.4f},  "
      f"rel.eta = {b8[5]*100:.1f}%")
print(f"  float reference: v_dir = {vf:.3f} deg,  rel.eta = {ef/np.linalg.norm(g_ls)*100:.1f}%")
if b8[2] < 3.0:
    print("  PASS  the bounce geometry survives 8-bit quantization")
else:
    print(f"  WARN  8-bit v_dir = {b8[2]:.2f} deg - geometry too coarse at 8 bits")

# ── Plot: eta vs bit depth ──────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
bb = np.array([r[0] for r in rows])
rel = np.array([r[5] for r in rows]) * 100.0
vdir = np.array([r[2] for r in rows])
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Integer-quantized spike-coding network: egomotion readout vs bit depth",
             fontweight="bold")
ax[0].semilogy(bb, rel, "o-", color="tab:blue")
ax[0].axhline(ef / np.linalg.norm(g_ls) * 100, color="tab:green", ls="--",
              label=f"float ({ef/np.linalg.norm(g_ls)*100:.1f}%)")
ax[0].axvline(8, color="0.6", ls=":")
ax[0].set_xlabel("membrane bits B"); ax[0].set_ylabel("readout error ||g-g_ls|| / ||g_ls||  (%)")
ax[0].set_title("discretization error eta vs bit depth"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].semilogy(bb, vdir, "o-", color="tab:purple")
ax[1].axhline(vf, color="tab:green", ls="--", label=f"float ({vf:.2f} deg)")
ax[1].axvline(8, color="0.6", ls=":")
ax[1].set_xlabel("membrane bits B"); ax[1].set_ylabel("translation-direction error vs LS (deg)")
ax[1].set_title("egomotion accuracy vs bit depth"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "pc_egomotion_scn_int_plots.png")
fig.savefig(out, dpi=120)
print(f"\n  Plot written to {out}")
