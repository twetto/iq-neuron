#!/usr/bin/env python3
"""
Known-depth 6-DoF egomotion via PC — 8-bit-faithful circuit (potential 0..255).

This is the hardware-realistic variant of test_pc_egomotion.py.  Every neuron
membrane is constrained to the native 8-bit range VMIN=0 .. VMAX=255, which
forces three changes over the float-register version:

1. PUSH-PULL VALUE NEURONS.  A [0,255] membrane with upward-only spikes and a
   VMIN=0 floor cannot hold a signed, bidirectionally-moving quantity.  Each
   motion component m_j is therefore the DIFFERENCE of two non-negative
   integrators, val+_j and val-_j (the same eps+/eps- trick used for errors):

       S+_j = max(0,  sum d_j),   S-_j = max(0, -sum d_j)       (shadow pots)
       m_j ∝ (S+_j - S-_j) / S_SCALE                            (exact: x = max(0,x)-max(0,-x))

   val+ is driven by +d_j = +(A_w^T eps)_j, val- by -d_j; the VMIN=0 clip is
   exactly the rectification, so the difference recovers the signed sum with no
   loss.  Magnitude lives in the (unbounded) spike COUNT; the membrane is just
   the <255 residue.

2. LEAK-FREE INTEGRATOR + NO SYNAPTIC LINGERING.  threshold=0, shift_b=15 gives
   f = x>>15 = 0 over [0,255] (the threshold=5000/shift_a=15 config leaks -1 per
   step).  Value-neuron surrogate_tau=1 makes the synapse deliver exactly one
   step's drive (no 8x accumulator build-up), so the membrane stays bounded.

3. BOUNDED WEIGHTS + RATE-DOMAIN DITHER.  To keep the per-step value drive
   d_j = sum_k w_kj·spike_k <= quantum (so one spike drains the membrane and it
   stays <=255), W_SCALE must be small -> coarse integer weights.  set_weight is
   destructive at runtime (it re-inits the accumulator), so weights are fixed;
   instead we recover sub-integer precision in the RATE domain with a
   sigma-delta dither on the error bias (e.g. target 1.5 -> emit 2,1,2,1,...).
   Toggle DITHER to measure its effect.
"""

import os
import tempfile
import numpy as np

# ── Geometry (shared with test_pc_egomotion.py) ───────────────────────────

def motion_field_rows(x, y, Z):
    A = np.array([[-1.0, 0.0, x],
                  [ 0.0, -1.0, y]])
    B = np.array([[ x*y, -(1.0 + x*x),  y],
                  [ 1.0 + y*y, -x*y,    -x]])
    return np.hstack([A / Z, B])  # 2x6


def generate_scene(n_points=40, noise_sigma=2e-3, seed=7):
    rng = np.random.RandomState(seed)
    v_gt = np.array([0.30, 0.05, 0.12])
    w_gt = np.array([0.02, -0.05, 0.015])
    m_gt = np.concatenate([v_gt, w_gt])
    xs = rng.uniform(-0.70, 0.70, n_points)
    ys = rng.uniform(-0.50, 0.50, n_points)
    Zs = rng.uniform(2.0, 10.0, n_points)
    rows = [motion_field_rows(x, y, Z) for x, y, Z in zip(xs, ys, Zs)]
    G = np.vstack(rows)
    U = G @ m_gt + rng.randn(2 * n_points) * noise_sigma
    return G, U, m_gt


def motion_errors(m_est, m_ref):
    v_est, w_est = m_est[:3], m_est[3:]
    v_ref, w_ref = m_ref[:3], m_ref[3:]
    cos = np.dot(v_est, v_ref) / (np.linalg.norm(v_est) * np.linalg.norm(v_ref) + 1e-12)
    v_ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    v_mag = abs(np.linalg.norm(v_est) - np.linalg.norm(v_ref))
    w_err = np.linalg.norm(w_est - w_ref)
    return v_ang, v_mag, w_err


# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Known-depth 6-DoF egomotion - 8-bit PC circuit (potential 0..255)")
print("=" * 70)

N_POINTS = 40
G, U, m_gt = generate_scene(n_points=N_POINTS, noise_sigma=2e-3)
N_ROWS = G.shape[0]
N_VAL = 6

Q, R_qr = np.linalg.qr(G, mode='reduced')
R_inv = np.linalg.inv(R_qr)
A_w = G @ R_inv                              # = Q, kappa = 1

m_ls, _, _, _ = np.linalg.lstsq(G, U, rcond=None)
g_ls = R_qr @ m_ls
PROBLEM_SCALE = 2.0 / np.max(np.abs(g_ls))
U_s = U * PROBLEM_SCALE
g_ls_s = g_ls * PROBLEM_SCALE
v_ang_ls, v_mag_ls, w_err_ls = motion_errors(m_ls, m_gt)

print(f"\n  {N_ROWS} flow equations, {N_VAL} unknowns,  problem scale = {PROBLEM_SCALE:.2f}")
print(f"  LS vs GT:  v_dir = {v_ang_ls:.3f} deg,  |v| err = {v_mag_ls:.4f},  "
      f"w err = {w_err_ls:.4f}")

from iqif import iqnet

# ── 8-bit constants ────────────────────────────────────────────────────────
VMAX = 255
VMIN = 0
RESET = 0                       # quantum = VMAX - RESET = 255
QUANTUM = VMAX - RESET
RATE_SCALE = 50.0
S_SCALE = 2000.0                # shadow units per (scaled) g-unit -> g res 1/2000
N_SPIKE_STEPS = 6000
SPIKE_CHECKS = [0, 100, 200, 500, 1000, 2000, 3000, 4000, 5999]

# W_SCALE chosen so the worst-case per-step value drive (all error neurons
# firing at rate 1) cannot exceed one quantum -> value membrane stays <=255.
col_l1_max = np.max(np.sum(np.abs(A_w), axis=0))     # max column 1-norm of Q
W_SCALE = max(1, int(QUANTUM / col_l1_max))
print(f"  max column 1-norm of A_w = {col_l1_max:.2f}  ->  W_SCALE = {W_SCALE} "
      f"(bounds value drive <= {QUANTUM})")

# Push-pull layout:
#   eps+  [0, N_ROWS)              rate ~ max(0,  eps_k)
#   eps-  [N_ROWS, 2N_ROWS)        rate ~ max(0, -eps_k)
#   val+  [2N_ROWS, 2N_ROWS+6)     integrates +d_j
#   val-  [2N_ROWS+6, 2N_ROWS+12)  integrates -d_j
N_ERR_TOT = 2 * N_ROWS
VP_OFFSET = N_ERR_TOT
VN_OFFSET = N_ERR_TOT + N_VAL
N_TOTAL = N_ERR_TOT + 2 * N_VAL


def build_files(tmpdir):
    par_path = os.path.join(tmpdir, "params.txt")
    con_path = os.path.join(tmpdir, "conn.txt")
    # rest=0, threshold=0, reset=0, shift_a=15, shift_b=15, noise=0
    #   -> f_min=0, x>=f_min always, f = x>>15 = 0 over [0,255] : leak-free.
    with open(par_path, "w") as pf:
        for i in range(N_TOTAL):
            pf.write(f"{i} 0 0 {RESET} 15 15 0\n")
    # Drive: val+_j gets +d_j, val-_j gets -d_j, where d_j = (A_w^T eps)_j is
    # carried by signed weights from the eps+/eps- pair.  tau=1 (no lingering).
    with open(con_path, "w") as cf:
        n_conn = 0
        for k in range(N_ROWS):
            for j in range(N_VAL):
                w = int(round(A_w[k, j] * W_SCALE))
                if w == 0:
                    continue
                # -> val+_j  (drive +d_j)
                cf.write(f"{k}          {VP_OFFSET + j} {+w} 1\n")
                cf.write(f"{N_ROWS + k} {VP_OFFSET + j} {-w} 1\n")
                # -> val-_j  (drive -d_j)
                cf.write(f"{k}          {VN_OFFSET + j} {-w} 1\n")
                cf.write(f"{N_ROWS + k} {VN_OFFSET + j} {+w} 1\n")
                n_conn += 4
        if n_conn == 0:
            cf.write("0 0 0 1\n")
    return par_path, con_path, n_conn


def run_circuit(dither, record=False):
    tmpdir = tempfile.mkdtemp(prefix="iq8bit_")
    par_path, con_path, n_conn = build_files(tmpdir)
    net = iqnet(par_path, con_path)

    for i in range(N_TOTAL):
        net.set_vmax(i, VMAX)
        net.set_vmin(i, VMIN)
    # Value neurons: kill synaptic lingering so per-step drive = one step only.
    for j in range(2 * N_VAL):
        net.set_surrogate_tau(VP_OFFSET + j, 1)

    # Init: split signed g_init across the val+ / val- shadows.
    g_init = np.random.RandomState(3).randn(N_VAL) * np.max(np.abs(g_ls_s)) * 0.8
    sp_init = np.maximum(0.0,  g_init) * S_SCALE      # -> val+
    sn_init = np.maximum(0.0, -g_init) * S_SCALE      # -> val-
    cum_p = np.zeros(N_VAL); cum_n = np.zeros(N_VAL)
    off_p = np.zeros(N_VAL); off_n = np.zeros(N_VAL)
    for j in range(N_VAL):
        rp = min(int(sp_init[j]), VMAX - 1); off_p[j] = sp_init[j] - rp
        rn = min(int(sn_init[j]), VMAX - 1); off_n[j] = sn_init[j] - rn
        net.set_potential(VP_OFFSET + j, rp)
        net.set_potential(VN_OFFSET + j, rn)

    def read_g():
        pot_p = np.array([net.potential(VP_OFFSET + j) for j in range(N_VAL)], float)
        pot_n = np.array([net.potential(VN_OFFSET + j) for j in range(N_VAL)], float)
        Sp = pot_p + cum_p * QUANTUM + off_p
        Sn = pot_n + cum_n * QUANTUM + off_n
        return (Sp - Sn) / S_SCALE              # scaled g

    res_p = np.zeros(N_ROWS); res_n = np.zeros(N_ROWS)   # sigma-delta carries
    conv = []
    rec = {} if record else None
    if record:
        rec["val_pot"] = np.zeros((N_SPIKE_STEPS, 2 * N_VAL))
        rec["err_pot"] = np.zeros((N_SPIKE_STEPS, N_ERR_TOT))
        rec["spikes"]  = np.zeros((N_SPIKE_STEPS, N_TOTAL), dtype=np.int16)
        rec["m"]       = np.zeros((N_SPIKE_STEPS, N_VAL))
    sp_err_pos = sp_err_neg = 0

    for step in range(N_SPIKE_STEPS):
        g_scaled = read_g()
        eps = U_s - A_w @ g_scaled
        target = eps * RATE_SCALE
        for k in range(N_ROWS):
            tp = max(0.0,  target[k]); tn = max(0.0, -target[k])
            if dither:                                   # error-feedback (sigma-delta)
                ap = tp + res_p[k]; bp = int(round(ap)); res_p[k] = ap - bp
                an = tn + res_n[k]; bn = int(round(an)); res_n[k] = an - bn
            else:
                bp = int(round(tp)); bn = int(round(tn))
            bp = min(max(bp, 0), VMAX); bn = min(max(bn, 0), VMAX)   # keep membrane bounded
            net.set_biascurrent(k,          bp)
            net.set_biascurrent(N_ROWS + k, bn)
        net.send_synapse()

        counts = net.get_all_spike_counts()              # one call; resets counters
        cum_p[:] += counts[VP_OFFSET:VN_OFFSET]
        cum_n[:] += counts[VN_OFFSET:N_TOTAL]
        sp_err_pos += int(counts[:N_ROWS].sum())
        sp_err_neg += int(counts[N_ROWS:N_ERR_TOT].sum())

        if record:
            rec["spikes"][step] = counts
            rec["err_pot"][step] = [net.potential(i) for i in range(N_ERR_TOT)]
            rec["val_pot"][step] = ([net.potential(VP_OFFSET + j) for j in range(N_VAL)] +
                                    [net.potential(VN_OFFSET + j) for j in range(N_VAL)])
            rec["m"][step] = (R_inv @ read_g()) / PROBLEM_SCALE
        if step in SPIKE_CHECKS:
            m_check = (R_inv @ g_scaled) / PROBLEM_SCALE
            conv.append((step,) + motion_errors(m_check, m_ls))

    # --- Two readouts, to quantify what a membrane probe buys you ---
    # (b) SHADOW: counts + final membrane residue + known init offset (exact).
    m_shadow = (R_inv @ read_g()) / PROBLEM_SCALE
    # (a) COUNT-ONLY: spike counters + known init offset, NO membrane read.
    #     The only thing dropped vs shadow is the live <quantum residue (pot),
    #     so the error is bounded by (pot_p - pot_n)/S_SCALE <= quantum/S_SCALE.
    g_count = ((cum_p * QUANTUM + off_p) - (cum_n * QUANTUM + off_n)) / S_SCALE
    m_count = (R_inv @ g_count) / PROBLEM_SCALE

    # Membrane bound check (post-update stored register value).
    if record:
        max_mem = max(rec["val_pot"].max(), rec["err_pot"].max())
        min_mem = min(rec["val_pot"].min(), rec["err_pot"].min())
    else:
        max_mem = min_mem = float('nan')
    return dict(m=m_shadow, m_count=m_count, conv=conv, rec=rec, n_conn=n_conn,
                sp_err=(sp_err_pos, sp_err_neg),
                cum=(cum_p.sum(), cum_n.sum()),
                mem_range=(min_mem, max_mem))


# ── Run without and with dither ─────────────────────────────────────────────
print("\n" + "-" * 70)
print("Running 8-bit circuit  (no dither) ...")
r_nodith = run_circuit(dither=False, record=False)
print("Running 8-bit circuit  (sigma-delta bias dither) ...")
r_dith = run_circuit(dither=True, record=True)

print("\n  Convergence with dither (motion vs LS):")
print(f"    {'step':>6s} {'v_dir(deg)':>12s} {'|v|err':>10s} {'w_err':>10s}")
for step, v_ang, v_mag, w_err in r_dith["conv"]:
    print(f"    {step:6d} {v_ang:12.4f} {v_mag:10.4f} {w_err:10.4f}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"  Neurons: {N_ERR_TOT} error + {2*N_VAL} value (push-pull) = {N_TOTAL}")
print(f"  Connections: {r_dith['n_conn']}   W_SCALE = {W_SCALE}   "
      f"VMAX={VMAX} VMIN={VMIN}")
mr = r_dith["mem_range"]
print(f"  Value+error membrane range (dither run): [{mr[0]:.0f}, {mr[1]:.0f}]  "
      f"{'OK <= 255' if mr[1] <= 255 else 'EXCEEDS 255!'}")
print(f"  Value spike counts: val+ total {r_dith['cum'][0]:.0f}, "
      f"val- total {r_dith['cum'][1]:.0f}")

# Extraction comparison: shadow (counts+membrane) vs count-only (counters only).
print(f"\n  Readout comparison (how the motion comes off the chip):")
print(f"  {'Readout':<34s} {'v_dir(deg)':>12s} {'|v|err':>10s} {'w_err':>10s}")
print(f"  {'-'*34} {'-'*12} {'-'*10} {'-'*10}")
for name, m_est in [
        ("(b) shadow: counts + membrane", r_dith["m"]),
        ("(a) count-only: spike counters", r_dith["m_count"])]:
    va, vm, we = motion_errors(m_est, m_ls)
    print(f"  {name + ' vs LS':<34s} {va:12.4f} {vm:10.4f} {we:10.4f}")
res_g = r_dith["m"] - r_dith["m_count"]
print(f"  count-only loses only the <quantum membrane residue "
      f"(quantum/S_SCALE = {QUANTUM/S_SCALE:.4f} in g-units)")

# On-chip "answer ready" signal: error-population firing rate decay.
err_rate = r_dith["rec"]["spikes"][:, :N_ERR_TOT].sum(axis=1)
smooth = np.convolve(err_rate, np.ones(50) / 50, mode="same")
thresh = 0.05 * smooth.max()
ready = int(np.argmax(smooth < thresh)) if np.any(smooth < thresh) else N_SPIKE_STEPS
print(f"\n  Convergence detector: error-population rate falls below 5% of peak "
      f"at step ~{ready}")
print(f"  (that crossing is the on-chip 'estimate ready' flag - no host monitoring needed)")

print(f"\n  {'Variant':<22s} {'v_dir(deg)':>12s} {'|v|err':>10s} {'w_err':>10s}")
print(f"  {'-'*22} {'-'*12} {'-'*10} {'-'*10}")
for name, r in [("8-bit no dither", r_nodith), ("8-bit + dither", r_dith)]:
    va, vm, we = motion_errors(r["m"], m_ls)
    print(f"  {name + ' vs LS':<22s} {va:12.4f} {vm:10.4f} {we:10.4f}")
va, vm, we = motion_errors(r_dith["m"], m_gt)
print(f"  {'8-bit+dither vs GT':<22s} {va:12.4f} {vm:10.4f} {we:10.4f}")

va_n, _, we_n = motion_errors(r_nodith["m"], m_ls)
va_d, _, we_d = motion_errors(r_dith["m"], m_ls)
print()
if r_dith["mem_range"][1] <= 255 and va_d < 5.0:
    print(f"  PASS  8-bit circuit (0..255) solved egomotion; "
          f"dither {va_n:.2f} -> {va_d:.2f} deg vs LS")
else:
    print(f"  WARN  8-bit vs LS: no-dither {va_n:.2f} deg, dither {va_d:.2f} deg "
          f"(membrane max {r_dith['mem_range'][1]:.0f})")

# ── Plots ───────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rec = r_dith["rec"]
steps = np.arange(N_SPIKE_STEPS)
LABELS = ["v1", "v2", "v3", "w1", "w2", "w3"]
COLORS = plt.cm.tab10(np.arange(N_VAL))

fig, ax = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle("8-bit egomotion PC circuit (potential 0..255, push-pull value, "
             "sigma-delta bias dither)", fontsize=13, fontweight="bold")

# (0,0) motion convergence
for j in range(N_VAL):
    ax[0, 0].plot(steps, rec["m"][:, j], color=COLORS[j], lw=1.2, label=LABELS[j])
    ax[0, 0].axhline(m_ls[j], color=COLORS[j], ls=":", lw=0.9, alpha=0.7)
ax[0, 0].set_title("Recovered motion m(t)  (dotted = numpy LS)")
ax[0, 0].set_xlabel("step"); ax[0, 0].set_ylabel("value"); ax[0, 0].legend(ncol=3, fontsize=8)

# (0,1) value membranes — must stay within [0,255]
for j in range(N_VAL):
    ax[0, 1].plot(steps, rec["val_pot"][:, j], color=COLORS[j], lw=0.8)
    ax[0, 1].plot(steps, rec["val_pot"][:, N_VAL + j], color=COLORS[j], lw=0.8, ls="--")
ax[0, 1].axhline(VMAX, color="k", ls="-", lw=1.0, label="VMAX=255")
ax[0, 1].axhline(VMIN, color="k", ls="-", lw=1.0)
ax[0, 1].set_ylim(-15, 285)
ax[0, 1].set_title("Value-neuron membranes (solid val+, dashed val-) — bounded 0..255")
ax[0, 1].set_xlabel("step"); ax[0, 1].set_ylabel("potential"); ax[0, 1].legend(fontsize=8)

# (1,0) sample error-neuron membranes (sawtooth, also 0..255)
act = rec["spikes"][:, :N_ERR_TOT].sum(axis=0)
for k in np.argsort(act[:N_ROWS])[-3:]:
    ax[1, 0].plot(steps, rec["err_pot"][:, k], lw=0.7, label=f"eps+ #{k}")
ax[1, 0].axhline(VMAX, color="k", ls="-", lw=1.0, label="VMAX=255")
ax[1, 0].set_ylim(-15, 285)
ax[1, 0].set_title("Error-neuron membrane (integrate-and-fire, 0..255)")
ax[1, 0].set_xlabel("step"); ax[1, 0].set_ylabel("potential"); ax[1, 0].legend(fontsize=7)

# (1,1) spike raster (error + value)
ev_s, ev_n = np.nonzero(rec["spikes"])
col = np.where(ev_n < N_ROWS, "tab:blue",
       np.where(ev_n < N_ERR_TOT, "tab:red", "tab:green"))
ax[1, 1].scatter(ev_s, ev_n, s=1.2, c=col, marker="|", linewidths=0.5)
ax[1, 1].axhline(N_ROWS - 0.5, color="k", lw=0.5)
ax[1, 1].axhline(N_ERR_TOT - 0.5, color="k", lw=0.8)
ax[1, 1].set_title(f"Spike raster (blue eps+, red eps-, green val+/-)  "
                   f"val idx >= {N_ERR_TOT}")
ax[1, 1].set_xlabel("step"); ax[1, 1].set_ylabel("neuron index"); ax[1, 1].set_ylim(-1, N_TOTAL)

# (2,0) dither vs no-dither convergence (energy)
def energy(m_hist):
    return np.array([np.sum((U_s - A_w @ (R_qr @ (m_hist[s] * PROBLEM_SCALE)))**2)
                     for s in range(N_SPIKE_STEPS)])
ax[2, 0].semilogy(steps, energy(rec["m"]) + 1e-12, color="tab:green", lw=1.2,
                  label="with dither")
ax[2, 0].set_title("PC free energy  F = ||U - G m||^2")
ax[2, 0].set_xlabel("step"); ax[2, 0].set_ylabel("energy"); ax[2, 0].legend(fontsize=8)

# (2,1) value spike-count difference = the represented state
cdiff = np.cumsum(rec["spikes"][:, VP_OFFSET:VN_OFFSET], axis=0) - \
        np.cumsum(rec["spikes"][:, VN_OFFSET:N_TOTAL], axis=0)
for j in range(N_VAL):
    ax[2, 1].plot(steps, cdiff[:, j], color=COLORS[j], lw=1.0, label=LABELS[j])
ax[2, 1].set_title("Value state carried by spike-count difference (C+ - C-)")
ax[2, 1].set_xlabel("step"); ax[2, 1].set_ylabel("count+ - count-"); ax[2, 1].legend(ncol=3, fontsize=7)

fig.tight_layout(rect=[0, 0, 1, 0.97])
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "pc_egomotion_8bit_plots.png")
fig.savefig(out_png, dpi=120)
print(f"\n  Plots written to {out_png}")
