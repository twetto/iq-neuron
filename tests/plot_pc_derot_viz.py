#!/usr/bin/env python3
"""
Meeting visuals for the IMU de-rotation egomotion circuit used by
`rust/crates/iqif-vio/examples/euroc_egomotion_imu_live.rs`, on ONE real EuRoC
frame. PC (predictive-coding) analogue of tests/plot_scn_viz.py.

The live example now runs the CLOSED-LOOP circuit: the camera-frame gyro rate
`omega` de-rotates the flow (subtract B(x)·omega), and the 8-bit predictive-
coding circuit solves the 3-DoF translation `v` with `G` realised as ON-CHIP
FEEDBACK weights — the host does no matmul (transliteration of `cl_relax` in
iqif-vio/src/lib.rs / tests/test_pc_egomotion_closedloop.py). `w1/w2/w3` are the
IMU rate itself (constant within the frame, not solved). This mirrors that exact
circuit on the IQIF chip and produces:

  - pc_derot_raster.png   : per-neuron membrane heatmap + spike raster + the
                            recovered 6-DoF motion m(t) over the relaxation
  - pc_derot_weights.png  : the single synaptic weight matrix W[post,pre] the
                            chip acts on — forward A_wᵀ (eps→hold, dense bottom),
                            feedback A_w (relay→eps, dense right), and the
                            hold→relay host-copy pseudo-block (bottom-right corner)

Three populations (8-bit 0..255, leak-free): eps+/- (tonic error, eps=U-A_w g),
hold+/- (hold g in their integrator shadow = membrane + count·quantum),
relay+/- (tonic readout of g whose spikes carry the prediction through the
feedback weights). Decode v = R⁻¹ g.

Usage:
    OBS_CSV=rust/derot_frame_150.csv python tests/plot_pc_derot_viz.py
    python tests/plot_pc_derot_viz.py rust/derot_frame_150.csv
The CSV is produced by the Rust example:
    DUMP_DEROT_PATH=derot_frame_150.csv DUMP_DEROT_FRAME=150 \
        cargo run -p iqif-vio --example euroc_egomotion_imu --release -- \
        /path/to/V1_01_easy 160
"""

import os
import sys
import tempfile
import numpy as np
from iqif import iqnet

# ── Closed-loop 8-bit constants (mirror iqif-vio/src/lib.rs cl_relax) ────────
VMAX, VMIN, RESET = 255, 0, 0
QUANTUM = VMAX - RESET            # 255
RS = 50.0                         # error & relay rate scale (bias per scaled unit)
WF = QUANTUM                      # feedback weight scale
S_SCALE = 2000.0                  # shadow units per scaled-g unit
N_STEPS = int(os.environ.get("PC_STEPS", "6000"))
N_VAL = 3                         # de-rotated solve: translation only


def motion_field_rows(x, y, Z):
    """Per-feature 2x6 block [ (1/Z)A | B ] (rows = u_x,u_y; cols = v1..w3)."""
    A = np.array([[-1.0, 0.0, x], [0.0, -1.0, y]])
    B = np.array([[x * y, -(1.0 + x * x), y], [1.0 + y * y, -x * y, -x]])
    return np.hstack([A / Z, B])


# ── Load the single real frame's de-rotated problem (obs + camera-frame omega)
OBS = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OBS_CSV", "rust/derot_frame_150.csv")
if not os.path.isabs(OBS):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OBS = OBS if os.path.exists(OBS) else os.path.join(root, OBS)

with open(OBS) as f:
    first = f.readline().strip()
if not first.startswith("#"):
    raise SystemExit(f"{OBS}: expected a '# omega_cam,wx,wy,wz' header line")
omega = np.array([float(v) for v in first.split(",")[1:4]])
rows = np.loadtxt(OBS, delimiter=",", comments="#", skiprows=2)  # x,y,z,ux,uy

# Subsample the feature points (evenly spread) so the error population doesn't
# dwarf the hold/relay bands — keeps the weight-matrix blocks legible.
MAX_OBS = int(os.environ.get("MAX_OBS", "30"))
if len(rows) > MAX_OBS:
    rows = rows[np.linspace(0, len(rows) - 1, MAX_OBS).round().astype(int)]

G = np.vstack([motion_field_rows(x, y, z) for x, y, z, _, _ in rows])   # 2N x 6
U = np.empty(2 * len(rows)); U[0::2] = rows[:, 3]; U[1::2] = rows[:, 4]  # flow
G_v = G[:, :3]                                   # (1/Z)A  : translation block
G_w = G[:, 3:]                                   # B       : rotation block
U_res = U - G_w @ omega                           # de-rotated flow (rotation removed)

N_ROWS = G_v.shape[0]
Q, R_qr = np.linalg.qr(G_v, mode="reduced")
R_inv = np.linalg.inv(R_qr)
A_w = Q                                          # kappa = 1, decode v = R^-1 g
g_ls = Q.T @ U_res
v_ls = (R_inv @ g_ls)                            # de-rotated translation LS
PROBLEM_SCALE = 2.0 / np.max(np.abs(g_ls))
U_s = U_res * PROBLEM_SCALE

col_l1_max = np.max(np.sum(np.abs(A_w), axis=0))
WB = max(1, int(QUANTUM / col_l1_max))           # forward scale; per-step hold drive <= quantum
cond = np.linalg.cond(G_v)
print(f"  {len(rows)} obs from {OBS}")
print(f"  omega_cam (rad/s) = [{omega[0]:.5f} {omega[1]:.5f} {omega[2]:.5f}]")
print(f"  3-DoF de-rotated system: {N_ROWS} flow eqs, cond(G_v) = {cond:.1f}, "
      f"WB={WB} WF={WF}")
print(f"  LS de-rotated v (m/s) = [{v_ls[0]:.4f} {v_ls[1]:.4f} {v_ls[2]:.4f}]")

# ── Build the IQIF network: eps+ | eps- | hold+ | hold- | relay+ | relay- ───
EP, EN = 0, N_ROWS
HP = 2 * N_ROWS
HN = HP + N_VAL
RP = HN + N_VAL
RN = RP + N_VAL
N_TOTAL = RN + N_VAL                              # 2N_ROWS + 4 N_VAL

td = tempfile.mkdtemp(prefix="iq_pcderot_cl_")
pp, cp = os.path.join(td, "p"), os.path.join(td, "c")
with open(pp, "w") as f:                          # leak-free: threshold=0, shift_b=15
    for i in range(N_TOTAL):
        f.write(f"{i} 0 0 {RESET} 15 15 0\n")
with open(cp, "w") as f:
    n_conn = 0
    for k in range(N_ROWS):
        for j in range(N_VAL):
            wb = int(round(A_w[k, j] * WB))
            wf = int(round(A_w[k, j] * WF))
            if wb != 0:                           # forward A_w^T : eps -> hold
                f.write(f"{EP + k} {HP + j} {+wb} 1\n")
                f.write(f"{EN + k} {HP + j} {-wb} 1\n")
                f.write(f"{EP + k} {HN + j} {-wb} 1\n")
                f.write(f"{EN + k} {HN + j} {+wb} 1\n")
                n_conn += 4
            if wf != 0:                           # feedback A_w : relay -> eps (delivers -p)
                f.write(f"{RP + j} {EP + k} {-wf} 1\n")
                f.write(f"{RN + j} {EP + k} {+wf} 1\n")
                f.write(f"{RP + j} {EN + k} {+wf} 1\n")
                f.write(f"{RN + j} {EN + k} {-wf} 1\n")
                n_conn += 4
    if n_conn == 0:
        f.write("0 0 0 1\n")

net = iqnet(pp, cp)
for i in range(N_TOTAL):
    net.set_vmax(i, VMAX); net.set_vmin(i, VMIN)
for i in range(2 * N_ROWS + 2 * N_VAL):           # eps + hold receive synapses: no lingering
    net.set_surrogate_tau(i, 1)

# Data U enters as a constant signed bias on the error neurons.
for k in range(N_ROWS):
    net.set_biascurrent(EP + k, int(round(U_s[k] * RS)))
    net.set_biascurrent(EN + k, int(round(-U_s[k] * RS)))

# Held estimate starts at zero (g_init = 0).
cum_p = np.zeros(N_VAL); cum_n = np.zeros(N_VAL)
off_p = np.zeros(N_VAL); off_n = np.zeros(N_VAL)
for j in range(N_VAL):
    net.set_potential(HP + j, 0); net.set_potential(HN + j, 0)


def read_g():
    pp_ = np.array([net.potential(HP + j) for j in range(N_VAL)], float)
    pn_ = np.array([net.potential(HN + j) for j in range(N_VAL)], float)
    Sp = pp_ + cum_p * QUANTUM + off_p
    Sn = pn_ + cum_n * QUANTUM + off_n
    return (Sp - Sn) / S_SCALE                    # scaled g


# ── Relaxation: record membrane, spikes, and recovered 6-DoF m(t) ───────────
MEM_DS = max(1, N_STEPS // 1500)                  # subsample the membrane probe
spike_hist = np.zeros((N_STEPS, N_TOTAL), dtype=np.int16)
m_hist = np.zeros((N_STEPS, 6))                    # [v1 v2 v3 | w1 w2 w3]
m_hist[:, 3:] = omega                              # w = IMU rate (constant)
pot_samples, pot_steps = [], []

for step in range(N_STEPS):
    g = read_g()
    for j in range(N_VAL):                         # relay tonic readout (local per-neuron copy)
        net.set_biascurrent(RP + j, max(0, int(round(g[j] * RS))))
        net.set_biascurrent(RN + j, max(0, int(round(-g[j] * RS))))
    net.send_synapse()
    counts = net.get_all_spike_counts()
    spike_hist[step] = counts
    cum_p[:] += counts[HP:HN]
    cum_n[:] += counts[HN:RP]
    m_hist[step, :3] = (R_inv @ read_g()) / PROBLEM_SCALE
    if step % MEM_DS == 0:
        pot_samples.append([net.potential(i) for i in range(N_TOTAL)])
        pot_steps.append(step)

pot_hist = np.array(pot_samples, dtype=np.int16)   # (n_samp, N_TOTAL)
v_final = m_hist[-1, :3]
v_dir_err = np.degrees(np.arccos(np.clip(
    np.dot(v_final, v_ls) / (np.linalg.norm(v_final) * np.linalg.norm(v_ls) + 1e-12),
    -1.0, 1.0)))
mem_max = int(pot_hist.max())
eps_spikes = int(spike_hist[:, :HP].sum())
hold_spikes = int(spike_hist[:, HP:RP].sum())
relay_spikes = int(spike_hist[:, RP:].sum())
print(f"  PC v_final (m/s) = [{v_final[0]:.4f} {v_final[1]:.4f} {v_final[2]:.4f}]"
      f"   v_dir err vs LS = {v_dir_err:.3f} deg")
print(f"  membrane max = {mem_max} ({'OK <= 255' if mem_max <= 255 else 'EXCEEDS 255!'})"
      f"   spikes: eps {eps_spikes}, hold {hold_spikes}, relay {relay_spikes}")

# ── Figure 1: membrane heatmap + spike raster + recovered 6-DoF m(t) ────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = ["v1", "v2", "v3", "w1", "w2", "w3"]
C = plt.cm.tab10(np.arange(6))
steps = np.arange(N_STEPS)
bands = [EN, HP, HN, RP, RN]                       # population boundaries

fig, ax = plt.subplots(3, 1, figsize=(15, 13),
                       gridspec_kw={"height_ratios": [3, 2, 2]})
fig.suptitle("IMU de-rotation egomotion on the IQIF chip (EuRoC frame): 8-bit "
             "CLOSED-LOOP predictive-coding relaxation\n"
             "membrane, spikes, recovered 6-DoF (v solved on-chip, w = IMU rate)",
             fontweight="bold")

# (0) membrane heatmap
im = ax[0].imshow(pot_hist.T, aspect="auto", origin="lower", cmap="viridis",
                  vmin=0, vmax=255, extent=[0, N_STEPS, 0, N_TOTAL],
                  interpolation="nearest")
for b in bands:
    ax[0].axhline(b, color="w", lw=0.8)
ax[0].set_title(f"membrane potential (0..255)   bands (bottom->top): eps+ | eps- | "
                f"hold+ | hold- | relay+ | relay-  (hold/relay = {N_VAL} each, near top)")
ax[0].set_ylabel("neuron index")
fig.colorbar(im, ax=ax[0], fraction=0.02, label="potential")

# (1) spike raster
ev_t, ev_n = np.nonzero(spike_hist)
col = np.where(ev_n < HP, "tab:red", np.where(ev_n < RP, "tab:green", "tab:blue"))
ax[1].scatter(ev_t, ev_n, s=1.0, c=col, marker="|", linewidths=0.4)
for b in bands:
    ax[1].axhline(b - 0.5, color="k", lw=0.5)
ax[1].set_ylim(-1, N_TOTAL)
ax[1].set_title("spike raster   (red eps, green hold, blue relay)   "
                "errors fire hardest early; relay holds a tonic rate carrying the prediction")
ax[1].set_ylabel("neuron index")

# (2) recovered 6-DoF motion m(t): v solved (dotted = LS), w = constant IMU rate
for j in range(3):
    ax[2].plot(steps, m_hist[:, j], color=C[j], lw=1.2, label=LAB[j])
    ax[2].axhline(v_ls[j], color=C[j], ls=":", lw=0.9, alpha=0.7)
for j in range(3, 6):
    ax[2].plot(steps, m_hist[:, j], color=C[j], lw=1.4, ls="--", label=f"{LAB[j]} (IMU)")
ax[2].axhline(0, color="0.8", lw=0.6)
ax[2].set_title("recovered 6-DoF m(t) = [R^-1 g / scale ; omega_IMU]   "
                "(dotted = de-rotated LS for v; w held at the gyro rate)")
ax[2].legend(ncol=6, fontsize=8); ax[2].set_xlabel("chip tick (relaxation step)")
ax[2].set_ylabel("m (v: m/s, w: rad/s)")

fig.tight_layout(rect=[0, 0, 1, 0.96])
out1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc_derot_raster.png")
fig.savefig(out1, dpi=120); print(f"  wrote {out1}")

# ── Figure 2: the single synaptic weight matrix W[post,pre] the chip acts on ─
# Rebuild W EXACTLY as written to the connection file. Closed-loop -> two dense
# blocks (forward eps->hold, feedback relay->eps) plus the host-copy hold->relay
# pseudo-block (NOT a synapse) marked for honesty.
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle

W = np.zeros((N_TOTAL, N_TOTAL))                   # W[post, pre]
for k in range(N_ROWS):
    for j in range(N_VAL):
        wb = int(round(A_w[k, j] * WB))
        wf = int(round(A_w[k, j] * WF))
        # forward A_w^T (eps -> hold): gradient path, "error -> value"
        W[HP + j, EP + k] += wb;  W[HP + j, EN + k] += -wb
        W[HN + j, EP + k] += -wb; W[HN + j, EN + k] += wb
        # feedback A_w (relay -> eps): prediction path, "value -> error"
        W[EP + k, RP + j] += -wf; W[EP + k, RN + j] += wf
        W[EN + k, RP + j] += wf;  W[EN + k, RN + j] += -wf

nz = int(np.count_nonzero(W))
print(f"  synaptic matrix W[post,pre]: {N_TOTAL}x{N_TOTAL}, {nz} nonzero "
      f"(forward eps->hold + feedback relay->eps)")

bounds = [0, EN, HP, HN, RP, RN, N_TOTAL]
pop_lab = ["eps+", "eps-", "hold+", "hold-", "relay+", "relay-"]
centres = [(bounds[i] + bounds[i + 1]) / 2 for i in range(6)]

fig2, ax3 = plt.subplots(figsize=(11, 10))
fig2.suptitle("IMU de-rotation CLOSED-LOOP PC circuit: the synaptic weight "
              "matrix the IQIF chip acts on   W[post, pre]", fontweight="bold")
wmax = max(1.0, np.max(np.abs(W)))
im3 = ax3.imshow(W, cmap="RdBu_r", origin="upper",
                 norm=SymLogNorm(linthresh=1.0, vmin=-wmax, vmax=wmax),
                 interpolation="nearest")
fig2.colorbar(im3, ax=ax3, fraction=0.046, label="weight (signed, symlog)")
for b in bounds[1:-1]:
    ax3.axhline(b - 0.5, color="0.4", lw=0.6)
    ax3.axvline(b - 0.5, color="0.4", lw=0.6)
ax3.set_xticks(centres); ax3.set_xticklabels(pop_lab, rotation=45, ha="right")
ax3.set_yticks(centres); ax3.set_yticklabels(pop_lab)
ax3.set_xlabel("presynaptic neuron  (source of spike)")
ax3.set_ylabel("postsynaptic neuron  (receives weight)")


def label_block(rlo, rhi, clo, chi, text, edge, ls="-"):
    ax3.add_patch(Rectangle((clo - 0.5, rlo - 0.5), chi - clo, rhi - rlo,
                            fill=False, edgecolor=edge, lw=2.0, ls=ls))
    ax3.annotate(text, xy=((clo + chi) / 2, (rlo + rhi) / 2), ha="center",
                 va="center", fontsize=9, fontweight="bold", color=edge,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=edge, alpha=0.85))


# forward gradient block: post = hold (dense BOTTOM), pre = eps
label_block(HP, RP, EP, HP, "A$_w^\\mathsf{T}$ (forward)\nerror $\\to$ value\n(dense bottom)",
            "tab:green")
# feedback prediction block: post = eps, pre = relay (dense RIGHT)
label_block(EP, HP, RP, N_TOTAL, "A$_w$ (feedback)\nvalue $\\to$ error\n(dense right)",
            "tab:blue")
# hold -> relay readout: host copy, not a synapse (bottom-right corner)
label_block(HP, RP, RP, N_TOTAL, "value $\\to$ relay\n(host copy,\nnot synaptic)",
            "0.4", ls="--")

ax3.set_title(f"forward A_wᵀ (eps→hold, dense bottom) + feedback A_w (relay→eps, "
              f"dense right) + hold→relay corner\n[{N_TOTAL}x{N_TOTAL}; hold/relay "
              f"= {N_VAL} neurons each, so the dense blocks are thin strips]", fontsize=9)

fig2.tight_layout(rect=[0, 0, 1, 0.95])
out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc_derot_weights.png")
fig2.savefig(out2, dpi=120); print(f"  wrote {out2}")
