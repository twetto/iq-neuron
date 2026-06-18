//! Neuromorphic VIO: predictive-coding 6-DoF egomotion on the IQIF spiking
//! substrate, fed by the Rudolf-V visual frontend.
//!
//! Pipeline:
//!
//! ```text
//! Rudolf-V (KLT tracks + sparse stereo depth)
//!   -> per feature: normalized coord x, flow u, depth Z
//!   -> build linear system  U = G m   (G from motion-field geometry + 1/Z)
//!   -> QR whiten             A_w = Q,  g = R m
//!   -> IQIF predictive-coding relaxation (iqif-core)  ->  g
//!   -> decode                m = R^-1 g / s  ->  (v, omega)
//! ```
//!
//! The solver below is a transliteration of `tests/test_pc_egomotion_8bit.py`
//! (the 8-bit, push-pull, sigma-delta-dithered open-loop circuit) onto the Rust
//! `iqif-core` network. The Rudolf-V frontend adapter (turning KLT tracks +
//! sparse stereo into [`FeatureObs`]) lands next; the dependency is wired now.

use iqif_core::IqNetwork;
use nalgebra::{DMatrix, DVector};

// ── 8-bit circuit constants (mirror the Python reference) ───────────────────
const VMAX: i32 = 255;
const VMIN: i32 = 0;
const RESET: i32 = 0;
const QUANTUM: i32 = VMAX - RESET; // 255
const RATE_SCALE: f64 = 50.0; // error magnitude -> bias current
const S_SCALE: f64 = 2000.0; // shadow units per scaled-g unit
const N_STEPS: usize = 6000;
const N_VAL: usize = 6;

/// One tracked feature: normalized (calibrated, undistorted) image coordinate
/// `(x, y)`, metric depth `z` (from sparse stereo), and measured flow `(ux, uy)`.
#[derive(Clone, Copy, Debug)]
pub struct FeatureObs {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub ux: f64,
    pub uy: f64,
}

/// One feature's `2x6` motion-field block `M(x) = [ (1/Z) A(x) | B(x) ]`.
///
/// Rows are the `[u_x; u_y]` flow equations; columns are `[v1 v2 v3 w1 w2 w3]`.
///
/// ```text
/// A(x) = [ -1   0   x ]      B(x) = [  x y    -(1+x^2)   y ]
///        [  0  -1   y ]             [ 1+y^2    -x y     -x ]
/// ```
pub fn motion_field_rows(x: f64, y: f64, z: f64) -> [[f64; 6]; 2] {
    [
        [-1.0 / z, 0.0, x / z, x * y, -(1.0 + x * x), y],
        [0.0, -1.0 / z, y / z, 1.0 + y * y, -x * y, -x],
    ]
}

/// Stack the per-feature blocks into the linear system `U = G m`.
/// `G` is `2N x 6`, `U` is `2N`.
pub fn build_system(features: &[FeatureObs]) -> (DMatrix<f64>, DVector<f64>) {
    let n = features.len();
    let mut g = DMatrix::<f64>::zeros(2 * n, 6);
    let mut u = DVector::<f64>::zeros(2 * n);
    for (i, f) in features.iter().enumerate() {
        let blk = motion_field_rows(f.x, f.y, f.z);
        for c in 0..6 {
            g[(2 * i, c)] = blk[0][c];
            g[(2 * i + 1, c)] = blk[1][c];
        }
        u[2 * i] = f.ux;
        u[2 * i + 1] = f.uy;
    }
    (g, u)
}

/// Read the held estimate `g` from the value-neuron shadow potentials
/// (`shadow = membrane + spike_count*quantum + init_offset`), differencing the
/// push-pull pair.
fn read_g(
    net: &IqNetwork,
    cum_p: &[f64],
    cum_n: &[f64],
    off_p: &[f64],
    off_n: &[f64],
    vp: i32,
    vn: i32,
) -> DVector<f64> {
    let n_val = cum_p.len();
    let mut g = DVector::<f64>::zeros(n_val);
    for j in 0..n_val {
        let pp = net.potential(vp + j as i32) as f64;
        let pn = net.potential(vn + j as i32) as f64;
        let sp = pp + cum_p[j] * QUANTUM as f64 + off_p[j];
        let sn = pn + cum_n[j] * QUANTUM as f64 + off_n[j];
        g[j] = (sp - sn) / S_SCALE;
    }
    g
}

/// How the design matrix is whitened to condition number 1 before the spiking
/// relaxation. Both yield `(q, r_inv)` with `q = G·r_inv` orthonormal-columned,
/// so the relaxation and decode (`m = r_inv·g`) are identical downstream.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WhitenMode {
    /// Host-side QR factorization (the classical crutch): `r_inv = R⁻¹`.
    Qr,
    /// On-chip-style **whitening plasticity**: a local Hebbian/anti-Hebbian
    /// network (Földiák / Pehlevan–Chklovskii) that learns the symmetric whitener
    /// from a stream of feature rows, using only pre/post-synaptic activity — no
    /// QR/SVD, no global matrix algebra. This is the "learned-plasticity" framing
    /// of the design notes (on-chip anti-Hebbian whitening replacing host QR).
    Plastic,
}

/// Learn the symmetric whitener for `GᵀG` with a **strictly local** plasticity
/// rule (Földiák 1990 / Pehlevan–Chklovskii anti-Hebbian whitening network).
///
/// The value neurons carry lateral inhibitory weights `M` (symmetric). For each
/// presented feature row `x` (a row of `G`) the output settles by lateral
/// inhibition — pure neural dynamics, *not* a learning step:
///
/// ```text
/// y = (I + M)⁻¹ x
/// ```
///
/// then every lateral synapse relaxes toward decorrelation + unit variance using
/// only the time-averaged correlation of the two neurons it connects:
///
/// ```text
/// ΔM_ij = η ( ⟨y_i y_j⟩ − δ_ij )
/// ```
///
/// Off-diagonal `⟨y_i y_j⟩` is anti-Hebbian decorrelation; the diagonal
/// `⟨y_i²⟩ − 1` is per-neuron homeostatic gain control to unit variance. The
/// running correlation `⟨y_i y_j⟩` is a quantity a slow synapse accumulates from
/// its own two endpoints over the feature stream — so the update is strictly
/// local; using its averaged value (rather than noisy per-sample `y_i y_j`) is
/// just what synaptic low-pass filtering does, and it removes the instability of
/// the rank-1 online form. Whitening is *not* computed — it emerges as the fixed
/// point `⟨y yᵀ⟩ = I`, where `(I+M)⁻¹` whitens the empirical covariance
/// `C_emp = GᵀG/n`. The effective transform is then symmetric
/// `W_eff = (I+M)⁻¹ = C_emp^{-1/2}`, so the whitener for `GᵀG` is `W_eff/√n`
/// (returned). The `(I+M)⁻¹` solve stands in for the lateral-inhibition network
/// settling to equilibrium.
fn learn_whitener_plastic(g_mat: &DMatrix<f64>) -> DMatrix<f64> {
    let n = g_mat.nrows();
    let k = g_mat.ncols();
    let id = DMatrix::<f64>::identity(k, k);

    // Local input correlations ⟨x_i x_j⟩ (each entry reads only neurons i,j).
    let c_emp = (g_mat.transpose() * g_mat) / (n as f64);
    // Scale by the mean neuron variance ⟨x_i²⟩ (a local homeostatic set-point) so
    // the lateral dynamics are conditioned the same way at any data magnitude.
    let s = (0..k).map(|i| c_emp[(i, i)]).sum::<f64>() / k as f64;
    let c_hat = &c_emp / s.max(1e-30);

    // Anti-Hebbian whitening flow on the lateral weights M, driving ⟨y yᵀ⟩ → I.
    // The additive flow `M += η(⟨yyᵀ⟩ − I)` is only stable for η < √λ_min(Ĉ)
    // (the 1/σ_min ill-conditioning the design notes flag), so we pace η by
    // backtracking on the scalar whitening residual — a global "slow down"
    // signal, not part of any synapse's update.
    let mut m = DMatrix::<f64>::zeros(k, k); // lateral inhibitory weights (symmetric)
    let mut eta = 0.2;
    let mut prev_res = f64::INFINITY;
    let mut best_m = m.clone();
    let mut best_res = f64::INFINITY;
    let n_epochs = 20_000usize;
    for _ in 0..n_epochs {
        let p_inv = match (&id + &m).clone().try_inverse() {
            Some(p) => p,
            None => break,
        };
        let yy = &p_inv * &c_hat * &p_inv; // ⟨y yᵀ⟩
        let drive = &yy - &id; // ΔM_ij ∝ ⟨y_i y_j⟩ − δ_ij  (local)
        let res = drive.norm();
        if res < best_res {
            best_res = res;
            best_m = m.clone();
        }
        if res < 1e-12 {
            break;
        }
        if res > prev_res {
            eta *= 0.5; // overshooting — slow the flow
            if eta < 1e-9 {
                break;
            }
        }
        prev_res = res;
        m += drive * eta;
    }

    // W_eff = (I+M)⁻¹ whitens Ĉ = C_emp/s; undo the s-scaling and the /n in
    // C_emp so the result whitens GᵀG: W_for_GtG = (I+M)⁻¹ / √(n·s).
    let w_eff = (&id + &best_m)
        .try_inverse()
        .expect("(I + M) singular after whitening plasticity");
    w_eff / (n as f64 * s).sqrt()
}

/// Whitener for the relaxation: returns `(q, r_inv)` such that `q = G·r_inv` has
/// orthonormal columns and the decode is `m = r_inv·g`.
fn whitener(g_mat: &DMatrix<f64>, mode: WhitenMode) -> (DMatrix<f64>, DMatrix<f64>) {
    match mode {
        WhitenMode::Qr => {
            let qr = g_mat.clone().qr();
            let q = qr.q(); // 2N x n_val, orthonormal columns
            let r = qr.r(); // n_val x n_val upper-triangular
            let r_inv = r.try_inverse().expect("R from QR is singular");
            (q, r_inv)
        }
        WhitenMode::Plastic => {
            let w = learn_whitener_plastic(g_mat); // symmetric W = (GᵀG)^{-1/2}
            let q = g_mat * &w; // orthonormal columns since WᵀGᵀGW = WCW = I
            (q, w)
        }
    }
}

/// Solve the least-squares system `U = G m` with the 8-bit push-pull
/// predictive-coding relaxation (QR whitening + sigma-delta dither). Works for
/// any number of unknowns (`= G.ncols()`); returns the decoded `m`.
fn pc_relax(g_mat: &DMatrix<f64>, u: &DVector<f64>, g_init: &[f64]) -> DVector<f64> {
    pc_relax_mode(g_mat, u, g_init, WhitenMode::Qr)
}

/// As [`pc_relax`], with an explicit whitening mode (QR or learned plasticity).
fn pc_relax_mode(
    g_mat: &DMatrix<f64>,
    u: &DVector<f64>,
    g_init: &[f64],
    mode: WhitenMode,
) -> DVector<f64> {
    let n_rows = g_mat.nrows();
    let n_val = g_mat.ncols();
    assert_eq!(g_init.len(), n_val);

    // Whiten to kappa = 1: A_w = q (orthonormal cols), solve g, decode m = r_inv g.
    let (q, r_inv) = whitener(g_mat, mode);

    // Closed-form whitened solution (g_ls = Q^T U), used ONLY to set integer
    // scales — not the estimate itself (the relaxation produces that).
    let g_ls = q.transpose() * u;
    let problem_scale = 2.0 / g_ls.amax();
    let u_s = u * problem_scale;

    // Bound the forward weight scale so the worst-case per-step value drive
    // (all error neurons firing) cannot exceed one quantum -> membrane <= 255.
    let mut col_l1_max = 0.0_f64;
    for j in 0..n_val {
        let mut s = 0.0;
        for k in 0..n_rows {
            s += q[(k, j)].abs();
        }
        col_l1_max = col_l1_max.max(s);
    }
    let wb = ((QUANTUM as f64 / col_l1_max).floor() as i32).max(1);

    // Index layout: eps+ | eps- | val+ | val-
    let ep = 0i32;
    let en = n_rows as i32;
    let vp = 2 * n_rows as i32;
    let vn = vp + n_val as i32;
    let n_total = 2 * n_rows + 2 * n_val;

    // Parameter file: leak-free integrators (threshold=0, shift_b=15 -> f=0).
    let mut par = String::new();
    for i in 0..n_total {
        par.push_str(&format!("{i} 0 0 {RESET} 15 15 0\n"));
    }

    // Connection file: forward A_w^T (eps -> val), signed via push-pull routing.
    let mut con = String::new();
    let mut n_conn = 0;
    for k in 0..n_rows {
        for j in 0..n_val {
            let w = (q[(k, j)] * wb as f64).round() as i32;
            if w == 0 {
                continue;
            }
            con.push_str(&format!("{} {} {} 1\n", ep + k as i32, vp + j as i32, w));
            con.push_str(&format!("{} {} {} 1\n", en + k as i32, vp + j as i32, -w));
            con.push_str(&format!("{} {} {} 1\n", ep + k as i32, vn + j as i32, -w));
            con.push_str(&format!("{} {} {} 1\n", en + k as i32, vn + j as i32, w));
            n_conn += 4;
        }
    }
    if n_conn == 0 {
        con.push_str("0 0 0 1\n");
    }

    let mut net = IqNetwork::from_text(&par, &con);
    for i in 0..n_total as i32 {
        net.set_vmax(i, VMAX);
        net.set_vmin(i, VMIN);
    }
    // Value neurons: kill synaptic lingering so per-step drive = one step only.
    for j in 0..(2 * n_val) as i32 {
        net.set_surrogate_tau_one(vp + j, 1);
    }

    // Initialise the held estimate: split signed g_init across val+ / val-.
    let mut cum_p = vec![0.0_f64; n_val];
    let mut cum_n = vec![0.0_f64; n_val];
    let mut off_p = vec![0.0_f64; n_val];
    let mut off_n = vec![0.0_f64; n_val];
    for j in 0..n_val {
        let sp = g_init[j].max(0.0) * S_SCALE;
        let sn = (-g_init[j]).max(0.0) * S_SCALE;
        let rp = (sp as i32).min(VMAX - 1);
        let rn = (sn as i32).min(VMAX - 1);
        off_p[j] = sp - rp as f64;
        off_n[j] = sn - rn as f64;
        net.set_potential(vp + j as i32, rp);
        net.set_potential(vn + j as i32, rn);
    }

    // Sigma-delta carries for sub-integer bias precision.
    let mut res_p = vec![0.0_f64; n_rows];
    let mut res_n = vec![0.0_f64; n_rows];

    for _step in 0..N_STEPS {
        let g = read_g(&net, &cum_p, &cum_n, &off_p, &off_n, vp, vn);
        let eps = &u_s - &(&q * &g); // whitened-space error
        for k in 0..n_rows {
            let target = eps[k] * RATE_SCALE;
            let tp = target.max(0.0);
            let tn = (-target).max(0.0);
            let ap = tp + res_p[k];
            let bp = ap.round();
            res_p[k] = ap - bp;
            let an = tn + res_n[k];
            let bn = an.round();
            res_n[k] = an - bn;
            net.set_biascurrent(ep + k as i32, (bp as i32).clamp(0, VMAX));
            net.set_biascurrent(en + k as i32, (bn as i32).clamp(0, VMAX));
        }
        net.send_synapse();
        let counts = net.get_all_spike_counts();
        for j in 0..n_val {
            cum_p[j] += counts[(vp + j as i32) as usize] as f64;
            cum_n[j] += counts[(vn + j as i32) as usize] as f64;
        }
    }

    let g_final = read_g(&net, &cum_p, &cum_n, &off_p, &off_n, vp, vn);
    (&r_inv * &g_final) / problem_scale
}

/// Solve for 6-DoF motion `m = [v; omega]` from known-depth flow features.
/// `g_init` seeds the held estimate (the whitened-space starting point).
pub fn solve_egomotion(features: &[FeatureObs], g_init: &[f64; N_VAL]) -> [f64; 6] {
    solve_egomotion_mode(features, g_init, WhitenMode::Qr)
}

/// As [`solve_egomotion`], selecting the whitener: host QR or learned
/// [`WhitenMode::Plastic`] (on-chip-style anti-Hebbian whitening).
pub fn solve_egomotion_mode(
    features: &[FeatureObs],
    g_init: &[f64; N_VAL],
    mode: WhitenMode,
) -> [f64; 6] {
    let (g_mat, u) = build_system(features);
    let m = pc_relax_mode(&g_mat, &u, g_init, mode);
    [m[0], m[1], m[2], m[3], m[4], m[5]]
}

// ── Inhibition-dominated spike-coding-network solver (MKM 2020) ─────────────
// Egomotion as the instantaneous population readout of a tight-balance SCN:
// min ½‖U − A_w g‖² with g = D r (r = filtered spikes ≥ 0). Recurrent lateral
// inhibition Ω = ΦᵀΦ (Φ = A_w D); a spike "bounces" the readout back toward the
// optimum. To survive the chip's SYNCHRONOUS one-tick-delayed update we make the
// recurrent strictly inhibition-dominated: every excitatory lateral weight is
// CLIPPED to zero, so a delayed spike can only inhibit, never trigger a
// follow-up (the MKM Gᵢⱼ≥0 delay-robustness condition). Decoder is a generic
// overcomplete random frame (no exactly-anti-parallel pairs). Validated on the
// IQIF chip in tests/test_pc_egomotion_scn_chip_inhib.py.

const SCN_N: usize = 72; // decoder neurons (overcomplete frame over 6-D g)
const SCN_STEPS: usize = 15_000; // relaxation ticks per frame
const SCN_LAM: f64 = 2.0; // drive / readout-filter rate
const SCN_DT: f64 = 1e-3;
const SCN_REST: i32 = 127; // QIF stable rest (V=0 operating point)
const SCN_SHIFT: i32 = 6; // membrane leak toward rest
                          // Noise 0 => deterministic per-frame readout. The all-inhibitory (clipped)
                          // recurrent + heterogeneous per-neuron bias desynchronize the population on
                          // their own, so no injected jitter is needed; removing it both removes the
                          // frame-to-frame readout noise AND improves accuracy (0.97° vs 2.87° vs LS).
const SCN_NOISE: i32 = 0;
// Exp-decay time constant on the LATERAL inhibition (post-synaptic surrogate
// tau, MUST be a power of two — the decay is a bit-shift). Smoothing the
// inhibitory input regularizes the firing (ISI CV 0.68 -> 0.14) and sharply
// cuts the per-frame error. Lateral weights are rescaled by 1/tau so the
// effective (time-integrated) inhibition strength is unchanged. The 3-DoF
// de-rotated solve needs more smoothing than the well-spanned 6-DoF one
// (A/B on EuRoC V1_01: 6-DoF best at 4, IMU 3-DoF best at 8 where it matches
// the host PC baseline). `SCN_TAU` env var overrides both for tuning.
const SCN_TAU: i32 = 4; // 6-DoF default
const SCN_TAU_DEROT: i32 = 8; // 3-DoF gyro-de-rotated default

/// Deterministic xorshift so the random decoder frame is reproducible.
fn scn_unit(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    ((x >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Core SCN relaxation: solve the least-squares `U = G m` (any number of
/// unknowns `= G.ncols()`) as the instantaneous population readout of an
/// inhibition-dominated tight-balance spiking network on the IQIF chip. Returns
/// the decoded `m`.
fn scn_core(g_mat: &DMatrix<f64>, u: &DVector<f64>, tau: i32) -> DVector<f64> {
    let n_val = g_mat.ncols();
    let (q, r_inv) = whitener(g_mat, WhitenMode::Qr); // q = A_w, decode m = r_inv·g

    let g_ls = q.transpose() * u; // whitened LS (scale reference only)
    let g_norm = g_ls.norm().max(1e-9);
    let d_scale = 0.02 * g_norm;

    // Overcomplete random decoder frame: unit columns in n_val-D g-space.
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut d_dirs = DMatrix::<f64>::zeros(n_val, SCN_N);
    for j in 0..SCN_N {
        let mut nrm = 0.0;
        for i in 0..n_val {
            let v = scn_unit(&mut rng);
            d_dirs[(i, j)] = v;
            nrm += v * v;
        }
        let nrm = nrm.sqrt().max(1e-12);
        for i in 0..n_val {
            d_dirs[(i, j)] /= nrm;
        }
    }
    let d_mat = &d_dirs * d_scale; // n_val x N decoder
    let phi = &q * &d_mat; // 2N x N measurement-space dictionary
    let fu = phi.transpose() * u; // ΦᵀU
    let cos = d_dirs.transpose() * &d_dirs; // pairwise cosines (= Ω / d_scale²)

    let t_thr = 0.5 * d_scale * d_scale; // uniform SCN threshold
    let s = VMAX as f64 / (2.0 * t_thr); // voltage -> membrane counts
    let drive = s * SCN_DT * SCN_LAM;

    // Parameters: QIF with stable rest = REST, fire at VMAX, small leak + noise.
    let mut par = String::new();
    for i in 0..SCN_N {
        par.push_str(&format!(
            "{i} {SCN_REST} {VMAX} {RESET} {SCN_SHIFT} {SCN_SHIFT} {SCN_NOISE}\n"
        ));
    }
    // Connections: lateral inhibition, CLIPPED to ≤0 (all-inhibitory) so a
    // delayed spike can only inhibit, never trigger a follow-up (delay-robust).
    let mut con = String::new();
    let mut n_conn = 0;
    for j in 0..SCN_N {
        for i in 0..SCN_N {
            if i == j {
                continue;
            }
            // all-inhibitory clip, rescaled by 1/tau (exp-decay synapse
            // integrates ~tau steps, so this keeps effective inhibition fixed).
            let w0 = (-(VMAX as f64 * cos[(i, j)]).round()).min(0.0);
            let w = (w0 / tau as f64).round() as i32;
            if w != 0 {
                con.push_str(&format!("{j} {i} {w} 1\n"));
                n_conn += 1;
            }
        }
    }
    if n_conn == 0 {
        con.push_str("0 0 0 1\n");
    }

    let mut net = IqNetwork::from_text(&par, &con);
    for i in 0..SCN_N as i32 {
        net.set_vmax(i, VMAX);
        net.set_vmin(i, VMIN);
        net.set_surrogate_tau_one(i, tau); // exp-decay lateral inhibition
        let bias = (drive * fu[i as usize]).round() as i32;
        net.set_biascurrent(i, bias);
        net.set_potential(i, SCN_REST);
    }

    // Run: integrate spikes into a leaky readout r; average g = D·r over the tail.
    let mut r = DVector::<f64>::zeros(SCN_N);
    let mut r_acc = DVector::<f64>::zeros(SCN_N);
    let tail = SCN_STEPS / 2;
    for step in 0..SCN_STEPS {
        net.send_synapse();
        let counts = net.get_all_spike_counts();
        for i in 0..SCN_N {
            r[i] += counts[i] as f64 - SCN_LAM * SCN_DT * r[i];
        }
        if step >= tail {
            r_acc += &r;
        }
    }
    let r_mean = r_acc / (SCN_STEPS - tail) as f64;
    let g_read = &d_mat * &r_mean; // readout in whitened g-space
    &r_inv * &g_read // decode to motion
}

/// Solve 6-DoF egomotion with the inhibition-dominated SCN on the IQIF chip.
/// Drop-in alternative to [`solve_egomotion`]: same `U = G m` least-squares, but
/// the answer is the instantaneous population readout of an all-inhibitory
/// spiking network rather than the labeled-line relaxation.
/// Lateral exp-decay tau, `SCN_TAU` env override (for A/B tuning) else `default`.
fn scn_tau(default: i32) -> i32 {
    std::env::var("SCN_TAU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

pub fn solve_egomotion_scn(features: &[FeatureObs]) -> [f64; 6] {
    let (g_mat, u) = build_system(features);
    let m = scn_core(&g_mat, &u, scn_tau(SCN_TAU));
    [m[0], m[1], m[2], m[3], m[4], m[5]]
}

/// IMU de-rotation with the SCN: given a KNOWN camera-frame angular velocity
/// `omega` (e.g. from the gyro), subtract the rotational flow `B(x) omega` and
/// solve only the 3-DoF translation `v` as the inhibition-dominated SCN
/// population readout. With rotation removed the system is well-conditioned
/// (no v↔ω ambiguity), so the SCN's lateral v1/v2 become observable — the
/// natural first on-chip SCN target.
pub fn solve_translation_known_rotation_scn(features: &[FeatureObs], omega: &[f64; 3]) -> [f64; 3] {
    let (g6, u) = build_system(features);
    let g_v = g6.columns(0, 3).into_owned(); // (1/Z) A : translation block
    let g_w = g6.columns(3, 3).into_owned(); // B : rotation block
    let u_res = &u - &(g_w * DVector::from_row_slice(omega)); // de-rotated flow
                                                              // tau=1 here: the 3-DoF de-rotated solve has a weakly-constrained forward
                                                              // axis that exp-decay smoothing destabilizes (v3 blows up), so no lingering.
    let v = scn_core(&g_v, &u_res, scn_tau(SCN_TAU_DEROT));
    [v[0], v[1], v[2]]
}

/// De-rotation solve: given a KNOWN camera-frame angular velocity `omega` (e.g.
/// from a gyro), subtract the rotational flow `B(x) omega` and solve only for
/// the 3-DoF translation `v`. With rotation removed, the residual flow is pure
/// `(1/Z) A v`, so lateral v1/v2 become observable instead of collapsing into
/// the translation-rotation ambiguity.
pub fn solve_translation_known_rotation(
    features: &[FeatureObs],
    omega: &[f64; 3],
    g_init: &[f64; 3],
) -> [f64; 3] {
    let (g6, u) = build_system(features);
    let g_v = g6.columns(0, 3).into_owned(); // (1/Z) A : translation block
    let g_w = g6.columns(3, 3).into_owned(); // B : rotation block
    let u_res = &u - &(g_w * DVector::from_row_slice(omega)); // de-rotated flow
    let v = pc_relax(&g_v, &u_res, g_init);
    [v[0], v[1], v[2]]
}

/// De-rotation solve with a temporal (Tikhonov) prior toward `prev_v`, turning
/// the independent per-frame solve into a light recursive filter (the filtering
/// lean of the design notes §7). The equilibrium is the MAP estimate
/// `v* = (Gᵀ G + λ I)⁻¹ (Gᵀ u_res + λ prev_v)`, realised by stacking
/// `√λ · I₃` rows onto the system before QR whitening — so the same 8-bit PC
/// circuit solves it, no internals changed.
///
/// `alpha` is a dimensionless smoothing strength: `0` reduces to the plain
/// per-frame solve; larger trades lag for less noise (mainly on the weak forward
/// channel v3). `λ = alpha · mean(diag(Gᵀ G))`, so it tracks the data curvature
/// and is invariant to feature count / depth scale.
pub fn solve_translation_known_rotation_filtered(
    features: &[FeatureObs],
    omega: &[f64; 3],
    prev_v: &[f64; 3],
    alpha: f64,
) -> [f64; 3] {
    let (g6, u) = build_system(features);
    let g_v = g6.columns(0, 3).into_owned();
    let g_w = g6.columns(3, 3).into_owned();
    let u_res = &u - &(g_w * DVector::from_row_slice(omega));

    if alpha <= 0.0 {
        let v = pc_relax(&g_v, &u_res, prev_v);
        return [v[0], v[1], v[2]];
    }

    // λ relative to the data curvature so `alpha` is scale-free.
    let gtg = g_v.transpose() * &g_v;
    let mean_diag = (gtg[(0, 0)] + gtg[(1, 1)] + gtg[(2, 2)]) / 3.0;
    let sl = (alpha * mean_diag).sqrt();

    let n = g_v.nrows();
    let mut g_aug = DMatrix::zeros(n + 3, 3);
    g_aug.rows_mut(0, n).copy_from(&g_v);
    let mut u_aug = DVector::zeros(n + 3);
    u_aug.rows_mut(0, n).copy_from(&u_res);
    for j in 0..3 {
        g_aug[(n + j, j)] = sl;
        u_aug[n + j] = sl * prev_v[j];
    }

    let v = pc_relax(&g_aug, &u_aug, prev_v);
    [v[0], v[1], v[2]]
}

// ── Closed-loop predictive-coding relaxation (G as on-chip feedback) ────────
// The open-loop `pc_relax` computes the prediction `A_w g` on the HOST each step
// and injects it as the error bias. The closed-loop circuit closes the loop on
// chip: a second weight matrix carries `A_w` as FEEDBACK (relay -> eps), so the
// error neurons compute `eps = U - A_w g` themselves. Both mat-vecs (`A_w g` and
// `A_wᵀ eps`) now live in the synaptic fabric; the host does no matmul. Three
// populations (8-bit, leak-free): eps+/- (tonic error), hold+/- (integrate the
// held estimate g in their shadow), relay+/- (tonic readout of g, whose spikes
// carry the prediction through the feedback weights). The hold->relay readout is
// a local per-neuron bias copy, not a synapse. Transliteration of
// tests/test_pc_egomotion_closedloop.py.

const CL_STEPS: usize = 6000;
const CL_RS: f64 = 50.0; // error & relay rate scale (bias units per scaled unit)

/// Closed-loop relaxation of the least-squares `U = G m`. Returns decoded `m`.
fn cl_relax(g_mat: &DMatrix<f64>, u: &DVector<f64>, g_init: &[f64]) -> DVector<f64> {
    let n_rows = g_mat.nrows();
    let n_val = g_mat.ncols();
    assert_eq!(g_init.len(), n_val);

    let (q, r_inv) = whitener(g_mat, WhitenMode::Qr);
    let g_ls = q.transpose() * u;
    let problem_scale = 2.0 / g_ls.amax();
    let u_s = u * problem_scale;

    // Forward weight scale bounded so per-step hold drive <= one quantum.
    let mut col_l1_max = 0.0_f64;
    for j in 0..n_val {
        let mut s = 0.0;
        for k in 0..n_rows {
            s += q[(k, j)].abs();
        }
        col_l1_max = col_l1_max.max(s);
    }
    let wb = ((QUANTUM as f64 / col_l1_max).floor() as i32).max(1);
    let wf = QUANTUM; // feedback weight scale: (wf * RS / QUANTUM) = RS -> delivers -p*RS

    // Index layout: eps+ | eps- | hold+ | hold- | relay+ | relay-
    let ep = 0i32;
    let en = n_rows as i32;
    let hp = 2 * n_rows as i32;
    let hn = hp + n_val as i32;
    let rp = hn + n_val as i32;
    let rn = rp + n_val as i32;
    let n_total = 2 * n_rows + 4 * n_val;

    // Leak-free integrators (threshold=0, shift_b=15 -> f=0).
    let mut par = String::new();
    for i in 0..n_total {
        par.push_str(&format!("{i} 0 0 {RESET} 15 15 0\n"));
    }

    let mut con = String::new();
    let mut n_conn = 0;
    for k in 0..n_rows {
        for j in 0..n_val {
            let wbf = (q[(k, j)] * wb as f64).round() as i32;
            let wff = (q[(k, j)] * wf as f64).round() as i32;
            if wbf != 0 {
                // forward A_wᵀ : eps -> hold (gradient path), push-pull.
                con.push_str(&format!("{} {} {} 1\n", ep + k as i32, hp + j as i32, wbf));
                con.push_str(&format!("{} {} {} 1\n", en + k as i32, hp + j as i32, -wbf));
                con.push_str(&format!("{} {} {} 1\n", ep + k as i32, hn + j as i32, -wbf));
                con.push_str(&format!("{} {} {} 1\n", en + k as i32, hn + j as i32, wbf));
                n_conn += 4;
            }
            if wff != 0 {
                // feedback A_w : relay -> eps (prediction path, delivers -p).
                con.push_str(&format!("{} {} {} 1\n", rp + j as i32, ep + k as i32, -wff));
                con.push_str(&format!("{} {} {} 1\n", rn + j as i32, ep + k as i32, wff));
                con.push_str(&format!("{} {} {} 1\n", rp + j as i32, en + k as i32, wff));
                con.push_str(&format!("{} {} {} 1\n", rn + j as i32, en + k as i32, -wff));
                n_conn += 4;
            }
        }
    }
    if n_conn == 0 {
        con.push_str("0 0 0 1\n");
    }

    let mut net = IqNetwork::from_text(&par, &con);
    for i in 0..n_total as i32 {
        net.set_vmax(i, VMAX);
        net.set_vmin(i, VMIN);
    }
    // eps + hold receive synapses: kill lingering so per-step drive = one step.
    for i in 0..(2 * n_rows + 2 * n_val) as i32 {
        net.set_surrogate_tau_one(i, 1);
    }

    // Data U enters as a constant signed bias on the error neurons.
    for k in 0..n_rows {
        net.set_biascurrent(ep + k as i32, (u_s[k] * CL_RS).round() as i32);
        net.set_biascurrent(en + k as i32, (-u_s[k] * CL_RS).round() as i32);
    }

    // Initialise the held estimate: split signed g_init across hold+ / hold-.
    let mut cum_p = vec![0.0_f64; n_val];
    let mut cum_n = vec![0.0_f64; n_val];
    let mut off_p = vec![0.0_f64; n_val];
    let mut off_n = vec![0.0_f64; n_val];
    for j in 0..n_val {
        let sp = g_init[j].max(0.0) * S_SCALE;
        let sn = (-g_init[j]).max(0.0) * S_SCALE;
        let rip = (sp as i32).min(VMAX - 1);
        let rin = (sn as i32).min(VMAX - 1);
        off_p[j] = sp - rip as f64;
        off_n[j] = sn - rin as f64;
        net.set_potential(hp + j as i32, rip);
        net.set_potential(hn + j as i32, rin);
    }

    for _step in 0..CL_STEPS {
        let g = read_g(&net, &cum_p, &cum_n, &off_p, &off_n, hp, hn);
        // Relay tonic readout: fire ~ held value (local per-neuron copy).
        for j in 0..n_val {
            let bp = (g[j] * CL_RS).round().clamp(0.0, VMAX as f64) as i32;
            let bn = (-g[j] * CL_RS).round().clamp(0.0, VMAX as f64) as i32;
            net.set_biascurrent(rp + j as i32, bp);
            net.set_biascurrent(rn + j as i32, bn);
        }
        net.send_synapse();
        let counts = net.get_all_spike_counts();
        for j in 0..n_val {
            cum_p[j] += counts[(hp + j as i32) as usize] as f64;
            cum_n[j] += counts[(hn + j as i32) as usize] as f64;
        }
    }

    let g_final = read_g(&net, &cum_p, &cum_n, &off_p, &off_n, hp, hn);
    (&r_inv * &g_final) / problem_scale
}

/// De-rotation solve with the CLOSED-LOOP circuit: given a known camera-frame
/// `omega`, subtract the rotational flow `B(x) omega` and solve the 3-DoF
/// translation `v` with `G` realised as on-chip feedback weights (no host
/// matmul). `g_init` seeds the held estimate. Drop-in for
/// [`solve_translation_known_rotation`].
pub fn solve_translation_known_rotation_closed_loop(
    features: &[FeatureObs],
    omega: &[f64; 3],
    g_init: &[f64; 3],
) -> [f64; 3] {
    let (g6, u) = build_system(features);
    let g_v = g6.columns(0, 3).into_owned();
    let g_w = g6.columns(3, 3).into_owned();
    let u_res = &u - &(g_w * DVector::from_row_slice(omega));
    let v = cl_relax(&g_v, &u_res, g_init);
    [v[0], v[1], v[2]]
}

/// Adapter: turn Rudolf-V frontend tracks + sparse stereo depth into the
/// [`FeatureObs`] the solver consumes. Holds the previous frame's normalized
/// feature positions (keyed by persistent track id) so it can form per-feature
/// optical flow.
pub mod frontend_adapter {
    use super::FeatureObs;
    use rudolf_v::camera::CameraIntrinsics;
    use rudolf_v::fast::Feature;
    use rudolf_v::stereo::StereoMatch;
    use std::collections::HashMap;

    #[derive(Default)]
    pub struct FlowDepthAdapter {
        prev_norm: HashMap<u64, (f64, f64)>,
    }

    impl FlowDepthAdapter {
        pub fn new() -> Self {
            Self {
                prev_norm: HashMap::new(),
            }
        }

        /// Assemble observations for the current frame. `features` and `matches`
        /// are index-aligned (as returned by `StereoMatcher::match_features`).
        /// `dt` is the inter-frame interval in seconds, so the recovered motion
        /// comes out in per-second units. A [`FeatureObs`] is emitted only for
        /// tracks with BOTH a valid stereo depth now AND a normalized position
        /// from the previous frame (needed to form the flow vector).
        pub fn observe(
            &mut self,
            features: &[Feature],
            matches: &[StereoMatch],
            cam: &CameraIntrinsics,
            dt: f64,
        ) -> Vec<FeatureObs> {
            let mut obs = Vec::new();
            let mut curr_norm = HashMap::with_capacity(features.len());
            for (f, m) in features.iter().zip(matches.iter()) {
                let (xn, yn) = cam.normalize_undistorted(f.x as f64, f.y as f64);
                curr_norm.insert(f.id, (xn, yn));
                if !m.matched || m.inv_depth <= 0.0 || dt <= 0.0 {
                    continue;
                }
                let z = 1.0 / m.inv_depth as f64;
                if let Some(&(xp, yp)) = self.prev_norm.get(&f.id) {
                    obs.push(FeatureObs {
                        x: xn,
                        y: yn,
                        z,
                        ux: (xn - xp) / dt,
                        uy: (yn - yp) / dt,
                    });
                }
            }
            self.prev_norm = curr_norm;
            obs
        }

        /// Forget all tracked positions (e.g. on a tracking reset).
        pub fn reset(&mut self) {
            self.prev_norm.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic LCG (for reproducible synthetic scenes/inits).
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed)
        }
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f64) / ((1u64 << 31) as f64) // [0, 1)
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }
    }

    fn v_dir_error_deg(a: &[f64; 6], b: &[f64; 6]) -> f64 {
        let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
        (dot / (na * nb + 1e-12))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    }

    fn w_error(a: &[f64; 6], b: &[f64; 6]) -> f64 {
        ((a[3] - b[3]).powi(2) + (a[4] - b[4]).powi(2) + (a[5] - b[5]).powi(2)).sqrt()
    }

    #[test]
    fn motion_field_block_shape_and_known_entries() {
        let m = motion_field_rows(0.10, -0.20, 4.0);
        assert!((m[0][0] - (-0.25)).abs() < 1e-12); // -1/Z
        assert!((m[1][5] - (-0.10)).abs() < 1e-12); // -x
    }

    #[test]
    fn solves_known_depth_egomotion_to_ground_truth() {
        // Synthetic scene mirroring tests/test_pc_egomotion_8bit.py.
        let m_gt = [0.30, 0.05, 0.12, 0.02, -0.05, 0.015];
        let mut rng = Lcg::new(7);
        let n = 40;
        let mut feats = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.range(-0.70, 0.70);
            let y = rng.range(-0.50, 0.50);
            let z = rng.range(2.0, 10.0);
            let blk = motion_field_rows(x, y, z);
            let ux: f64 = (0..6).map(|c| blk[0][c] * m_gt[c]).sum();
            let uy: f64 = (0..6).map(|c| blk[1][c] * m_gt[c]).sum();
            feats.push(FeatureObs { x, y, z, ux, uy });
        }

        // Off-axis init so both eps+/eps- and val+/val- populations engage.
        let mut rng2 = Lcg::new(3);
        let mut g_init = [0.0_f64; 6];
        for v in g_init.iter_mut() {
            *v = rng2.range(-1.6, 1.6);
        }

        let m_est = solve_egomotion(&feats, &g_init);

        let v_ang = v_dir_error_deg(&m_est, &m_gt);
        let w_err = w_error(&m_est, &m_gt);
        println!("v_dir_err = {v_ang:.4} deg, w_err = {w_err:.5}, m = {m_est:?}");
        assert!(
            v_ang < 3.0,
            "translation direction error too large: {v_ang} deg"
        );
        assert!(w_err < 0.03, "angular velocity error too large: {w_err}");
    }

    /// The learned whitener whitens to kappa=1 and equals the QR whitener in the
    /// sense that matters: `r_inv·r_invᵀ = (GᵀG)⁻¹` (the decode is identical LS).
    #[test]
    fn plastic_whitening_conditions_to_identity() {
        let mut rng = Lcg::new(11);
        let n = 30;
        let mut g = DMatrix::<f64>::zeros(2 * n, 6);
        for i in 0..n {
            let x = rng.range(-0.7, 0.7);
            let y = rng.range(-0.5, 0.5);
            let z = rng.range(2.0, 10.0);
            let blk = motion_field_rows(x, y, z);
            for c in 0..6 {
                g[(2 * i, c)] = blk[0][c];
                g[(2 * i + 1, c)] = blk[1][c];
            }
        }

        let w = learn_whitener_plastic(&g);
        let c = g.transpose() * &g;
        println!("|W|max = {:.3e}, C cond ~ {:.2e}", w.amax(), {
            let ev = c.clone().symmetric_eigenvalues();
            ev.amax() / ev.amin()
        });

        // WCW = I  (columns of G·W are orthonormal -> kappa = 1).
        let wcw = &w * &c * &w;
        let mut max_off = 0.0_f64;
        for i in 0..6 {
            for j in 0..6 {
                let t = if i == j { 1.0 } else { 0.0 };
                let d = (wcw[(i, j)] - t).abs();
                if d.is_nan() || d > max_off {
                    max_off = d;
                }
            }
        }
        println!("max|WCW - I| = {max_off:.3e}");
        // Local plasticity converges approximately (not to machine precision):
        // resulting condition number 1+max_off should be close to 1.
        assert!(
            max_off < 0.05,
            "plastic whitener not conditioning to ~kappa=1: {max_off}"
        );

        // Decode equivalence: r_inv·r_invᵀ should track (GᵀG)⁻¹ (relative).
        let recon = &w * w.transpose();
        let c_inv = c.try_inverse().unwrap();
        let err = (&recon - &c_inv).amax() / c_inv.amax();
        println!("rel max|WWᵀ - C⁻¹| = {err:.3e}");
        assert!(err < 0.05, "plastic decode drifts from LS decode: {err}");
    }

    /// End-to-end: the plastic-whitened spiking solve hits the same accuracy
    /// bounds as the QR-whitened one on the synthetic scene.
    #[test]
    fn plastic_whitening_solves_egomotion() {
        let m_gt = [0.30, 0.05, 0.12, 0.02, -0.05, 0.015];
        let mut rng = Lcg::new(7);
        let n = 40;
        let mut feats = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.range(-0.70, 0.70);
            let y = rng.range(-0.50, 0.50);
            let z = rng.range(2.0, 10.0);
            let blk = motion_field_rows(x, y, z);
            let ux: f64 = (0..6).map(|c| blk[0][c] * m_gt[c]).sum();
            let uy: f64 = (0..6).map(|c| blk[1][c] * m_gt[c]).sum();
            feats.push(FeatureObs { x, y, z, ux, uy });
        }
        let mut rng2 = Lcg::new(3);
        let mut g_init = [0.0_f64; 6];
        for v in g_init.iter_mut() {
            *v = rng2.range(-1.6, 1.6);
        }

        let m_est = solve_egomotion_mode(&feats, &g_init, WhitenMode::Plastic);
        let v_ang = v_dir_error_deg(&m_est, &m_gt);
        let w_err = w_error(&m_est, &m_gt);
        println!("[plastic] v_dir_err = {v_ang:.4} deg, w_err = {w_err:.5}, m = {m_est:?}");
        assert!(
            v_ang < 3.0,
            "translation direction error too large: {v_ang} deg"
        );
        assert!(w_err < 0.03, "angular velocity error too large: {w_err}");
    }

    /// Closed-loop de-rotation translation solve: with the rotation known, the
    /// on-chip-feedback circuit recovers the translation direction.
    #[test]
    fn closed_loop_translation_solves_known_rotation() {
        let m_gt = [0.30, 0.05, 0.12, 0.02, -0.05, 0.015];
        let omega = [m_gt[3], m_gt[4], m_gt[5]];
        let mut rng = Lcg::new(7);
        let n = 40;
        let mut feats = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.range(-0.70, 0.70);
            let y = rng.range(-0.50, 0.50);
            let z = rng.range(2.0, 10.0);
            let blk = motion_field_rows(x, y, z);
            let ux: f64 = (0..6).map(|c| blk[0][c] * m_gt[c]).sum();
            let uy: f64 = (0..6).map(|c| blk[1][c] * m_gt[c]).sum();
            feats.push(FeatureObs { x, y, z, ux, uy });
        }

        let v = solve_translation_known_rotation_closed_loop(&feats, &omega, &[0.0; 3]);
        let v6 = [v[0], v[1], v[2], 0.0, 0.0, 0.0];
        let v_ang = v_dir_error_deg(&v6, &m_gt);
        println!("[closed-loop] v_dir_err = {v_ang:.4} deg, v = {v:?}");
        assert!(v_ang < 5.0, "closed-loop translation direction error: {v_ang} deg");
    }
}
