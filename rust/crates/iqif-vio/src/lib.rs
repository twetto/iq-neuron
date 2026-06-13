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

/// Solve the least-squares system `U = G m` with the 8-bit push-pull
/// predictive-coding relaxation (QR whitening + sigma-delta dither). Works for
/// any number of unknowns (`= G.ncols()`); returns the decoded `m`.
fn pc_relax(g_mat: &DMatrix<f64>, u: &DVector<f64>, g_init: &[f64]) -> DVector<f64> {
    let n_rows = g_mat.nrows();
    let n_val = g_mat.ncols();
    assert_eq!(g_init.len(), n_val);

    // QR whitening: A_w = Q (kappa = 1), solve for g = R m, decode m = R^-1 g.
    let qr = g_mat.clone().qr();
    let q = qr.q(); // 2N x n_val, orthonormal columns
    let r = qr.r(); // n_val x n_val upper-triangular
    let r_inv = r.try_inverse().expect("R from QR is singular");

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
    let (g_mat, u) = build_system(features);
    let m = pc_relax(&g_mat, &u, g_init);
    [m[0], m[1], m[2], m[3], m[4], m[5]]
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
}
