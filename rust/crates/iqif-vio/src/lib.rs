//! Neuromorphic VIO: predictive-coding 6-DoF egomotion on the IQIF spiking
//! substrate, fed by the Rudolf-V visual frontend.
//!
//! Planned pipeline:
//!
//! ```text
//! Rudolf-V (KLT tracks + sparse stereo depth)
//!   -> per feature: normalized coord x, flow u, depth Z
//!   -> build linear system  U = G m   (G from motion-field geometry + 1/Z)
//!   -> QR whiten             A_w = Q,  g = R m
//!   -> IQIF predictive-coding relaxation (iqif-core)  ->  g
//!   -> decode                m = R^-1 g  ->  (v, omega)
//! ```
//!
//! This crate currently scaffolds the dependency wiring (git-pinned Rudolf-V +
//! path iqif-core) and the motion-field geometry. The PC relaxation and the
//! Rudolf-V frontend adapter land next.

// Wire the dependencies into the build graph. Replaced by real references as
// the adapter (Rudolf-V) and solver (iqif-core) are filled in.
use iqif_core as _;
use rudolf_v as _;

/// One feature's `2x6` motion-field block `M(x) = [ (1/Z) A(x) | B(x) ]`.
///
/// Rows are the `[u_x; u_y]` flow equations; columns are `[v1 v2 v3 w1 w2 w3]`.
/// `x, y` are *normalized* (calibrated, undistorted) image coordinates and `z`
/// is the metric depth of the feature (from sparse stereo).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motion_field_block_shape_and_known_entries() {
        let m = motion_field_rows(0.10, -0.20, 4.0);
        // Translation block scales by 1/Z; rotation block is depth-independent.
        assert!((m[0][0] - (-0.25)).abs() < 1e-12); // -1/Z
        assert!((m[1][5] - (-0.10)).abs() < 1e-12); // -x
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].len(), 6);
    }
}
