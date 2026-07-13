//! TUM-VI stereo egomotion via predictive coding (headless + CSV log).
//!
//! Same pipeline as `euroc_egomotion`, adapted to the TUM-VI dataset:
//!   * Fisheye (equidistant) cameras, calibrated from the Kalibr
//!     `dso/camchain.yaml`. Rudolf-V's geometry-first `pixel_to_bearing` handles
//!     the wide-FOV unprojection; the frontend adapter drops each bearing to
//!     pinhole-normalized `(x/z, y/z)` for the motion-field solver.
//!   * Ground truth from `mav0/mocap0/data.csv`, which stores pose only, so the
//!     world-frame velocity is derived by central-differencing the mocap
//!     positions.
//!
//! No IMU is used: the 6-DoF egomotion is solved purely from stereo optical flow
//! + sparse stereo depth (`solve_egomotion`).
//!
//! Usage:
//!     cargo run -p iqif-vio --example tumvi_egomotion --release -- /path/to/dataset-room1_512_16 [num_frames] [out.csv]

use iqif_vio::frontend_adapter::FlowDepthAdapter;
use iqif_vio::{
    solve_egomotion, solve_egomotion_bearing, solve_translation_known_rotation_bearing,
};

use rudolf_v::camera::{CameraIntrinsics, StereoRig};
use rudolf_v::frontend::{DetectorType, Frontend, FrontendConfig, LbpPolicy};
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image;
use rudolf_v::klt::LkMethod;
use rudolf_v::stereo::{StereoConfig, StereoMatcher};

use nalgebra::{Matrix3, Vector3};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn load_grayscale(path: &Path) -> Image<u8> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()))
        .into_luma8();
    Image::from_vec(img.width() as usize, img.height() as usize, img.into_raw())
}

fn list_pngs(cam_dir: &Path) -> Vec<PathBuf> {
    let data_dir = cam_dir.join("data");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&data_dir)
        .unwrap_or_else(|_| panic!("Expected data dir at {}", data_dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e == "png"))
        .collect();
    files.sort();
    files
}

/// TUM-VI (like EuRoC) image filenames are nanosecond timestamps.
fn ts_ns(path: &Path) -> u64 {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Load TUM-VI mocap ground truth and derive world-frame velocity.
///
/// `mocap0/data.csv` columns: `timestamp_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z`
/// (pose only). Velocity is estimated per sample by central-differencing the
/// positions, so the returned rows match the EuRoC layout
/// (timestamp_ns, world-velocity [vx,vy,vz], orientation quaternion
/// [qw,qx,qy,qz] = R_world_body). Empty if the file is absent.
fn load_mocap_gt(path: &Path) -> Vec<(u64, [f64; 3], [f64; 4])> {
    let Ok(txt) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut poses: Vec<(u64, [f64; 3], [f64; 4])> = Vec::new();
    for line in txt.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 8 {
            continue;
        }
        let p = |i: usize| c[i].trim().parse::<f64>().ok();
        if let (Ok(ts), Some(px), Some(py), Some(pz), Some(qw), Some(qx), Some(qy), Some(qz)) = (
            c[0].trim().parse::<u64>(),
            p(1),
            p(2),
            p(3),
            p(4),
            p(5),
            p(6),
            p(7),
        ) {
            poses.push((ts, [px, py, pz], [qw, qx, qy, qz]));
        }
    }

    // Central-difference velocity: v[i] = (p[i+1] - p[i-1]) / (t[i+1] - t[i-1]).
    let n = poses.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lo = i.saturating_sub(1);
        let hi = (i + 1).min(n.saturating_sub(1));
        let dt = (poses[hi].0 as f64 - poses[lo].0 as f64) * 1e-9;
        let v = if dt > 0.0 {
            let (p0, p1) = (poses[lo].1, poses[hi].1);
            [
                (p1[0] - p0[0]) / dt,
                (p1[1] - p0[1]) / dt,
                (p1[2] - p0[2]) / dt,
            ]
        } else {
            [0.0; 3]
        };
        out.push((poses[i].0, v, poses[i].2));
    }
    out
}

/// Index of the nearest-timestamp ground-truth row (assumes sorted).
fn nearest_gt(gt: &[(u64, [f64; 3], [f64; 4])], ts: u64) -> Option<usize> {
    if gt.is_empty() {
        return None;
    }
    let idx = gt.partition_point(|&(t, _, _)| t < ts);
    let mut best = idx.min(gt.len() - 1);
    if idx > 0 {
        let d_prev = (ts as i128 - gt[idx - 1].0 as i128).abs();
        let d_best = (gt[best].0 as i128 - ts as i128).abs();
        if d_prev < d_best {
            best = idx - 1;
        }
    }
    Some(best)
}

/// Camera<-body(imu) rotation from the Kalibr camchain `T_cam_imu` 4x4 of `cam`.
fn parse_r_cam_imu(camchain: &Path, cam: &str) -> Matrix3<f64> {
    let txt = std::fs::read_to_string(camchain).expect("read camchain.yaml");
    let sect = &txt[txt
        .find(&format!("{cam}:"))
        .expect("cam section in camchain")..];
    let after =
        &sect[sect.find("T_cam_imu:").expect("T_cam_imu in camchain") + "T_cam_imu:".len()..];
    let mut n: Vec<f64> = Vec::new();
    for line in after.lines() {
        let l = line.trim();
        if !(l.starts_with('-') && l.contains('[')) {
            if !n.is_empty() {
                break;
            }
            continue;
        }
        for tok in l
            .trim_start_matches('-')
            .trim()
            .trim_matches(|c| c == '[' || c == ']')
            .split(',')
        {
            if let Ok(v) = tok.trim().parse::<f64>() {
                n.push(v);
            }
        }
        if n.len() >= 16 {
            break;
        }
    }
    Matrix3::new(n[0], n[1], n[2], n[4], n[5], n[6], n[8], n[9], n[10])
}

/// TUM-VI `imu0/data.csv`: (ts_ns, gyro [wx,wy,wz] rad/s in the IMU/body frame).
fn load_imu(path: &Path) -> Vec<(u64, [f64; 3])> {
    let mut out = Vec::new();
    let Ok(txt) = std::fs::read_to_string(path) else {
        return out;
    };
    for line in txt.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 4 {
            continue;
        }
        let p = |i: usize| c[i].trim().parse::<f64>().ok();
        if let (Ok(ts), Some(wx), Some(wy), Some(wz)) =
            (c[0].trim().parse::<u64>(), p(1), p(2), p(3))
        {
            out.push((ts, [wx, wy, wz]));
        }
    }
    out
}

/// Mean gyro over the half-open interval `(t0, t1]`; nearest sample if none inside.
fn mean_gyro(imu: &[(u64, [f64; 3])], t0: u64, t1: u64) -> Option<[f64; 3]> {
    if imu.is_empty() {
        return None;
    }
    let mut sum = [0.0; 3];
    let mut n = 0u32;
    for &(t, w) in imu {
        if t > t0 && t <= t1 {
            sum[0] += w[0];
            sum[1] += w[1];
            sum[2] += w[2];
            n += 1;
        }
    }
    if n > 0 {
        return Some([sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64]);
    }
    let idx = imu.partition_point(|&(t, _)| t < t1);
    let mut best = idx.min(imu.len() - 1);
    if idx > 0
        && (t1 as i128 - imu[idx - 1].0 as i128).abs() < (imu[best].0 as i128 - t1 as i128).abs()
    {
        best = idx - 1;
    }
    Some(imu[best].1)
}

/// Constant gyro bias from the first `window_s` seconds (stationary-start assumption).
fn estimate_bias(imu: &[(u64, [f64; 3])], window_s: f64) -> [f64; 3] {
    if imu.is_empty() {
        return [0.0; 3];
    }
    let cutoff = imu[0].0 + (window_s * 1e9) as u64;
    let mut sum = [0.0; 3];
    let mut n = 0u32;
    for &(t, w) in imu {
        if t <= cutoff {
            sum[0] += w[0];
            sum[1] += w[1];
            sum[2] += w[2];
            n += 1;
        }
    }
    if n == 0 {
        return [0.0; 3];
    }
    [sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: tumvi_egomotion <tumvi_dataset_path> [num_frames] [out.csv]");
        eprintln!(
            "  dataset dir must contain mav0/cam0, mav0/cam1, dso/camchain.yaml, mav0/mocap0"
        );
        std::process::exit(1);
    }
    // Default the PC relaxation to 3000 steps (2x faster than the 6000 library
    // default, ~same accuracy); IQIF_PC_STEPS still overrides.
    if std::env::var("IQIF_PC_STEPS").is_err() {
        std::env::set_var("IQIF_PC_STEPS", "3000");
    }
    let data_dir = PathBuf::from(&args[1]);
    let max_frames: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let csv_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "tumvi_egomotion_log.csv".to_string());

    let cam0_dir = data_dir.join("mav0/cam0");
    let cam1_dir = data_dir.join("mav0/cam1");

    // TUM-VI calibration lives in the Kalibr camchain (fisheye/equidistant),
    // not per-camera EuRoC sensor.yaml files.
    let camchain = data_dir.join("dso/camchain.yaml");
    let rig = StereoRig::from_kalibr_camchain(&camchain)
        .expect("load stereo rig from TUM-VI dso/camchain.yaml");
    let cam = CameraIntrinsics::from_kalibr_camchain(&camchain, "cam0")
        .expect("load cam0 intrinsics from camchain");
    let r_cb = parse_r_cam_imu(&camchain, "cam0"); // camera <- body(imu)
    println!(
        "Stereo rig: baseline = {:.4} m, cam0 {}x{} fx={:.1} model={:?}",
        rig.baseline_meters(),
        rig.cam0.resolution[0],
        rig.cam0.resolution[1],
        rig.cam0.fx,
        rig.cam0.model,
    );

    let cam0_files = list_pngs(&cam0_dir);
    let cam1_files = list_pngs(&cam1_dir);
    let num_frames = cam0_files.len().min(cam1_files.len()).min(max_frames);
    let (w, h) = (rig.cam0.resolution[0], rig.cam0.resolution[1]);

    let gt = load_mocap_gt(&data_dir.join("mav0/mocap0/data.csv"));

    // IMU de-rotation: gyro omega + 3-DoF bearing translation solve. On by
    // default when imu0 is present; TUMVI_USE_IMU=0 forces the full 6-DoF solve.
    let imu = load_imu(&data_dir.join("mav0/imu0/data.csv"));
    let use_imu = !imu.is_empty()
        && std::env::var("TUMVI_USE_IMU")
            .map(|s| s != "0")
            .unwrap_or(true);
    let bias = estimate_bias(&imu, 0.5);
    println!(
        "Frames: {num_frames}, resolution {w}x{h}, mocap {} / imu {} samples",
        gt.len(),
        imu.len()
    );
    if use_imu {
        println!(
            "Mode: IMU de-rotation (gyro omega + bearing 3-DoF translation), gyro bias (body) [{:.5} {:.5} {:.5}]\n",
            bias[0], bias[1], bias[2]
        );
    }

    let frontend_config = FrontendConfig {
        detector: DetectorType::Fast,
        fast_threshold: 20,
        max_features: 200,
        cell_size: 32,
        pyramid_levels: 3,
        klt_method: LkMethod::InverseCompositional,
        histeq: HistEqMethod::Global,
        camera: Some(cam.clone()),
        lbp_policy: LbpPolicy::SoftPenalty,
        enable_internal_ransac: true, // drop geometric outliers before the solve
        ..FrontendConfig::default()
    };
    let mut frontend = Frontend::new(frontend_config, w, h);

    let stereo_config = StereoConfig {
        pyramid_levels: 3,
        patch_half_size: 4,
        max_iterations: 30,
        histeq: HistEqMethod::Global,
        ..StereoConfig::default()
    };
    let mut matcher = StereoMatcher::new(rig, stereo_config, w, h);

    // Fisheye rays near the periphery break the pinhole-normalized motion-field
    // linearization, so restrict the solve to a central FOV cone. Tunable via
    // TUMVI_FOV_DEG (half-angle in degrees); 50 deg is a good default here.
    let fov_deg: f64 = std::env::var("TUMVI_FOV_DEG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(85.0);
    println!("Central FOV limit: {fov_deg:.0} deg half-angle");
    // Solver: "bearing" (default, spherical/fisheye-correct) or "pinhole"
    // (normalized z=1-plane) via TUMVI_SOLVER, for A/B comparison. Ignored when
    // IMU de-rotation is active (that always uses the bearing translation solve).
    let use_bearing = std::env::var("TUMVI_SOLVER")
        .map(|s| s.to_lowercase() != "pinhole")
        .unwrap_or(true);
    if !use_imu {
        println!(
            "Solver: {}\n",
            if use_bearing {
                "bearing-first (spherical)"
            } else {
                "pinhole-normalized"
            }
        );
    }
    let mut adapter = FlowDepthAdapter::new().with_fov_limit_deg(fov_deg);
    let g_init = [0.0_f64; 6];
    let mut prev_v = [0.0_f64; 3];
    let mut prev_ts: Option<u64> = None;

    let mut sum_abs_err = 0.0;
    let mut n_scored = 0usize;

    let mut csv = std::io::BufWriter::new(
        std::fs::File::create(&csv_path).unwrap_or_else(|e| panic!("create {csv_path}: {e}")),
    );
    writeln!(
        csv,
        "frame,ts_ns,vx,vy,vz,wx,wy,wz,gt_vx,gt_vy,gt_vz,qw,qx,qy,qz"
    )
    .unwrap();

    println!("frame  n_obs   v_est (m/s)               |v_est|   |v_gt|   |w_est|   solve_ms");
    for i in 0..num_frames {
        let f0 = load_grayscale(&cam0_files[i]);
        let f1 = load_grayscale(&cam1_files[i]);

        let (feats, _stats) = frontend.process(&f0);
        let feats = feats.to_vec();
        let matches = matcher.match_features(&f1, &feats, frontend.current_pyramid());

        let ts = ts_ns(&cam0_files[i]);
        let t_prev = prev_ts;
        let dt = t_prev.map_or(0.0, |pt| (ts as f64 - pt as f64) * 1e-9);
        prev_ts = Some(ts);

        // Each observe advances per-track history; we call only the selected one.
        let (m, n_obs, solve_ms) = if use_imu {
            let omega = t_prev.and_then(|pt| mean_gyro(&imu, pt, ts)).map(|g| {
                let w_body = Vector3::new(g[0] - bias[0], g[1] - bias[1], g[2] - bias[2]);
                let w_cam = r_cb * w_body; // body(imu) -> camera
                [w_cam[0], w_cam[1], w_cam[2]]
            });
            let obs = adapter.observe_bearing(&feats, &matches, &cam, dt);
            let Some(omega) = omega else { continue };
            if obs.len() < 8 || dt <= 0.0 {
                continue;
            }
            let t0 = Instant::now();
            let v = solve_translation_known_rotation_bearing(&obs, &omega, &prev_v);
            prev_v = v;
            let solve_ms = t0.elapsed().as_secs_f64() * 1000.0;
            (
                [v[0], v[1], v[2], omega[0], omega[1], omega[2]],
                obs.len(),
                solve_ms,
            )
        } else if use_bearing {
            let obs = adapter.observe_bearing(&feats, &matches, &cam, dt);
            if obs.len() < 8 || dt <= 0.0 {
                continue;
            }
            let t0 = Instant::now();
            let m = solve_egomotion_bearing(&obs, &g_init);
            (m, obs.len(), t0.elapsed().as_secs_f64() * 1000.0)
        } else {
            let obs = adapter.observe(&feats, &matches, &cam, dt);
            if obs.len() < 8 || dt <= 0.0 {
                continue;
            }
            let t0 = Instant::now();
            let m = solve_egomotion(&obs, &g_init);
            (m, obs.len(), t0.elapsed().as_secs_f64() * 1000.0)
        };

        let v = [m[0], m[1], m[2]];
        let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let wn = (m[3] * m[3] + m[4] * m[4] + m[5] * m[5]).sqrt();

        let gt_idx = nearest_gt(&gt, ts);
        let gt_speed = gt_idx.map(|k| {
            let g = gt[k].1;
            (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt()
        });
        if let Some(g) = gt_speed {
            sum_abs_err += (speed - g).abs();
            n_scored += 1;
        }

        // CSV: estimate (camera frame) + derived GT world-velocity + orientation.
        let (gv, q) = gt_idx.map_or(([f64::NAN; 3], [f64::NAN; 4]), |k| (gt[k].1, gt[k].2));
        writeln!(
            csv,
            "{i},{ts},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            v[0], v[1], v[2], m[3], m[4], m[5], gv[0], gv[1], gv[2], q[0], q[1], q[2], q[3]
        )
        .unwrap();

        let gt_str = gt_speed.map_or("   n/a".to_string(), |g| format!("{g:6.3}"));
        println!(
            "{i:5}  {:5}  [{:6.3} {:6.3} {:6.3}]   {speed:6.3}   {gt_str}   {wn:6.3}   {solve_ms:6.1}",
            n_obs,
            v[0],
            v[1],
            v[2],
        );
    }

    csv.flush().unwrap();
    println!("\nWrote per-frame log to {csv_path}");
    if n_scored > 0 {
        println!(
            "Mean |speed_est - speed_gt| over {n_scored} frames: {:.3} m/s",
            sum_abs_err / n_scored as f64
        );
    }
}
