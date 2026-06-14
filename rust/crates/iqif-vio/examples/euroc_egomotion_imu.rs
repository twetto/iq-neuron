//! IMU-driven de-rotation: the real predict layer.
//!
//! This is the de-rotation diagnostic (`euroc_egomotion_derot`) with the *only*
//! change that matters for a real system: the angular velocity comes from the
//! EuRoC gyro (`mav0/imu0`), not from the ground-truth orientation. We average
//! the gyro samples over each inter-frame interval (200 Hz IMU vs ~20 Hz frames),
//! subtract a constant bias estimate, rotate the body-frame rate into the camera
//! frame with `R_BS^T`, subtract `B(x) omega` from the measured flow, and let the
//! 8-bit predictive-coding circuit solve only the 3-DoF translation `v`.
//!
//! If lateral v1/v2 track GT here (as they did with GT omega in the de-rotation
//! diagnostic), the gyro predict layer is doing its job and the
//! translation-rotation ambiguity is broken by a real sensor.
//!
//! Bias: by default a constant gyro bias is estimated from the first
//! `--bias-window` seconds of IMU data (assumes a roughly stationary start; the
//! omega-vs-GT panels in the plot reveal whether this holds). Pass an explicit
//! `bx,by,bz` to override.
//!
//! Logs the same CSV schema as `euroc_egomotion` (with w = the IMU omega used),
//! so `tests/plot_egomotion.py` plots it directly.
//!
//! Usage:
//!     cargo run -p iqif-vio --example euroc_egomotion_imu --release -- \
//!         /path/to/V1_01_easy [num_frames] [out.csv] [bx,by,bz]

use iqif_vio::frontend_adapter::FlowDepthAdapter;
use iqif_vio::{solve_translation_known_rotation, solve_translation_known_rotation_filtered};

use rudolf_v::camera::{CameraIntrinsics, StereoRig};
use rudolf_v::frontend::{DetectorType, Frontend, FrontendConfig, LbpPolicy};
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image;
use rudolf_v::klt::LkMethod;
use rudolf_v::stereo::{StereoConfig, StereoMatcher};

use nalgebra::{Matrix3, Vector3};
use std::io::Write;
use std::path::{Path, PathBuf};

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

fn ts_ns(path: &Path) -> u64 {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// IMU samples: (timestamp_ns, gyro_body [rad/s]).
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

/// Mean gyro over the half-open interval `(t0, t1]` (the flow integration window).
/// Falls back to the nearest sample if no IMU sample lands inside.
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
    // No sample inside the window: take the nearest one to t1.
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
    let t0 = imu[0].0;
    let cutoff = t0 + (window_s * 1e9) as u64;
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

fn load_gt(path: &Path) -> Vec<(u64, [f64; 3], [f64; 4])> {
    let mut out = Vec::new();
    let Ok(txt) = std::fs::read_to_string(path) else {
        return out;
    };
    for line in txt.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 11 {
            continue;
        }
        let p = |i: usize| c[i].trim().parse::<f64>().ok();
        if let (Ok(ts), Some(qw), Some(qx), Some(qy), Some(qz), Some(vx), Some(vy), Some(vz)) = (
            c[0].trim().parse::<u64>(),
            p(4),
            p(5),
            p(6),
            p(7),
            p(8),
            p(9),
            p(10),
        ) {
            out.push((ts, [vx, vy, vz], [qw, qx, qy, qz]));
        }
    }
    out
}

fn nearest_gt(gt: &[(u64, [f64; 3], [f64; 4])], ts: u64) -> Option<usize> {
    if gt.is_empty() {
        return None;
    }
    let idx = gt.partition_point(|&(t, _, _)| t < ts);
    let mut best = idx.min(gt.len() - 1);
    if idx > 0
        && (ts as i128 - gt[idx - 1].0 as i128).abs() < (gt[best].0 as i128 - ts as i128).abs()
    {
        best = idx - 1;
    }
    Some(best)
}

fn parse_r_bs(sensor_yaml: &Path) -> Matrix3<f64> {
    let txt = std::fs::read_to_string(sensor_yaml).expect("read sensor.yaml");
    let after = &txt[txt.find("T_BS:").expect("T_BS in sensor.yaml")..];
    let block = &after[after.find("data:").expect("T_BS data")..];
    let lb = block.find('[').unwrap();
    let rb = block.find(']').unwrap();
    let n: Vec<f64> = block[lb + 1..rb]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    Matrix3::new(n[0], n[1], n[2], n[4], n[5], n[6], n[8], n[9], n[10])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: euroc_egomotion_imu <euroc_dataset_path> [num_frames] [out.csv] [bx,by,bz] [alpha]"
        );
        std::process::exit(1);
    }
    let data_dir = PathBuf::from(&args[1]);
    let max_frames: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let csv_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "egomotion_imu.csv".to_string());

    let cam0_dir = data_dir.join("mav0/cam0");
    let cam1_dir = data_dir.join("mav0/cam1");

    let rig = StereoRig::from_euroc(&cam0_dir.join("sensor.yaml"), &cam1_dir.join("sensor.yaml"))
        .expect("load stereo rig");
    let cam = CameraIntrinsics::from_euroc_yaml(&cam0_dir.join("sensor.yaml")).expect("load cam0");
    let r_bs = parse_r_bs(&cam0_dir.join("sensor.yaml"));
    let (w, h) = (rig.cam0.resolution[0], rig.cam0.resolution[1]);

    let cam0_files = list_pngs(&cam0_dir);
    let cam1_files = list_pngs(&cam1_dir);
    let num_frames = cam0_files.len().min(cam1_files.len()).min(max_frames);
    let gt = load_gt(&data_dir.join("mav0/state_groundtruth_estimate0/data.csv"));
    let imu = load_imu(&data_dir.join("mav0/imu0/data.csv"));

    // Gyro bias: explicit arg "bx,by,bz" or estimate from a stationary-start window.
    let bias: [f64; 3] = args
        .get(4)
        .and_then(|s| {
            let v: Vec<f64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            if v.len() == 3 {
                Some([v[0], v[1], v[2]])
            } else {
                None
            }
        })
        .unwrap_or_else(|| estimate_bias(&imu, 0.5));

    // Temporal-prior strength (filtering lean): 0 = per-frame solve.
    let alpha: f64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    println!(
        "IMU de-rotation: {num_frames} frames, imu samples {}, gt samples {}",
        imu.len(),
        gt.len()
    );
    println!(
        "  gyro bias (body, rad/s): [{:.5} {:.5} {:.5}], alpha {alpha}",
        bias[0], bias[1], bias[2]
    );

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
        enable_internal_ransac: true,
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

    let mut adapter = FlowDepthAdapter::new();
    let mut prev_ts: Option<u64> = None;
    let mut prev_v = [0.0_f64; 3]; // recursive-filter state (previous translation)

    let mut csv = std::io::BufWriter::new(std::fs::File::create(&csv_path).expect("create csv"));
    writeln!(
        csv,
        "frame,ts_ns,vx,vy,vz,wx,wy,wz,gt_vx,gt_vy,gt_vz,qw,qx,qy,qz"
    )
    .unwrap();

    println!("frame  n_obs   v_derot (m/s)              w_imu(cam) (rad/s)");
    for i in 0..num_frames {
        let f0 = load_grayscale(&cam0_files[i]);
        let f1 = load_grayscale(&cam1_files[i]);

        let (feats, _stats) = frontend.process(&f0);
        let feats = feats.to_vec();
        let matches = matcher.match_features(&f1, &feats, frontend.current_pyramid());

        let ts = ts_ns(&cam0_files[i]);
        let dt = prev_ts.map_or(0.0, |pt| (ts as f64 - pt as f64) * 1e-9);
        let obs = adapter.observe(&feats, &matches, &cam, dt);

        // IMU gyro averaged over the flow interval (prev_ts, ts], bias-corrected,
        // rotated body -> camera frame.
        let w_cam = match prev_ts {
            Some(pt) if ts > pt => mean_gyro(&imu, pt, ts).map(|g| {
                let w_body = Vector3::new(g[0] - bias[0], g[1] - bias[1], g[2] - bias[2]);
                r_bs.transpose() * w_body
            }),
            _ => None,
        };
        prev_ts = Some(ts);

        let (Some(w_cam), true) = (w_cam, obs.len() >= 8 && dt > 0.0) else {
            continue;
        };
        let Some(k) = nearest_gt(&gt, ts) else {
            continue;
        };

        let omega = [w_cam[0], w_cam[1], w_cam[2]];
        // alpha == 0: original per-frame solve; alpha > 0: temporal-prior filter.
        let v = if alpha > 0.0 {
            solve_translation_known_rotation_filtered(&obs, &omega, &prev_v, alpha)
        } else {
            solve_translation_known_rotation(&obs, &omega, &[0.0; 3])
        };
        prev_v = v;

        let gv = gt[k].1;
        let q = gt[k].2;
        writeln!(
            csv,
            "{i},{ts},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            v[0], v[1], v[2], omega[0], omega[1], omega[2], gv[0], gv[1], gv[2], q[0], q[1], q[2],
            q[3]
        )
        .unwrap();

        if i % 50 == 0 {
            println!(
                "{i:5}  {:5}  [{:6.3} {:6.3} {:6.3}]   [{:6.3} {:6.3} {:6.3}]",
                obs.len(),
                v[0],
                v[1],
                v[2],
                omega[0],
                omega[1],
                omega[2]
            );
        }
    }

    csv.flush().unwrap();
    println!("\nWrote {csv_path}");
    println!(
        "  python tests/plot_egomotion.py {csv_path} {}",
        cam0_dir.join("sensor.yaml").display()
    );
}
