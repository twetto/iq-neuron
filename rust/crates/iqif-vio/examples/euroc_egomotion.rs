//! EuRoC stereo egomotion via predictive coding.
//!
//! Per frame: run the Rudolf-V CPU frontend on cam0 (FAST detect + KLT track),
//! match each track into cam1 for sparse stereo depth, turn the tracks + depth
//! into known-depth flow observations, and solve the 6-DoF egomotion with the
//! IQIF predictive-coding circuit. Estimated linear-speed magnitude is compared
//! against the EuRoC ground-truth velocity as a sanity check.
//!
//! Usage:
//!     cargo run -p iqif-vio --example euroc_egomotion --release -- /path/to/V1_01_easy [num_frames]

use iqif_vio::frontend_adapter::FlowDepthAdapter;
use iqif_vio::solve_egomotion;

use rudolf_v::camera::{CameraIntrinsics, StereoRig};
use rudolf_v::frontend::{DetectorType, Frontend, FrontendConfig, LbpPolicy};
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image;
use rudolf_v::klt::LkMethod;
use rudolf_v::stereo::{StereoConfig, StereoMatcher};

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

/// EuRoC image filenames are nanosecond timestamps.
fn ts_ns(path: &Path) -> u64 {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Load ground-truth (timestamp_ns, speed |v| m/s) from
/// `mav0/state_groundtruth_estimate0/data.csv` (cols 0 and 8..11). Empty if absent.
fn load_gt_speed(path: &Path) -> Vec<(u64, f64)> {
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
        if let (Ok(ts), Ok(vx), Ok(vy), Ok(vz)) = (
            c[0].trim().parse::<u64>(),
            c[8].trim().parse::<f64>(),
            c[9].trim().parse::<f64>(),
            c[10].trim().parse::<f64>(),
        ) {
            out.push((ts, (vx * vx + vy * vy + vz * vz).sqrt()));
        }
    }
    out
}

/// Nearest-timestamp ground-truth speed (assumes `gt` sorted by timestamp).
fn nearest_gt_speed(gt: &[(u64, f64)], ts: u64) -> Option<f64> {
    if gt.is_empty() {
        return None;
    }
    let idx = gt.partition_point(|&(t, _)| t < ts);
    let mut best = idx.min(gt.len() - 1);
    if idx > 0 {
        let d_prev = (ts as i128 - gt[idx - 1].0 as i128).abs();
        let d_best = (gt[best].0 as i128 - ts as i128).abs();
        if d_prev < d_best {
            best = idx - 1;
        }
    }
    Some(gt[best].1)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: euroc_egomotion <euroc_dataset_path> [num_frames]");
        std::process::exit(1);
    }
    let data_dir = PathBuf::from(&args[1]);
    let max_frames: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);

    let cam0_dir = data_dir.join("mav0/cam0");
    let cam1_dir = data_dir.join("mav0/cam1");

    let rig = StereoRig::from_euroc(&cam0_dir.join("sensor.yaml"), &cam1_dir.join("sensor.yaml"))
        .expect("load stereo rig from EuRoC sensor.yaml");
    let cam = CameraIntrinsics::from_euroc_yaml(&cam0_dir.join("sensor.yaml"))
        .expect("load cam0 intrinsics");
    println!(
        "Stereo rig: baseline = {:.4} m, cam0 {}x{} fx={:.1}",
        rig.baseline_meters(),
        rig.cam0.resolution[0],
        rig.cam0.resolution[1],
        rig.cam0.fx
    );

    let cam0_files = list_pngs(&cam0_dir);
    let cam1_files = list_pngs(&cam1_dir);
    let num_frames = cam0_files.len().min(cam1_files.len()).min(max_frames);
    let (w, h) = (rig.cam0.resolution[0], rig.cam0.resolution[1]);

    let gt = load_gt_speed(&data_dir.join("mav0/state_groundtruth_estimate0/data.csv"));
    println!(
        "Frames: {num_frames}, resolution {w}x{h}, ground-truth samples: {}\n",
        gt.len()
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
        lbp_policy: LbpPolicy::HardReject,
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

    let mut adapter = FlowDepthAdapter::new();
    let g_init = [0.0_f64; 6];
    let mut prev_ts: Option<u64> = None;

    let mut sum_abs_err = 0.0;
    let mut n_scored = 0usize;

    println!("frame  n_obs   v_est (m/s)               |v_est|   |v_gt|   |w_est|   solve_ms");
    for i in 0..num_frames {
        let f0 = load_grayscale(&cam0_files[i]);
        let f1 = load_grayscale(&cam1_files[i]);

        let (feats, _stats) = frontend.process(&f0);
        let feats = feats.to_vec();
        let matches = matcher.match_features(&f1, &feats, frontend.current_pyramid());

        let ts = ts_ns(&cam0_files[i]);
        let dt = prev_ts.map_or(0.0, |pt| (ts as f64 - pt as f64) * 1e-9);
        let obs = adapter.observe(&feats, &matches, &cam, dt);
        prev_ts = Some(ts);

        if obs.len() < 8 || dt <= 0.0 {
            continue;
        }

        let t0 = Instant::now();
        let m = solve_egomotion(&obs, &g_init);
        let solve_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let v = [m[0], m[1], m[2]];
        let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let wn = (m[3] * m[3] + m[4] * m[4] + m[5] * m[5]).sqrt();
        let gt_speed = nearest_gt_speed(&gt, ts);

        if let Some(g) = gt_speed {
            sum_abs_err += (speed - g).abs();
            n_scored += 1;
        }
        let gt_str = gt_speed.map_or("   n/a".to_string(), |g| format!("{g:6.3}"));
        println!(
            "{i:5}  {:5}  [{:6.3} {:6.3} {:6.3}]   {speed:6.3}   {gt_str}   {wn:6.3}   {solve_ms:6.1}",
            obs.len(),
            v[0],
            v[1],
            v[2],
        );
    }

    if n_scored > 0 {
        println!(
            "\nMean |speed_est - speed_gt| over {n_scored} frames: {:.3} m/s",
            sum_abs_err / n_scored as f64
        );
        println!(
            "(camera-frame estimate vs body world-frame GT magnitude — a coarse sanity check;\n full direction/ATE evaluation needs the body<-cam extrinsic alignment.)"
        );
    }
}
