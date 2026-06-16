//! EuRoC stereo egomotion via an INHIBITION-DOMINATED spike-coding network.
//!
//! Same pipeline as `euroc_egomotion` (Rudolf-V frontend + sparse stereo depth
//! -> known-depth flow), but the 6-DoF solve is the instantaneous population
//! readout of an all-inhibitory tight-balance SCN running on the IQIF chip
//! (Mancoo-Keemink-Machens 2020), rather than the labeled-line relaxation.
//! Logs est-vs-GT to a CSV for `tests/plot_egomotion.py`.
//!
//! Usage:
//!     cargo run -p iqif-vio --example euroc_egomotion_scn --release -- /path/to/V1_01_easy [num_frames] [out.csv]

use iqif_vio::frontend_adapter::FlowDepthAdapter;
use iqif_vio::solve_egomotion_scn;

use rudolf_v::camera::{CameraIntrinsics, StereoRig};
use rudolf_v::frontend::{DetectorType, Frontend, FrontendConfig, LbpPolicy};
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image;
use rudolf_v::klt::LkMethod;
use rudolf_v::stereo::{StereoConfig, StereoMatcher};

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

fn ts_ns(path: &Path) -> u64 {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
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
    if idx > 0 {
        let d_prev = (ts as i128 - gt[idx - 1].0 as i128).abs();
        let d_best = (gt[best].0 as i128 - ts as i128).abs();
        if d_prev < d_best {
            best = idx - 1;
        }
    }
    Some(best)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: euroc_egomotion_scn <euroc_dataset_path> [num_frames] [out.csv]");
        std::process::exit(1);
    }
    let data_dir = PathBuf::from(&args[1]);
    let max_frames: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let csv_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "egomotion_scn_log.csv".to_string());

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

    let gt = load_gt(&data_dir.join("mav0/state_groundtruth_estimate0/data.csv"));
    println!(
        "Frames: {num_frames}, resolution {w}x{h}, ground-truth samples: {}",
        gt.len()
    );
    println!("Solver: inhibition-dominated spike-coding network (IQIF chip)\n");

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
        let dt = prev_ts.map_or(0.0, |pt| (ts as f64 - pt as f64) * 1e-9);
        let obs = adapter.observe(&feats, &matches, &cam, dt);
        prev_ts = Some(ts);

        if obs.len() < 8 || dt <= 0.0 {
            continue;
        }

        let t0 = Instant::now();
        let m = solve_egomotion_scn(&obs);
        let solve_ms = t0.elapsed().as_secs_f64() * 1000.0;

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
            obs.len(),
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
    println!("Plot est-vs-GT (6-DoF, frame-aligned):");
    println!(
        "  python tests/plot_egomotion.py {csv_path} {}",
        cam0_dir.join("sensor.yaml").display()
    );
}
