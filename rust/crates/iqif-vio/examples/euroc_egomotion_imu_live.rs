//! Live IMU-driven de-rotation egomotion, with real-time minifb visualization
//! and faded ground-truth overlay.
//!
//! Same display as `euroc_egomotion_live`, but the angular velocity is taken
//! from the EuRoC gyro (`mav0/imu0`) instead of the full 6-DoF solve: the gyro
//! is averaged over each inter-frame interval, bias-corrected, rotated body->cam
//! with `R_BS^T`, used to de-rotate the flow, and the 8-bit CLOSED-LOOP
//! predictive-coding circuit (G realised as on-chip feedback weights, no host
//! matmul) solves only the 3-DoF translation `v`. The estimate's w1/w2/w3 charts
//! show the IMU rate actually used; v1/v2/v3 show the recovered translation.
//!
//! Left panel: cam0 with tracked features colored by sparse stereo depth.
//! Right panel: six scrolling charts, top->bottom v1 v2 v3 (m/s), w1 w2 w3
//! (rad/s). Bright = estimate (PC translation + IMU rate), faded = ground truth.
//!
//! Usage:
//!     cargo run -p iqif-vio --example euroc_egomotion_imu_live --release -- \
//!         /path/to/V1_01_easy [num_frames] [bx,by,bz]
//! Controls: Q / Esc = quit, Space = pause/step.

use iqif_vio::frontend_adapter::FlowDepthAdapter;
use iqif_vio::solve_translation_known_rotation_closed_loop;

use rudolf_v::camera::{CameraIntrinsics, StereoRig};
use rudolf_v::fast::Feature;
use rudolf_v::frontend::{DetectorType, Frontend, FrontendConfig, LbpPolicy};
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image;
use rudolf_v::klt::LkMethod;
use rudolf_v::stereo::{StereoConfig, StereoMatch, StereoMatcher};

use minifb::{Key, Window, WindowOptions};
use nalgebra::{Matrix3, Quaternion, Rotation3, UnitQuaternion, Vector3};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

const GAP: usize = 8;
const CHART_W: usize = 460;
const PLOT_WINDOW: usize = 300; // frames shown in the scrolling charts
const NEAR_M: f64 = 0.5;
const FAR_M: f64 = 8.0;

const DOF_COLORS: [u32; 6] = [
    0xFFE6194B, 0xFF3CB44B, 0xFF4363D8, // v1 v2 v3 (red green blue)
    0xFF42D4F4, 0xFFF032E6, 0xFFFFE119, // w1 w2 w3 (cyan magenta yellow)
];

/// One chart sample: the estimate and (optionally) frame-aligned ground truth.
type Sample = ([f64; 6], Option<[f64; 6]>);

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

/// Ground truth: (ts_ns, world velocity, quaternion [qw,qx,qy,qz] = R_world_body).
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

/// Body<-camera rotation R_bs from the cam0 `sensor.yaml` `T_BS` 4x4 (row-major).
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
            "Usage: euroc_egomotion_imu_live <euroc_dataset_path> [num_frames] [bx,by,bz]"
        );
        std::process::exit(1);
    }
    let data_dir = PathBuf::from(&args[1]);
    let max_frames: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let cam0_dir = data_dir.join("mav0/cam0");
    let cam1_dir = data_dir.join("mav0/cam1");

    let rig = StereoRig::from_euroc(&cam0_dir.join("sensor.yaml"), &cam1_dir.join("sensor.yaml"))
        .expect("load stereo rig from EuRoC sensor.yaml");
    let cam = CameraIntrinsics::from_euroc_yaml(&cam0_dir.join("sensor.yaml"))
        .expect("load cam0 intrinsics");
    let r_bs = parse_r_bs(&cam0_dir.join("sensor.yaml"));
    let (w, h) = (rig.cam0.resolution[0], rig.cam0.resolution[1]);

    let cam0_files = list_pngs(&cam0_dir);
    let cam1_files = list_pngs(&cam1_dir);
    let num_frames = cam0_files.len().min(cam1_files.len()).min(max_frames);
    let gt = load_gt(&data_dir.join("mav0/state_groundtruth_estimate0/data.csv"));
    let imu = load_imu(&data_dir.join("mav0/imu0/data.csv"));

    let bias: [f64; 3] = args
        .get(3)
        .and_then(|s| {
            let v: Vec<f64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            if v.len() == 3 {
                Some([v[0], v[1], v[2]])
            } else {
                None
            }
        })
        .unwrap_or_else(|| estimate_bias(&imu, 0.5));

    println!(
        "Live IMU de-rotation (closed-loop PC): {num_frames} frames, {w}x{h}, imu {}, gt {}",
        imu.len(),
        gt.len()
    );
    println!(
        "  gyro bias (body, rad/s): [{:.5} {:.5} {:.5}]",
        bias[0], bias[1], bias[2]
    );
    println!("Charts top->bottom: v1 v2 v3 (m/s), w1 w2 w3 (rad/s). Bright=estimate (PC v + IMU w), faded=GT. Q/Esc quit, Space pause.");

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
    let mut prev_gt: Option<(u64, Matrix3<f64>)> = None; // (ts, R_world_body) for GT angular rate
    let mut prev_v = [0.0_f64; 3]; // recursive-filter state (previous translation)
    let mut history: VecDeque<Sample> = VecDeque::with_capacity(PLOT_WINDOW);

    let win_w = w + GAP + CHART_W;
    let win_h = h;
    let mut window = Window::new(
        "iqif-vio - EuRoC IMU de-rotation egomotion (live)",
        win_w,
        win_h,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .expect("failed to create window");
    window.set_target_fps(30);
    let mut fb = vec![0xFF1A1A1Au32; win_w * win_h];

    let mut paused = false;
    let mut i = 0usize;
    while window.is_open() && !window.is_key_down(Key::Escape) && !window.is_key_down(Key::Q) {
        if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
            paused = !paused;
        }

        if !paused && i < num_frames {
            let f0 = load_grayscale(&cam0_files[i]);
            let f1 = load_grayscale(&cam1_files[i]);

            let (feats, _stats) = frontend.process(&f0);
            let feats = feats.to_vec();
            let matches = matcher.match_features(&f1, &feats, frontend.current_pyramid());

            let ts = ts_ns(&cam0_files[i]);
            let dt = prev_ts.map_or(0.0, |pt| (ts as f64 - pt as f64) * 1e-9);
            let obs = adapter.observe(&feats, &matches, &cam, dt);

            // IMU gyro over the flow interval -> camera frame.
            let w_cam = match prev_ts {
                Some(pt) if ts > pt => mean_gyro(&imu, pt, ts).map(|g| {
                    let w_body = Vector3::new(g[0] - bias[0], g[1] - bias[1], g[2] - bias[2]);
                    r_bs.transpose() * w_body
                }),
                _ => None,
            };
            prev_ts = Some(ts);

            if let (Some(w_cam), true) = (w_cam, obs.len() >= 8 && dt > 0.0) {
                let omega = [w_cam[0], w_cam[1], w_cam[2]];
                // Closed-loop circuit: G realised as on-chip feedback weights
                // (no host matmul). prev_v seeds the held estimate.
                let v = solve_translation_known_rotation_closed_loop(&obs, &omega, &prev_v);
                prev_v = v;
                let est = [v[0], v[1], v[2], omega[0], omega[1], omega[2]];

                // Frame-aligned ground truth (camera frame), if available.
                let gt_cam = nearest_gt(&gt, ts).map(|k| {
                    let q = gt[k].2;
                    let r_wb =
                        *UnitQuaternion::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3]))
                            .to_rotation_matrix()
                            .matrix();
                    let v_w = Vector3::new(gt[k].1[0], gt[k].1[1], gt[k].1[2]);
                    let v_cam = r_bs.transpose() * (r_wb.transpose() * v_w); // body vel -> cam frame
                    let w_gt = match prev_gt {
                        Some((pt, r_prev)) if ts > pt => {
                            let dtg = (ts - pt) as f64 * 1e-9;
                            let w_body =
                                Rotation3::from_matrix_unchecked(r_prev.transpose() * r_wb)
                                    .scaled_axis()
                                    / dtg;
                            r_bs.transpose() * w_body
                        }
                        _ => Vector3::zeros(),
                    };
                    prev_gt = Some((ts, r_wb));
                    [v_cam[0], v_cam[1], v_cam[2], w_gt[0], w_gt[1], w_gt[2]]
                });

                if history.len() == PLOT_WINDOW {
                    history.pop_front();
                }
                history.push_back((est, gt_cam));
            }

            fb.fill(0xFF1A1A1A);
            let disp = frontend.preprocessed_image().unwrap_or(&f0);
            render_camera(&mut fb, win_w, win_h, disp, &feats, &matches);
            render_charts(&mut fb, win_w, win_h, w + GAP, &history);

            i += 1;
        }

        window
            .update_with_buffer(&fb, win_w, win_h)
            .expect("window update failed");

        if i >= num_frames && !paused {
            paused = true; // hold the final frame until quit
        }
    }
}

// ── Camera panel ────────────────────────────────────────────────────────────

fn render_camera(
    fb: &mut [u32],
    win_w: usize,
    win_h: usize,
    img: &Image<u8>,
    feats: &[Feature],
    matches: &[StereoMatch],
) {
    for y in 0..img.height().min(win_h) {
        for x in 0..img.width().min(win_w) {
            let c = img.get(x, y) as u32;
            fb[y * win_w + x] = 0xFF000000 | (c << 16) | (c << 8) | c;
        }
    }
    for (f, m) in feats.iter().zip(matches.iter()) {
        let (px, py) = (f.x.round() as i32, f.y.round() as i32);
        if m.matched && m.inv_depth > 0.0 {
            draw_disk(
                fb,
                win_w,
                win_h,
                px,
                py,
                2,
                color_for_depth(1.0 / m.inv_depth as f64),
            );
        } else {
            draw_cross(fb, win_w, win_h, px, py, 2, 0xFFFF3030);
        }
    }
}

// ── 6-DoF scrolling line charts (estimate + faded GT) ────────────────────────

fn render_charts(
    fb: &mut [u32],
    win_w: usize,
    win_h: usize,
    x0: usize,
    history: &VecDeque<Sample>,
) {
    let rows = 6;
    let row_h = win_h / rows;
    let pw = CHART_W.saturating_sub(2 * GAP);
    let samples: Vec<Sample> = history.iter().copied().collect();
    let n = samples.len();

    for d in 0..rows {
        let py0 = d * row_h + 2;
        let ph = row_h.saturating_sub(4);
        let px0 = x0 + GAP;

        fill_rect(fb, win_w, win_h, px0, py0, pw, ph, 0xFF101010);
        draw_rect(fb, win_w, win_h, px0, py0, pw, ph, 0xFF404040);

        // Symmetric autoscale over BOTH estimate and GT so both fit.
        let mut m = 0.05_f64;
        for s in &samples {
            m = m.max(s.0[d].abs());
            if let Some(g) = s.1 {
                m = m.max(g[d].abs());
            }
        }

        let yz = (py0 + ph / 2) as i32;
        draw_hline(fb, win_w, win_h, px0, px0 + pw, yz, 0xFF303030);
        for t in 0..6usize {
            fill_rect(fb, win_w, win_h, px0 + 3, py0 + 3 + t, 8, 1, DOF_COLORS[d]);
        }

        if n < 2 {
            continue;
        }
        let half = (ph / 2).saturating_sub(2) as f64;
        let map_y = |v: f64| (py0 + ph / 2) as i32 - ((v / m) * half).round() as i32;
        let xat = |k: usize| (px0 + k * (pw - 1) / (n - 1)) as i32;

        // GT first (faded), behind the estimate.
        for k in 1..n {
            if let (Some(a), Some(b)) = (samples[k - 1].1, samples[k].1) {
                draw_line(
                    fb,
                    win_w,
                    win_h,
                    xat(k - 1),
                    map_y(a[d]),
                    xat(k),
                    map_y(b[d]),
                    fade(DOF_COLORS[d], 0.40),
                );
            }
        }
        // Estimate on top (full color).
        for k in 1..n {
            draw_line(
                fb,
                win_w,
                win_h,
                xat(k - 1),
                map_y(samples[k - 1].0[d]),
                xat(k),
                map_y(samples[k].0[d]),
                DOF_COLORS[d],
            );
        }
    }
}

// ── Framebuffer primitives ───────────────────────────────────────────────────

fn set_px(fb: &mut [u32], win_w: usize, win_h: usize, x: i32, y: i32, c: u32) {
    if x < 0 || y < 0 || x as usize >= win_w || y as usize >= win_h {
        return;
    }
    fb[y as usize * win_w + x as usize] = c;
}

fn fill_rect(
    fb: &mut [u32],
    win_w: usize,
    win_h: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    c: u32,
) {
    for yy in y..(y + h).min(win_h) {
        for xx in x..(x + w).min(win_w) {
            fb[yy * win_w + xx] = c;
        }
    }
}

fn draw_rect(
    fb: &mut [u32],
    win_w: usize,
    win_h: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    c: u32,
) {
    draw_hline(fb, win_w, win_h, x, x + w, y as i32, c);
    draw_hline(fb, win_w, win_h, x, x + w, (y + h) as i32, c);
    for yy in y..(y + h).min(win_h) {
        set_px(fb, win_w, win_h, x as i32, yy as i32, c);
        set_px(fb, win_w, win_h, (x + w) as i32, yy as i32, c);
    }
}

fn draw_hline(fb: &mut [u32], win_w: usize, win_h: usize, x0: usize, x1: usize, y: i32, c: u32) {
    for x in x0..x1.min(win_w) {
        set_px(fb, win_w, win_h, x as i32, y, c);
    }
}

fn draw_line(
    fb: &mut [u32],
    win_w: usize,
    win_h: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    c: u32,
) {
    let (mut x0, mut y0) = (x0, y0);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        set_px(fb, win_w, win_h, x0, y0, c);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn draw_disk(fb: &mut [u32], win_w: usize, win_h: usize, cx: i32, cy: i32, r: i32, c: u32) {
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                set_px(fb, win_w, win_h, cx + dx, cy + dy, c);
            }
        }
    }
}

fn draw_cross(fb: &mut [u32], win_w: usize, win_h: usize, cx: i32, cy: i32, r: i32, c: u32) {
    for d in -r..=r {
        set_px(fb, win_w, win_h, cx + d, cy + d, c);
        set_px(fb, win_w, win_h, cx + d, cy - d, c);
    }
}

/// Blend `c` toward the dark background. `t=1` keeps `c`, `t=0` is background.
fn fade(c: u32, t: f64) -> u32 {
    let bg = 26.0_f64; // 0x1A background channel value
    let ch = |sh: u32| {
        let v = ((c >> sh) & 0xFF) as f64;
        (bg + (v - bg) * t).round().clamp(0.0, 255.0) as u32
    };
    0xFF000000 | (ch(16) << 16) | (ch(8) << 8) | ch(0)
}

fn color_for_depth(depth_m: f64) -> u32 {
    let t = ((FAR_M - depth_m) / (FAR_M - NEAR_M)).clamp(0.0, 1.0);
    jet(t)
}

fn jet(t: f64) -> u32 {
    const J: [(f64, u8, u8, u8); 6] = [
        (0.0, 0, 0, 128),
        (0.125, 0, 0, 255),
        (0.375, 0, 255, 255),
        (0.625, 255, 255, 0),
        (0.875, 255, 0, 0),
        (1.0, 128, 0, 0),
    ];
    let mut idx = J.len() - 2;
    for i in 0..J.len() - 1 {
        if t <= J[i + 1].0 {
            idx = i;
            break;
        }
    }
    let (t0, r0, g0, b0) = J[idx];
    let (t1, r1, g1, b1) = J[idx + 1];
    let lt = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
    let lerp = |a: u8, b: u8| (a as f64 + (b as f64 - a as f64) * lt).round() as u32;
    0xFF000000 | (lerp(r0, r1) << 16) | (lerp(g0, g1) << 8) | lerp(b0, b1)
}
