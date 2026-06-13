//! Live EuRoC stereo egomotion via predictive coding, with real-time minifb
//! visualization.
//!
//! Left panel: cam0 with tracked features colored by sparse stereo depth
//! (jet: near = red, far = blue; unmatched = red cross).
//! Right panel: six scrolling line charts of the recovered 6-DoF motion,
//! top to bottom: v1 v2 v3 (m/s), w1 w2 w3 (rad/s).
//!
//! Usage:
//!     cargo run -p iqif-vio --example euroc_egomotion_live --release -- /path/to/V1_01_easy [num_frames]
//! Controls: Q / Esc = quit, Space = pause/step.

use iqif_vio::frontend_adapter::FlowDepthAdapter;
use iqif_vio::solve_egomotion;

use rudolf_v::camera::{CameraIntrinsics, StereoRig};
use rudolf_v::fast::Feature;
use rudolf_v::frontend::{DetectorType, Frontend, FrontendConfig, LbpPolicy};
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image;
use rudolf_v::klt::LkMethod;
use rudolf_v::stereo::{StereoConfig, StereoMatch, StereoMatcher};

use minifb::{Key, Window, WindowOptions};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

const GAP: usize = 8;
const CHART_W: usize = 460;
const PLOT_WINDOW: usize = 300; // frames shown in the scrolling charts
const NEAR_M: f64 = 0.5;
const FAR_M: f64 = 8.0;

const DOF_LABELS: [&str; 6] = ["v1", "v2", "v3", "w1", "w2", "w3"];
const DOF_COLORS: [u32; 6] = [
    0xFFE6194B, 0xFF3CB44B, 0xFF4363D8, // v1 v2 v3 (red green blue)
    0xFF42D4F4, 0xFFF032E6, 0xFFFFE119, // w1 w2 w3 (cyan magenta yellow)
];

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: euroc_egomotion_live <euroc_dataset_path> [num_frames]");
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
    let (w, h) = (rig.cam0.resolution[0], rig.cam0.resolution[1]);

    let cam0_files = list_pngs(&cam0_dir);
    let cam1_files = list_pngs(&cam1_dir);
    let num_frames = cam0_files.len().min(cam1_files.len()).min(max_frames);
    println!(
        "Live egomotion: {num_frames} frames, {w}x{h}, baseline {:.4} m",
        rig.baseline_meters()
    );
    println!("Charts top->bottom: v1 v2 v3 (m/s), w1 w2 w3 (rad/s). Q/Esc quit, Space pause.");

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
    let g_init = [0.0_f64; 6];
    let mut prev_ts: Option<u64> = None;
    let mut history: VecDeque<[f64; 6]> = VecDeque::with_capacity(PLOT_WINDOW);

    let win_w = w + GAP + CHART_W;
    let win_h = h;
    let mut window = Window::new(
        "iqif-vio - EuRoC PC egomotion (live)",
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
            prev_ts = Some(ts);

            if obs.len() >= 8 && dt > 0.0 {
                let m = solve_egomotion(&obs, &g_init);
                if history.len() == PLOT_WINDOW {
                    history.pop_front();
                }
                history.push_back(m);
            }

            // Render: clear, camera panel, depth-colored features, charts.
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
            paused = true; // hold the final frame on screen until quit
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

// ── 6-DoF scrolling line charts ──────────────────────────────────────────────

fn render_charts(
    fb: &mut [u32],
    win_w: usize,
    win_h: usize,
    x0: usize,
    history: &VecDeque<[f64; 6]>,
) {
    let rows = 6;
    let row_h = win_h / rows;
    let pw = CHART_W.saturating_sub(2 * GAP);
    let vals: Vec<[f64; 6]> = history.iter().copied().collect();

    for d in 0..rows {
        let py0 = d * row_h + 2;
        let ph = row_h.saturating_sub(4);
        let px0 = x0 + GAP;

        // Panel background + border.
        fill_rect(fb, win_w, win_h, px0, py0, pw, ph, 0xFF101010);
        draw_rect(fb, win_w, win_h, px0, py0, pw, ph, 0xFF404040);

        // Per-DoF symmetric autoscale (floor avoids a flat line dominating).
        let m = vals
            .iter()
            .map(|v| v[d].abs())
            .fold(0.0_f64, f64::max)
            .max(if d < 3 { 0.05 } else { 0.05 });

        // Zero axis.
        let yz = (py0 + ph / 2) as i32;
        draw_hline(fb, win_w, win_h, px0, px0 + pw, yz, 0xFF303030);

        // Label tag (a short colored bar; text-free framebuffer).
        for t in 0..6usize {
            fill_rect(fb, win_w, win_h, px0 + 3, py0 + 3 + t, 8, 1, DOF_COLORS[d]);
        }
        let _ = DOF_LABELS;

        if vals.len() >= 2 {
            let map_y = |v: f64| -> i32 {
                let half = (ph / 2).saturating_sub(2) as f64;
                (py0 + ph / 2) as i32 - ((v / m) * half).round() as i32
            };
            let n = vals.len();
            for k in 1..n {
                let xa = px0 + (k - 1) * (pw - 1) / (n - 1);
                let xb = px0 + k * (pw - 1) / (n - 1);
                draw_line(
                    fb,
                    win_w,
                    win_h,
                    xa as i32,
                    map_y(vals[k - 1][d]),
                    xb as i32,
                    map_y(vals[k][d]),
                    DOF_COLORS[d],
                );
            }
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
