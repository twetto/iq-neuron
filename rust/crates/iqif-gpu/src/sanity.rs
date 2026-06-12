//! Phase-2 validation: prove WGSL `i32` `>>`, `/`, `%` (and wrapping `*`) match
//! the CPU on *negative* operands, on whatever device wgpu picks here.
//!
//! This is the "step 0" gate from `rust/PLAN.md`: nothing else in the GPU
//! backend is worth writing until the actual driver is confirmed to honor the
//! integer semantics the bit-exact port depends on. It is intentionally
//! device-agnostic — it must pass identically on the NVIDIA (Vulkan/DX12) and
//! RADV (Vulkan) boxes this project is developed across.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// One operand pair fed to the sanity kernel. Layout mirrors the WGSL `Pair`
/// (16 bytes, std430-compatible).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Pair {
    a: i32,
    b: i32,
    shift: u32,
    _pad: u32,
}

/// A single lane where the device disagreed with the CPU.
#[derive(Clone, Copy)]
pub struct Mismatch {
    pub a: i32,
    pub b: i32,
    pub shift: u32,
    pub op: &'static str,
    pub gpu: i32,
    pub cpu: i32,
}

/// Result of a sanity run: the device it ran on, how many pairs were checked,
/// and every lane that diverged (empty == bit-exact, the Phase-2 pass criterion).
pub struct SanityReport {
    pub adapter: String,
    pub backend: String,
    pub pairs_checked: usize,
    pub mismatches: Vec<Mismatch>,
}

/// Build the operand set: a hand-picked sign-mixed grid that exercises the
/// edge cases where naive (logical-shift / round-toward-negative-infinity)
/// implementations would diverge from C/Rust. Excludes the two WGSL
/// integer-indeterminate cases (`b == 0`, and `INT_MIN / -1`).
fn build_pairs() -> Vec<Pair> {
    let operands: [i32; 13] = [
        i32::MIN,
        -1_000_000_007,
        -65_536,
        -255,
        -8,
        -3,
        -1,
        0,
        1,
        3,
        255,
        65_536,
        i32::MAX,
    ];
    let shifts: [u32; 6] = [0, 1, 3, 7, 16, 31];

    let mut pairs = Vec::new();
    let mut s = 0usize;
    for &a in &operands {
        for &b in &operands {
            // Skip the cases WGSL explicitly leaves indeterminate.
            if b == 0 || (a == i32::MIN && b == -1) {
                continue;
            }
            let shift = shifts[s % shifts.len()];
            s += 1;
            pairs.push(Pair { a, b, shift, _pad: 0 });
        }
    }
    pairs
}

/// What the device *should* produce for a pair, computed with Rust's i32
/// semantics (which match the WGSL spec). `wrapping_*` mirrors WGSL's modulo-2^32
/// arithmetic and keeps the host side panic-free in debug builds. Lanes mirror
/// the WGSL `Out` struct; lane 4 (`rem_via_div`) must equal lane 2 (`rem`).
fn expected(p: &Pair) -> [i32; 8] {
    let rem = p.a.wrapping_rem(p.b);
    [
        p.a >> p.shift,
        p.a.wrapping_div(p.b),
        rem,
        p.a.wrapping_mul(p.b),
        rem, // rem_via_div: a - (a/b)*b is identical to truncated rem
        0,
        0,
        0,
    ]
}

/// Lane index -> op name, matching the WGSL `Out` struct order.
const LANE_OPS: [&str; 8] = ["shr", "div", "rem", "mul", "rem_via_div", "", "", ""];

/// Run the sanity kernel on whatever backend wgpu picks by default.
///
/// Returns `Ok(report)` with any divergences listed, or `Err(msg)` if no
/// adapter is available. Synchronous: blocks on the GPU via `pollster`.
pub fn check_integer_semantics() -> Result<SanityReport, String> {
    check_integer_semantics_on(wgpu::Backends::all())
}

/// Same as [`check_integer_semantics`] but pins the wgpu backend(s), so callers
/// can compare e.g. DX12 vs Vulkan codegen of the synthesized integer ops on
/// the same physical GPU.
pub fn check_integer_semantics_on(backends: wgpu::Backends) -> Result<SanityReport, String> {
    let pairs = build_pairs();

    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| format!("no compatible GPU adapter: {e}"))?;

    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("iqif-sanity-device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| format!("request_device failed: {e}"))?;

    let pairs_bytes = bytemuck::cast_slice(&pairs);
    let results_size = (pairs.len() * std::mem::size_of::<[i32; 8]>()) as u64;

    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sanity-pairs"),
        contents: pairs_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sanity-results"),
        size: results_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sanity-readback"),
        size: results_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sanity-kernel"),
        source: wgpu::ShaderSource::Wgsl(include_str!("sanity.wgsl").into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sanity-pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sanity-bind-group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sanity-encoder") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sanity-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = pairs.len().div_ceil(64) as u32;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, results_size);
    queue.submit(Some(encoder.finish()));

    // Map and wait. poll(Wait) blocks until the GPU is done with the readback.
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("device poll failed: {e}"))?;

    let data = slice.get_mapped_range();
    let got: &[[i32; 8]] = bytemuck::cast_slice(&data);

    let mut mismatches = Vec::new();
    for (p, g) in pairs.iter().zip(got.iter()) {
        let want = expected(p);
        for k in 0..8 {
            if LANE_OPS[k].is_empty() {
                continue; // padding lane
            }
            if g[k] != want[k] {
                mismatches.push(Mismatch {
                    a: p.a,
                    b: p.b,
                    shift: p.shift,
                    op: LANE_OPS[k],
                    gpu: g[k],
                    cpu: want[k],
                });
            }
        }
    }
    drop(data);
    readback.unmap();

    Ok(SanityReport {
        adapter: info.name,
        backend: format!("{:?}", info.backend),
        pairs_checked: pairs.len(),
        mismatches,
    })
}

/// Operations the GPU kernels are allowed to emit. Bit-exactness here is the
/// Phase-2 pass criterion. Raw signed `%` is deliberately absent: it is broken
/// on NVIDIA+Vulkan for negative operands (see `RELIED_ON_OPS` note below), so
/// the kernels compute remainder as `rem_via_div` (`a - (a/b)*b`) instead.
pub const RELIED_ON_OPS: [&str; 4] = ["shr", "div", "mul", "rem_via_div"];

#[cfg(test)]
mod tests {
    use super::*;

    /// Run on one backend; return the report, or `None` if no adapter exists
    /// for it (e.g. DX12 on Linux). Logs a full characterization either way.
    fn report(label: &str, backends: wgpu::Backends) -> Option<SanityReport> {
        let r = match check_integer_semantics_on(backends) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[{label}] no adapter: {e}");
                return None;
            }
        };
        eprintln!(
            "[{label}] {} pairs on {} ({}); {} mismatching lanes",
            r.pairs_checked,
            r.adapter,
            r.backend,
            r.mismatches.len()
        );
        for op in ["shr", "div", "rem", "mul", "rem_via_div"] {
            let n = r.mismatches.iter().filter(|m| m.op == op).count();
            if n > 0 {
                let tag = if RELIED_ON_OPS.contains(&op) { "RELIED-ON" } else { "unused" };
                eprintln!("  [{label}] {op}: {n} mismatches ({tag})");
            }
        }
        Some(r)
    }

    /// Phase-2 gate: on every backend an adapter exists for, the ops the kernels
    /// actually emit (`RELIED_ON_OPS`) must be bit-exact with the CPU. Raw `%`
    /// is allowed to diverge — we never emit it. Skips (does not fail) when no
    /// GPU is present at all, so it is safe in headless CI.
    #[test]
    fn relied_on_ops_are_bit_exact_on_every_backend() {
        let reports = [
            report("vulkan", wgpu::Backends::VULKAN),
            report("dx12", wgpu::Backends::DX12),
            report("metal", wgpu::Backends::METAL),
            report("gl", wgpu::Backends::GL),
        ];

        let mut any = false;
        for r in reports.into_iter().flatten() {
            any = true;
            let bad: Vec<_> = r
                .mismatches
                .iter()
                .filter(|m| RELIED_ON_OPS.contains(&m.op))
                .collect();
            assert!(
                bad.is_empty(),
                "{} ({}): {} relied-on-op mismatches, first: {} a={} b={} -> GPU {} != CPU {}",
                r.adapter,
                r.backend,
                bad.len(),
                bad[0].op,
                bad[0].a,
                bad[0].b,
                bad[0].gpu,
                bad[0].cpu,
            );
        }

        if !any {
            eprintln!("no GPU adapter on any backend; Phase-2 gate skipped");
        }
    }
}
