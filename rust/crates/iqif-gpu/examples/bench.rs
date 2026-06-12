//! Phase-5 throughput benchmark: CPU (`iqif_core`) vs resident GPU
//! (`iqif_gpu`), plus a workgroup-size sweep.
//!
//! Run in release (debug timings are meaningless and the core panics on
//! overflow):
//!
//! ```text
//! cargo run --release -p iqif-gpu --example bench
//! ```
//!
//! The GPU is timed the way it is meant to be used: state stays resident,
//! `run(steps)` queues every step, and we sync once at the end (a single bulk
//! readback). Per-neuron getters in the hot loop would instead pay a PCIe
//! round-trip per call and dominate — exactly what the cache in normal use, and
//! this benchmark's structure, avoid.

use iqif_core::IqNetwork;
use iqif_gpu::GpuNetwork;
use std::fmt::Write as _;
use std::time::Instant;

/// Generate a random sparse network as (params, connections) text: `n` neurons,
/// each projecting to `fan_out` uniformly-random targets (weight 1, tau 8).
/// Bounded, deterministic dynamics; an LCG keeps it dependency-free.
fn gen_network(n: usize, fan_out: usize, seed: u64) -> (String, String) {
    let mut par = String::with_capacity(n * 20);
    for i in 0..n {
        // rest=0 threshold=128 reset=128 shift_a=15 shift_b=1 noise=0
        let _ = writeln!(par, "{i} 0 128 128 15 1 0");
    }

    let mut rng = seed | 1;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng >> 33) as usize
    };

    let mut con = String::with_capacity(n * fan_out * 12);
    for s in 0..n {
        for _ in 0..fan_out {
            let t = next() % n;
            let _ = writeln!(con, "{s} {t} 1 8");
        }
    }
    (par, con)
}

/// A core driven so both backends do real propagation work (bias on every
/// neuron makes them fire, so the CPU scatter isn't trivially skipped).
fn build_core(n: usize, fan_out: usize) -> IqNetwork {
    let (par, con) = gen_network(n, fan_out, 0xC0FFEE);
    let mut core = IqNetwork::from_text(&par, &con);
    for i in 0..n {
        core.set_biascurrent(i as i32, 20);
    }
    core
}

fn time_cpu(core: &mut IqNetwork, steps: usize) -> f64 {
    let t = Instant::now();
    for _ in 0..steps {
        core.send_synapse();
    }
    t.elapsed().as_secs_f64()
}

/// Time `steps` resident GPU steps; the final bulk read forces a full sync so
/// the elapsed time includes all queued compute.
fn time_gpu(gpu: &GpuNetwork, steps: usize) -> f64 {
    let t = Instant::now();
    gpu.run(steps);
    let _ = gpu.potentials(); // single readback == device sync
    t.elapsed().as_secs_f64()
}

fn mupd_per_s(n: usize, steps: usize, secs: f64) -> f64 {
    (n as f64 * steps as f64) / secs / 1.0e6
}

fn main() {
    let steps = 200;
    let fan_out = 16;

    println!("IQIF throughput — CPU vs resident GPU (RTX 3060 class)");
    println!("steps={steps}, fan_out={fan_out}, metric = million neuron-updates/sec\n");
    println!("{:>9}  {:>10}  {:>12}  {:>12}  {:>9}", "N", "edges", "CPU Mupd/s", "GPU Mupd/s", "speedup");

    for &n in &[1_000usize, 10_000, 100_000, 500_000] {
        let mut core = build_core(n, fan_out);
        let gpu = match GpuNetwork::from_core(&core) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("GPU unavailable: {e}");
                return;
            }
        };
        gpu.run(10); // warm up pipelines/allocator
        let _ = gpu.potentials();

        let cpu_s = time_cpu(&mut core, steps);
        let gpu_s = time_gpu(&gpu, steps);
        println!(
            "{:>9}  {:>10}  {:>12.1}  {:>12.1}  {:>8.2}x",
            n,
            n * fan_out,
            mupd_per_s(n, steps, cpu_s),
            mupd_per_s(n, steps, gpu_s),
            cpu_s / gpu_s,
        );
    }

    // ── workgroup-size sweep at the largest size ─────────────────────────
    let n = 500_000;
    let core = build_core(n, fan_out);
    println!("\nWorkgroup sweep at N={n}:");
    println!("{:>7}  {:>12}", "wg_size", "GPU Mupd/s");
    for &wg in &[32u32, 64, 128, 256] {
        let gpu = match GpuNetwork::from_core_with_workgroup(&core, wg) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("GPU unavailable: {e}");
                return;
            }
        };
        gpu.run(10);
        let _ = gpu.potentials();
        let s = time_gpu(&gpu, steps);
        println!("{:>7}  {:>12.1}", wg, mupd_per_s(n, steps, s));
    }
}
