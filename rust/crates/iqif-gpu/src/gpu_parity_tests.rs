//! Phase-3 parity: the GPU two-pass timestep must reproduce the CPU reference
//! (`iqif_core`) bit-for-bit. We build both from the *same* core, step them in
//! lockstep, and compare resident state each step. Skips (does not fail) when
//! no GPU adapter is present, so it is safe in headless CI.

use super::*;

/// True when the error means "this machine has no usable GPU", in which case
/// the parity test should skip rather than fail.
fn is_no_adapter(err: &str) -> bool {
    err.contains("no compatible GPU adapter")
}

fn cpu_potentials(core: &IqNetwork) -> Vec<i32> {
    (0..core.num_neurons()).map(|i| core.potential(i)).collect()
}
fn cpu_accumulators(core: &IqNetwork) -> Vec<i32> {
    (0..core.num_neurons()).map(|i| core.get_current_accumulator(i)).collect()
}
fn cpu_firing(core: &IqNetwork) -> Vec<i32> {
    (0..core.num_neurons()).map(|i| core.get_is_firing(i)).collect()
}

/// Step CPU core and GPU network in lockstep for `steps`, asserting potentials,
/// accumulators, and is_firing match every step. Returns total spikes observed
/// (CPU side) so callers can assert the firing path was actually exercised.
fn run_parity(mut core: IqNetwork, steps: usize) -> Option<i64> {
    let gpu = match GpuNetwork::from_core(&core) {
        Ok(g) => g,
        Err(e) if is_no_adapter(&e) => {
            eprintln!("skipping GPU parity: {e}");
            return None;
        }
        Err(e) => panic!("GpuNetwork build failed: {e}"),
    };

    let mut total_spikes: i64 = 0;
    for step in 0..steps {
        core.send_synapse();
        gpu.step();

        let (cp, gp) = (cpu_potentials(&core), gpu.potentials());
        let (ca, ga) = (cpu_accumulators(&core), gpu.accumulators());
        let (cf, gf) = (cpu_firing(&core), gpu.is_firing());

        assert_eq!(cp, gp, "potential mismatch at step {step}");
        assert_eq!(ca, ga, "accumulator mismatch at step {step}");
        assert_eq!(cf, gf, "is_firing mismatch at step {step}");

        total_spikes += gf.iter().map(|&f| f as i64).sum::<i64>();
    }
    Some(total_spikes)
}

#[test]
fn feedforward_network_with_firing_matches_cpu() {
    // Feed-forward chain 0->1->2. Neuron 0 is driven by bias so it fires and
    // propagates downstream, exercising the CSC gather. No excitatory loop:
    // with the >>3 synapse decay amplifying sustained input ~8x, a ring would
    // run x past the soft reset and overflow i32 (the debug CPU core panics on
    // overflow, while WGSL `+` wraps — they only agree while in range).
    let par = "0 0 128 128 15 1 0\n1 0 128 128 15 1 0\n2 0 128 128 15 1 0\n";
    let con = "0 1 8 8\n1 2 8 8\n";
    let mut core = IqNetwork::from_text(par, con);
    core.set_biascurrent(0, 25);

    if let Some(spikes) = run_parity(core, 200) {
        assert!(spikes > 0, "test network never fired; propagate path not exercised");
        eprintln!("feed-forward parity OK over 200 steps, {spikes} spikes");
    }
}

#[test]
fn noisy_neuron_matches_cpu_lcg() {
    // Single neuron with noise > 1, so the per-step LCG noise term is active.
    // Parity here proves the GPU replicates the CPU's integer noise stream.
    let par = "0 0 128 128 15 1 37\n";
    let con = "0 0 0 8\n"; // self-loop, weight 0 (placeholder)
    let mut core = IqNetwork::from_text(par, con);
    core.set_biascurrent(0, 20);

    if let Some(_spikes) = run_parity(core, 500) {
        eprintln!("noisy-neuron LCG parity OK over 500 steps");
    }
}
