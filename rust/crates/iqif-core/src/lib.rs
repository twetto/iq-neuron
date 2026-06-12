//! Bit-exact Rust port of the IQIF (Integer Quadratic Integrate-and-Fire)
//! neuron and network from the C++ reference (`src/iq_neuron.cpp`,
//! `src/iq_network.cpp`).
//!
//! Everything is plain `i32` integer arithmetic, mirroring C++ `int`:
//!   - `>>` on signed ints is arithmetic shift in both languages,
//!   - `/` and `%` truncate toward zero in both languages,
//! so the dynamics reproduce the C++ output exactly (for the deterministic
//! `noise <= 1` case used by the test suite).
//!
//! This module is deliberately free of any Python/PyO3 dependency so a future
//! GPU (wgpu) backend can build on the same reference semantics.

/// Post-synaptic current accumulator with timer-gated bit-shift decay.
/// Mirror of the C++ `SynapseGroup` struct.
#[derive(Clone, Debug)]
pub struct SynapseGroup {
    pub current_accumulator: i32,
    pub timer: i32,
    pub timer_threshold: i32,
    pub apparent_tau: i32,
    pub surrogate_tau: i32,
    pub decay_shift_k: i32,
}

impl SynapseGroup {
    pub fn new() -> Self {
        SynapseGroup {
            current_accumulator: 0,
            timer: 0,
            timer_threshold: 0,
            apparent_tau: 0,
            surrogate_tau: 0,
            decay_shift_k: 0,
        }
    }

    pub fn init(&mut self, app_tau: i32, sur_tau: i32) {
        self.apparent_tau = app_tau;
        self.surrogate_tau = sur_tau;
        self.current_accumulator = 0;
        self.timer = 0;
        self.recalculate_params();
    }

    pub fn recalculate_params(&mut self) {
        // surrogate_tau is preferred to be a power of 2 (2, 4, 8, 16, ...).
        if self.surrogate_tau <= 0 {
            self.surrogate_tau = 1;
        }
        // C++: (int)log2((float)surrogate_tau). Use f64 so exact powers of two
        // (the intended inputs) land precisely on the integer boundary.
        self.decay_shift_k = (self.surrogate_tau as f64).log2() as i32;

        if self.apparent_tau <= self.surrogate_tau {
            self.timer_threshold = 0;
        } else {
            // C++ stores these intermediates as `float`; mirror that width so
            // truncation to int matches the reference.
            let decay_factor: f32 = 1.0 - (1.0 / ((1i32 << self.decay_shift_k) as f32));
            let num: f32 = (decay_factor as f64).log10() as f32;
            let den: f32 =
                (((self.apparent_tau as f32 - 1.0) / self.apparent_tau as f32) as f64).log10() as f32;

            if den == 0.0 {
                self.timer_threshold = 0;
            } else {
                self.timer_threshold = (num / den) as i32;
            }
        }
    }

    pub fn set_surrogate_tau(&mut self, s_tau: i32) {
        self.surrogate_tau = s_tau;
        self.recalculate_params();
    }

    pub fn set_apparent_tau(&mut self, app_tau: i32) {
        self.apparent_tau = app_tau;
        self.recalculate_params();
    }

    pub fn step(&mut self) {
        if self.timer > self.timer_threshold {
            let decay = self.current_accumulator >> self.decay_shift_k;
            if decay != 0 {
                self.current_accumulator -= decay;
            } else {
                // "Leak by 1" for small magnitudes.
                if self.current_accumulator > 0 {
                    self.current_accumulator -= 1;
                } else if self.current_accumulator < 0 {
                    self.current_accumulator += 1;
                }
            }
            self.timer = 0;
        }
        self.timer += 1;
    }

    pub fn add_input(&mut self, weight: i32) {
        self.current_accumulator += weight;
    }
}

/// Minimal LCG so `noise > 1` neurons are still stochastic. This does NOT
/// reproduce C `rand()`, so it is only exercised when noise actually matters;
/// the deterministic test suite uses `noise <= 1`, where the noise term is 0.
#[derive(Clone, Debug)]
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> i32 {
        // Numerical Recipes constants; returns a value in [0, 2^31).
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 1) as i32
    }
}

/// Single IQIF neuron. Mirror of the C++ `iq_neuron` class.
#[derive(Clone, Debug)]
pub struct IqNeuron {
    t_neuron: i32,
    rest: i32,
    threshold: i32,
    shift_a: i32,
    shift_b: i32,
    reset: i32,
    noise: i32,
    x: i32,
    f_min: i32,
    spike_count: i32,
    vmax: i32,
    vmin: i32,
    is_set: bool,
    pub is_firing: bool,
    pub synapse: SynapseGroup,
    rng: Lcg,
}

impl IqNeuron {
    pub fn new() -> Self {
        IqNeuron {
            t_neuron: 0,
            rest: 0,
            threshold: 0,
            shift_a: 0,
            shift_b: 0,
            reset: 0,
            noise: 1,
            x: 0,
            f_min: 0,
            spike_count: 0,
            vmax: 255,
            vmin: 0,
            is_set: false,
            is_firing: false,
            synapse: SynapseGroup::new(),
            rng: Lcg(0x12345678),
        }
    }

    pub fn is_set(&self) -> bool {
        self.is_set
    }

    pub fn set(&mut self, rest: i32, threshold: i32, reset: i32, shift_a: i32, shift_b: i32, noise: i32) {
        self.x = rest; // initialize with rest potential
        self.t_neuron = 0;

        // f_min = (a*rest + b*threshold)/(a+b), with a = 1/(1<<shift_a),
        // b = 1/(1<<shift_b). Cross-multiplying by the shifts:
        let weight_a = 1i32 << shift_b; // weight for rest
        let weight_b = 1i32 << shift_a; // weight for threshold
        self.f_min = (weight_a * rest + weight_b * threshold) / (weight_a + weight_b);

        self.shift_a = shift_a;
        self.shift_b = shift_b;
        self.rest = rest;
        self.threshold = threshold;
        self.reset = reset;

        let mut noise = noise;
        if noise == 0 {
            noise += 1;
        } else if noise < 0 {
            noise = -noise;
        }
        self.noise = noise;
        self.is_set = true;
        self.synapse.init(32, 8); // default; network init overrides
    }

    pub fn set_vmax(&mut self, vmax: i32) {
        self.vmax = vmax;
    }

    pub fn set_vmin(&mut self, vmin: i32) {
        self.vmin = vmin;
    }

    pub fn update_state(&mut self, external_current: i32) {
        // Capture undecayed input from t-1, then decay for t+1.
        let current_val = self.synapse.current_accumulator;
        self.synapse.step();

        let total_input = current_val + external_current;

        let f = if self.x < self.f_min {
            (self.rest - self.x) >> self.shift_a
        } else {
            (self.x - self.threshold) >> self.shift_b
        };

        let noise_term = if self.noise > 1 {
            self.rng.next() % self.noise - (self.noise >> 1)
        } else {
            0
        };

        self.x += f + total_input + noise_term;

        self.is_firing = false;
        if self.x >= self.vmax {
            self.spike_count += 1;
            self.is_firing = true;
            self.x -= self.vmax - self.reset; // soft reset
        }
        if self.x < self.vmin {
            self.x = self.vmin;
        }
        self.t_neuron += 1;
    }

    pub fn receive_spike(&mut self, weight: i32) {
        self.synapse.add_input(weight);
    }

    pub fn set_synapse_tau(&mut self, apparent_tau: i32, s_tau: i32) {
        self.synapse.init(apparent_tau, s_tau);
    }

    pub fn set_surrogate_tau(&mut self, s_tau: i32) {
        self.synapse.set_surrogate_tau(s_tau);
    }

    pub fn get_surrogate_tau(&self) -> i32 {
        self.synapse.surrogate_tau
    }

    pub fn get_decay_threshold(&self) -> i32 {
        self.synapse.timer_threshold
    }

    pub fn potential(&self) -> i32 {
        self.x
    }

    pub fn set_potential(&mut self, value: i32) {
        self.x = value;
    }

    /// Reads and clears the spike counter (matches C++ side-effecting getter).
    pub fn spike_count(&mut self) -> i32 {
        let count = self.spike_count;
        self.spike_count = 0;
        count
    }

    pub fn spike_rate(&mut self) -> f32 {
        let denom = if self.t_neuron != 0 { self.t_neuron } else { 1 };
        let r = self.spike_count as f32 / denom as f32;
        self.t_neuron = 0;
        self.spike_count = 0;
        r
    }

    /// Internal LCG state, exposed so an alternate backend can carry forward
    /// the exact same noise stream. Only advances when `noise > 1`.
    pub fn rng_state(&self) -> u32 {
        self.rng.0
    }

    /// Flat copy of every field a GPU backend needs to reproduce this neuron's
    /// dynamics bit-for-bit. Plain data, no behavior — the GPU crate packs it
    /// into its own buffer layout. `is_firing` is 1/0; raw `spike_count` is the
    /// running count (not the side-effecting reset getter).
    pub fn snapshot(&self) -> NeuronSnapshot {
        NeuronSnapshot {
            rest: self.rest,
            threshold: self.threshold,
            shift_a: self.shift_a,
            shift_b: self.shift_b,
            reset: self.reset,
            noise: self.noise,
            f_min: self.f_min,
            vmax: self.vmax,
            vmin: self.vmin,
            timer_threshold: self.synapse.timer_threshold,
            decay_shift_k: self.synapse.decay_shift_k,
            x: self.x,
            accumulator: self.synapse.current_accumulator,
            timer: self.synapse.timer,
            is_firing: self.is_firing as i32,
            spike_count: self.spike_count,
            t_neuron: self.t_neuron,
            rng_state: self.rng.0,
        }
    }
}

/// Plain-data view of an [`IqNeuron`], the bridge to a non-CPU backend. Splits
/// naturally into setup params (constant after `set`) and mutable per-step
/// state, but is kept as one flat struct so the source of truth stays here.
#[derive(Clone, Copy, Debug)]
pub struct NeuronSnapshot {
    // Setup params (constant during a run).
    pub rest: i32,
    pub threshold: i32,
    pub shift_a: i32,
    pub shift_b: i32,
    pub reset: i32,
    pub noise: i32,
    pub f_min: i32,
    pub vmax: i32,
    pub vmin: i32,
    pub timer_threshold: i32,
    pub decay_shift_k: i32,
    // Mutable per-step state.
    pub x: i32,
    pub accumulator: i32,
    pub timer: i32,
    pub is_firing: i32,
    pub spike_count: i32,
    pub t_neuron: i32,
    pub rng_state: u32,
}

/// Transposed (CSC) adjacency: incoming edges grouped by *post*-synaptic
/// neuron. `offsets` has `num_neurons + 1` entries; for post `j`, the slice
/// `sources[offsets[j]..offsets[j+1]]` (with matching `weights`) lists the
/// presynaptic neuron and weight of each incoming edge. This is the gather dual
/// of the CSR scatter used by [`IqNetwork::send_synapse`].
#[derive(Clone, Debug)]
pub struct Csc {
    pub offsets: Vec<i32>,
    pub sources: Vec<i32>,
    pub weights: Vec<i32>,
}

/// IQIF network with CSR-stored synapses. Mirror of the C++ `iq_network`.
pub struct IqNetwork {
    num_neurons: usize,
    s_tau: i32,
    csr_offsets: Vec<i32>,
    csr_targets: Vec<i32>,
    csr_weights: Vec<i32>,
    biascurrent: Vec<i32>,
    neurons: Vec<IqNeuron>,
}

struct Conn {
    pre: i32,
    post: i32,
    weight: i32,
    tau: i32,
}

impl IqNetwork {
    /// Build a network from the raw text of a parameter file and a connection
    /// file (whitespace-separated integers, same formats as the C++ loader).
    pub fn from_text(par_text: &str, con_text: &str) -> Self {
        let par: Vec<i32> = par_text
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        // Parameter rows are 7 ints each; row count == neuron count.
        let num_neurons = par.len() / 7;

        let mut net = IqNetwork {
            num_neurons,
            s_tau: 8,
            csr_offsets: Vec::new(),
            csr_targets: Vec::new(),
            csr_weights: Vec::new(),
            biascurrent: vec![0; num_neurons],
            neurons: vec![IqNeuron::new(); num_neurons],
        };

        net.set_neurons(&par);
        net.get_weight(con_text);
        net
    }

    fn set_neurons(&mut self, par: &[i32]) {
        for row in par.chunks_exact(7) {
            let (i, rest, threshold, reset, shift_a, shift_b, noise) =
                (row[0], row[1], row[2], row[3], row[4], row[5], row[6]);
            if (i as usize) < self.num_neurons {
                self.neurons[i as usize].set(rest, threshold, reset, shift_a, shift_b, noise);
            }
        }
    }

    fn get_weight(&mut self, con_text: &str) {
        let nums: Vec<i32> = con_text
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();

        let mut raw: Vec<Conn> = nums
            .chunks_exact(4)
            .map(|c| Conn { pre: c[0], post: c[1], weight: c[2], tau: c[3] })
            .collect();

        // Apply per-neuron tau in file order (last connection to a given post
        // neuron wins), matching the C++ loader which does this before sorting.
        let s_tau = self.s_tau;
        for c in &raw {
            self.neurons[c.post as usize].set_synapse_tau(c.tau, s_tau);
        }

        // Group by presynaptic neuron. (Order within a group is irrelevant:
        // propagation just sums weights into accumulators.)
        raw.sort_by_key(|c| c.pre);

        let num_synapses = raw.len();
        self.csr_offsets = vec![0; self.num_neurons + 1];
        self.csr_targets = vec![0; num_synapses];
        self.csr_weights = vec![0; num_synapses];

        let mut current_pre: usize = 0;
        for (k, c) in raw.iter().enumerate() {
            while current_pre < c.pre as usize {
                self.csr_offsets[current_pre + 1] = k as i32;
                current_pre += 1;
            }
            self.csr_targets[k] = c.post;
            self.csr_weights[k] = c.weight;
        }
        while current_pre < self.num_neurons {
            self.csr_offsets[current_pre + 1] = num_synapses as i32;
            current_pre += 1;
        }
    }

    pub fn num_neurons(&self) -> i32 {
        self.num_neurons as i32
    }

    /// Per-neuron flat snapshots, index == neuron id. The bridge a GPU backend
    /// uploads from (single source of truth for setup + initial state).
    pub fn neuron_snapshots(&self) -> Vec<NeuronSnapshot> {
        self.neurons.iter().map(|n| n.snapshot()).collect()
    }

    /// Per-neuron bias current, index == neuron id.
    pub fn biascurrents(&self) -> &[i32] {
        &self.biascurrent
    }

    /// Transpose the CSR adjacency into CSC (group incoming edges by post). The
    /// GPU `propagate` kernel gathers over this instead of scattering over CSR,
    /// so spike propagation is race-free and order-independent. Within a post
    /// group, edge order is irrelevant — propagation only sums weights.
    pub fn build_csc(&self) -> Csc {
        let n = self.num_neurons;
        let m = self.csr_targets.len();

        // Count incoming edges per post neuron, then prefix-sum to offsets.
        let mut offsets = vec![0i32; n + 1];
        for &post in &self.csr_targets {
            offsets[post as usize + 1] += 1;
        }
        for j in 0..n {
            offsets[j + 1] += offsets[j];
        }

        // Scatter each CSR edge (pre -> post, weight) into its post bucket.
        let mut sources = vec![0i32; m];
        let mut weights = vec![0i32; m];
        let mut cursor: Vec<i32> = offsets[..n].to_vec();
        for pre in 0..n {
            let start = self.csr_offsets[pre] as usize;
            let end = self.csr_offsets[pre + 1] as usize;
            for k in start..end {
                let post = self.csr_targets[k] as usize;
                let slot = cursor[post] as usize;
                sources[slot] = pre as i32;
                weights[slot] = self.csr_weights[k];
                cursor[post] += 1;
            }
        }

        Csc { offsets, sources, weights }
    }

    fn in_range(&self, i: i32) -> bool {
        i >= 0 && (i as usize) < self.num_neurons
    }

    pub fn send_synapse(&mut self) {
        // Phase 1: propagate spikes using is_firing from t-1.
        for i in 0..self.num_neurons {
            if self.neurons[i].is_firing {
                let start = self.csr_offsets[i] as usize;
                let end = self.csr_offsets[i + 1] as usize;
                for k in start..end {
                    let target = self.csr_targets[k] as usize;
                    let weight = self.csr_weights[k];
                    self.neurons[target].receive_spike(weight);
                }
            }
        }
        // Phase 2: decay + solve + set firing for t.
        for i in 0..self.num_neurons {
            let bias = self.biascurrent[i];
            self.neurons[i].update_state(bias);
        }
    }

    pub fn set_biascurrent(&mut self, i: i32, biascurrent: i32) -> i32 {
        if self.in_range(i) {
            self.biascurrent[i as usize] = biascurrent;
            1
        } else {
            0
        }
    }

    pub fn set_neuron(
        &mut self,
        i: i32,
        rest: i32,
        threshold: i32,
        reset: i32,
        shift_a: i32,
        shift_b: i32,
        noise: i32,
    ) -> i32 {
        if self.in_range(i) {
            self.neurons[i as usize].set(rest, threshold, reset, shift_a, shift_b, noise);
            1
        } else {
            0
        }
    }

    pub fn set_weight(&mut self, pre: i32, post: i32, weight: i32, tau: i32) -> i32 {
        if !self.in_range(pre) || !self.in_range(post) {
            return 0;
        }
        let start = self.csr_offsets[pre as usize] as usize;
        let end = self.csr_offsets[pre as usize + 1] as usize;
        for k in start..end {
            if self.csr_targets[k] == post {
                self.csr_weights[k] = weight;
                let s_tau = self.s_tau;
                self.neurons[post as usize].set_synapse_tau(tau, s_tau);
                return 1;
            }
        }
        0
    }

    pub fn set_surrogate_tau_all(&mut self, s_tau: i32) -> i32 {
        self.s_tau = s_tau;
        for n in &mut self.neurons {
            n.set_surrogate_tau(s_tau);
        }
        1
    }

    pub fn set_surrogate_tau_one(&mut self, i: i32, s_tau: i32) -> i32 {
        if self.in_range(i) {
            self.neurons[i as usize].set_surrogate_tau(s_tau);
            1
        } else {
            0
        }
    }

    pub fn get_surrogate_tau(&self, i: i32) -> i32 {
        self.neurons[i as usize].get_surrogate_tau()
    }

    pub fn get_current_accumulator(&self, i: i32) -> i32 {
        if self.in_range(i) {
            self.neurons[i as usize].synapse.current_accumulator
        } else {
            0
        }
    }

    pub fn set_current_accumulator(&mut self, i: i32, value: i32) -> i32 {
        if self.in_range(i) {
            self.neurons[i as usize].synapse.current_accumulator = value;
            1
        } else {
            0
        }
    }

    pub fn get_all_current_accumulators(&self) -> Vec<i32> {
        self.neurons.iter().map(|n| n.synapse.current_accumulator).collect()
    }

    pub fn set_all_current_accumulators(&mut self, values: &[i32]) {
        for (n, &v) in self.neurons.iter_mut().zip(values.iter()) {
            n.synapse.current_accumulator = v;
        }
    }

    pub fn get_decay_threshold(&self, i: i32) -> i32 {
        if self.in_range(i) {
            self.neurons[i as usize].get_decay_threshold()
        } else {
            0
        }
    }

    pub fn set_vmax(&mut self, i: i32, vmax: i32) -> i32 {
        if self.in_range(i) {
            self.neurons[i as usize].set_vmax(vmax);
            0
        } else {
            1
        }
    }

    pub fn set_vmin(&mut self, i: i32, vmin: i32) -> i32 {
        if self.in_range(i) {
            self.neurons[i as usize].set_vmin(vmin);
            0
        } else {
            1
        }
    }

    pub fn potential(&self, i: i32) -> i32 {
        self.neurons[i as usize].potential()
    }

    pub fn set_potential(&mut self, i: i32, value: i32) -> i32 {
        if self.in_range(i) {
            self.neurons[i as usize].set_potential(value);
            1
        } else {
            0
        }
    }

    pub fn get_is_firing(&self, i: i32) -> i32 {
        if self.in_range(i) && self.neurons[i as usize].is_firing {
            1
        } else {
            0
        }
    }

    pub fn set_is_firing(&mut self, i: i32, value: i32) -> i32 {
        if self.in_range(i) {
            self.neurons[i as usize].is_firing = value != 0;
            1
        } else {
            0
        }
    }

    pub fn get_synapse_timer(&self, i: i32) -> i32 {
        if self.in_range(i) {
            self.neurons[i as usize].synapse.timer
        } else {
            0
        }
    }

    pub fn set_synapse_timer(&mut self, i: i32, value: i32) -> i32 {
        if self.in_range(i) {
            self.neurons[i as usize].synapse.timer = value;
            1
        } else {
            0
        }
    }

    pub fn spike_count(&mut self, i: i32) -> i32 {
        self.neurons[i as usize].spike_count()
    }

    pub fn get_all_spike_counts(&mut self) -> Vec<i32> {
        self.neurons.iter_mut().map(|n| n.spike_count()).collect()
    }

    pub fn spike_rate(&mut self, i: i32) -> f32 {
        self.neurons[i as usize].spike_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_toward_zero_leaks_by_one_near_zero() {
        // tau=8 default surrogate -> decay_shift_k=3, threshold=0.
        let mut s = SynapseGroup::new();
        s.init(8, 8);
        s.current_accumulator = 200;
        // First step: timer(0) not > threshold(0), so no decay, timer->1.
        s.step();
        assert_eq!(s.current_accumulator, 200);
        // Subsequent steps decay by acc>>3 each time.
        s.step();
        assert_eq!(s.current_accumulator, 200 - (200 >> 3));
    }

    #[test]
    fn csc_is_the_transpose_of_csr() {
        // 3 neurons; edges 0->1, 0->2, 2->1 (weights 10, 20, 30).
        let par = "0 128 255 128 8 8 0\n1 128 255 128 8 8 0\n2 128 255 128 8 8 0\n";
        let con = "0 1 10 8\n0 2 20 8\n2 1 30 8\n";
        let net = IqNetwork::from_text(par, con);
        let csc = net.build_csc();

        // Collect (source, weight) incoming to each post, order-independent.
        let incoming = |post: usize| {
            let mut v: Vec<(i32, i32)> = (csc.offsets[post]..csc.offsets[post + 1])
                .map(|k| (csc.sources[k as usize], csc.weights[k as usize]))
                .collect();
            v.sort();
            v
        };
        assert_eq!(incoming(0), vec![]); // nothing targets neuron 0
        assert_eq!(incoming(1), vec![(0, 10), (2, 30)]);
        assert_eq!(incoming(2), vec![(0, 20)]);
        assert_eq!(csc.offsets, vec![0, 0, 2, 3]);
    }

    #[test]
    fn subthreshold_integrator_accumulates_without_firing() {
        // shift_a=15, shift_b=1: leak is negligible, neuron integrates input.
        let par = "0 128 255 128 15 1 0\n";
        let con = "0 0 0 8\n"; // self-loop placeholder, weight 0
        let mut net = IqNetwork::from_text(par, con);
        net.set_biascurrent(0, 5);
        let start = net.potential(0);
        for _ in 0..10 {
            net.send_synapse();
        }
        assert!(net.potential(0) > start);
        assert!(net.potential(0) < 255);
    }
}
