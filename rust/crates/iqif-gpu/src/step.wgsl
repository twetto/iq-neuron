// One IQIF timestep, split into two compute entry points that share one bind
// group (see rust/PLAN.md, Phase 3). The host runs them as two passes per step:
//
//   1. propagate   — gather incoming spikes over the transposed (CSC) adjacency
//   2. update_state — decay synapse, integrate, fire/reset
//
// The pass boundary serializes them, so `is_firing` written by step N's
// `update_state` is exactly what step N+1's `propagate` reads. All state stays
// resident in `state`; the host only reads it back in bulk on demand.
//
// Bit-exactness rules (Phase 2 finding): never emit raw signed `%` — NVIDIA
// Vulkan/GL miscompile it for negatives. The one remainder here (noise) is done
// as `r - (r/noise)*noise`. `>>` on i32 is the validated arithmetic shift.

// Constant per-neuron setup. Packed (12 i32) to fit storage-buffer limits;
// mirrors the host `GpuParams`.
struct Params {
    rest: i32,
    threshold: i32,
    shift_a: i32,
    shift_b: i32,
    reset: i32,
    noise: i32,
    f_min: i32,
    vmax: i32,
    vmin: i32,
    timer_threshold: i32,
    decay_shift_k: i32,
    biascurrent: i32,
};

// Mutable per-neuron state (8 words; mirrors host `GpuState`).
struct State {
    x: i32,
    accumulator: i32,
    timer: i32,
    is_firing: i32,
    spike_count: i32,
    t_neuron: i32,
    rng_state: u32,
    _pad: u32,
};

struct Meta {
    num_neurons: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read>        params: array<Params>;
@group(0) @binding(1) var<storage, read_write>  state: array<State>;
@group(0) @binding(2) var<storage, read>        csc_offsets: array<i32>;
@group(0) @binding(3) var<storage, read>        csc_edges: array<vec2<i32>>; // (source, weight)
@group(0) @binding(4) var<uniform>              dims: Meta;

// Gather: each post-neuron sums the weights of incoming edges whose source
// fired last step, then adds the sum into its accumulator. Race-free (each post
// is written by exactly one invocation) and deterministic (integer add is
// associative), matching the CSR scatter in the CPU reference.
@compute @workgroup_size(64)
fn propagate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let post = gid.x;
    if (post >= dims.num_neurons) {
        return;
    }
    let start = csc_offsets[post];
    let end = csc_offsets[post + 1u];
    var sum: i32 = 0;
    for (var k: i32 = start; k < end; k = k + 1) {
        let e = csc_edges[k];           // e.x = source, e.y = weight
        if (state[e.x].is_firing != 0) {
            sum = sum + e.y;
        }
    }
    if (sum != 0) {
        state[post].accumulator = state[post].accumulator + sum;
    }
}

// Faithful port of `IqNeuron::update_state` (+ `SynapseGroup::step`).
@compute @workgroup_size(64)
fn update_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.num_neurons) {
        return;
    }
    let p = params[i];
    var s = state[i];

    // Capture undecayed input from t-1, then decay the synapse for t+1.
    let current_val = s.accumulator;
    if (s.timer > p.timer_threshold) {
        let decay = s.accumulator >> u32(p.decay_shift_k);
        if (decay != 0) {
            s.accumulator = s.accumulator - decay;
        } else if (s.accumulator > 0) {
            s.accumulator = s.accumulator - 1;
        } else if (s.accumulator < 0) {
            s.accumulator = s.accumulator + 1;
        }
        s.timer = 0;
    }
    s.timer = s.timer + 1;

    let total_input = current_val + p.biascurrent;

    var f: i32;
    if (s.x < p.f_min) {
        f = (p.rest - s.x) >> u32(p.shift_a);
    } else {
        f = (s.x - p.threshold) >> u32(p.shift_b);
    }

    var noise_term: i32 = 0;
    if (p.noise > 1) {
        // LCG (Numerical Recipes constants); u32 wraps mod 2^32 like the CPU.
        s.rng_state = s.rng_state * 1664525u + 1013904223u;
        let r = i32(s.rng_state >> 1u);          // in [0, 2^31)
        let q = r / p.noise;                     // truncating divide (validated)
        noise_term = (r - q * p.noise) - (p.noise >> 1u); // r % noise, no raw `%`
    }

    s.x = s.x + f + total_input + noise_term;

    s.is_firing = 0;
    if (s.x >= p.vmax) {
        s.spike_count = s.spike_count + 1;
        s.is_firing = 1;
        s.x = s.x - (p.vmax - p.reset);          // soft reset
    }
    if (s.x < p.vmin) {
        s.x = p.vmin;
    }
    s.t_neuron = s.t_neuron + 1;

    state[i] = s;
}
