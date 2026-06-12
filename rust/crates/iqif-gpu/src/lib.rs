//! GPU (wgpu) backend for the integer IQIF network.
//!
//! `GpuNetwork` builds from the same text formats as the CPU backend and, via
//! `iqif_core`'s snapshot/CSC export, uploads a bit-exact copy of the network to
//! the device. One timestep is two compute passes — `propagate` (gather spikes
//! over the transposed adjacency) then `update_state` — and all per-neuron state
//! stays resident on the GPU between steps; the host reads it back only in bulk,
//! on demand (see `rust/PLAN.md`, Phase 3).

use bytemuck::{Pod, Zeroable};
use iqif_core::{IqNetwork, NeuronSnapshot};
use std::sync::Mutex;
use wgpu::util::DeviceExt;

mod sanity;
pub use sanity::{
    check_integer_semantics, check_integer_semantics_on, Mismatch, SanityReport, RELIED_ON_OPS,
};

/// Constant per-neuron setup, packed to mirror the WGSL `Params` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
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
}

/// Mutable per-neuron state, mirrors the WGSL `State` struct (8 words).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuState {
    x: i32,
    accumulator: i32,
    timer: i32,
    is_firing: i32,
    spike_count: i32,
    t_neuron: i32,
    rng_state: u32,
    _pad: u32,
}

/// One incoming CSC edge; matches WGSL `vec2<i32>` (source, weight).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuEdge {
    source: i32,
    weight: i32,
}

/// Uniform; padded to 16 bytes as uniform buffers require.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuMeta {
    num_neurons: u32,
    _pad: [u32; 3],
}

fn params_of(s: &NeuronSnapshot, biascurrent: i32) -> GpuParams {
    GpuParams {
        rest: s.rest,
        threshold: s.threshold,
        shift_a: s.shift_a,
        shift_b: s.shift_b,
        reset: s.reset,
        noise: s.noise,
        f_min: s.f_min,
        vmax: s.vmax,
        vmin: s.vmin,
        timer_threshold: s.timer_threshold,
        decay_shift_k: s.decay_shift_k,
        biascurrent,
    }
}

fn state_of(s: &NeuronSnapshot) -> GpuState {
    GpuState {
        x: s.x,
        accumulator: s.accumulator,
        timer: s.timer,
        is_firing: s.is_firing,
        spike_count: s.spike_count,
        t_neuron: s.t_neuron,
        rng_state: s.rng_state,
        _pad: 0,
    }
}

/// Host mirror of the GPU state buffer, kept coherent lazily (PyTorch-style):
/// reads pull a fresh copy only when the device has advanced; writes are batched
/// and flushed to the device just before the next step. This turns the parity
/// test's per-neuron getter storm into one readback per timestep instead of one
/// PCIe round-trip per call.
struct HostSync {
    /// Mirror of `state_buf`. Authoritative for *state* reads whenever `fresh`.
    state: Vec<GpuState>,
    /// `state` reflects the latest device values (no step since last readback).
    fresh: bool,
    /// `state` has host writes not yet uploaded to the device.
    dirty: bool,
}

/// GPU-resident IQIF network. The wgpu device owns the live state; a retained
/// `IqNetwork` core is the authority for *setup/topology* (so connectivity- or
/// param-changing setters re-derive and re-upload cleanly), while the GPU plus
/// the [`HostSync`] cache are the authority for evolving *state*.
pub struct GpuNetwork {
    /// Setup/topology authority. Its *state* fields go stale once the GPU steps;
    /// only params/edges are ever re-derived from it.
    core: IqNetwork,
    device: wgpu::Device,
    queue: wgpu::Queue,
    propagate: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    state_buf: wgpu::Buffer,
    edges_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    num_neurons: usize,
    state_bytes: u64,
    workgroups: u32,
    sync: Mutex<HostSync>,
}

fn acquire_device() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| format!("no compatible GPU adapter: {e}"))?;

    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("iqif-gpu-device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| format!("request_device failed: {e}"))
}

impl GpuNetwork {
    /// Build the CPU reference from text, then upload a bit-exact mirror to the
    /// GPU. Fallible because device acquisition can fail (no adapter, etc.).
    pub fn from_text(par: &str, con: &str) -> Result<Self, String> {
        Self::build(IqNetwork::from_text(par, con))
    }

    /// Upload a clone of an existing CPU network's setup + initial state.
    pub fn from_core(core: &IqNetwork) -> Result<Self, String> {
        Self::build(core.clone())
    }

    /// Take ownership of a core and upload a bit-exact mirror to the GPU.
    fn build(core: IqNetwork) -> Result<Self, String> {
        let num_neurons = core.num_neurons() as usize;
        if num_neurons == 0 {
            return Err("cannot build a GpuNetwork with zero neurons".into());
        }

        let snaps = core.neuron_snapshots();
        let bias = core.biascurrents();
        let csc = core.build_csc();

        let params: Vec<GpuParams> = snaps
            .iter()
            .enumerate()
            .map(|(i, s)| params_of(s, bias[i]))
            .collect();
        let state: Vec<GpuState> = snaps.iter().map(state_of).collect();
        let mut edges: Vec<GpuEdge> = csc
            .sources
            .iter()
            .zip(csc.weights.iter())
            .map(|(&source, &weight)| GpuEdge { source, weight })
            .collect();
        // Storage buffers must be non-empty even for an edgeless network; the
        // offsets keep every gather loop empty so the dummy is never read.
        if edges.is_empty() {
            edges.push(GpuEdge { source: 0, weight: 0 });
        }
        let meta = GpuMeta { num_neurons: num_neurons as u32, _pad: [0; 3] };

        let (device, queue) = acquire_device()?;

        // params/state/edges take COPY_DST so setters can patch them via
        // queue.write_buffer (params/edges re-derived from core; state flushed
        // from the host cache). state also needs COPY_SRC for readback.
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("iqif-params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let state_bytes = std::mem::size_of_val(state.as_slice()) as u64;
        let state_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("iqif-state"),
            contents: bytemuck::cast_slice(&state),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("iqif-csc-offsets"),
            contents: bytemuck::cast_slice(&csc.offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let edges_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("iqif-csc-edges"),
            contents: bytemuck::cast_slice(&edges),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("iqif-meta"),
            contents: bytemuck::bytes_of(&meta),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iqif-readback"),
            size: state_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Explicit layout so both pipelines share one bind group.
        let storage_ro = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iqif-bind-layout"),
            entries: &[
                storage_ro(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_ro(2),
                storage_ro(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iqif-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: edges_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: meta_buf.as_entire_binding() },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iqif-step"),
            source: wgpu::ShaderSource::Wgsl(include_str!("step.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iqif-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let propagate = make_pipeline("propagate");
        let update = make_pipeline("update_state");

        Ok(GpuNetwork {
            core,
            device,
            queue,
            propagate,
            update,
            bind_group,
            params_buf,
            state_buf,
            edges_buf,
            readback_buf,
            num_neurons,
            state_bytes,
            workgroups: num_neurons.div_ceil(64) as u32,
            // Cache starts coherent with the just-uploaded initial state.
            sync: Mutex::new(HostSync { state, fresh: true, dirty: false }),
        })
    }

    pub fn num_neurons(&self) -> i32 {
        self.num_neurons as i32
    }

    fn in_range(&self, i: i32) -> bool {
        i >= 0 && (i as usize) < self.num_neurons
    }

    /// One timestep: flush any pending host writes, then `propagate` +
    /// `update_state` as two compute passes (the pass boundary orders the
    /// accumulator and `is_firing` handoff). State stays on the device; the
    /// host cache is marked stale so the next read pulls fresh values.
    pub fn step(&self) {
        {
            let mut g = self.sync.lock().unwrap();
            if g.dirty {
                self.queue.write_buffer(&self.state_buf, 0, bytemuck::cast_slice(&g.state));
                g.dirty = false;
            }
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("iqif-step") });
        for (label, pipeline) in [("propagate", &self.propagate), ("update_state", &self.update)] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.sync.lock().unwrap().fresh = false;
    }

    /// Run `steps` timesteps back-to-back, all on-device.
    pub fn run(&self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    // ── host cache sync ──────────────────────────────────────────────────

    /// Bulk-copy the state buffer to the host (one PCIe transfer). No locking;
    /// callers hold the `sync` guard.
    fn read_state_raw(&self) -> Vec<GpuState> {
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("iqif-readback") },
        );
        encoder.copy_buffer_to_buffer(&self.state_buf, 0, &self.readback_buf, 0, self.state_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll failed during readback");
        let data = slice.get_mapped_range();
        let out: Vec<GpuState> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.readback_buf.unmap();
        out
    }

    /// Read from the cache, refreshing it from the device first if stale.
    fn read<R>(&self, f: impl FnOnce(&[GpuState]) -> R) -> R {
        let mut g = self.sync.lock().unwrap();
        if !g.fresh {
            g.state = self.read_state_raw();
            g.fresh = true;
        }
        f(&g.state)
    }

    /// Mutate the cache (refreshing first so untouched fields stay correct) and
    /// mark it dirty for upload before the next step.
    fn write<R>(&self, f: impl FnOnce(&mut [GpuState]) -> R) -> R {
        let mut g = self.sync.lock().unwrap();
        if !g.fresh {
            g.state = self.read_state_raw();
            g.fresh = true;
        }
        let r = f(&mut g.state);
        g.dirty = true;
        r
    }

    /// Re-derive the params buffer from the (authoritative) core and upload it.
    fn reupload_params(&self) {
        let snaps = self.core.neuron_snapshots();
        let bias = self.core.biascurrents();
        let params: Vec<GpuParams> =
            snaps.iter().enumerate().map(|(i, s)| params_of(s, bias[i])).collect();
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&params));
    }

    /// Re-derive the CSC edge buffer from the core and upload it. Edge *count*
    /// is unchanged by weight edits, so the buffer size still matches.
    fn reupload_edges(&self) {
        let csc = self.core.build_csc();
        let mut edges: Vec<GpuEdge> = csc
            .sources
            .iter()
            .zip(csc.weights.iter())
            .map(|(&source, &weight)| GpuEdge { source, weight })
            .collect();
        if edges.is_empty() {
            edges.push(GpuEdge { source: 0, weight: 0 });
        }
        self.queue.write_buffer(&self.edges_buf, 0, bytemuck::cast_slice(&edges));
    }

    /// Re-initialize one neuron's GPU state from the core (used after
    /// `set_neuron`, which resets `x`, accumulator, and timer).
    fn reset_state_slot(&self, i: usize) {
        let snap = self.core.neuron_snapshots()[i];
        self.write(|s| s[i] = state_of(&snap));
    }

    // ── bulk state readback (whole-network) ──────────────────────────────

    /// Membrane potentials (`x`) for all neurons.
    pub fn potentials(&self) -> Vec<i32> {
        self.read(|s| s.iter().map(|e| e.x).collect())
    }

    /// Synapse current accumulators for all neurons.
    pub fn accumulators(&self) -> Vec<i32> {
        self.read(|s| s.iter().map(|e| e.accumulator).collect())
    }

    /// `is_firing` (1/0) for all neurons.
    pub fn is_firing(&self) -> Vec<i32> {
        self.read(|s| s.iter().map(|e| e.is_firing).collect())
    }

    // ── per-neuron state getters/setters (cache-backed) ──────────────────

    pub fn potential(&self, i: i32) -> i32 {
        if !self.in_range(i) {
            return 0;
        }
        self.read(|s| s[i as usize].x)
    }

    pub fn set_potential(&self, i: i32, v: i32) -> i32 {
        if !self.in_range(i) {
            return 0;
        }
        self.write(|s| s[i as usize].x = v);
        1
    }

    pub fn get_current_accumulator(&self, i: i32) -> i32 {
        if !self.in_range(i) {
            return 0;
        }
        self.read(|s| s[i as usize].accumulator)
    }

    pub fn set_current_accumulator(&self, i: i32, v: i32) -> i32 {
        if !self.in_range(i) {
            return 0;
        }
        self.write(|s| s[i as usize].accumulator = v);
        1
    }

    pub fn get_all_current_accumulators(&self) -> Vec<i32> {
        self.read(|s| s.iter().map(|e| e.accumulator).collect())
    }

    pub fn set_all_current_accumulators(&self, values: &[i32]) {
        self.write(|s| {
            for (e, &v) in s.iter_mut().zip(values.iter()) {
                e.accumulator = v;
            }
        });
    }

    pub fn get_is_firing(&self, i: i32) -> i32 {
        if !self.in_range(i) {
            return 0;
        }
        self.read(|s| s[i as usize].is_firing)
    }

    pub fn set_is_firing(&self, i: i32, v: i32) -> i32 {
        if !self.in_range(i) {
            return 0;
        }
        self.write(|s| s[i as usize].is_firing = (v != 0) as i32);
        1
    }

    pub fn get_synapse_timer(&self, i: i32) -> i32 {
        if !self.in_range(i) {
            return 0;
        }
        self.read(|s| s[i as usize].timer)
    }

    pub fn set_synapse_timer(&self, i: i32, v: i32) -> i32 {
        if !self.in_range(i) {
            return 0;
        }
        self.write(|s| s[i as usize].timer = v);
        1
    }

    /// Read-and-clear one neuron's spike counter (matches the C++ getter).
    pub fn spike_count(&self, i: i32) -> i32 {
        if !self.in_range(i) {
            return 0;
        }
        self.write(|s| {
            let c = s[i as usize].spike_count;
            s[i as usize].spike_count = 0;
            c
        })
    }

    /// Read-and-clear every neuron's spike counter (bulk; one round-trip).
    pub fn get_all_spike_counts(&self) -> Vec<i32> {
        self.write(|s| {
            s.iter_mut()
                .map(|e| {
                    let c = e.spike_count;
                    e.spike_count = 0;
                    c
                })
                .collect()
        })
    }

    /// Spike rate since the last call; resets the window (matches C++).
    pub fn spike_rate(&self, i: i32) -> f32 {
        if !self.in_range(i) {
            return 0.0;
        }
        self.write(|s| {
            let e = &mut s[i as usize];
            let denom = if e.t_neuron != 0 { e.t_neuron } else { 1 };
            let r = e.spike_count as f32 / denom as f32;
            e.t_neuron = 0;
            e.spike_count = 0;
            r
        })
    }

    // ── params getters (from the core; constant w.r.t. stepping) ─────────

    pub fn get_decay_threshold(&self, i: i32) -> i32 {
        self.core.get_decay_threshold(i)
    }

    pub fn get_surrogate_tau(&self, i: i32) -> i32 {
        self.core.get_surrogate_tau(i)
    }

    // ── setup mutators: apply to core, re-derive + re-upload buffers ──────

    pub fn set_biascurrent(&mut self, i: i32, v: i32) -> i32 {
        let r = self.core.set_biascurrent(i, v);
        if r == 1 {
            self.reupload_params();
        }
        r
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_neuron(&mut self, i: i32, rest: i32, threshold: i32, reset: i32, a: i32, b: i32, noise: i32) -> i32 {
        let r = self.core.set_neuron(i, rest, threshold, reset, a, b, noise);
        if r == 1 {
            self.reupload_params();
            self.reset_state_slot(i as usize); // set() re-inits x/accumulator/timer
        }
        r
    }

    pub fn set_weight(&mut self, pre: i32, post: i32, weight: i32, tau: i32) -> i32 {
        let r = self.core.set_weight(pre, post, weight, tau);
        if r == 1 {
            self.reupload_params(); // tau change -> timer_threshold/decay_shift_k
            self.reupload_edges(); // weight change -> CSC edge
        }
        r
    }

    pub fn set_surrogate_tau_all(&mut self, s_tau: i32) -> i32 {
        let r = self.core.set_surrogate_tau_all(s_tau);
        self.reupload_params();
        r
    }

    pub fn set_surrogate_tau_one(&mut self, i: i32, s_tau: i32) -> i32 {
        let r = self.core.set_surrogate_tau_one(i, s_tau);
        if r == 1 {
            self.reupload_params();
        }
        r
    }

    pub fn set_vmax(&mut self, i: i32, v: i32) -> i32 {
        let r = self.core.set_vmax(i, v); // core: 0 ok, 1 oob
        if r == 0 {
            self.reupload_params();
        }
        r
    }

    pub fn set_vmin(&mut self, i: i32, v: i32) -> i32 {
        let r = self.core.set_vmin(i, v); // core: 0 ok, 1 oob
        if r == 0 {
            self.reupload_params();
        }
        r
    }
}

#[cfg(test)]
mod gpu_parity_tests;
