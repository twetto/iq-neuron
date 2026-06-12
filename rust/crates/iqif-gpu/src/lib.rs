//! GPU (wgpu) backend for the integer IQIF network — skeleton.
//!
//! The compute kernels are not implemented yet (see `rust/PLAN.md`, Phase 2+).
//! For now `GpuNetwork` constructs from the same text formats and holds an
//! `iqif_core::IqNetwork` so that, once kernels land, setup (f_min, decay
//! params, CSR/CSC adjacency) has a single source of truth shared with the CPU
//! backend. Everything except construction/`num_neurons` is `unimplemented!()`.

use iqif_core::IqNetwork;

mod sanity;
pub use sanity::{
    check_integer_semantics, check_integer_semantics_on, Mismatch, SanityReport, RELIED_ON_OPS,
};

/// Placeholder GPU-resident IQIF network.
pub struct GpuNetwork {
    /// CPU reference network; kernels will read setup params off this and
    /// upload GPU-resident state buffers in a later phase.
    core: IqNetwork,
}

impl GpuNetwork {
    pub fn from_text(par: &str, con: &str) -> Self {
        GpuNetwork { core: IqNetwork::from_text(par, con) }
    }

    pub fn num_neurons(&self) -> i32 {
        self.core.num_neurons()
    }
}
