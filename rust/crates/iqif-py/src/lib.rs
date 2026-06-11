//! PyO3 bindings for the integer IQIF network.
//!
//! Exposes a single `iqnet` class with a PyTorch-style `device=` selector. The
//! simulation logic lives in `iqif_core` (CPU) and, behind the optional `gpu`
//! feature, `iqif_gpu` (GPU). Both implement the object-safe [`Backend`] trait,
//! so `iqnet` dispatches at runtime to whichever device it was constructed on.
//!
//! Module is named `iqif_rs`, kept deliberately distinct from the C++ `iqif`
//! wrapper so the two never collide on `import`.

use iqif_core::IqNetwork;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::fs;

/// Object-safe interface every backend implements. Both the CPU
/// (`iqif_core::IqNetwork`) and the GPU (`iqif_gpu::GpuNetwork`) networks plug
/// in here; `iqnet` holds a `Box<dyn Backend>` and forwards to it.
///
/// `Send + Sync` supertrait so the boxed object satisfies `#[pyclass]`'s
/// requirement (the network types hold only plain data, no interior mutability).
trait Backend: Send + Sync {
    fn num_neurons(&self) -> i32;
    fn send_synapse(&mut self);
    fn set_biascurrent(&mut self, i: i32, v: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn set_neuron(&mut self, i: i32, rest: i32, threshold: i32, reset: i32, a: i32, b: i32, noise: i32) -> i32;
    fn set_weight(&mut self, pre: i32, post: i32, weight: i32, tau: i32) -> i32;
    fn set_surrogate_tau_all(&mut self, s_tau: i32) -> i32;
    fn set_surrogate_tau_one(&mut self, i: i32, s_tau: i32) -> i32;
    fn get_surrogate_tau(&self, i: i32) -> i32;
    fn get_current_accumulator(&self, i: i32) -> i32;
    fn set_current_accumulator(&mut self, i: i32, v: i32) -> i32;
    fn get_all_current_accumulators(&self) -> Vec<i32>;
    fn set_all_current_accumulators(&mut self, values: &[i32]);
    fn get_decay_threshold(&self, i: i32) -> i32;
    fn set_vmax(&mut self, i: i32, v: i32) -> i32;
    fn set_vmin(&mut self, i: i32, v: i32) -> i32;
    fn potential(&self, i: i32) -> i32;
    fn set_potential(&mut self, i: i32, v: i32) -> i32;
    fn get_is_firing(&self, i: i32) -> i32;
    fn set_is_firing(&mut self, i: i32, v: i32) -> i32;
    fn get_synapse_timer(&self, i: i32) -> i32;
    fn set_synapse_timer(&mut self, i: i32, v: i32) -> i32;
    fn spike_count(&mut self, i: i32) -> i32;
    fn get_all_spike_counts(&mut self) -> Vec<i32>;
    fn spike_rate(&mut self, i: i32) -> f32;
}

// CPU backend: forward to the inherent methods. Each call is qualified as
// `IqNetwork::method(self, ..)` so it resolves to the inherent method rather
// than recursing into this trait impl.
impl Backend for IqNetwork {
    fn num_neurons(&self) -> i32 { IqNetwork::num_neurons(self) }
    fn send_synapse(&mut self) { IqNetwork::send_synapse(self) }
    fn set_biascurrent(&mut self, i: i32, v: i32) -> i32 { IqNetwork::set_biascurrent(self, i, v) }
    fn set_neuron(&mut self, i: i32, rest: i32, threshold: i32, reset: i32, a: i32, b: i32, noise: i32) -> i32 {
        IqNetwork::set_neuron(self, i, rest, threshold, reset, a, b, noise)
    }
    fn set_weight(&mut self, pre: i32, post: i32, weight: i32, tau: i32) -> i32 { IqNetwork::set_weight(self, pre, post, weight, tau) }
    fn set_surrogate_tau_all(&mut self, s_tau: i32) -> i32 { IqNetwork::set_surrogate_tau_all(self, s_tau) }
    fn set_surrogate_tau_one(&mut self, i: i32, s_tau: i32) -> i32 { IqNetwork::set_surrogate_tau_one(self, i, s_tau) }
    fn get_surrogate_tau(&self, i: i32) -> i32 { IqNetwork::get_surrogate_tau(self, i) }
    fn get_current_accumulator(&self, i: i32) -> i32 { IqNetwork::get_current_accumulator(self, i) }
    fn set_current_accumulator(&mut self, i: i32, v: i32) -> i32 { IqNetwork::set_current_accumulator(self, i, v) }
    fn get_all_current_accumulators(&self) -> Vec<i32> { IqNetwork::get_all_current_accumulators(self) }
    fn set_all_current_accumulators(&mut self, values: &[i32]) { IqNetwork::set_all_current_accumulators(self, values) }
    fn get_decay_threshold(&self, i: i32) -> i32 { IqNetwork::get_decay_threshold(self, i) }
    fn set_vmax(&mut self, i: i32, v: i32) -> i32 { IqNetwork::set_vmax(self, i, v) }
    fn set_vmin(&mut self, i: i32, v: i32) -> i32 { IqNetwork::set_vmin(self, i, v) }
    fn potential(&self, i: i32) -> i32 { IqNetwork::potential(self, i) }
    fn set_potential(&mut self, i: i32, v: i32) -> i32 { IqNetwork::set_potential(self, i, v) }
    fn get_is_firing(&self, i: i32) -> i32 { IqNetwork::get_is_firing(self, i) }
    fn set_is_firing(&mut self, i: i32, v: i32) -> i32 { IqNetwork::set_is_firing(self, i, v) }
    fn get_synapse_timer(&self, i: i32) -> i32 { IqNetwork::get_synapse_timer(self, i) }
    fn set_synapse_timer(&mut self, i: i32, v: i32) -> i32 { IqNetwork::set_synapse_timer(self, i, v) }
    fn spike_count(&mut self, i: i32) -> i32 { IqNetwork::spike_count(self, i) }
    fn get_all_spike_counts(&mut self) -> Vec<i32> { IqNetwork::get_all_spike_counts(self) }
    fn spike_rate(&mut self, i: i32) -> f32 { IqNetwork::spike_rate(self, i) }
}

// GPU backend: present so the runtime-switch plumbing is real, but the compute
// kernels are not written yet. Only construction/`num_neurons` work; the rest
// is `unimplemented!()` (see rust/PLAN.md, Phases 2-4).
#[cfg(feature = "gpu")]
impl Backend for iqif_gpu::GpuNetwork {
    fn num_neurons(&self) -> i32 { iqif_gpu::GpuNetwork::num_neurons(self) }
    fn send_synapse(&mut self) { unimplemented!("GPU send_synapse: kernels not implemented yet") }
    fn set_biascurrent(&mut self, _i: i32, _v: i32) -> i32 { unimplemented!() }
    fn set_neuron(&mut self, _i: i32, _r: i32, _t: i32, _re: i32, _a: i32, _b: i32, _n: i32) -> i32 { unimplemented!() }
    fn set_weight(&mut self, _pre: i32, _post: i32, _w: i32, _tau: i32) -> i32 { unimplemented!() }
    fn set_surrogate_tau_all(&mut self, _s: i32) -> i32 { unimplemented!() }
    fn set_surrogate_tau_one(&mut self, _i: i32, _s: i32) -> i32 { unimplemented!() }
    fn get_surrogate_tau(&self, _i: i32) -> i32 { unimplemented!() }
    fn get_current_accumulator(&self, _i: i32) -> i32 { unimplemented!() }
    fn set_current_accumulator(&mut self, _i: i32, _v: i32) -> i32 { unimplemented!() }
    fn get_all_current_accumulators(&self) -> Vec<i32> { unimplemented!() }
    fn set_all_current_accumulators(&mut self, _values: &[i32]) { unimplemented!() }
    fn get_decay_threshold(&self, _i: i32) -> i32 { unimplemented!() }
    fn set_vmax(&mut self, _i: i32, _v: i32) -> i32 { unimplemented!() }
    fn set_vmin(&mut self, _i: i32, _v: i32) -> i32 { unimplemented!() }
    fn potential(&self, _i: i32) -> i32 { unimplemented!() }
    fn set_potential(&mut self, _i: i32, _v: i32) -> i32 { unimplemented!() }
    fn get_is_firing(&self, _i: i32) -> i32 { unimplemented!() }
    fn set_is_firing(&mut self, _i: i32, _v: i32) -> i32 { unimplemented!() }
    fn get_synapse_timer(&self, _i: i32) -> i32 { unimplemented!() }
    fn set_synapse_timer(&mut self, _i: i32, _v: i32) -> i32 { unimplemented!() }
    fn spike_count(&mut self, _i: i32) -> i32 { unimplemented!() }
    fn get_all_spike_counts(&mut self) -> Vec<i32> { unimplemented!() }
    fn spike_rate(&mut self, _i: i32) -> f32 { unimplemented!() }
}

#[cfg(feature = "gpu")]
fn make_gpu(par: &str, con: &str) -> PyResult<Box<dyn Backend + Send + Sync>> {
    Ok(Box::new(iqif_gpu::GpuNetwork::from_text(par, con)))
}

#[cfg(not(feature = "gpu"))]
fn make_gpu(_par: &str, _con: &str) -> PyResult<Box<dyn Backend + Send + Sync>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "iqif_rs was built without GPU support; rebuild with the `gpu` feature \
         (e.g. maturin build --features gpu)",
    ))
}

/// IQIF spiking network. `device` selects the backend ("cpu" or "gpu"), like a
/// PyTorch tensor's device. Bit-exact with the C++ `iq_network` on the CPU.
#[pyclass]
#[allow(non_camel_case_types)] // lowercase to match the Python API (`iqif.iqnet`)
struct iqnet {
    inner: Box<dyn Backend + Send + Sync>,
    #[pyo3(get)]
    device: String,
}

#[pymethods]
impl iqnet {
    #[new]
    #[pyo3(signature = (par, con, device="cpu"))]
    fn new(par: &str, con: &str, device: &str) -> PyResult<Self> {
        let par_text = fs::read_to_string(par)
            .map_err(|e| PyIOError::new_err(format!("cannot read {par}: {e}")))?;
        let con_text = fs::read_to_string(con)
            .map_err(|e| PyIOError::new_err(format!("cannot read {con}: {e}")))?;

        let inner: Box<dyn Backend + Send + Sync> = match device {
            "cpu" => Box::new(IqNetwork::from_text(&par_text, &con_text)),
            "gpu" => make_gpu(&par_text, &con_text)?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown device '{other}', expected 'cpu' or 'gpu'"
                )))
            }
        };
        Ok(iqnet { inner, device: device.to_string() })
    }

    fn num_neurons(&self) -> i32 { self.inner.num_neurons() }
    fn send_synapse(&mut self) { self.inner.send_synapse() }
    fn set_biascurrent(&mut self, neuron_index: i32, biascurrent: i32) -> i32 {
        self.inner.set_biascurrent(neuron_index, biascurrent)
    }

    #[allow(clippy::too_many_arguments)]
    fn set_neuron(&mut self, neuron_index: i32, rest: i32, threshold: i32, reset: i32, a: i32, b: i32, noise: i32) -> i32 {
        self.inner.set_neuron(neuron_index, rest, threshold, reset, a, b, noise)
    }

    fn set_weight(&mut self, pre: i32, post: i32, weight: i32, tau: i32) -> i32 {
        self.inner.set_weight(pre, post, weight, tau)
    }

    #[pyo3(signature = (arg1, arg2=None))]
    fn set_surrogate_tau(&mut self, arg1: i32, arg2: Option<i32>) -> i32 {
        match arg2 {
            None => self.inner.set_surrogate_tau_all(arg1),
            Some(s_tau) => self.inner.set_surrogate_tau_one(arg1, s_tau),
        }
    }

    fn get_surrogate_tau(&self, neuron_index: i32) -> i32 { self.inner.get_surrogate_tau(neuron_index) }
    fn get_current_accumulator(&self, neuron_index: i32) -> i32 { self.inner.get_current_accumulator(neuron_index) }
    fn set_current_accumulator(&mut self, neuron_index: i32, value: i32) -> i32 { self.inner.set_current_accumulator(neuron_index, value) }
    fn get_all_current_accumulators(&self) -> Vec<i32> { self.inner.get_all_current_accumulators() }
    fn set_all_current_accumulators(&mut self, values: Vec<i32>) { self.inner.set_all_current_accumulators(&values) }
    fn get_decay_threshold(&self, neuron_index: i32) -> i32 { self.inner.get_decay_threshold(neuron_index) }
    fn set_vmax(&mut self, neuron_index: i32, vmax: i32) -> i32 { self.inner.set_vmax(neuron_index, vmax) }
    fn set_vmin(&mut self, neuron_index: i32, vmin: i32) -> i32 { self.inner.set_vmin(neuron_index, vmin) }
    fn potential(&self, neuron_index: i32) -> i32 { self.inner.potential(neuron_index) }
    fn set_potential(&mut self, neuron_index: i32, value: i32) -> i32 { self.inner.set_potential(neuron_index, value) }
    fn get_is_firing(&self, neuron_index: i32) -> i32 { self.inner.get_is_firing(neuron_index) }
    fn set_is_firing(&mut self, neuron_index: i32, value: i32) -> i32 { self.inner.set_is_firing(neuron_index, value) }
    fn get_synapse_timer(&self, neuron_index: i32) -> i32 { self.inner.get_synapse_timer(neuron_index) }
    fn set_synapse_timer(&mut self, neuron_index: i32, value: i32) -> i32 { self.inner.set_synapse_timer(neuron_index, value) }
    fn spike_count(&mut self, neuron_index: i32) -> i32 { self.inner.spike_count(neuron_index) }
    fn get_all_spike_counts(&mut self) -> Vec<i32> { self.inner.get_all_spike_counts() }
    fn spike_rate(&mut self, neuron_index: i32) -> f32 { self.inner.spike_rate(neuron_index) }

    /// Accepted for API compatibility; the CPU backend is single-threaded.
    fn set_num_threads(&mut self, _num_threads: i32) {}
}

#[pymodule]
fn iqif_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<iqnet>()?;
    Ok(())
}
