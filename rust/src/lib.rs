//! PyO3 bindings for the Rust IQIF port.
//!
//! Exposes an `iqnet` class mirroring the existing `iqif.iqnet` Python API
//! (see `iqif/__init__.py`) so the existing-style tests run unchanged against
//! the Rust implementation. The simulation logic lives in `core`, which has no
//! Python dependency.

mod core;

use crate::core::IqNetwork;
use pyo3::prelude::*;
use std::fs;

/// IQIF spiking network, bit-exact with the C++ `iq_network`.
#[pyclass]
struct iqnet {
    net: IqNetwork,
}

#[pymethods]
impl iqnet {
    /// Construct from a parameter file and a connection table file (same text
    /// formats as the C++ loader).
    #[new]
    fn new(par: &str, con: &str) -> PyResult<Self> {
        let par_text = fs::read_to_string(par).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("cannot read {par}: {e}"))
        })?;
        let con_text = fs::read_to_string(con).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("cannot read {con}: {e}"))
        })?;
        Ok(iqnet { net: IqNetwork::from_text(&par_text, &con_text) })
    }

    fn num_neurons(&self) -> i32 {
        self.net.num_neurons()
    }

    fn send_synapse(&mut self) {
        self.net.send_synapse();
    }

    fn set_biascurrent(&mut self, neuron_index: i32, biascurrent: i32) -> i32 {
        self.net.set_biascurrent(neuron_index, biascurrent)
    }

    #[allow(clippy::too_many_arguments)]
    fn set_neuron(
        &mut self,
        neuron_index: i32,
        rest: i32,
        threshold: i32,
        reset: i32,
        a: i32,
        b: i32,
        noise: i32,
    ) -> i32 {
        // C++/Python expose the bit-shift params as `a`/`b`.
        self.net.set_neuron(neuron_index, rest, threshold, reset, a, b, noise)
    }

    fn set_weight(&mut self, pre: i32, post: i32, weight: i32, tau: i32) -> i32 {
        self.net.set_weight(pre, post, weight, tau)
    }

    /// Network-wide (`arg2` omitted) or per-neuron surrogate tau, matching the
    /// overloaded Python signature `set_surrogate_tau(arg1, arg2=None)`.
    #[pyo3(signature = (arg1, arg2=None))]
    fn set_surrogate_tau(&mut self, arg1: i32, arg2: Option<i32>) -> i32 {
        match arg2 {
            None => self.net.set_surrogate_tau_all(arg1),
            Some(s_tau) => self.net.set_surrogate_tau_one(arg1, s_tau),
        }
    }

    fn get_surrogate_tau(&self, neuron_index: i32) -> i32 {
        self.net.get_surrogate_tau(neuron_index)
    }

    fn get_current_accumulator(&self, neuron_index: i32) -> i32 {
        self.net.get_current_accumulator(neuron_index)
    }

    fn set_current_accumulator(&mut self, neuron_index: i32, value: i32) -> i32 {
        self.net.set_current_accumulator(neuron_index, value)
    }

    fn get_all_current_accumulators(&self) -> Vec<i32> {
        self.net.get_all_current_accumulators()
    }

    fn set_all_current_accumulators(&mut self, values: Vec<i32>) {
        self.net.set_all_current_accumulators(&values);
    }

    fn get_decay_threshold(&self, neuron_index: i32) -> i32 {
        self.net.get_decay_threshold(neuron_index)
    }

    fn set_vmax(&mut self, neuron_index: i32, vmax: i32) -> i32 {
        self.net.set_vmax(neuron_index, vmax)
    }

    fn set_vmin(&mut self, neuron_index: i32, vmin: i32) -> i32 {
        self.net.set_vmin(neuron_index, vmin)
    }

    fn potential(&self, neuron_index: i32) -> i32 {
        self.net.potential(neuron_index)
    }

    fn set_potential(&mut self, neuron_index: i32, value: i32) -> i32 {
        self.net.set_potential(neuron_index, value)
    }

    fn get_is_firing(&self, neuron_index: i32) -> i32 {
        self.net.get_is_firing(neuron_index)
    }

    fn set_is_firing(&mut self, neuron_index: i32, value: i32) -> i32 {
        self.net.set_is_firing(neuron_index, value)
    }

    fn get_synapse_timer(&self, neuron_index: i32) -> i32 {
        self.net.get_synapse_timer(neuron_index)
    }

    fn set_synapse_timer(&mut self, neuron_index: i32, value: i32) -> i32 {
        self.net.set_synapse_timer(neuron_index, value)
    }

    fn spike_count(&mut self, neuron_index: i32) -> i32 {
        self.net.spike_count(neuron_index)
    }

    fn get_all_spike_counts(&mut self) -> Vec<i32> {
        self.net.get_all_spike_counts()
    }

    fn spike_rate(&mut self, neuron_index: i32) -> f32 {
        self.net.spike_rate(neuron_index)
    }

    /// Accepted for API compatibility; the Rust port is single-threaded.
    fn set_num_threads(&mut self, _num_threads: i32) {}
}

#[pymodule]
fn iqif_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<iqnet>()?;
    Ok(())
}
