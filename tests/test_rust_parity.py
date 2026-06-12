#!/usr/bin/env python3
"""
Bit-exact parity: Rust port (iqif_rs) vs C++ reference (iqif).

The two implementations are intentionally separate top-level modules:
  - `import iqif`     -> the original C++ ctypes wrapper (repo iqif/ dir)
  - `import iqif_rs`  -> the Rust/PyO3 extension (installed into site-packages)
so there is no way to confuse them. This test drives identical scenarios
through both and asserts every observable integer matches step-by-step.

The Rust extension is checked on every backend it was built with: `device="cpu"`
always, and `device="gpu"` too when iqif_rs was compiled with the `gpu` feature
and a GPU adapter is present (otherwise that backend is skipped, not failed).
To exercise the GPU path: `maturin develop --features gpu`.

All neurons use noise=0 (deterministic), where the IQIF noise term is exactly
zero, so the integer dynamics must agree on the nose.
"""

import os
import tempfile

from iqif import iqnet as CppNet
from iqif_rs import iqnet as RustNet

passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL  {name}")


def write_net(par_lines, con_lines):
    d = tempfile.mkdtemp(prefix="iqparity_")
    par = os.path.join(d, "params.txt")
    con = os.path.join(d, "conn.txt")
    with open(par, "w") as f:
        f.write("".join(par_lines))
    with open(con, "w") as f:
        f.write("".join(con_lines))
    return par, con


def rust_backends(par, con):
    """Rust backends to verify against the C++ ground truth: always CPU, plus
    GPU when this build supports it and an adapter exists (else skipped)."""
    backs = [("rust-cpu", RustNet(par, con, device="cpu"))]
    try:
        backs.append(("rust-gpu", RustNet(par, con, device="gpu")))
    except Exception as e:  # NotImplementedError (no gpu feature) or no adapter
        print(f"  (device=gpu skipped: {e})")
    return backs


# ── Scenario A: synaptic decay (sub-threshold, no firing) ────────────────
print("=== Scenario A: synaptic decay ===")
par, con = write_net(
    ["0 128 255 128 15 1 0\n", "1 128 255 128 15 1 0\n"],
    ["0 1 100 8\n"],
)
cpp = CppNet(par, con)
rusts = rust_backends(par, con)
for n in [cpp] + [r for _, r in rusts]:
    n.set_biascurrent(0, 0)
    n.set_biascurrent(1, 0)
    n.set_current_accumulator(1, 200)

for t in range(60):
    cpp.send_synapse()
    for _, r in rusts:
        r.send_synapse()
    for i in (0, 1):
        for name, r in rusts:
            check(f"A pot n{i} t{t} [{name}]", cpp.potential(i) == r.potential(i))
            check(f"A acc n{i} t{t} [{name}]",
                  cpp.get_current_accumulator(i) == r.get_current_accumulator(i))
            check(f"A timer n{i} t{t} [{name}]",
                  cpp.get_synapse_timer(i) == r.get_synapse_timer(i))

# decay_threshold derivation (log2/log10 path) must match too
for i in (0, 1):
    for name, r in rusts:
        check(f"A decay_threshold n{i} [{name}]",
              cpp.get_decay_threshold(i) == r.get_decay_threshold(i))


# ── Scenario B: driven network with firing + CSR propagation ─────────────
# Realistic multi-neuron config from the repo inputs; drive neuron 0 hard so
# it crosses VMAX, soft-resets, and spikes propagate through the CSR table.
print("=== Scenario B: driven firing network ===")
par_b = os.path.join("inputs", "neuronParameter_IQIF.txt")
con_b = os.path.join("inputs", "Connection_Table_IQIF.txt")
cpp = CppNet(par_b, con_b)
rusts = rust_backends(par_b, con_b)
N = cpp.num_neurons()
for name, r in rusts:
    check(f"B same num_neurons [{name}]", N == r.num_neurons())
for n in [cpp] + [r for _, r in rusts]:
    n.set_biascurrent(0, 13)

cpp_fires = 0
rust_fires = {name: 0 for name, _ in rusts}
for t in range(1000):
    cpp.send_synapse()
    for _, r in rusts:
        r.send_synapse()
    for i in range(N):
        for name, r in rusts:
            check(f"B pot n{i} t{t} [{name}]", cpp.potential(i) == r.potential(i))
            check(f"B firing n{i} t{t} [{name}]",
                  cpp.get_is_firing(i) == r.get_is_firing(i))
    cpp_fires += cpp.get_is_firing(0)
    for name, r in rusts:
        rust_fires[name] += r.get_is_firing(0)

check("B neuron 0 actually fired", cpp_fires > 0)
for name, _ in rusts:
    check(f"B total fires match [{name}]", cpp_fires == rust_fires[name])


# ── Scenario C: bulk accumulator get/set + spike-count semantics ─────────
print("=== Scenario C: bulk accessors ===")
par, con = write_net(
    ["0 62 130 145 3 3 0\n", "1 62 130 145 3 3 0\n", "2 62 130 145 3 3 0\n"],
    ["0 1 5 8\n", "1 2 5 8\n"],
)
cpp = CppNet(par, con)
rusts = rust_backends(par, con)
for n in [cpp] + [r for _, r in rusts]:
    n.set_all_current_accumulators([10, 20, 30])
for name, r in rusts:
    check(f"C bulk get matches [{name}]",
          list(cpp.get_all_current_accumulators()) ==
          list(r.get_all_current_accumulators()))

for n in [cpp] + [r for _, r in rusts]:
    n.set_biascurrent(0, 20)
for _ in range(200):
    cpp.send_synapse()
    for _, r in rusts:
        r.send_synapse()
# spike_count is read-and-reset; compare the batched snapshot
cpp_counts = list(cpp.get_all_spike_counts())
for name, r in rusts:
    check(f"C spike counts match [{name}]",
          cpp_counts == list(r.get_all_spike_counts()))


print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
