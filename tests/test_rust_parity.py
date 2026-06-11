#!/usr/bin/env python3
"""
Bit-exact parity: Rust port (iqif_rs) vs C++ reference (iqif).

The two implementations are intentionally separate top-level modules:
  - `import iqif`     -> the original C++ ctypes wrapper (repo iqif/ dir)
  - `import iqif_rs`  -> the Rust/PyO3 extension (installed into site-packages)
so there is no way to confuse them. This test drives identical scenarios
through both and asserts every observable integer matches step-by-step.

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


# ── Scenario A: synaptic decay (sub-threshold, no firing) ────────────────
print("=== Scenario A: synaptic decay ===")
par, con = write_net(
    ["0 128 255 128 15 1 0\n", "1 128 255 128 15 1 0\n"],
    ["0 1 100 8\n"],
)
cpp, rust = CppNet(par, con), RustNet(par, con)
for n in (cpp, rust):
    n.set_biascurrent(0, 0)
    n.set_biascurrent(1, 0)
    n.set_current_accumulator(1, 200)

for t in range(60):
    cpp.send_synapse()
    rust.send_synapse()
    for i in (0, 1):
        check(f"A pot n{i} t{t}", cpp.potential(i) == rust.potential(i))
        check(f"A acc n{i} t{t}",
              cpp.get_current_accumulator(i) == rust.get_current_accumulator(i))
        check(f"A timer n{i} t{t}",
              cpp.get_synapse_timer(i) == rust.get_synapse_timer(i))

# decay_threshold derivation (log2/log10 path) must match too
for i in (0, 1):
    check(f"A decay_threshold n{i}",
          cpp.get_decay_threshold(i) == rust.get_decay_threshold(i))


# ── Scenario B: driven network with firing + CSR propagation ─────────────
# Realistic multi-neuron config from the repo inputs; drive neuron 0 hard so
# it crosses VMAX, soft-resets, and spikes propagate through the CSR table.
print("=== Scenario B: driven firing network ===")
par_b = os.path.join("inputs", "neuronParameter_IQIF.txt")
con_b = os.path.join("inputs", "Connection_Table_IQIF.txt")
cpp, rust = CppNet(par_b, con_b), RustNet(par_b, con_b)
N = cpp.num_neurons()
check("B same num_neurons", N == rust.num_neurons())
for n in (cpp, rust):
    n.set_biascurrent(0, 13)

cpp_fires = 0
rust_fires = 0
for t in range(1000):
    cpp.send_synapse()
    rust.send_synapse()
    for i in range(N):
        check(f"B pot n{i} t{t}", cpp.potential(i) == rust.potential(i))
        check(f"B firing n{i} t{t}",
              cpp.get_is_firing(i) == rust.get_is_firing(i))
    cpp_fires += cpp.get_is_firing(0)
    rust_fires += rust.get_is_firing(0)

check("B neuron 0 actually fired", cpp_fires > 0)
check("B total fires match", cpp_fires == rust_fires)


# ── Scenario C: bulk accumulator get/set + spike-count semantics ─────────
print("=== Scenario C: bulk accessors ===")
par, con = write_net(
    ["0 62 130 145 3 3 0\n", "1 62 130 145 3 3 0\n", "2 62 130 145 3 3 0\n"],
    ["0 1 5 8\n", "1 2 5 8\n"],
)
cpp, rust = CppNet(par, con), RustNet(par, con)
cpp.set_all_current_accumulators([10, 20, 30])
rust.set_all_current_accumulators([10, 20, 30])
check("C bulk get matches",
      list(cpp.get_all_current_accumulators()) ==
      list(rust.get_all_current_accumulators()))

cpp.set_biascurrent(0, 20)
rust.set_biascurrent(0, 20)
for _ in range(200):
    cpp.send_synapse()
    rust.send_synapse()
# spike_count is read-and-reset; compare the batched snapshot
check("C spike counts match",
      list(cpp.get_all_spike_counts()) == list(rust.get_all_spike_counts()))


print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
