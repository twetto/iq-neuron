#!/usr/bin/env python3
"""GPU<->CPU parity for iqif_rs, driven entirely through the Python API.

The C++ reference module (`iqif`) isn't always installed, but the Rust CPU
backend is already proven bit-exact against it by tests/test_rust_parity.py.
So asserting device="gpu" == device="cpu" here transitively certifies the GPU
backend, exercising the exact PyO3 -> Backend -> GpuNetwork path Python sees.

Run after `maturin develop --release --features gpu` (from rust/).
"""

import os
import tempfile

from iqif_rs import iqnet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root

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
    d = tempfile.mkdtemp(prefix="iqgpu_")
    par = os.path.join(d, "params.txt")
    con = os.path.join(d, "conn.txt")
    with open(par, "w") as f:
        f.write("".join(par_lines))
    with open(con, "w") as f:
        f.write("".join(con_lines))
    return par, con


# Fail fast (not silently skip) if the GPU backend isn't usable: the whole
# point of this script is to test it.
par0, con0 = write_net(["0 0 128 128 15 1 0\n"], ["0 0 0 8\n"])
try:
    iqnet(par0, con0, device="gpu")
except Exception as e:
    print(f"GPU backend unavailable: {e}")
    print("(build with: maturin develop --release --features gpu)")
    raise SystemExit(2)


# ── Scenario A: synaptic decay ───────────────────────────────────────────
print("=== Scenario A: synaptic decay ===")
par, con = write_net(
    ["0 128 255 128 15 1 0\n", "1 128 255 128 15 1 0\n"],
    ["0 1 100 8\n"],
)
cpu = iqnet(par, con, device="cpu")
gpu = iqnet(par, con, device="gpu")
for n in (cpu, gpu):
    n.set_biascurrent(0, 0)
    n.set_biascurrent(1, 0)
    n.set_current_accumulator(1, 200)

for t in range(60):
    cpu.send_synapse()
    gpu.send_synapse()
    for i in (0, 1):
        check(f"A pot n{i} t{t}", cpu.potential(i) == gpu.potential(i))
        check(f"A acc n{i} t{t}",
              cpu.get_current_accumulator(i) == gpu.get_current_accumulator(i))
        check(f"A timer n{i} t{t}",
              cpu.get_synapse_timer(i) == gpu.get_synapse_timer(i))
for i in (0, 1):
    check(f"A decay_threshold n{i}",
          cpu.get_decay_threshold(i) == gpu.get_decay_threshold(i))


# ── Scenario B: driven firing network (real repo inputs) ─────────────────
print("=== Scenario B: driven firing network ===")
par_b = os.path.join(ROOT, "inputs", "neuronParameter_IQIF.txt")
con_b = os.path.join(ROOT, "inputs", "Connection_Table_IQIF.txt")
cpu = iqnet(par_b, con_b, device="cpu")
gpu = iqnet(par_b, con_b, device="gpu")
N = cpu.num_neurons()
check("B same num_neurons", N == gpu.num_neurons())
for n in (cpu, gpu):
    n.set_biascurrent(0, 13)

cpu_fires = 0
gpu_fires = 0
for t in range(1000):
    cpu.send_synapse()
    gpu.send_synapse()
    for i in range(N):
        check(f"B pot n{i} t{t}", cpu.potential(i) == gpu.potential(i))
        check(f"B firing n{i} t{t}", cpu.get_is_firing(i) == gpu.get_is_firing(i))
    cpu_fires += cpu.get_is_firing(0)
    gpu_fires += gpu.get_is_firing(0)

check("B neuron 0 actually fired", cpu_fires > 0)
check("B total fires match", cpu_fires == gpu_fires)


# ── Scenario C: bulk accessors + spike counts ────────────────────────────
print("=== Scenario C: bulk accessors ===")
par, con = write_net(
    ["0 62 130 145 3 3 0\n", "1 62 130 145 3 3 0\n", "2 62 130 145 3 3 0\n"],
    ["0 1 5 8\n", "1 2 5 8\n"],
)
cpu = iqnet(par, con, device="cpu")
gpu = iqnet(par, con, device="gpu")
for n in (cpu, gpu):
    n.set_all_current_accumulators([10, 20, 30])
check("C bulk get matches",
      list(cpu.get_all_current_accumulators()) ==
      list(gpu.get_all_current_accumulators()))

for n in (cpu, gpu):
    n.set_biascurrent(0, 20)
for _ in range(200):
    cpu.send_synapse()
    gpu.send_synapse()
check("C spike counts match",
      list(cpu.get_all_spike_counts()) == list(gpu.get_all_spike_counts()))


print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
