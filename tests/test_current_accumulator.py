#!/usr/bin/env python3
"""
Test script for current_accumulator get/set.

Setup:
  neuron 0: rest=80 thresh=235 reset=80 shift_a=3 shift_b=4 noise=0
  neuron 1: rest=80 thresh=235 reset=80 shift_a=3 shift_b=4 noise=0
  connection 0->1: weight=+10  tau=32  (excitatory)
  connection 1->0: weight=-5   tau=64  (inhibitory)
  bias current: neuron 0 = 13, neuron 1 = 12
  surrogate tau: 8 (default)

Expected behavior:
  - No equilibrium exists for either neuron, so both ramp up and fire.
  - Neuron 0 fires first (~20 steps), depositing +10 into neuron 1's
    accumulator. Neuron 1 fires slightly later, depositing -5 into
    neuron 0's accumulator.
"""

import os
import sys
import tempfile
import numpy as np
from iqif import iqnet

# ---------------------------------------------------------------------------
# 1. Create temporary config files
# ---------------------------------------------------------------------------
tmpdir = tempfile.mkdtemp(prefix="iqtest_")

par_path = os.path.join(tmpdir, "params.txt")
con_path = os.path.join(tmpdir, "conn.txt")

with open(par_path, "w") as f:
    # index  rest  threshold  reset  shift_a  shift_b  noise
    f.write("0 80 235 80 3 4 0\n")
    f.write("1 80 235 80 3 4 0\n")

with open(con_path, "w") as f:
    # pre  post  weight  tau
    f.write("0 1 10 32\n")
    f.write("1 0 -5 64\n")

print(f"Config files in: {tmpdir}")

# ---------------------------------------------------------------------------
# 2. Tests
# ---------------------------------------------------------------------------
passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        failed += 1


# --- Test A: per-neuron set/get round-trip ---
print("\n=== Test A: per-neuron set/get round-trip ===")
net = iqnet(par_path, con_path)
n = net.num_neurons()
check("num_neurons == 2", n == 2)

net.set_current_accumulator(0, 42)
net.set_current_accumulator(1, -17)
check("get[0] == 42", net.get_current_accumulator(0) == 42)
check("get[1] == -17", net.get_current_accumulator(1) == -17)


# --- Test B: bulk set/get round-trip ---
print("\n=== Test B: bulk set/get round-trip ===")
net = iqnet(par_path, con_path)

net.set_all_current_accumulators([100, -200])
vals = net.get_all_current_accumulators()
check("bulk get[0] == 100", vals[0] == 100)
check("bulk get[1] == -200", vals[1] == -200)


# --- Test C: bulk and per-neuron agree ---
print("\n=== Test C: bulk and per-neuron consistency ===")
net = iqnet(par_path, con_path)

net.set_current_accumulator(0, 7)
net.set_current_accumulator(1, -3)
vals = net.get_all_current_accumulators()
check("per-neuron set -> bulk get[0]", vals[0] == 7)
check("per-neuron set -> bulk get[1]", vals[1] == -3)

net.set_all_current_accumulators([55, 66])
check("bulk set -> per-neuron get[0]", net.get_current_accumulator(0) == 55)
check("bulk set -> per-neuron get[1]", net.get_current_accumulator(1) == 66)


# --- Test D: simulation run, observe accumulator after spike ---
print("\n=== Test D: simulation with spike propagation ===")
net = iqnet(par_path, con_path)

net.set_biascurrent(0, 13)
net.set_biascurrent(1, 12)

spike_times = {0: [], 1: []}
acc_log = {0: [], 1: []}
potential_log = {0: [], 1: []}
steps = 100

for t in range(steps):
    net.send_synapse()

    for idx in range(2):
        potential_log[idx].append(net.potential(idx))
        acc_log[idx].append(net.get_current_accumulator(idx))

    # spike_count resets on read, so nonzero means spike this step
    for idx in range(2):
        sc = net.spike_count(idx)
        if sc > 0:
            spike_times[idx].append(t)

print(f"  Neuron 0 spike times: {spike_times[0]}")
print(f"  Neuron 1 spike times: {spike_times[1]}")

check("neuron 0 fired at least once", len(spike_times[0]) > 0)
check("neuron 1 fired at least once", len(spike_times[1]) > 0)

# After neuron 0 fires, neuron 1's accumulator should have received +10
if spike_times[0]:
    t0 = spike_times[0][0]
    if t0 + 1 < steps:
        acc1_after = acc_log[1][t0 + 1]
        print(f"  Neuron 1 accumulator at t={t0+1} (after N0 first spike): {acc1_after}")
        check("neuron 1 accumulator nonzero after N0 spike", acc1_after != 0)


# --- Test E: set accumulator mid-simulation ---
print("\n=== Test E: inject current via set_current_accumulator ===")
net = iqnet(par_path, con_path)

# No bias current -- neuron sits at rest
# Inject a large value into the accumulator and see if it affects potential
v_before = net.potential(0)
net.set_current_accumulator(0, 50)
net.send_synapse()
v_after = net.potential(0)

print(f"  V before inject: {v_before}, V after one step with acc=50: {v_after}")
check("potential changed after accumulator injection", v_after > v_before)


# ---------------------------------------------------------------------------
# 3. Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed!")
else:
    print("Some tests FAILED.")
    sys.exit(1)
