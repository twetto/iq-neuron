#!/usr/bin/env python3
"""
Test set_weight() for iq_network (CSR mode).

Setup:
  neuron 0: rest=80 thresh=235 reset=80 shift_a=3 shift_b=4 noise=0
  neuron 1: rest=80 thresh=235 reset=80 shift_a=3 shift_b=4 noise=0
  neuron 2: rest=80 thresh=235 reset=80 shift_a=3 shift_b=4 noise=0
  0 -> 1: weight=+10  tau=32
  0 -> 2: weight=+10  tau=32
  1 -> 0: weight=-5   tau=64
"""

import os
import sys
import tempfile
from iqif import iqnet

tmpdir = tempfile.mkdtemp(prefix="iqtest_sw_")
par_path = os.path.join(tmpdir, "params.txt")
con_path = os.path.join(tmpdir, "conn.txt")

with open(par_path, "w") as f:
    f.write("0 80 235 80 3 4 0\n")
    f.write("1 80 235 80 3 4 0\n")
    f.write("2 80 235 80 3 4 0\n")

with open(con_path, "w") as f:
    f.write("0 1 10 32\n")
    f.write("0 2 10 32\n")
    f.write("1 0 -5 64\n")

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

# --- Test A: set_weight returns 1 for existing synapse ---
print("\n=== Test A: return value ===")
net = iqnet(par_path, con_path)
check("existing synapse 0->1 returns 1", net.set_weight(0, 1, 20, 32) == 1)
check("existing synapse 1->0 returns 1", net.set_weight(1, 0, -3, 64) == 1)
check("nonexistent synapse 2->0 returns 0", net.set_weight(2, 0, 5, 32) == 0)
check("out-of-range pre returns 0", net.set_weight(-1, 0, 5, 32) == 0)
check("out-of-range post returns 0", net.set_weight(0, 99, 5, 32) == 0)

# --- Test B: weight change affects spike propagation ---
print("\n=== Test B: weight change affects dynamics ===")

def run_and_record(weight_0_1, steps=60):
    """Run network with given 0->1 weight, return neuron 1 potentials."""
    net = iqnet(par_path, con_path)
    net.set_weight(0, 1, weight_0_1, 32)
    net.set_biascurrent(0, 13)
    net.set_biascurrent(1, 0)  # neuron 1 only driven by synapse from 0
    net.set_biascurrent(2, 0)
    potentials = []
    for _ in range(steps):
        net.send_synapse()
        potentials.append(net.potential(1))
    return potentials

v_strong = run_and_record(50)
v_weak = run_and_record(1)

# With a strong weight, neuron 1 should reach higher potentials
check("strong weight drives N1 higher than weak",
      max(v_strong) > max(v_weak))

# --- Test C: zeroing a weight silences the synapse ---
print("\n=== Test C: zero weight silences synapse ===")
net = iqnet(par_path, con_path)
net.set_weight(0, 1, 0, 32)
net.set_biascurrent(0, 13)
net.set_biascurrent(1, 0)
net.set_biascurrent(2, 0)

for _ in range(60):
    net.send_synapse()

# Neuron 1 has no bias and zero-weight synapse from 0, should stay near rest
check("zero weight: N1 stays near rest (V<=82)",
      net.potential(1) <= 82)

# --- Test D: sign flip (excitatory -> inhibitory) ---
print("\n=== Test D: sign flip ===")
net = iqnet(par_path, con_path)
net.set_weight(0, 1, -20, 32)  # flip 0->1 to inhibitory
net.set_biascurrent(0, 13)
net.set_biascurrent(1, 5)  # small drive so N1 is above VMIN
net.set_biascurrent(2, 0)

# Record N1 potential right after N0's first spike
potentials = []
for _ in range(60):
    net.send_synapse()
    potentials.append(net.potential(1))

# After N0 fires, the inhibitory weight should push N1 down.
# With excitatory input N1 would climb; with inhibitory it should dip.
v_min = min(potentials)
check("inhibitory weight pushes N1 below rest (80)", v_min < 80)

# --- Test E: modifying one target doesn't affect another ---
print("\n=== Test E: independent targets ===")
net = iqnet(par_path, con_path)
net.set_weight(0, 1, 50, 32)  # boost 0->1
# 0->2 should still be original weight=10
net.set_biascurrent(0, 13)
net.set_biascurrent(1, 0)
net.set_biascurrent(2, 0)

v1_list = []
v2_list = []
for _ in range(60):
    net.send_synapse()
    v1_list.append(net.potential(1))
    v2_list.append(net.potential(2))

check("boosted N1 climbs higher than untouched N2",
      max(v1_list) > max(v2_list))

# ---------------------------------------------------------------------------
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed!")
else:
    print("Some tests FAILED.")
    sys.exit(1)
