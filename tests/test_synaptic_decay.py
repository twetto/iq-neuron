#!/usr/bin/env python3
"""
Test synaptic decay behavior of SynapseGroup.

Verifies that the current_accumulator decays toward zero over time
with the expected bit-shift dynamics, and that different tau settings
produce different decay rates.

SynapseGroup.step() logic:
  - A timer counts up each call.
  - Decay fires only when timer > timer_threshold, then timer resets.
  - timer_threshold = 0 when apparent_tau <= surrogate_tau, meaning
    decay fires every step except the very first (timer inits to 0).
  - Decay formula: acc -= acc >> decay_shift_k
  - decay_shift_k = (int)log2(surrogate_tau)

Setup: sub-threshold integrator neurons (shift_a=15, shift_b=1)
so the only dynamics are the synaptic decay + integration.
"""

import os
import sys
import tempfile
from iqif import iqnet

tmpdir = tempfile.mkdtemp(prefix="iqtest_decay_")
par_path = os.path.join(tmpdir, "params.txt")
con_path = os.path.join(tmpdir, "conn.txt")

with open(par_path, "w") as f:
    f.write("0 128 255 128 15 1 0\n")
    f.write("1 128 255 128 15 1 0\n")

with open(con_path, "w") as f:
    # tau=8, surrogate_tau defaults to 8 -> timer_threshold=0
    # decay_shift_k = log2(8) = 3
    f.write("0 1 100 8\n")

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

# --- Test A: accumulator decays toward zero ---
print("\n=== Test A: decay toward zero ===")
net = iqnet(par_path, con_path)
net.set_biascurrent(0, 0)
net.set_biascurrent(1, 0)

net.set_current_accumulator(1, 200)
acc_values = [200]

for t in range(50):
    net.send_synapse()
    acc_values.append(net.get_current_accumulator(1))

print(f"  Accumulator trace: {acc_values[:15]}...")
check("accumulator decreases over time", acc_values[5] < acc_values[0])
check("accumulator reaches zero by step 50", acc_values[-1] == 0)

# --- Test B: negative accumulator decays toward zero ---
print("\n=== Test B: negative decay ===")
net = iqnet(par_path, con_path)
net.set_biascurrent(0, 0)
net.set_biascurrent(1, 0)

net.set_current_accumulator(1, -200)
acc_values = [-200]

for t in range(50):
    net.send_synapse()
    acc_values.append(net.get_current_accumulator(1))

print(f"  Accumulator trace: {acc_values[:15]}...")
check("negative accumulator increases toward zero", acc_values[5] > acc_values[0])
check("negative accumulator reaches zero by step 50", acc_values[-1] == 0)

# --- Test C: verify bit-shift decay formula ---
# With apparent_tau=8, surrogate_tau=8 -> timer_threshold=0.
# Timer inits to 0, so first step: timer(0) > 0? No -> no decay.
# All subsequent steps: timer(1) > 0? Yes -> decay, reset.
# Decay: acc -= acc >> 3
print("\n=== Test C: verify bit-shift decay formula ===")
net = iqnet(par_path, con_path)
net.set_biascurrent(0, 0)
net.set_biascurrent(1, 0)

initial = 256
net.set_current_accumulator(1, initial)

# Simulate: first step has no decay, then acc -= acc >> 3 each step
expected_acc = initial
actual_trace = []
expected_trace = []

for t in range(12):
    net.send_synapse()
    actual = net.get_current_accumulator(1)
    actual_trace.append(actual)

    if t == 0:
        # First step: no decay (timer starts at 0, threshold is 0)
        pass
    else:
        decay = expected_acc >> 3
        if decay != 0:
            expected_acc -= decay
        elif expected_acc > 0:
            expected_acc -= 1
        elif expected_acc < 0:
            expected_acc += 1
    expected_trace.append(expected_acc)

print(f"  Actual:   {actual_trace}")
print(f"  Expected: {expected_trace}")
check("decay matches bit-shift formula",
      actual_trace == expected_trace)

# --- Test D: different apparent_tau -> different decay rate ---
# apparent_tau=8 with surrogate_tau=8 -> timer_threshold=0 -> decay every step
# apparent_tau=64 with surrogate_tau=8 -> timer_threshold>0 -> decay less often
print("\n=== Test D: tau comparison ===")

def measure_decay_time(tau, initial=200, target=50):
    """Steps until accumulator drops below target."""
    par = os.path.join(tmpdir, "params_d.txt")
    con = os.path.join(tmpdir, f"conn_d_{tau}.txt")
    with open(par, "w") as f:
        f.write("0 128 255 128 15 1 0\n")
        f.write("1 128 255 128 15 1 0\n")
    with open(con, "w") as f:
        f.write(f"0 1 100 {tau}\n")
    net = iqnet(par, con)
    net.set_biascurrent(0, 0)
    net.set_biascurrent(1, 0)
    net.set_current_accumulator(1, initial)
    for t in range(500):
        net.send_synapse()
        if net.get_current_accumulator(1) < target:
            return t + 1
    return 500

def collect_decay_trace(tau, initial=200, steps=30):
    """Collect accumulator trace for debugging."""
    par = os.path.join(tmpdir, "params_d.txt")
    con = os.path.join(tmpdir, f"conn_d_trace_{tau}.txt")
    with open(par, "w") as f:
        f.write("0 128 255 128 15 1 0\n")
        f.write("1 128 255 128 15 1 0\n")
    with open(con, "w") as f:
        f.write(f"0 1 100 {tau}\n")
    net = iqnet(par, con)
    net.set_biascurrent(0, 0)
    net.set_biascurrent(1, 0)
    net.set_current_accumulator(1, initial)
    trace = [initial]
    for t in range(steps):
        net.send_synapse()
        trace.append(net.get_current_accumulator(1))
    return trace

trace_fast = collect_decay_trace(8)
trace_slow = collect_decay_trace(64)
print(f"  tau=8  trace: {trace_fast[:20]}")
print(f"  tau=64 trace: {trace_slow[:20]}")

fast_decay = measure_decay_time(8)
slow_decay = measure_decay_time(64)

print(f"  tau=8  decays to <50 in {fast_decay} steps")
print(f"  tau=64 decays to <50 in {slow_decay} steps")
check("larger tau decays slower", slow_decay > fast_decay)

# ---------------------------------------------------------------------------
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed!")
else:
    print("Some tests FAILED.")
    sys.exit(1)
