#!/usr/bin/env python3
"""
Test sub-threshold integrator behavior for predictive coding.

With large shift_a and small shift_b, and threshold placed at VMAX,
the IQIF neuron should behave as a near-ideal discrete integrator:
    x(t+1) ≈ x(t) + input

Setup:
  - shift_a=15 (huge: restoring force ≈ 0)
  - shift_b=1  (small, but f_min ≈ VMAX so never reached)
  - rest=128, threshold=255, reset=128, VMAX=255, noise=0
  - Two neurons: 0 drives 1 via a known weight.
"""

import os
import sys
import tempfile
from iqif import iqnet

tmpdir = tempfile.mkdtemp(prefix="iqtest_sub_")
par_path = os.path.join(tmpdir, "params.txt")
con_path = os.path.join(tmpdir, "conn.txt")

# shift_a=15, shift_b=1
# f_min = (rest*(1<<shift_a) + threshold*(1<<shift_b)) / ((1<<shift_a) + (1<<shift_b))
#       = (128*2 + 255*32768) / (2 + 32768) ≈ 255  (≈ VMAX)
# So the neuron is always in the x < f_min regime,
# where f = (rest - x) >> 15 ≈ 0 for any x in [0, 255].
with open(par_path, "w") as f:
    f.write("0 128 255 128 15 1 0\n")
    f.write("1 128 255 128 15 1 0\n")

with open(con_path, "w") as f:
    f.write("0 1 5 32\n")

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

# --- Test A: bias current integrates linearly ---
print("\n=== Test A: bias current integration ===")
net = iqnet(par_path, con_path)
net.set_biascurrent(0, 3)
net.set_biascurrent(1, 0)

v_prev = net.potential(0)  # should be 128 (rest)
check("initial potential is rest=128", v_prev == 128)

deltas = []
for t in range(10):
    net.send_synapse()
    v = net.potential(0)
    deltas.append(v - v_prev)
    v_prev = v

# Each step adds bias + f where f = (rest - x) >> 15.
# At rest, f=0 so first delta is exactly 3. Once x drifts from rest,
# f = -1 at most (tiny restoring force). This is acceptable for PC use:
# the "leak" is negligible relative to synaptic drive.
print(f"  Per-step deltas: {deltas}")
check("first delta equals bias current exactly (3)", deltas[0] == 3)
check("subsequent deltas within 1 of bias (restoring force ≤ 1)",
      all(abs(d - 3) <= 1 for d in deltas[1:]))

# --- Test B: no input means no drift ---
print("\n=== Test B: zero-input stability ===")
net = iqnet(par_path, con_path)
net.set_biascurrent(0, 0)
net.set_biascurrent(1, 0)

v0 = net.potential(0)
for _ in range(50):
    net.send_synapse()

check("potential unchanged after 50 steps with zero input",
      net.potential(0) == v0)

# --- Test C: negative bias integrates downward ---
print("\n=== Test C: negative bias ===")
net = iqnet(par_path, con_path)
net.set_biascurrent(0, -2)
net.set_biascurrent(1, 0)

v_prev = net.potential(0)
deltas = []
for t in range(10):
    net.send_synapse()
    v = net.potential(0)
    deltas.append(v - v_prev)
    v_prev = v

print(f"  Per-step deltas: {deltas}")
check("all deltas equal bias current (-2)",
      all(d == -2 for d in deltas))

# --- Test D: synaptic input accumulates correctly ---
print("\n=== Test D: synaptic spike integration ===")
# Use set_current_accumulator to inject a known value into neuron 1
# (simulating what a spike through weight=5 would do)
net = iqnet(par_path, con_path)
net.set_biascurrent(0, 0)
net.set_biascurrent(1, 0)

v_before = net.potential(1)
net.set_current_accumulator(1, 7)
net.send_synapse()
v_after = net.potential(1)

delta = v_after - v_before
print(f"  Injected acc=7: V {v_before} -> {v_after} (delta={delta})")
check("potential increases by injected current (7)", delta == 7)

# --- Test E: superposition (bias + synaptic input) ---
print("\n=== Test E: superposition ===")
net = iqnet(par_path, con_path)
net.set_biascurrent(1, 4)

v_before = net.potential(1)
net.set_current_accumulator(1, 3)
net.send_synapse()
v_after = net.potential(1)

delta = v_after - v_before
print(f"  bias=4 + acc=3: V {v_before} -> {v_after} (delta={delta})")
check("delta equals bias + injected (4+3=7)", delta == 7)

# --- Test F: VMIN clamp prevents underflow ---
print("\n=== Test F: VMIN clamp ===")
net = iqnet(par_path, con_path)
net.set_biascurrent(0, -200)
net.set_biascurrent(1, 0)

for _ in range(5):
    net.send_synapse()

v = net.potential(0)
print(f"  After large negative bias: V={v}")
check("potential clamped at VMIN (0)", v == 0)

# ---------------------------------------------------------------------------
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed!")
else:
    print("Some tests FAILED.")
    sys.exit(1)
