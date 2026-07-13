#!/usr/bin/env python3
"""Regression test for the v0.4.0 synaptic decay phase change.

Locks in two things:

  1. Observability (the fix): the synaptic leak now runs at the head of
     send_synapse(), so get_current_accumulator() reports the TRUE input the
     neuron integrated this step (pre-decay), not the post-decay residual.

  2. Golden spike trains for two deterministic configs, one in each decay
     regime, so future changes that perturb the dynamics are caught:
       - timer_threshold == 0  (synapse tau <= surrogate_tau): decay every step
       - timer_threshold  > 0  (synapse tau  > surrogate_tau): periodic decay

Run from tests/ so the installed `iqif` is imported, not the repo's ./iqif
shadow. noise=0 in the param file is forced to 1 internally, and rand()%1 == 0,
so the simulation is fully deterministic.
"""
import os
import sys
import tempfile
from iqif import iqnet

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        print(f"  PASS  {name}"); passed += 1
    else:
        print(f"  FAIL  {name}"); failed += 1


def make_cfg(tau_a, tau_b):
    d = tempfile.mkdtemp(prefix="iqdecay_")
    par = os.path.join(d, "p.txt"); con = os.path.join(d, "c.txt")
    with open(par, "w") as f:
        f.write("0 80 235 80 3 4 0\n")   # rest thr reset shift_a shift_b noise
        f.write("1 80 235 80 3 4 0\n")
    with open(con, "w") as f:
        f.write(f"0 1 10 {tau_a}\n")     # pre post weight tau  (exc)
        f.write(f"1 0 -5 {tau_b}\n")     # (inh)
    return par, con


def spike_trains(tau_a, tau_b, steps=150):
    par, con = make_cfg(tau_a, tau_b)
    net = iqnet(par, con)
    net.set_biascurrent(0, 13)
    net.set_biascurrent(1, 12)
    s0, s1 = [], []
    for t in range(steps):
        net.send_synapse()
        if net.get_is_firing(0): s0.append(t)
        if net.get_is_firing(1): s1.append(t)
    return s0, s1, [net.get_decay_threshold(0), net.get_decay_threshold(1)]


# --- Test 1: observability invariant (the fix) ---
# A single presynaptic spike deposits weight=10 into neuron 1. With the leak at
# the head of the step, the accumulator read after that step is the undecayed
# 10. (The pre-fix trailing decay would have returned 10 - (10>>3) = 9.)
print("\n=== Test 1: accumulator reports pre-decay input ===")
par, con = make_cfg(8, 8)
net = iqnet(par, con)
net.set_biascurrent(0, 0); net.set_biascurrent(1, 0)
net.set_is_firing(0, 1)            # force neuron 0 to deposit into neuron 1
net.send_synapse()
acc = net.get_current_accumulator(1)
check("accumulator == deposited weight (10, not decayed 9)", acc == 10)

# --- Test 2: golden spike trains, timer_threshold == 0 ---
print("\n=== Test 2: spike trains, timer_threshold == 0 (tau 8/8) ===")
s0, s1, thr = spike_trains(8, 8)
check("decay thresholds are [0, 0]", thr == [0, 0])
check("neuron0 spikes", s0 == [18, 40, 61, 83, 105, 126, 147])
check("neuron1 spikes", s1 == [20, 35, 49, 66, 85, 100, 114, 131, 149])

# --- Test 3: golden spike trains, timer_threshold > 0 ---
print("\n=== Test 3: spike trains, timer_threshold > 0 (tau 32/64) ===")
s0, s1, thr = spike_trains(32, 64)
check("decay thresholds are [8, 4]", thr == [8, 4])
check("neuron0 spikes", s0 == [18])
check("neuron1 spikes", s1 == [20, 30, 42, 57, 77, 98, 119, 141])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
