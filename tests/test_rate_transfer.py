#!/usr/bin/env python3
"""
Test the rate-coded transfer curve: does postsynaptic accumulator
current scale linearly with presynaptic firing rate?

Setup:
  - Neuron 0: driven by varying bias current -> different firing rates
  - Neuron 1: receives spikes from neuron 0 via weight W, tau T
  - Measure neuron 1's steady-state accumulator as a function of
    neuron 0's firing rate.

If the transfer is roughly linear, rate-based predictive coding
is viable on this architecture.
"""

import os
import sys
import tempfile
import numpy as np
from iqif import iqnet

tmpdir = tempfile.mkdtemp(prefix="iqtest_rate_")
par_path = os.path.join(tmpdir, "params.txt")
con_path = os.path.join(tmpdir, "conn.txt")

# Standard spiking neurons (not sub-threshold)
# rest=80, threshold=235, reset=80, shift_a=3, shift_b=4, noise=0
with open(par_path, "w") as f:
    f.write("0 80 235 80 3 4 0\n")
    f.write("1 80 235 80 3 4 0\n")

WEIGHT = 20
TAU = 32

with open(con_path, "w") as f:
    f.write(f"0 1 {WEIGHT} {TAU}\n")

# --- Sweep bias current and measure transfer curve ---
print("=== Rate-coded transfer curve: bias -> rate -> accumulator ===\n")
print(f"  Weight = {WEIGHT}, Tau = {TAU}")
print(f"  {'Bias':>6s}  {'Rate':>8s}  {'Mean Acc':>10s}  {'Std Acc':>10s}")
print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}")

WARMUP = 200   # steps to reach steady state
MEASURE = 500  # steps to measure over

bias_values = [5, 8, 10, 13, 16, 20, 25, 30, 40, 50]
rates = []
mean_accs = []

for bias in bias_values:
    net = iqnet(par_path, con_path)
    net.set_biascurrent(0, bias)
    net.set_biascurrent(1, 0)

    # Warmup
    for _ in range(WARMUP):
        net.send_synapse()
    # Reset spike count after warmup
    net.spike_count(0)

    # Measure
    acc_samples = []
    for t in range(MEASURE):
        net.send_synapse()
        acc_samples.append(net.get_current_accumulator(1))

    spikes = net.spike_count(0)
    rate = spikes / MEASURE
    mean_acc = np.mean(acc_samples)
    std_acc = np.std(acc_samples)

    rates.append(rate)
    mean_accs.append(mean_acc)

    print(f"  {bias:6d}  {rate:8.4f}  {mean_acc:10.2f}  {std_acc:10.2f}")

# --- Linearity analysis ---
print("\n=== Linearity analysis ===")

# Only use points where neuron is actually firing
firing_mask = [r > 0 for r in rates]
firing_rates = [r for r, m in zip(rates, firing_mask) if m]
firing_accs = [a for a, m in zip(mean_accs, firing_mask) if m]

if len(firing_rates) >= 3:
    # Linear regression: acc = slope * rate + intercept
    coeffs = np.polyfit(firing_rates, firing_accs, 1)
    slope, intercept = coeffs

    # R² goodness of fit
    predicted = np.polyval(coeffs, firing_rates)
    ss_res = np.sum((np.array(firing_accs) - predicted) ** 2)
    ss_tot = np.sum((np.array(firing_accs) - np.mean(firing_accs)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"  Linear fit: acc = {slope:.2f} * rate + {intercept:.2f}")
    print(f"  R² = {r_squared:.6f}")
    print(f"  Points used: {len(firing_rates)} (of {len(rates)} total)")

    if r_squared > 0.95:
        print("\n  PASS  Transfer curve is approximately linear (R² > 0.95)")
        print("        Rate-based predictive coding is viable.")
    elif r_squared > 0.85:
        print("\n  WARN  Transfer curve is somewhat linear (R² > 0.85)")
        print("        May work but expect convergence degradation.")
    else:
        print("\n  FAIL  Transfer curve is nonlinear (R² < 0.85)")
        print("        Rate-based PC will likely not converge reliably.")
else:
    print("  Not enough firing data points for analysis.")
    sys.exit(1)

# --- Monotonicity check ---
print("\n=== Monotonicity check ===")
monotonic = all(firing_accs[i] <= firing_accs[i+1]
                for i in range(len(firing_accs) - 1))
print(f"  Accumulator monotonically increasing with rate: {monotonic}")
if monotonic:
    print("  PASS  Monotonic transfer — gradient signal preserved.")
else:
    print("  WARN  Non-monotonic — check for saturation or resonance.")
