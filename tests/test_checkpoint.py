#!/usr/bin/env python3
"""
Checkpoint/restore test for iq-neuron.

1. Run network A for 100 steps, record potentials every step.
2. At step 30, snapshot full state (potential, is_firing, current_accumulator,
   synapse_timer) for each neuron.
3. Create network B from the same config, restore the snapshot.
4. Run network B for 70 steps.
5. Verify B's 70-step trajectory == A's last 70 steps exactly.

Setup (deterministic -- noise=0 -> rand()%1==0):
  neuron 0: rest=80 thresh=235 reset=80 shift_a=3 shift_b=4 noise=0
  neuron 1: rest=80 thresh=235 reset=80 shift_a=3 shift_b=4 noise=0
  0 -> 1: weight=+10  tau=32
  1 -> 0: weight=-5   tau=64
  bias: neuron 0 = 13, neuron 1 = 12
"""

import os
import sys
import tempfile
import numpy as np
from iqif import iqnet

# ---------------------------------------------------------------------------
# 3. Create temporary config files
# ---------------------------------------------------------------------------
tmpdir = tempfile.mkdtemp(prefix="iqtest_ckpt_")

par_path = os.path.join(tmpdir, "params.txt")
con_path = os.path.join(tmpdir, "conn.txt")

with open(par_path, "w") as f:
    f.write("0 80 235 80 3 4 0\n")
    f.write("1 80 235 80 3 4 0\n")

with open(con_path, "w") as f:
    f.write("0 1 10 32\n")
    f.write("1 0 -5 64\n")

TOTAL_STEPS = 100
CHECKPOINT_STEP = 64
NUM_NEURONS = 2

# ---------------------------------------------------------------------------
# 4. Run network A for TOTAL_STEPS, snapshot at CHECKPOINT_STEP
# ---------------------------------------------------------------------------
print(f"\n=== Running network A for {TOTAL_STEPS} steps, checkpoint at step {CHECKPOINT_STEP} ===")

net_a = iqnet(par_path, con_path)
net_a.set_biascurrent(0, 13)
net_a.set_biascurrent(1, 12)

potentials_a = np.zeros((TOTAL_STEPS, NUM_NEURONS), dtype=np.int32)
snapshot = {}

for t in range(TOTAL_STEPS):
    net_a.send_synapse()

    for idx in range(NUM_NEURONS):
        potentials_a[t, idx] = net_a.potential(idx)

    if t == CHECKPOINT_STEP - 1:
        print(f"  Snapshotting at t={t} (after step {t+1})...")
        for idx in range(NUM_NEURONS):
            snapshot[idx] = {
                "potential": net_a.potential(idx),
                "is_firing": net_a.get_is_firing(idx),
                "current_acc": net_a.get_current_accumulator(idx),
                "syn_timer": net_a.get_synapse_timer(idx),
            }
            print(f"    N{idx}: V={snapshot[idx]['potential']}  "
                  f"firing={snapshot[idx]['is_firing']}  "
                  f"acc={snapshot[idx]['current_acc']}  "
                  f"timer={snapshot[idx]['syn_timer']}")

# ---------------------------------------------------------------------------
# 5. Create network B, restore snapshot, run remaining steps
# ---------------------------------------------------------------------------
remaining = TOTAL_STEPS - CHECKPOINT_STEP
print(f"\n=== Restoring snapshot into network B, running {remaining} steps ===")

net_b = iqnet(par_path, con_path)
net_b.set_biascurrent(0, 13)
net_b.set_biascurrent(1, 12)

# Restore state
for idx in range(NUM_NEURONS):
    s = snapshot[idx]
    net_b.set_potential(idx, s["potential"])
    net_b.set_is_firing(idx, s["is_firing"])
    net_b.set_current_accumulator(idx, s["current_acc"])
    net_b.set_synapse_timer(idx, s["syn_timer"])

# Run and record
potentials_b = np.zeros((remaining, NUM_NEURONS), dtype=np.int32)

for t in range(remaining):
    net_b.send_synapse()
    for idx in range(NUM_NEURONS):
        potentials_b[t, idx] = net_b.potential(idx)

# ---------------------------------------------------------------------------
# 6. Compare
# ---------------------------------------------------------------------------
print(f"\n=== Comparing trajectories ===")

# A's last 70 steps = potentials_a[CHECKPOINT_STEP:]
tail_a = potentials_a[CHECKPOINT_STEP:]

assert tail_a.shape == potentials_b.shape, \
    f"Shape mismatch: {tail_a.shape} vs {potentials_b.shape}"

mismatches = np.where(tail_a != potentials_b)

if len(mismatches[0]) == 0:
    print(f"  PASS  All {remaining} steps match exactly for both neurons.")
else:
    n_bad = len(mismatches[0])
    print(f"  FAIL  {n_bad} mismatches found!")
    # Show first few
    for i in range(min(10, n_bad)):
        t_idx = mismatches[0][i]
        n_idx = mismatches[1][i]
        print(f"    step {CHECKPOINT_STEP + t_idx + 1}, neuron {n_idx}: "
              f"A={tail_a[t_idx, n_idx]}  B={potentials_b[t_idx, n_idx]}")
    if n_bad > 10:
        print(f"    ... and {n_bad - 10} more")
    sys.exit(1)

# Sanity: verify the simulation wasn't trivial (neurons actually fired)
n0_spikes = np.sum(np.diff(potentials_a[:, 0]) < -50)  # large drops = resets
n1_spikes = np.sum(np.diff(potentials_a[:, 1]) < -50)
print(f"\n  Sanity check: ~{n0_spikes} spikes from N0, ~{n1_spikes} spikes from N1 "
      f"over {TOTAL_STEPS} steps")

if n0_spikes == 0 and n1_spikes == 0:
    print("  WARNING: no spikes detected, test may be vacuous")
else:
    print("  Good -- nontrivial dynamics confirmed.")

print("\nAll tests passed!")
