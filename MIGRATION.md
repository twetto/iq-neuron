# Migration Guide

## v0.2.x → v0.3.0: Bit-shift Slope Parameters

### Overview

Version 0.3.0 replaces the `a` and `b` multiplier parameters with `shift_a` and `shift_b` bit-shift parameters. This change enables direct hardware implementation using barrel shifters instead of multipliers.

## Parameter File Format Change

### Old format (v0.2.x)
```
index rest threshold reset a b noise
```

### New format (v0.3.0)
```
index rest threshold reset shift_a shift_b noise
```

## Migration Formula

```
shift = 3 - log2(a_old)
```

Or use this lookup table:

| Old `a` | New `shift` | Effective Rate |
|---------|-------------|----------------|
| 1       | 3           | 1/8            |
| 2       | 2           | 1/4            |
| 4       | 1           | 1/2            |
| 8       | 0           | 1              |

The same conversion applies to `b` → `shift_b`.

## Example

Old `neuronParameter_IQIF.txt`:
```
0 62 130 145 1 1 0
1 62 130 50 2 2 0
```

New `neuronParameter_IQIF.txt`:
```
0 62 130 145 3 3 0
1 62 130 50 2 2 0
```

## Limitations

- Effective rates are now restricted to powers of 2: 1, 1/2, 1/4, 1/8, 1/16, ...
- Old configurations with `a` values not equal to 1, 2, 4, or 8 will need approximation
- Maximum effective rate is 1 (`shift=0`), equivalent to old `a=8`

## API Changes

### C++ API
```cpp
// Old
set_neuron(index, rest, threshold, reset, a, b, noise);

// New
set_neuron(index, rest, threshold, reset, shift_a, shift_b, noise);
```

### C API
```c
// Old
iq_network_set_neuron(network, index, rest, threshold, reset, a, b, noise);

// New
iq_network_set_neuron(network, index, rest, threshold, reset, shift_a, shift_b, noise);
```

## Internal Changes

### Old dynamics
```cpp
if (x < f_min)
    f = _a * (_rest - x);
else
    f = _b * (x - _threshold);
x += (f >> 3) + input + noise;
```

### New dynamics
```cpp
if (x < f_min)
    f = (_rest - x) >> _shift_a;
else
    f = (x - _threshold) >> _shift_b;
x += f + input + noise;
```

## v0.3.x → v0.4.0: Synaptic Decay Phase Change

The synaptic current decay was moved from the **tail of `update_state()`** to the
**head of `send_synapse()`** (a new decay pass that runs *before* spikes are
accumulated).

### Why

`get_current_accumulator()` / `get_all_current_accumulators()` previously returned
the **post-decay residual** carried into the next step, not the post-synaptic
current the neuron actually integrated. The accumulator was read into a local and
then immediately overwritten by the trailing decay, so the integrated value was
never externally observable.

With the leak applied at the head of the step, `current_accumulator` now persists
the **true input the neuron integrated this step** (`decay(I_{t-1}) + S_t`). This
is also the more hardware-faithful form: on a digital datapath the accumulator
register holds the synaptic current the soma integrates at the clock edge, and the
leak is a next-cycle register operation.

### Behavioral impact — check `get_decay_threshold(i)`

| `timer_threshold` (per neuron) | When | Effect on dynamics |
|---|---|---|
| `== 0` | synapse `tau <= surrogate_tau` (decay fires every step) | **Bit-identical** — membrane potentials and spike trains unchanged. |
| `> 0`  | synapse `tau > surrogate_tau` (periodic decay) | Decay now lands **one step earlier**. Spike timing can shift by ~1 step. **Re-validate / re-tune these configs.** |

The decay phase is a sub-step discretization detail; neither phase is "more
correct," so the new (hardware-natural) ordering is now the canonical one rather
than the legacy simulator's accidental phase. Configs that ran `tau <= surrogate_tau`
are unaffected.

### Readout semantics changed

`get_current_accumulator()` now reports the **pre-decay integrated input** instead
of the post-decay residual. Update any analysis code, fixtures, or saved checkpoints
that depended on the old value. `set_current_accumulator()` still restores the
persisted accumulator (the two are symmetric within this version).

## Questions?

Contact: chen_fu_yeh@lolab-nthu.org
