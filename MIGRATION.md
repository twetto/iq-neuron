# Migration Guide: v0.2.x to v0.3.0

## Overview

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

## Questions?

Contact: chen_fu_yeh@lolab-nthu.org
