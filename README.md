# Integer Quadratic Integrate-and-Fire Neuron

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Library for IQIF neuron simulation. Includes shared libraries `libiq-network`, `libiz-network`, and `liblif-network` (C++), a Python API via `iqif`, and a standalone binary `libiq` for direct C++ usage.

## Dependencies

**Runtime:** OpenMP >= 4.5, Python >= 3.7, NumPy

**Build:** GCC (C++11), CMake >= 3.15, pip

**macOS only:** `brew install cmake libomp`

## Installation

### Python (recommended)

Builds the C++ libraries and installs everything in one step:

```bash
pip install .
```

Uninstall with `pip uninstall python-iqif`.

### System-wide C++ libraries only

```bash
mkdir build && cd build
cmake .. (-DCMAKE_INSTALL_PREFIX=<your preferred directory>)
make -j
sudo make install
```

#### Debian-based packaging

```bash
sudo checkinstall --pkgname iq-neuron
```

Uninstall with `sudo dpkg -r iq-neuron`.

#### Arch-based packaging

Download the PKGBUILD, go to the working directory, then:

```bash
makepkg -si
```

Uninstall with `sudo pacman -Rs iq-neuron`.

## Quick Start (Python)

```python
from iqif import iqnet

net = iqnet("params.txt", "conn.txt")
net.set_biascurrent(0, 13)

for t in range(1000):
    net.send_synapse()
    print(net.potential(0))
```

## Quick Start (C++)

```bash
mkdir build && cd build
cmake ..
make -j
./libiq < ../inputs/session.txt
../utils/iq_plot.py
```

## Configuration

### Neuron Parameters

Each line in the neuron parameter file:

```
neuron_index  rest  threshold  reset  shift_a  shift_b  noise
```

Where `shift_a` and `shift_b` control the bit-shift dynamics (larger = slower rate):

| `shift` | Effective Rate |
|---------|----------------|
| 0       | 1              |
| 1       | 1/2            |
| 2       | 1/4            |
| 3       | 1/8            |
| 4       | 1/16           |

### Connection Table

Each line in the connection table:

```
pre-synapse_index  post-synapse_index  weight  time_constant
```

> **Note:** If upgrading from v0.2.x, see [MIGRATION.md](MIGRATION.md) for parameter conversion.

## Other Models

Izhikevich and Leaky Integrate-and-Fire models are included for comparison. See [iz_network.h](include/iz_network.h) and [lif_network.h](include/lif_network.h). For the standalone binary, modify [main.cpp](src/main.cpp) to select the desired model.

## Notes

It is recommended to use multithreading only when the number of neurons is large (>100).

![IQIF & Izhikevich performing WTA](WTA.png)
