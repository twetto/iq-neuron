# Integer Quadratic Integrate-and-Fire Neuron

## Buildtime Dependencies:

* gcc

* cmake

* python 3

## Compiling & Running:

```bash
mkdir build && cd build
cmake ..
make
./main
../utils/theta_plot.py
```

You can change the synaptic weights in the [Connection Table](inputs/Connection_Table.txt).

You can change the neuron parameters in the [neuron parameter file](inputs/neuronParameter.txt). The parameters in each lines are `neuron index, rest potential, threshold potential, reset potential, noise strength` respectively.

