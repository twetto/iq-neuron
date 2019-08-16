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
../utils/iq_plot.py
```

By default there are two neurons inhibiting each other to perform the winner-take-all behavior.

You can change the synaptic weights in the [Connection Table](inputs/Connection_Table_IQIF.txt).

You can change the neuron parameters in the [neuron parameter file](inputs/neuronParameter_IQIF.txt). The parameters in each lines are `neuron index, rest potential, threshold potential, reset potential, noise strength` respectively.

I also have a [Izhikevich model](include/iz_network.h) for comparison. You can check how it performs WTA as well using `../utils/iz_plot.py`.

![IQIF & Izhikevich performing WTA](WTA.png)

