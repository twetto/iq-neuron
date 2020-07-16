# Integer Quadratic Integrate-and-Fire Neuron

## Buildtime Dependencies:

* gcc (C++11)

* OpenMP >= 4.5

* cmake >= 3.9

* python 3

## Compiling & Running:

```bash
mkdir build && cd build
cmake ..
make -j
./libiq < ../inputs/session.txt (or use your custom session)
../utils/iq_plot.py
```

You can change the synaptic weights in the [Connection Table](inputs/Connection_Table_IQIF.txt). The numbers in each lines are `pre-synapse neuron index, post-synapse neuron index, weight, time constant` respectively.

You can change the neuron parameters in the [neuron parameter file](inputs/neuronParameter_IQIF.txt). The parameters in each lines are `neuron index, rest potential, threshold potential, reset potential, noise strength` respectively.

It is recommended to use multithreading only when number of neurons is large (>100 for example).

I also have an [Izhikevich model](include/iz_network.h) for comparison. You need to change the [main code](src/main.cpp) to let it work though. You can also uncomment `add_library` in the [CMakeLists](CMakeLists.txt) to get a shared library binary.

![IQIF & Izhikevich performing WTA](WTA.png)

