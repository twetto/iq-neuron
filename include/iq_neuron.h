#ifndef IQ_NEURON_H
#define IQ_NEURON_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

#define MAX_POTENTIAL 65535

class iq_neuron
{
public:
    iq_neuron() {};
    iq_neuron(int rest, int threshold,      // Set neuron nullcline
              int reset, int a, int b);
    bool is_set();
    void set(int rest, int threshold,       // Set neuron nullcline
             int reset, int a, int b);
    void iq();                              // Solve ODE
    void iq(int external_current);          // Solve ODE with external input
    int potential();
    bool is_fired();
    float spike_rate();
    void reset_time();                      // Remember to reset accordingly
    void reset_spike_count();               // to get proper spiking rate

private:
    int t_neuron;                           // Iterator of timestep
    int _rest, _threshold, _a, _b, _reset;  // Iq neuron parameters
    int x , nullcline_min, spike_count;
    bool _is_set = false, _is_fired = false;
};

#endif

