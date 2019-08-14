#ifndef IQ_NEURON_H
#define IQ_NEURON_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

#define MAX_POTENTIAL 256

class iq_neuron
{
public:
    iq_neuron() {};
    iq_neuron(int rest, int threshold,      // Set equation & noise strength
              int reset, int a, int b, int noise);
    bool is_set();
    void set(int rest, int threshold,       // Set equation & noise strength
             int reset, int a, int b, int noise);
    void iq();                              // Solve ODE
    void iq(int external_current);          // Solve ODE with external input
    int potential();
    bool is_firing();
    int spike_count();
    float spike_rate();
    void reset_time();                      // Remember to reset accordingly
    void reset_spike_count();               // to get proper spiking rate

private:
    int t_neuron;                                   // Iterator of timestep
    int _rest, _threshold, _a, _b, _reset, _noise;  // IQ neuron parameters
    int x , f_min, _spike_count;
    bool _is_set = false, _is_firing = false;
};

#endif

