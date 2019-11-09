/* IQIF neuron object
 * Chen-Fu Yeh, 2019/11/09
 */

#ifndef IQ_NEURON_H
#define IQ_NEURON_H
#include <stdio.h>
#include <stdlib.h>

#define MAX_POTENTIAL 255

class iq_network;

class iq_neuron
{
public:
    iq_neuron() {};
    iq_neuron(int rest, int threshold,      // Set equation & noise strength
              int reset, int a, int b, int noise);
    bool is_set();
    void set(int rest, int threshold,       // Set equation & noise strength
             int reset, int a, int b, int noise);
    void iq(int external_current);          // Solve ODE
    int potential();
    bool is_firing();
    int spike_count();
    float spike_rate();

    friend class iq_network;                // For direct access to _is_firing

private:
    int t_neuron;                                   // Iterator of timestep
    int _rest, _threshold, _a, _b, _reset, _noise;  // IQ neuron parameters
    int x , f_min, _spike_count = 0;
    bool _is_set = false, _is_firing = false;
};

#endif

