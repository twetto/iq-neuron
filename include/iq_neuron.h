/* IQIF neuron object
 * Chen-Fu Yeh, 2019/11/09
 */

#ifndef IQ_NEURON_H
#define IQ_NEURON_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

class iq_network;

struct SynapseGroup
{
    int current_accumulator;
    int timer;
    int timer_threshold; 
    
    int apparent_tau;   
    int surrogate_tau;
    int decay_num;      
    int decay_den;      

    void init(int app_tau, int sur_tau) {
        apparent_tau = app_tau;
        surrogate_tau = sur_tau;
        current_accumulator = 0;
        timer = 0;
        recalculate_params();
    }

    void recalculate_params() {
        decay_num = surrogate_tau - 1;
        decay_den = surrogate_tau;

        if (apparent_tau <= surrogate_tau) {
             timer_threshold = 0; 
        } else {
             // Exact logic from original iq_network.cpp
             float num = log10((float)decay_num / decay_den);
             float den = log10(((float)apparent_tau - 1) / apparent_tau);
             timer_threshold = (int)(num / den);
        }
    }

    // Updates s_tau without changing apparent_tau
    void set_surrogate_tau(int s_tau) {
        surrogate_tau = s_tau;
        recalculate_params();
    }
    
    // Updates apparent_tau (e.g. set_weight) without resetting s_tau
    void set_apparent_tau(int app_tau) {
        apparent_tau = app_tau;
        recalculate_params();
    }

    void step() {
        if (timer > timer_threshold) {
             if (decay_den != 0)
                current_accumulator = (current_accumulator * decay_num) / decay_den;
             timer = 0;
        }
        timer++;
    }
    
    void add_input(int weight) {
        #pragma omp atomic
        current_accumulator += weight;
    }
};

class iq_neuron
{
public:
    iq_neuron() {};
    iq_neuron(int rest, int threshold,      // Set equation & noise strength
              int reset, int a, int b, int noise);
    bool is_set();
    void set(int rest, int threshold,       // Set equation & noise strength
             int reset, int a, int b, int noise);
    void set_vmax(int vmax);
    void set_vmin(int vmin);

    // Per-neuron synaptic decay logic
    void update_state(int external_current); 
    void receive_spike(int weight);
    void set_synapse_tau(int apparent_tau, int s_tau);
    void set_surrogate_tau(int s_tau);

    int get_surrogate_tau();
    int get_decay_threshold();

    int potential();
    bool is_firing();
    int spike_count();
    float spike_rate();

    friend class iq_network;                // For direct access to _is_firing

private:
    int t_neuron;                                   // Iterator of timestep
    int _rest, _threshold, _a, _b, _reset, _noise;  // IQ neuron parameters
    int x , f_min, _spike_count = 0;
    int VMAX = 255;
    int VMIN = 0;
    bool _is_set = false, _is_firing = false;
    
    SynapseGroup _synapse;  // Manages post-synaptic currents
};

#endif

