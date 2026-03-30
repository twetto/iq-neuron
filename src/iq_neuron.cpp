/* IQIF neuron object
 * Chen-Fu Yeh, 2019/11/09
 */

#include "iq_neuron.h"

using namespace std;

iq_neuron::iq_neuron(int rest, int threshold,
                     int reset, int shift_a, int shift_b, int noise)
{
    set(rest, threshold, reset, shift_a, shift_b, noise);
    return;
}

bool iq_neuron::is_set()
{
    return _is_set;
}

void iq_neuron::set(int rest, int threshold,
                    int reset, int shift_a, int shift_b, int noise)
{
    x = rest;                               // initialize with rest potential
    t_neuron = 0;
    
    // dV/dt and others
    // f_min = (a*rest + b*threshold) / (a+b)
    // With a = 1/(1<<shift_a), b = 1/(1<<shift_b):
    // f_min = (rest*(1<<shift_b) + threshold*(1<<shift_a)) / ((1<<shift_a) + (1<<shift_b))
    int weight_a = 1 << shift_b;    // weight for rest (inverse of a's shift)
    int weight_b = 1 << shift_a;    // weight for threshold (inverse of b's shift)
    f_min = (weight_a * rest + weight_b * threshold) / (weight_a + weight_b);

    _shift_a = shift_a;
    _shift_b = shift_b;
    _rest = rest;
    _threshold = threshold;
    _reset = reset;
    if(noise == 0) noise++;                 // set noise strength
    else if(noise < 0) noise = -noise;
    _noise = noise;
    _is_set = true;
    _synapse.init(32, 8);                   // Default safety.
                                            // Will be set again in network init
    return;
}

void iq_neuron::set_vmax(int vmax)
{
    VMAX = vmax;
    return;
}

void iq_neuron::set_vmin(int vmin)
{
    VMIN = vmin;
    return;
}

void iq_neuron::update_state(int external_current)
{
    // Capture the Input Current (undecayed sum of spikes from t-1)
    int current_val = _synapse.current_accumulator;

    // Decay the accumulator (Preparing it for t+1)
    _synapse.step();
    
    // Solve ODE
    int f;
    int total_input = current_val + external_current;

    if(x < f_min)
        f = (_rest - x) >> _shift_a;
    else
        f = (x - _threshold) >> _shift_b;
    
    x += f + total_input + rand()%_noise - (_noise >> 1);

    _is_firing = false;
    if(x >= VMAX) {
        _spike_count++;
        _is_firing = true;
        x = x - (VMAX - _reset);
    }
    if(x < VMIN) x = VMIN;
    t_neuron++;
}

void iq_neuron::receive_spike(int weight)
{
    _synapse.add_input(weight);
}

void iq_neuron::set_synapse_tau(int apparent_tau, int s_tau)
{
    _synapse.init(apparent_tau, s_tau);
}

void iq_neuron::set_surrogate_tau(int s_tau) {
    _synapse.set_surrogate_tau(s_tau);
}

int iq_neuron::get_surrogate_tau(){
    return _synapse.surrogate_tau;
}

int iq_neuron::get_decay_threshold()
{
    return _synapse.timer_threshold;
}

int iq_neuron::potential()
{
    return x;
}

bool iq_neuron::is_firing()
{
    return _is_firing;
}

int iq_neuron::spike_count()
{
    int count = _spike_count;
    _spike_count = 0;
    return count;
}

float iq_neuron::spike_rate()
{
    float r = _spike_count / (float) (t_neuron ? t_neuron : 1);
    t_neuron = 0;
    _spike_count = 0;
    return r;
}

