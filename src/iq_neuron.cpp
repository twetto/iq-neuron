/* IQIF neuron object
 * Chen-Fu Yeh, 2019/11/09
 */

#include "iq_neuron.h"

using namespace std;

iq_neuron::iq_neuron(int rest, int threshold,
                     int reset, int a, int b, int noise)
{
    x = rest;                               // initialize with rest potential
    t_neuron = 0;
    f_min = (a*rest + b*threshold) / (a+b); // dV/dt and others
    _a = a;
    _b = b;
    _rest = rest;
    _threshold = threshold;
    _reset = reset;
    if(noise == 0) noise++;                 // set noise strength
    else if(noise < 0) noise = -noise;
    _noise = noise;
    _is_set = true;
    return;
}

bool iq_neuron::is_set()
{
    return _is_set;
}

void iq_neuron::set(int rest, int threshold,
                    int reset, int a, int b, int noise)
{
    x = rest;
    t_neuron = 0;
    f_min = (a*rest + b*threshold) / (a+b);
    _a = a;
    _b = b;
    _rest = rest;
    _threshold = threshold;
    _reset = reset;
    if(noise == 0) noise++;
    else if(noise < 0) noise = -noise;
    _noise = noise;
    _is_set = true;
    return;
}

void iq_neuron::iq(int external_current)
{
    int f;

    /* solving dV/dt */
    if(x < f_min)
        f = _a * (_rest - x);
    else
        f = _b * (x - _threshold);
    x += f/10 + external_current + rand()%_noise-_noise/2;

    /* fire if exceeding action potential */
    _is_firing = false;
    if(x > MAX_POTENTIAL) {
        _spike_count++;
        _is_firing = true;
        x = _reset;
    }
    //else if(x < 0) x = 0;
    t_neuron++;
    return;
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
    float r = _spike_count / (float) t_neuron;
    t_neuron = 0;
    _spike_count = 0;
    return r;
}

