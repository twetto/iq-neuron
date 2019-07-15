#include "iq_neuron.h"

using namespace std;

iq_neuron::iq_neuron(int rest, int threshold,
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
    _noise = noise;
    _is_set = true;
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
    _noise = noise;
    _is_set = true;
}

void iq_neuron::iq()
{
    int f;

    if(x < f_min)
        f = _a * (_rest - x);
    else
        f = _b * (x - _threshold);
    x += f/10 + rand()%_noise-_noise/2;
    _is_firing = false;
    if(x > MAX_POTENTIAL) {
        spike_count++;
        _is_firing = true;
        x = _reset;
    }
    t_neuron++;
    return;
}

void iq_neuron::iq(int external_current)
{
    int f;

    if(x < f_min)
        f = _a * (_rest - x);
    else
        f = _b * (x - _threshold);
    x += f/10 + external_current + rand()%_noise-_noise/2;
    _is_firing = false;
    if(x > MAX_POTENTIAL) {
        spike_count++;
        _is_firing = true;
        x = _reset;
    }
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

float iq_neuron::spike_rate()
{
    float t = (float) t_neuron;
    float s = (float) spike_count;
    return s / t;
}

void iq_neuron::reset_time()
{
    t_neuron = 0;
    return;
}

void iq_neuron::reset_spike_count()
{
    spike_count = 0;
    return;
}

