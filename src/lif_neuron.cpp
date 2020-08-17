/* LIF neuron object
 * Chen-Fu Yeh, 2020/07/02
 */

#include "lif_neuron.h"

using namespace std;

bool lif_neuron::is_set()
{
    return _is_set;
}

void lif_neuron::set(float g, float rest, float threshold,
                     float reset, int noise)
{
    t_neuron = 0;
    _g = g;
    _rest = rest;
    _threshold = threshold;
    _reset = reset;
    if(noise == 0) noise++;
    else if(noise < 0) noise = -noise;
    _noise = noise;
    _v = rest;
    _is_set = true;
    return;
}

void lif_neuron::lif_rk4(float external_current)
{
    float fa1, fa2, fa3, fa4;

    funca(fa1, external_current, 0.0, 0.0);
    
    funca(fa2, external_current, 0.5, fa1);

    funca(fa3, external_current, 0.5, fa2);

    funca(fa4, external_current, 1.0, fa3);

    _v += (fa1 + 2 * fa2 + 2 * fa3 + fa4) / 6.0;
    
    _is_firing = false;
    if(_v > _threshold) {
        _spike_count++;
        _is_firing = true;
        _v = _reset;
    }
    //else if(_v < 0) _v = 0;

    t_neuron++;
    return;
}

void lif_neuron::lif_euler(float external_current)
{
    if(_r_count == 0) {
        _v += _g * (_v - _rest) + external_current + rand()%_noise-_noise/2;
    }
    else {
        _r_count--;
    }
    
    _is_firing = false;
    if(_v > _threshold) {
        _spike_count++;
        _is_firing = true;
        _v = _reset;
        _r_count = _r_period;
    }
    //else if(_v < 0) _v = 0;

    t_neuron++;
    return;
}

float lif_neuron::potential()
{
    return _v;
}

bool lif_neuron::is_firing()
{
    return _is_firing;
}

int lif_neuron::spike_count()
{
    int count = _spike_count;
    _spike_count = 0;
    return count;
}

float lif_neuron::spike_rate()
{
    float r = _spike_count / (float) t_neuron;
    t_neuron = 0;
    _spike_count = 0;
    return r;
}

void lif_neuron::funca(float &fa, float &I, const float dtt,
                       const float arg)
{
    float tmpv;
    tmpv = _v + dtt * arg;
    fa = _g * (tmpv - _rest) + I + rand()%_noise-_noise/2;
    return;

}

/*
void iz_neuron::funcb(float &fb, const float dtt,
                      const float arg1, const float arg2)
{
    float tmpv, tmpu;
    tmpv = _v + dtt * arg1;
    tmpu = _u + dtt * arg2;
    fb = _a * ( _b * (tmpv - _rest) - tmpu);
    return;

}
*/
 
