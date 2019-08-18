#include "iz_neuron.h"

using namespace std;

bool iz_neuron::is_set()
{
    return _is_set;
}

void iz_neuron::set(float a, float b, float c, float d, float k,
                    float rest, float threshold, int noise)
{
    t_neuron = 0;
    _a = a;
    _b = b;
    _c = c;
    _d = d;
    _k = k;
    _rest = rest;
    _threshold = threshold;
    if(noise == 0) noise++;
    else if(noise < 0) noise = -noise;
    _noise = noise;
    _v = rest;
    _u = 0;
    _is_set = true;
    //printf("a = %f\nb = %f\nc = %f\nd = %f\nk = %f\nrest = %f\nthreshold = %f\nnoise = %d\n", _a, _b, _c, _d, _k, _rest, _threshold, _noise);
    return;
}

void iz_neuron::iz_rk4(float external_current)
{
    float fa1, fa2, fa3, fa4;
    float fb1, fb2, fb3, fb4;

    funca(fa1, external_current, 0.0, 0.0, 0.0);
    funcb(fb1, 0.0, 0.0, 0.0);
    
    funca(fa2, external_current, 0.5, fa1, fb1);
    funcb(fb2, 0.5, fa1, fb1);

    funca(fa3, external_current, 0.5, fa2, fb2);
    funcb(fb3, 0.5, fa2, fb2);

    funca(fa4, external_current, 1.0, fa3, fb3);
    funcb(fb4, 1.0, fa3, fb3);

    _v += (fa1 + 2 * fa2 + 2 * fa3 + fa4) / 6.0;
    _u += (fb1 + 2 * fb2 + 2 * fb3 + fb4) / 6.0;
    
    _is_firing = false;
    if(_v > VMAX) {
        _spike_count++;
        _is_firing = true;
        _v = _c;
        _u += _d;
        //printf("firing...\n");
    }
    //else if(_v < 0) _v = 0;

    t_neuron++;
    return;
}

void iz_neuron::iz_euler(float external_current)
{
    _v += _k * (_v - _rest) * (_v - _threshold) - _u + external_current + rand()%_noise-_noise/2;
    _u += _a * ( _b * (_v - _rest) - _u);
    
    _is_firing = false;
    if(_v > VMAX) {
        _spike_count++;
        _is_firing = true;
        _v = _c;
        _u += _d;
    }
    //else if(_v < 0) _v = 0;

    t_neuron++;
    return;
}

float iz_neuron::potential()
{
    return _v;
}

float iz_neuron::adaptive_term()
{
    return _u;
}

bool iz_neuron::is_firing()
{
    return _is_firing;
}

int iz_neuron::spike_count()
{
    return _spike_count;
}

float iz_neuron::spike_rate()
{
    float r = _spike_count / (float) t_neuron;
    reset_time();
    reset_spike_count();
    return r;
}

void iz_neuron::reset_time()
{
    t_neuron = 0;
    return;
}

void iz_neuron::reset_spike_count()
{
    _spike_count = 0;
    return;
}

void iz_neuron::funca(float &fa, const float I, const float dtt,
                      const float arg1, const float arg2)
{
    float tmpv, tmpu;
    tmpv = _v + dtt * arg1;
    tmpu = _u + dtt * arg2;
    fa = _k * (tmpv - _rest) * (tmpv - _threshold) - tmpu + I + rand()%_noise-_noise/2;
    return;

}

void iz_neuron::funcb(float &fb, const float dtt,
                      const float arg1, const float arg2)
{
    float tmpv, tmpu;
    tmpv = _v + dtt * arg1;
    tmpu = _u + dtt * arg2;
    fb = _a * ( _b * (tmpv - _rest) - tmpu);
    return;

}
 
