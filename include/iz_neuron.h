#ifndef IZ_NEURON_H
#define IZ_NEURON_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

class iz_neuron
{
public:
    iz_neuron() {};
    bool is_set();
    void set(float a, float b, float c, float d, float k,
             float rest, float threshold, int noise);
    void iz_rk4(float external_current);
    void iz_euler(float external_current);
    float potential();
    float adaptive_term();
    bool is_firing();
    int spike_count();
    float spike_rate();
    void reset_time();
    void reset_spike_count();

private:
    void funca(float &fa, const float I, const float dtt,
               const float arg1, const float arg2);
    void funcb(float &fb, const float dtt,
               const float arg1, const float arg2);
    int t_neuron;
    float _v = 0, _u = 0, _a, _b, _c, _d;
    float _k, _rest, _threshold;
    int _noise;
    const float VMAX = 255;
    int _spike_count;
    bool _is_set = false, _is_firing = false;
};

#endif

