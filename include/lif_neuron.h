#ifndef LIF_NEURON_H
#define LIF_NEURON_H
#include <stdio.h>
#include <stdlib.h>

class lif_neuron
{
public:
    lif_neuron() {};
    bool is_set();
    void set(float g, float rest, float threshold, float reset, int noise);
    void lif_rk4(float external_current);
    void lif_euler(float external_current);
    float potential();
    bool is_firing();
    int spike_count();
    float spike_rate();

private:
    void funca(float &fa, float &I, const float dtt,
               const float arg);
    int t_neuron;
    float _v = 0;
    float _g, _rest, _threshold, _reset;
    int _noise;
    float VMAX = 255;
    int _spike_count = 0;
    int _r_count = 0, _r_period = 0;
    bool _is_set = false, _is_firing = false;
};

class ilif_neuron
{
public:
    ilif_neuron() {};
    bool is_set();
    //void set(float g, float rest, float threshold, float reset, int noise);
    void set(int inv_g, int rest, int threshold, int reset, int noise);
    //void ilif_rk4(float external_current);
    //void ilif_euler(float external_current);
    void ilif_rk4(int external_current);
    void ilif_euler(int external_current);
    //float potential();
    int potential();
    bool is_firing();
    int spike_count();
    float spike_rate();

private:
    //void funca(float &fa, float &I, const float dtt,
    //           const float arg);
    int t_neuron;
    //float _v = 0;
    //float _g, _rest, _threshold, _reset;
    int _v = 0;
    int _inv_g, _rest, _threshold, _reset;
    int _noise;
    //float VMAX = 255;
    int VMAX = 255;
    int _spike_count = 0;
    int _r_count = 0, _r_period = 0;
    bool _is_set = false, _is_firing = false;
};

#endif

