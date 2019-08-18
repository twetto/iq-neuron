#ifndef IZ_NETWORK_H
#define IZ_NETWORK_H
#include "iz_neuron.h"
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <time.h>
#include <math.h>

class iz_network
{
public:
    iz_network();
    ~iz_network();
    int set_neurons();
    int get_weight();
    int num_neurons();
    void send_synapse();
    void printfile(FILE **fp_potential, FILE **fp_adaptive_term);
    void set_biascurrent(int neuron_index, float biascurrent);

private:
    int linenum_neuronParameter();
    int _num_neurons;
    int _tau = 0;
    float *_weight, *_current, *_biascurrent;
    iz_neuron *_neurons;

};



#endif

