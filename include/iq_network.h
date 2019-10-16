#ifndef IQ_NETWORK_H
#define IQ_NETWORK_H
#include "iq_neuron.h"
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <time.h>
#include <math.h>

class iq_network
{
public:
    iq_network();
    ~iq_network();
    int set_neurons();
    int get_weight();
    int num_neurons();
    void send_synapse();
    void printfile(FILE **fp);
    void set_biascurrent(int neuron_index, int biascurrent);
    int potential(int neuron_index);
    int spike_count(int neuron_index);
    float spike_rate(int neuron_index);

private:
    int linenum_neuronParameter();
    int _num_neurons;
    int *_tau, *_f, *_n;
    int *_weight, *_scurrent, *_ncurrent, *_biascurrent;
    iq_neuron *_neurons;

};

#endif

