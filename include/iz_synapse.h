#ifndef IZ_SYNAPSE_H
#define IZ_SYNAPSE_H
#include "iz_neuron.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int linenum_neuronParameter();
void set_neurons(int num_neurons, iz_neuron *neurons);
void get_weight(int num_neurons, float *weight);
void send_synapse(int num_neurons, iz_neuron *neurons,
                  float *weight, int tau, float *current,
                  float *biascurrent);
void delete_all(float *weight, float *current, iz_neuron *neurons);

#endif

