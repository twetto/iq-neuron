#ifndef IQ_SYNAPSE_H
#define IQ_SYNAPSE_H
#include "iq_neuron.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

class iq_synapse
{
    public:
        iq_synapse() {};
    private:

};

void get_weight(FILE *fp, int num_neurons, int *weight);
void send_synapse(int num_neurons, iq_neuron *neurons,      // Add current
                  int *weight, int tau, int *current);      // before this
void delete_all(int *weight, int *current, iq_neuron *neurons);

#endif

