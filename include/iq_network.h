/* IQIF network
 * Chen-Fu Yeh, 2019/11/09
 */

#ifndef IQ_NETWORK_H
#define IQ_NETWORK_H
#include "iq_neuron.h"
#include "weight_index_list.h"
#include <math.h>
#include <omp.h>

class iq_network
{
public:
    iq_network();
    ~iq_network();
    int num_neurons();
    void send_synapse();                    // proceed one timestep
    void printfile(FILE **fp);              // output potentials
    void set_biascurrent(int neuron_index, int biascurrent);
    int potential(int neuron_index);
    int spike_count(int neuron_index);
    float spike_rate(int neuron_index);
    void set_num_threads(int num_threads);  // for multithreading

private:
    int linenum_neuronParameter();  // get # of neurons by # of lines in file
    int set_neurons();              // read from input/neuronParameter.txt
    int get_weight();               // read from input/connectionTable.txt
    int _num_neurons;
    int *_tau, *_f, *_n;            // synapse decay time constant & siblings
    int *_weight;                   // synapse weight matrix
    int *_scurrent;                 // synapse current matrix
    int *_ncurrent;                 // neuron input synapse current
    int *_biascurrent;              // neuron bias current
    iq_neuron *_neurons;
    weight_index_list *_wlist;      // axon index for each neurons
    int _num_threads = 1;
};

#endif

