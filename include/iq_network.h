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
    iq_network(const char *par = "../inputs/neuronParameter_IQIF.txt",
               const char *con = "../inputs/Connection_Table_IQIF.txt");
    ~iq_network();
    int num_neurons();
    void send_synapse();                    // proceed one timestep
    void printfile(FILE **fp);              // output potentials
    void set_biascurrent(int neuron_index, int biascurrent);
    int potential(int neuron_index);
    int spike_count(int neuron_index);
    float spike_rate(int neuron_index);
    void set_num_threads(int num_threads);  // for multithreading

protected:  // non-copyable and non-movable
    iq_network(const iq_network& other) = delete;
    iq_network(iq_network&& other) = delete;
    iq_network& operator=(const iq_network& other) = delete;
    iq_network& operator=(iq_network&& other) = delete;

private:
    int linenum_neuronParameter(const char *par);   // get # of neurons
    int set_neurons(const char *par);   // read from file par
    int get_weight(const char *con);    // read from file con
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

