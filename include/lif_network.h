/* LIF/ILIF network
 * Chen-Fu Yeh, 2021/04/07
 */

#ifndef LIF_NETWORK_H
#define LIF_NETWORK_H
#include "lif_neuron.h"
#include "weight_index_list.h"
#include <math.h>
#include <omp.h>

class lif_network
{
public:
    lif_network(const char *par = "../inputs/neuronParameter_LIF.txt",
                const char *con = "../inputs/Connection_Table_LIF.txt");
    ~lif_network();
    int num_neurons();
    void send_synapse();
    void printfile(FILE **fp);
    int set_biascurrent(int neuron_index, float biascurrent);
    int set_neuron(int neuron_index, float g, float rest,
                   float threshold, float reset, int noise);
    int set_weight(int pre, int post, float weight, int tau);
    float potential(int neuron_index);
    int spike_count(int neuron_index);
    float spike_rate(int neuron_index);
    void set_num_threads(int num_threads);

protected:  // non-copyable and non-movable
    lif_network(const lif_network& other) = delete;
    lif_network(lif_network&& other) = delete;
    lif_network& operator=(const lif_network& other) = delete;
    lif_network& operator=(lif_network&& other) = delete;

private:
    int linenum_neuronParameter(const char *par);
    int set_neurons(const char *par);
    int get_weight(const char *con);
    int _num_neurons;
    int *_tau;
    float *_weight, *_scurrent, *_ncurrent, *_biascurrent;
    lif_neuron *_neurons;
    weight_index_list *_wlist;
    int _num_threads = 1;
};

class ilif_network
{
public:
    ilif_network(const char *par = "../inputs/neuronParameter_ILIF.txt",
                const char *con = "../inputs/Connection_Table_ILIF.txt");
    ~ilif_network();
    int num_neurons();
    void send_synapse();
    void printfile(FILE **fp);
    int set_biascurrent(int neuron_index, int biascurrent);
    int set_neuron(int neuron_index, int inv_g, int rest,
                   int threshold, int reset, int noise);
    int set_weight(int pre, int post, int weight, int tau);
    int potential(int neuron_index);
    int spike_count(int neuron_index);
    float spike_rate(int neuron_index);
    void set_num_threads(int num_threads);

protected:  // non-copyable and non-movable
    ilif_network(const ilif_network& other) = delete;
    ilif_network(ilif_network&& other) = delete;
    ilif_network& operator=(const ilif_network& other) = delete;
    ilif_network& operator=(ilif_network&& other) = delete;

private:
    int linenum_neuronParameter(const char *par);
    int set_neurons(const char *par);
    int get_weight(const char *con);
    int _num_neurons;
    int *_tau, *_f, *_n;
    int *_weight, *_scurrent, *_ncurrent, *_biascurrent;
    ilif_neuron *_neurons;
    weight_index_list *_wlist;
    int _num_threads = 1;
};

#endif

