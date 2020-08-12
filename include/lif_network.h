/* LIF network
 * Chen-Fu Yeh, 2020/07/02
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
    void set_biascurrent(int neuron_index, float biascurrent);
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

#endif

