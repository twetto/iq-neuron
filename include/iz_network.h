/* Izhikevich network
 * Chen-Fu Yeh, 2019/12/04
 */

#ifndef IZ_NETWORK_H
#define IZ_NETWORK_H
#include "iz_neuron.h"
#include "weight_index_list.h"
#include <math.h>
#include <omp.h>

class iz_network
{
public:
    iz_network(const char *par = "../inputs/neuronParameter_Izhikevich.txt",
               const char *con = "../inputs/Connection_Table_Izhikevich.txt");
    ~iz_network();
    int num_neurons();
    void send_synapse();
    void printfile(FILE **fp_potential, FILE **fp_adaptive_term);
    int set_biascurrent(int neuron_index, float biascurrent);
    int set_neuron(int neuron_index, float a, float b, float c,
                   float d, float k, float rest, float threshold, int noise);
    int set_weight(int pre, int post, float weight, int tau);
    float potential(int neuron_index);
    float adaptive_term(int neuron_index);
    int spike_count(int neuron_index);
    float spike_rate(int neuron_index);
    void set_num_threads(int num_threads);

protected:  // non-copyable and non-movable
    iz_network(const iz_network& other) = delete;
    iz_network(iz_network&& other) = delete;
    iz_network& operator=(const iz_network& other) = delete;
    iz_network& operator=(iz_network&& other) = delete;

private:
    int linenum_neuronParameter(const char *par);
    int set_neurons(const char *par);
    int get_weight(const char *con);
    int _num_neurons;
    int *_tau;
    float *_weight, *_scurrent, *_ncurrent, *_biascurrent;
    iz_neuron *_neurons;
    weight_index_list *_wlist;
    int _num_threads = 1;
};

#endif

