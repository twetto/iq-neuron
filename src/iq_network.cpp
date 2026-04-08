/* IQIF network
 * Chen-Fu Yeh, 2019/11/09
 */

#if defined(_WIN32) && defined(iq_network_EXPORTS)
#    define DLLEXPORTIQ __declspec (dllexport)
#else
#    define DLLEXPORTIQ
#endif

#include "iq_network.h"
#include <stdlib.h>

using namespace std;

// Helper for sorting connections during loading
struct Conn
{ 
    int pre; 
    int post; 
    int weight; 
    int tau; 
};

int compare_conn(const void* a, const void* b)
{
    return ((Conn*)a)->pre - ((Conn*)b)->pre;
}

iq_network::iq_network(const char *par, const char *con)
{
    _num_neurons = linenum_neuronParameter(par);
    _neurons = new iq_neuron[_num_neurons];
    _biascurrent = new int[_num_neurons]();

    _csr_offsets = NULL;
    _csr_targets = NULL;
    _csr_weights = NULL;
    _num_synapses = 0;

    set_neurons(par);
    get_weight(con);
    return;
}

iq_network::~iq_network()
{
    delete[] _neurons;
    delete[] _biascurrent;
    if(_csr_offsets) delete[] _csr_offsets;
    if(_csr_targets) delete[] _csr_targets;
    if(_csr_weights) delete[] _csr_weights;
    return;
}

int iq_network::linenum_neuronParameter(const char *par)
{
    int i[7], linenum = 0;
    FILE *fp = fopen(par, "r");
    if(fp == NULL) {
        printf("IQIF parameter file not opened\n");
        return -1;
    }
    
    while(fscanf(fp, " %d %d %d %d %d %d %d", &i[0], &i[1], &i[2],
            &i[3], &i[4], &i[5], &i[6]) == 7) {
        linenum++;
    }
    fclose(fp);
    return linenum;
}

int iq_network::set_neurons(const char *par)
{
    int i, rest, threshold, reset, shift_a, shift_b, noise;
    FILE *fp;

    /*
    if(*(_tau + 0) == 0) {
        printf("synapse decay time constant _tau = %d\n", _tau);
        printf("error: please get_weight() first to set _tau.\n");
        return 1;
    }
    */
    fp = fopen(par, "r");
    while(fscanf(fp, " %d %d %d %d %d %d %d", &i, &rest, &threshold,
            &reset, &shift_a, &shift_b, &noise) == 7) {
        if(i < _num_neurons)
            (_neurons + i)->set(rest, threshold, reset, shift_a, shift_b, noise);
    }
    fclose(fp);
    return 0;
}

int iq_network::get_weight(const char *con)
{
    FILE *fp = fopen(con, "r");
    if(fp == NULL) {
        printf("IQIF connection table file not opened\n");
        return 1;
    }

    // PASS 1: Count lines
    int i, j, weight, tau;
    int count = 0;
    for(i = 0; i < _num_neurons; i++) {
        *(_biascurrent + i) = 0;
    }
    while(fscanf(fp, "%d %d %d %d", &i, &j, &weight, &tau) == 4) {
        count++;
    }
    _num_synapses = count;
    rewind(fp);

    // Temp storage for sorting
    Conn* raw_list = new Conn[_num_synapses];
    int idx = 0;
    while(fscanf(fp, "%d %d %d %d", &i, &j, &weight, &tau) == 4) {
        raw_list[idx].pre = i;
        raw_list[idx].post = j;
        raw_list[idx].weight = weight;
        raw_list[idx].tau = tau;
        idx++;
    }
    fclose(fp);

    // Initialize Neurons (Tau settings)
    // Note: This overwrites if multiple connections have different taus
    // (Only the last tau applys)
    for(int k=0; k<_num_synapses; k++) {
        (_neurons + raw_list[k].post)->set_synapse_tau(raw_list[k].tau, _s_tau);
    }

    // Sort by pre-synaptic neuron
    qsort(raw_list, _num_synapses, sizeof(Conn), compare_conn);

    // PASS 2: Build CSR Arrays
    _csr_offsets = new int[_num_neurons + 1]();
    _csr_targets = new int[_num_synapses];
    _csr_weights = new int[_num_synapses];

    int current_pre = 0;
    for(int k=0; k<_num_synapses; k++) {
        Conn c = raw_list[k];
        
        // Fill offsets for any neurons we skipped
        while(current_pre < c.pre) {
            _csr_offsets[current_pre + 1] = k;
            current_pre++;
        }

        _csr_targets[k] = c.post;
        _csr_weights[k] = c.weight;
    }
    
    // Fill remaining offsets
    while(current_pre < _num_neurons) {
        _csr_offsets[current_pre + 1] = _num_synapses;
        current_pre++;
    }

    delete[] raw_list;
    return 0;
}

int iq_network::set_surrogate_tau(int s_tau)
{
    _s_tau = s_tau; 
    for(int i = 0; i < _num_neurons; i++) {
        (_neurons + i)->set_surrogate_tau(s_tau);
    }
    return 1;
}

// Per-neuron overload
int iq_network::set_surrogate_tau(int neuron_index, int s_tau)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        (_neurons + neuron_index)->set_surrogate_tau(s_tau);
        return 1;
    }
    else return 0;
}

int iq_network::get_surrogate_tau(int neuron_index)
{
    return (_neurons + neuron_index)->get_surrogate_tau();
}

int iq_network::num_neurons()
{
    return _num_neurons;
}

void iq_network::send_synapse()
{
    // Propagate Spikes (using is_firing from t-1)
    #pragma omp parallel for
    for(int i = 0; i < _num_neurons; i++) {
        if((_neurons + i)->is_firing()) {
            
            int start = _csr_offsets[i];
            int end = _csr_offsets[i+1];

            // Linear loop over contiguous memory
            for(int k = start; k < end; k++) {
                int target_idx = _csr_targets[k];
                int weight = _csr_weights[k];

                // Atomicity handled inside receive_spike
                (_neurons + target_idx)->receive_spike(weight);
            }
        }
    }

    // Update Neurons (Decay + Solve + Set Fire for t)
    #pragma omp parallel for
    for(int i = 0; i < _num_neurons; i++) {
        (_neurons + i)->update_state(*(_biascurrent + i));
    }
}

void iq_network::printfile(FILE **fp)
{
    int i;

    for(i = 0; i < _num_neurons; i++) {
        fprintf(fp[i], "%d\n", (_neurons+i)->potential());
    }
    return;
}

int iq_network::get_current_accumulator(int neuron_index) {
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        return (_neurons + neuron_index)->_synapse.current_accumulator;
    }
    return 0;
}

int iq_network::set_current_accumulator(int neuron_index, int value)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        (_neurons + neuron_index)->_synapse.current_accumulator = value;
        return 1;
    }
    return 0;
}

void iq_network::get_all_current_accumulators(int* output_array)
{
    for(int i = 0; i < _num_neurons; i++) {
        output_array[i] = (_neurons + i)->_synapse.current_accumulator;
    }
}

void iq_network::set_all_current_accumulators(const int* input_array)
{
    for(int i = 0; i < _num_neurons; i++) {
        (_neurons + i)->_synapse.current_accumulator = input_array[i];
    }
}

int iq_network::get_decay_threshold(int neuron_index) {
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        return (_neurons + neuron_index)->get_decay_threshold();
    }
    return 0;
}

int iq_network::set_biascurrent(int neuron_index, int biascurrent)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        *(_biascurrent + neuron_index) = biascurrent;
        return 1;
    }
    else {
        printf("Neuron index out of range.\n");
        printf("Please select index within 0 ~ %d\n", _num_neurons-1);
        return 0;
    }
}

int iq_network::set_neuron(int neuron_index, int rest, int threshold,
                           int reset, int shift_a, int shift_b, int noise)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        (_neurons + neuron_index)->set(rest, threshold, reset, shift_a, shift_b, noise);
        return 1;
    }
    else return 0;
}

int iq_network::set_weight(int pre, int post, int weight, int tau) {
    if(pre < 0 || pre >= _num_neurons || post < 0 || post >= _num_neurons)
        return 0;

    int start = _csr_offsets[pre];
    int end   = _csr_offsets[pre + 1];

    for(int k = start; k < end; k++) {
        if(_csr_targets[k] == post) {
            _csr_weights[k] = weight;
            (_neurons + post)->set_synapse_tau(tau, _s_tau);
            return 1;
        }
    }

    printf("Warning: synapse %d -> %d not found in connection table.\n", pre, post);
    return 0;
}

int iq_network::set_vmax(int neuron_index, int vmax)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        (_neurons + neuron_index)->set_vmax(vmax);
        return 0;
    }
    else return 1;
}

int iq_network::set_vmin(int neuron_index, int vmin)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        (_neurons + neuron_index)->set_vmin(vmin);
        return 0;
    }
    else return 1;
}

int iq_network::potential(int neuron_index)
{
    return (_neurons + neuron_index)->potential();
}

int iq_network::set_potential(int neuron_index, int value)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        (_neurons + neuron_index)->x = value;
        return 1;
    }
    return 0;
}

int iq_network::get_is_firing(int neuron_index)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        return (_neurons + neuron_index)->_is_firing ? 1 : 0;
    }
    return 0;
}

int iq_network::set_is_firing(int neuron_index, int value)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        (_neurons + neuron_index)->_is_firing = (value != 0);
        return 1;
    }
    return 0;
}

int iq_network::get_synapse_timer(int neuron_index)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        return (_neurons + neuron_index)->_synapse.timer;
    }
    return 0;
}

int iq_network::set_synapse_timer(int neuron_index, int value)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        (_neurons + neuron_index)->_synapse.timer = value;
        return 1;
    }
    return 0;
}

int iq_network::spike_count(int neuron_index)
{
    return (_neurons + neuron_index)->spike_count();
}

void iq_network::get_all_spike_counts(int* output_array)
{
    for(int i = 0; i < _num_neurons; i++) {
        output_array[i] = (_neurons + i)->spike_count();
    }
}

float iq_network::spike_rate(int neuron_index)
{
    return (_neurons + neuron_index)->spike_rate();
}

void iq_network::set_num_threads(int num_threads)
{
    _num_threads = num_threads;
    omp_set_num_threads(num_threads);
    return;
}

extern "C"
{
    DLLEXPORTIQ iq_network* iq_network_new(const char *par, const char *con) {return new iq_network(par, con);}
    DLLEXPORTIQ void iq_network_delete(iq_network* network) {delete network;}
    DLLEXPORTIQ int iq_network_num_neurons(iq_network* network) {return network->num_neurons();}
    DLLEXPORTIQ void iq_network_send_synapse(iq_network* network) {return network->send_synapse();}
    DLLEXPORTIQ int iq_network_set_biascurrent(iq_network* network, int neuron_index, int biascurrent) {return network->set_biascurrent(neuron_index, biascurrent);}
    DLLEXPORTIQ int iq_network_set_neuron(iq_network* network, int neuron_index, int rest, int threshold, int reset, int shift_a, int shift_b, int noise) {return network->set_neuron(neuron_index, rest, threshold, reset, shift_a, shift_b, noise);}
    DLLEXPORTIQ int iq_network_set_weight(iq_network* network, int pre, int post, int weight, int tau) {return network->set_weight(pre, post, weight, tau);}
    DLLEXPORTIQ int iq_network_set_surrogate_tau(iq_network* network, int s_tau) {return network->set_surrogate_tau(s_tau);}
    DLLEXPORTIQ int iq_network_set_neuron_surrogate_tau(iq_network* network, int neuron_index, int s_tau) { return network->set_surrogate_tau(neuron_index, s_tau); }
    DLLEXPORTIQ int iq_network_get_neuron_surrogate_tau(iq_network* network, int neuron_index) { return network->get_surrogate_tau(neuron_index); }
    DLLEXPORTIQ int iq_network_get_current_accumulator(iq_network* network, int neuron_index) { return network->get_current_accumulator(neuron_index); }
    DLLEXPORTIQ int iq_network_set_current_accumulator(iq_network* network, int neuron_index, int value) { return network->set_current_accumulator(neuron_index, value); }
    DLLEXPORTIQ void iq_network_get_all_current_accumulators(iq_network* network, int* output_array) { network->get_all_current_accumulators(output_array); }
    DLLEXPORTIQ void iq_network_set_all_current_accumulators(iq_network* network, const int* input_array) { network->set_all_current_accumulators(input_array); }
    DLLEXPORTIQ int iq_network_get_decay_threshold(iq_network* network, int neuron_index) { return network->get_decay_threshold(neuron_index); }
    DLLEXPORTIQ int iq_network_set_vmax(iq_network* network, int neuron_index, int vmax) {return network->set_vmax(neuron_index, vmax);}
    DLLEXPORTIQ int iq_network_set_vmin(iq_network* network, int neuron_index, int vmin) {return network->set_vmin(neuron_index, vmin);}
    DLLEXPORTIQ int iq_network_potential(iq_network* network, int neuron_index) {return network->potential(neuron_index);}
    DLLEXPORTIQ int iq_network_set_potential(iq_network* network, int neuron_index, int value) {return network->set_potential(neuron_index, value);}
    DLLEXPORTIQ int iq_network_get_is_firing(iq_network* network, int neuron_index) {return network->get_is_firing(neuron_index);}
    DLLEXPORTIQ int iq_network_set_is_firing(iq_network* network, int neuron_index, int value) {return network->set_is_firing(neuron_index, value);}
    DLLEXPORTIQ int iq_network_get_synapse_timer(iq_network* network, int neuron_index) {return network->get_synapse_timer(neuron_index);}
    DLLEXPORTIQ int iq_network_set_synapse_timer(iq_network* network, int neuron_index, int value) {return network->set_synapse_timer(neuron_index, value);}
    DLLEXPORTIQ int iq_network_spike_count(iq_network* network, int neuron_index) {return network->spike_count(neuron_index);}
    DLLEXPORTIQ void iq_network_get_all_spike_counts(iq_network* network, int* output_array) {return network->get_all_spike_counts(output_array);}
    DLLEXPORTIQ float iq_network_spike_rate(iq_network* network, int neuron_index) {return network->spike_rate(neuron_index);}
    DLLEXPORTIQ void iq_network_set_num_threads(iq_network* network, int num_threads) {return network->set_num_threads(num_threads);}
}
