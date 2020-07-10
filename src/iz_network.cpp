/* Izhikevich network
 * Chen-Fu Yeh, 2019/12/04
 */

#if defined(_WIN32) && defined(iz_network_EXPORTS)
#    define DLLEXPORTIZ __declspec (dllexport)
#else
#    define DLLEXPORTIZ
#endif

#include "iz_network.h"

using namespace std;

iz_network::iz_network()
{
    _num_neurons = linenum_neuronParameter();
    _neurons = new iz_neuron[_num_neurons];
    _tau = new int[_num_neurons * _num_neurons]();
    _weight = new float[_num_neurons * _num_neurons]();
    _wlist = new weight_index_list[_num_neurons];
    _scurrent = new float[_num_neurons * _num_neurons]();
    _ncurrent = new float[_num_neurons]();
    _biascurrent = new float[_num_neurons]();

    get_weight();
    set_neurons();
    return;
}

iz_network::~iz_network()
{
    delete[] _neurons;
    delete[] _tau;
    delete[] _weight;
    delete[] _wlist;
    delete[] _scurrent;
    delete[] _ncurrent;
    delete[] _biascurrent;
    return;
}

int iz_network::linenum_neuronParameter()
{
    int linenum = 0;
    float f[9];
    FILE *fp = fopen("../inputs/neuronParameter_Izhikevich.txt", "r");
    if(fp == NULL) {
        printf("neuronParameter_Izhikevich.txt file not opened\n");
        return -1;
    }
    
    while(fscanf(fp, " %f %f %f %f %f %f %f %f %f", &f[0], &f[1], &f[2],
            &f[3], &f[4], &f[5], &f[6], &f[7], &f[8]) == 9) {
        linenum++;
    }
    fclose(fp);
    return linenum;
}

int iz_network::set_neurons()
{
    int i;
    float a, b, c, d, k, rest, threshold;
    int noise;
    FILE *fp;

    /*
    if(_tau == 0) {
        printf("synapse decay time constant _tau = %d\n", _tau);
        printf("error: please get_weight() first to set _tau.\n");
        return 1;
    }
    */
    fp = fopen("../inputs/neuronParameter_Izhikevich.txt", "r");
    while(fscanf(fp, " %d %f %f %f %f %f %f %f %d", &i, &a, &b, &c, &d, &k, &rest, &threshold, &noise) == 9) {
        (_neurons + i)->set(a, b, c, d, k, rest, threshold, noise);
    }
    fclose(fp);
    return 0;
}

int iz_network::get_weight()
{
    float temp;
    int i, j, temptwo;
    FILE *fp;
    for(i = 0; i < _num_neurons; i++) {
        for(j = 0; j < _num_neurons; j++) {
            *(_scurrent + _num_neurons*i + j) = 0;
            *(_weight + _num_neurons*i + j) = 0;
            *(_tau + _num_neurons*i + j) = 1;
        }
        *(_biascurrent + i) = 0;
        *(_ncurrent + i) = 0;
    }
    fp = fopen("../inputs/Connection_Table_Izhikevich.txt", "r");
    if(fp == NULL) {
        printf("Connection_Table_Izhikevich.txt file not opened\n");
        return 1;
    }

    while(fscanf(fp, "%d %d %f %d", &i, &j, &temp, &temptwo) == 4) {
        *(_weight + _num_neurons*i + j) = temp;
        *(_tau + _num_neurons*i + j) = temptwo;
        (_wlist + i)->push_front(j);
        if(temptwo < 1) {
            printf("tau[%d][%d] = %d\n", i, j, *(_tau + _num_neurons*i + j));
            printf("error: synapse time constant cannot be less than 1!\n");
            return 1;
        }
    }
    fclose(fp);
    return 0;
}

int iz_network::num_neurons()
{
    return _num_neurons;
}

void iz_network::send_synapse()
{
    /* accumulating/decaying synapse current */
    if(_num_threads > 1) {
        #pragma omp parallel
        {
            float *ncurrent_private = new float[_num_neurons]();
            #pragma omp for
            for(int i = 0; i < _num_neurons; i++) {
                float *pts = _scurrent + _num_neurons*i;
                int *ptt = _tau + _num_neurons*i;
                if((_neurons + i)->is_firing()) {
                    float *ptw = _weight + _num_neurons*i;
                    weight_index_node *j = (_wlist + i)->_first;
                    while(j != NULL) {
                        *(pts + j->_data) += *(ptw + j->_data);
                        ncurrent_private[j->_data] += *(pts + j->_data);
                        *(pts + j->_data) = *(pts + j->_data) * (*(ptt + j->_data) - 1) / *(ptt + j->_data);
                        j = j->_next;
                    }
                }
                else {
                    weight_index_node *j = (_wlist + i)->_first;
                    while(j != NULL) {
                        ncurrent_private[j->_data] += *(pts + j->_data);
                        *(pts + j->_data) = *(pts + j->_data) * (*(ptt + j->_data) - 1) / *(ptt + j->_data);
                        j = j->_next;
                    }
                }
            }
            #pragma omp critical
            {
                for(int i = 0; i < _num_neurons; i++) {
                    *(_ncurrent + i) += ncurrent_private[i];
                }
            }
            delete[] ncurrent_private;
        }
    }
    else {
        for(int i = 0; i < _num_neurons; i++) {
            float *pts = _scurrent + _num_neurons*i;
            int *ptt = _tau + _num_neurons*i;
            if((_neurons + i)->is_firing()) {
                float *ptw = _weight + _num_neurons*i;
                weight_index_node *j = (_wlist + i)->_first;
                while(j != NULL) {
                    *(pts + j->_data) += *(ptw + j->_data);
                    *(_ncurrent + j->_data) += *(pts + j->_data);
                    *(pts + j->_data) = *(pts + j->_data) * (*(ptt + j->_data) - 1) / *(ptt + j->_data);
                    j = j->_next;
                }
            }
            else {
                weight_index_node *j = (_wlist + i)->_first;
                while(j != NULL) {
                    *(_ncurrent + j->_data) += *(pts + j->_data);
                    *(pts + j->_data) = *(pts + j->_data) * (*(ptt + j->_data) - 1) / *(ptt + j->_data);
                    j = j->_next;
                }
            }
        }
    }

    /* solving DE, reset post-syn current */
    for(int i = 0; i < _num_neurons; i++) {
        (_neurons + i)->iz_rk4(*(_ncurrent + i) + *(_biascurrent + i));
        *(_ncurrent + i) = 0;
    }
    return;
}

void iz_network::printfile(FILE **fp_potential, FILE **fp_adaptive_term)
{
    int i;

    for(i = 0; i < _num_neurons; i++) {
        fprintf(fp_potential[i], "%f\n", (_neurons+i)->potential());
    }
    for(i = 0; i < _num_neurons; i++) {
        fprintf(fp_adaptive_term[i], "%f\n", (_neurons+i)->adaptive_term());
    }
    return;
}

void iz_network::set_biascurrent(int neuron_index, float biascurrent)
{
    *(_biascurrent + neuron_index) = biascurrent;
    return;
}

float iz_network::potential(int neuron_index)
{
    return (_neurons + neuron_index)->potential();
}

float iz_network::adaptive_term(int neuron_index)
{
    return (_neurons + neuron_index)->adaptive_term();
}

int iz_network::spike_count(int neuron_index)
{
    return (_neurons + neuron_index)->spike_count();
}

float iz_network::spike_rate(int neuron_index)
{
    return (_neurons + neuron_index)->spike_rate();
}

void iz_network::set_num_threads(int num_threads)
{
    _num_threads = num_threads;
    omp_set_num_threads(num_threads);
    return;
}

extern "C"
{
    DLLEXPORTIZ iz_network* iz_network_new() {return new iz_network();}
    DLLEXPORTIZ int iz_network_num_neurons(iz_network* network) {return network->num_neurons();}
    DLLEXPORTIZ void iz_network_send_synapse(iz_network* network) {return network->send_synapse();}
    DLLEXPORTIZ void iz_network_set_biascurrent(iz_network* network, int neuron_index, int biascurrent) {return network->set_biascurrent(neuron_index, biascurrent);}
    DLLEXPORTIZ float iz_network_potential(iz_network* network, int neuron_index) {return network->potential(neuron_index);}
    DLLEXPORTIZ float iz_network_adaptive_term(iz_network* network, int neuron_index) {return network->adaptive_term(neuron_index);}
    DLLEXPORTIZ int iz_network_spike_count(iz_network* network, int neuron_index) {return network->spike_count(neuron_index);}
    DLLEXPORTIZ float iz_network_spike_rate(iz_network* network, int neuron_index) {return network->spike_rate(neuron_index);}
    DLLEXPORTIZ void iz_network_set_num_threads(iz_network* network, int num_threads) {return network->set_num_threads(num_threads);}
}

