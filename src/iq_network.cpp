/* IQIF network
 * Chen-Fu Yeh, 2019/11/09
 */

#include "iq_network.h"

using namespace std;

iq_network::iq_network()
{
    _num_neurons = linenum_neuronParameter();
    _neurons = new iq_neuron[_num_neurons];
    _tau = new int[_num_neurons * _num_neurons]();
    _f = new int[_num_neurons * _num_neurons]();
    _n = new int[_num_neurons * _num_neurons]();
    _weight = new int[_num_neurons * _num_neurons]();
    _wlist = new weight_index_list[_num_neurons];
    _scurrent = new int[_num_neurons * _num_neurons]();
    _ncurrent = new int[_num_neurons]();
    _biascurrent = new int[_num_neurons]();

    get_weight();
    set_neurons();
    return;
}

iq_network::~iq_network()
{
    delete[] _neurons;
    delete[] _tau;
    delete[] _f;
    delete[] _n;
    delete[] _weight;
    delete[] _wlist;
    delete[] _scurrent;
    delete[] _ncurrent;
    delete[] _biascurrent;
    return;
}

int iq_network::linenum_neuronParameter()
{
    int i[7], linenum = 0;
    FILE *fp = fopen("../inputs/neuronParameter_IQIF.txt", "r");
    if(fp == NULL) {
        printf("neuronParameter_IQIF.txt file not opened\n");
        return -1;
    }
    
    while(fscanf(fp, " %d %d %d %d %d %d %d", &i[0], &i[1], &i[2],
            &i[3], &i[4], &i[5], &i[6]) == 7) {
        linenum++;
    }
    fclose(fp);
    return linenum;
}

int iq_network::set_neurons()
{
    int i, rest, threshold, reset, a, b, noise;
    FILE *fp;

    /*
    if(*(_tau + 0) == 0) {
        printf("synapse decay time constant _tau = %d\n", _tau);
        printf("error: please get_weight() first to set _tau.\n");
        return 1;
    }
    */
    fp = fopen("../inputs/neuronParameter_IQIF.txt", "r");
    while(fscanf(fp, " %d %d %d %d %d %d %d", &i, &rest, &threshold,
            &reset, &a, &b, &noise) == 7) {
        (_neurons + i)->set(rest, threshold, reset, a, b, noise);
    }
    fclose(fp);
    return 0;
}

int iq_network::get_weight()
{
    int i, j, weight, tau;
    FILE *fp;
    for(i = 0; i < _num_neurons; i++) {
        for(j = 0; j < _num_neurons; j++) {
            *(_scurrent + _num_neurons*i + j) = 0;
            *(_weight + _num_neurons*i + j) = 0;
            *(_tau + _num_neurons*i + j) = 0;
            *(_f + _num_neurons*i + j) = 0;
            *(_n + _num_neurons*i + j) = 0;
        }
        *(_biascurrent + i) = 0;
        *(_ncurrent + i) = 0;
    }

    fp = fopen("../inputs/Connection_Table_IQIF.txt", "r");
    if(fp == NULL) {
        printf("Connection_Table_IQIF.txt file not opened\n");
        return 1;
    }

    while(fscanf(fp, "%d %d %d %d", &i, &j, &weight, &tau) == 4) {
        *(_weight + _num_neurons*i + j) = weight;
        *(_tau + _num_neurons*i + j) = tau;
        (_wlist + i)->push_front(j);
        if(tau >= 10) {
            *(_f + _num_neurons*i + j) = (int) (log10(0.875) / log10((tau-1)/(float) tau));
        }
        else {
            printf("tau[%d][%d] = %d\n", i, j, *(_tau + _num_neurons*i + j));
            printf("error: synapse time constant cannot be less than 10!\n");
            return 1;
        }
    }
    fclose(fp);
    return 0;
}

int iq_network::num_neurons()
{
    return _num_neurons;
}

void iq_network::send_synapse()
{
    /* accumulating/decaying synapse current */
    if(_num_threads > 1) {
    // parallel mode
        #pragma omp parallel
        {
            int *ncurrent_private = new int[_num_neurons]();
            #pragma omp for
            for(int i = 0; i < _num_neurons; i++) {
                int *pts = _scurrent + _num_neurons*i;
                int *ptn = _n + _num_neurons*i;
                int *ptf = _f + _num_neurons*i;
                if((_neurons + i)->_is_firing) {
                    int *ptw = _weight + _num_neurons*i;

                    /* parse through axon index */
                    weight_index_node *j = (_wlist + i)->_first;
                    while(j != NULL) {

                        /* accumulate weight if fired */
                        *(pts + j->_data) += *(ptw + j->_data);

                        /* add current to neuron input */
                        ncurrent_private[j->_data] += *(pts + j->_data);

                        /* synapse decay */
                        if(*(ptn + j->_data) > *(ptf + j->_data)) {
                            *(ptn + j->_data) = 0;
                            *(pts + j->_data) = *(pts + j->_data) * 7 / 8;
                        }
                        (*(ptn + j->_data))++;

                        j = j->_next;
                    }
                }
                else {
                    weight_index_node *j = (_wlist + i)->_first;
                    while(j != NULL) {
                        ncurrent_private[j->_data] += *(pts + j->_data);
                        if(*(ptn + j->_data) > *(ptf + j->_data)) {
                            *(ptn + j->_data) = 0;
                            *(pts + j->_data) = *(pts + j->_data) * 7 / 8;
                        }
                        (*(ptn + j->_data))++;
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
    // single thread mode
        for(int i = 0; i < _num_neurons; i++) {
            int *pts = _scurrent + _num_neurons*i;
            int *ptn = _n + _num_neurons*i;
            int *ptf = _f + _num_neurons*i;
            if((_neurons + i)->_is_firing) {
                int *ptw = _weight + _num_neurons*i;
                weight_index_node *j = (_wlist + i)->_first;
                while(j != NULL) {
                    *(pts + j->_data) += *(ptw + j->_data);
                    *(_ncurrent + j->_data) += *(pts + j->_data);
                    if(*(ptn + j->_data) > *(ptf + j->_data)) {
                        *(ptn + j->_data) = 0;
                        *(pts + j->_data) = *(pts + j->_data) * 7 / 8;
                    }
                    (*(ptn + j->_data))++;
                    j = j->_next;
                }
            }
            else {
                weight_index_node *j = (_wlist + i)->_first;
                while(j != NULL) {
                    *(_ncurrent + j->_data) += *(pts + j->_data);
                    if(*(ptn + j->_data) > *(ptf + j->_data)) {
                        *(ptn + j->_data) = 0;
                        *(pts + j->_data) = *(pts + j->_data) * 7 / 8;
                    }
                    (*(ptn + j->_data))++;
                    j = j->_next;
                }
            }
        }
    }

    /* solving DE, reset post-syn current */
    for(int i = 0; i < _num_neurons; i++) {
        (_neurons + i)->iq(*(_ncurrent + i) + *(_biascurrent + i));
        *(_ncurrent + i) = 0;
    }
    return;
}

void iq_network::printfile(FILE **fp)
{
    int i;

    for(i = 0; i < _num_neurons; i++) {
        fprintf(fp[i], "%d\n", (_neurons+i)->potential());
    }
    return;
}

void iq_network::set_biascurrent(int neuron_index, int biascurrent)
{
    *(_biascurrent + neuron_index) = biascurrent;
    return;
}

int iq_network::potential(int neuron_index)
{
    return (_neurons + neuron_index)->potential();
}

int iq_network::spike_count(int neuron_index)
{
    return (_neurons + neuron_index)->spike_count();
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
    iq_network* iq_network_new() {return new iq_network();}
    int iq_network_num_neurons(iq_network* network) {return network->num_neurons();}
    void iq_network_send_synapse(iq_network* network) {return network->send_synapse();}
    void iq_network_set_biascurrent(iq_network* network, int neuron_index, int biascurrent) {return network->set_biascurrent(neuron_index, biascurrent);}
    int iq_network_potential(iq_network* network, int neuron_index) {return network->potential(neuron_index);}
    int iq_network_spike_count(iq_network* network, int neuron_index) {return network->spike_count(neuron_index);}
    float iq_network_spike_rate(iq_network* network, int neuron_index) {return network->spike_rate(neuron_index);}
    void iq_network_set_num_threads(iq_network* network, int num_threads) {return network->set_num_threads(num_threads);}
}

