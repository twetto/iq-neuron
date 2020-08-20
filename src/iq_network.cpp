/* IQIF network
 * Chen-Fu Yeh, 2019/11/09
 */

#if defined(_WIN32) && defined(iq_network_EXPORTS)
#    define DLLEXPORTIQ __declspec (dllexport)
#else
#    define DLLEXPORTIQ
#endif

#include "iq_network.h"

using namespace std;

iq_network::iq_network(const char *par, const char *con)
{
    _num_neurons = linenum_neuronParameter(par);
    _neurons = new iq_neuron[_num_neurons];
    _tau = new int[_num_neurons * _num_neurons]();
    _f = new int[_num_neurons * _num_neurons]();
    _n = new int[_num_neurons * _num_neurons]();
    _weight = new int[_num_neurons * _num_neurons]();
    _wlist = new weight_index_list[_num_neurons];
    _scurrent = new int[_num_neurons * _num_neurons]();
    _ncurrent = new int[_num_neurons]();
    _biascurrent = new int[_num_neurons]();

    get_weight(con);
    set_neurons(par);
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
    int i, rest, threshold, reset, a, b, noise;
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
            &reset, &a, &b, &noise) == 7) {
        (_neurons + i)->set(rest, threshold, reset, a, b, noise);
    }
    fclose(fp);
    return 0;
}

int iq_network::get_weight(const char *con)
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

    fp = fopen(con, "r");
    if(fp == NULL) {
        printf("IQIF connection table file not opened\n");
        return 1;
    }

    while(fscanf(fp, "%d %d %d %d", &i, &j, &weight, &tau) == 4) {
        *(_weight + _num_neurons*i + j) = weight;
        *(_tau + _num_neurons*i + j) = tau;
        (_wlist + i)->push_front(j);
        if(tau >= 10) {
            *(_f + _num_neurons*i + j) = (int) (log10(0.875) / log10((tau-1)/(float) tau));
            printf("synapse[%d][%d]: decays every %d steps\n", i, j, *(_f + _num_neurons*i + j));
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
                    //printf("neuron %d has fired\n", i);
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
                //printf("neuron %d has fired\n", i);
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
                           int reset, int a, int b, int noise)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        (_neurons + neuron_index)->set(rest, threshold, reset, a, b, noise);
        return 1;
    }
    else return 0;
}

int iq_network::set_weight(int pre, int post, int weight, int tau)
{
    if(pre >= 0 && pre < _num_neurons && post >= 0 && post < _num_neurons && tau >= 10) {
        *(_weight + _num_neurons*pre + post) = weight;
        *(_tau + _num_neurons*pre + post) = tau;
        *(_f + _num_neurons*pre + post) = (int) (log10(0.875) / log10((tau-1)/(float) tau));
        return 1;
    }
    else {
        printf("Pre/post index out of range or tau not greater than 10.\n");
        printf("Please select index within 0 ~ %d.\n", _num_neurons-1);
        printf("Bad tau[%d][%d] = %d\n", pre, post, *(_tau + _num_neurons*pre + post));
        return 0;
    }
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
    DLLEXPORTIQ iq_network* iq_network_new(const char *par, const char *con) {return new iq_network(par, con);}
    DLLEXPORTIQ int iq_network_num_neurons(iq_network* network) {return network->num_neurons();}
    DLLEXPORTIQ void iq_network_send_synapse(iq_network* network) {return network->send_synapse();}
    DLLEXPORTIQ int iq_network_set_biascurrent(iq_network* network, int neuron_index, int biascurrent) {return network->set_biascurrent(neuron_index, biascurrent);}
    DLLEXPORTIQ int iq_network_set_neuron(iq_network* network, int neuron_index, int rest, int threshold, int reset, int a, int b, int noise) {return network->set_neuron(neuron_index, rest, threshold, reset, a, b, noise);}
    DLLEXPORTIQ int iq_network_set_weight(iq_network* network, int pre, int post, int weight, int tau) {return network->set_weight(pre, post, weight, tau);}
    DLLEXPORTIQ int iq_network_potential(iq_network* network, int neuron_index) {return network->potential(neuron_index);}
    DLLEXPORTIQ int iq_network_spike_count(iq_network* network, int neuron_index) {return network->spike_count(neuron_index);}
    DLLEXPORTIQ float iq_network_spike_rate(iq_network* network, int neuron_index) {return network->spike_rate(neuron_index);}
    DLLEXPORTIQ void iq_network_set_num_threads(iq_network* network, int num_threads) {return network->set_num_threads(num_threads);}
}

