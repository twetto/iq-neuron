/* LIF network
 * Chen-Fu Yeh, 2020/07/02
 */

#if defined(_WIN32) && defined(lif_network_EXPORTS)
#    define DLLEXPORTLIF __declspec (dllexport)
#else
#    define DLLEXPORTLIF
#endif

#include "lif_network.h"

using namespace std;

lif_network::lif_network(const char *par, const char *con)
{
    _num_neurons = linenum_neuronParameter(par);
    _neurons = new lif_neuron[_num_neurons];
    _tau = new int[_num_neurons * _num_neurons]();
    _weight = new float[_num_neurons * _num_neurons]();
    _wlist = new weight_index_list[_num_neurons];
    _scurrent = new float[_num_neurons * _num_neurons]();
    _ncurrent = new float[_num_neurons]();
    _biascurrent = new float[_num_neurons]();

    get_weight(con);
    set_neurons(par);
    return;
}

lif_network::~lif_network()
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

int lif_network::linenum_neuronParameter(const char *par)
{
    int linenum = 0;
    float f[6];
    FILE *fp = fopen(par, "r");
    if(fp == NULL) {
        printf("LIF parameter file not opened\n");
        return -1;
    }
    
    while(fscanf(fp, " %f %f %f %f %f %f", &f[0], &f[1], &f[2],
            &f[3], &f[4], &f[5]) == 6) {
        linenum++;
    }
    fclose(fp);
    return linenum;
}

int lif_network::set_neurons(const char *par)
{
    int i;
    float g, rest, threshold, reset;
    int noise;
    FILE *fp;

    /*
    if(_tau == 0) {
        printf("synapse decay time constant _tau = %d\n", _tau);
        printf("error: please get_weight() first to set _tau.\n");
        return 1;
    }
    */
    fp = fopen(par, "r");
    while(fscanf(fp, " %d %f %f %f %f %d", &i, &g, &rest, &threshold, &reset, &noise) == 6) {
        (_neurons + i)->set(g, rest, threshold, reset, noise);
    }
    fclose(fp);
    return 0;
}

int lif_network::get_weight(const char *con)
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
    fp = fopen(con, "r");
    if(fp == NULL) {
        printf("LIF connection table file not opened\n");
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

int lif_network::num_neurons()
{
    return _num_neurons;
}

void lif_network::send_synapse()
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
        (_neurons + i)->lif_euler(*(_ncurrent + i) + *(_biascurrent + i));
        *(_ncurrent + i) = 0;
    }
    return;
}

void lif_network::printfile(FILE **fp)
{
    int i;

    for(i = 0; i < _num_neurons; i++) {
        fprintf(fp[i], "%f\n", (_neurons+i)->potential());
    }
    return;
}

int lif_network::set_biascurrent(int neuron_index, float biascurrent)
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

int lif_network::set_neuron(int neuron_index, float g, float rest,
                            float threshold, float reset, int noise)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        (_neurons + neuron_index)->set(g, rest, threshold, reset, noise);
        return 1;
    }
    else return 0;
}

int lif_network::set_weight(int pre, int post, float weight, int tau)
{
    if(pre >= 0 && pre < _num_neurons && post >= 0 && post < _num_neurons && tau > 1) {
        *(_weight + _num_neurons*pre + post) = weight;
        *(_tau + _num_neurons*pre + post) = tau;
        return 1;
    }
    else {
        printf("Pre/post index out of range or tau not greater than 1.\n");
        printf("Please select index within 0 ~ %d.\n", _num_neurons-1);
        return 0;
    }
}


float lif_network::potential(int neuron_index)
{
    return (_neurons + neuron_index)->potential();
}

int lif_network::spike_count(int neuron_index)
{
    return (_neurons + neuron_index)->spike_count();
}

float lif_network::spike_rate(int neuron_index)
{
    return (_neurons + neuron_index)->spike_rate();
}

void lif_network::set_num_threads(int num_threads)
{
    _num_threads = num_threads;
    omp_set_num_threads(num_threads);
    return;
}

extern "C"
{
    DLLEXPORTLIF lif_network* lif_network_new(const char *par, const char *con) {return new lif_network(par, con);}
    DLLEXPORTLIF int lif_network_num_neurons(lif_network* network) {return network->num_neurons();}
    DLLEXPORTLIF void lif_network_send_synapse(lif_network* network) {return network->send_synapse();}
    DLLEXPORTLIF int lif_network_set_biascurrent(lif_network* network, int neuron_index, float biascurrent) {return network->set_biascurrent(neuron_index, biascurrent);}
    DLLEXPORTLIF int lif_network_set_neuron(lif_network* network, int neuron_index, float g, float rest, float threshold, float reset, int noise) {return network->set_neuron(neuron_index, g, rest, threshold, reset, noise);}
    DLLEXPORTLIF int lif_network_set_weight(lif_network* network, int pre, int post, float weight, int tau) {return network->set_weight(pre, post, weight, tau);}
    DLLEXPORTLIF float lif_network_potential(lif_network* network, int neuron_index) {return network->potential(neuron_index);}
    DLLEXPORTLIF int lif_network_spike_count(lif_network* network, int neuron_index) {return network->spike_count(neuron_index);}
    DLLEXPORTLIF float lif_network_spike_rate(lif_network* network, int neuron_index) {return network->spike_rate(neuron_index);}
    DLLEXPORTLIF void lif_network_set_num_threads(lif_network* network, int num_threads) {return network->set_num_threads(num_threads);}
}

