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

int lif_network::set_vmax(int neuron_index, float vmax)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        (_neurons + neuron_index)->set_vmax(vmax);
        return 0;
    }
    else return 1;
}

int lif_network::set_vmin(int neuron_index, float vmin)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        (_neurons + neuron_index)->set_vmin(vmin);
        return 0;
    }
    else return 1;
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

ilif_network::ilif_network(const char *par, const char *con)
{
    _num_neurons = linenum_neuronParameter(par);
    _neurons = new ilif_neuron[_num_neurons];
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

ilif_network::~ilif_network()
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

int ilif_network::linenum_neuronParameter(const char *par)
{
    int i[6], linenum = 0;
    FILE *fp = fopen(par, "r");
    if(fp == NULL) {
        printf("ILIF parameter file not opened\n");
        return -1;
    }
    
    while(fscanf(fp, " %d %d %d %d %d %d", &i[0], &i[1], &i[2],
            &i[3], &i[4], &i[5]) == 6) {
        linenum++;
    }
    fclose(fp);
    return linenum;
}

int ilif_network::set_neurons(const char *par)
{
    int i, inv_g, rest, threshold, reset;
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
    while(fscanf(fp, " %d %d %d %d %d %d", &i, &inv_g, &rest, &threshold, &reset, &noise) == 6) {
        (_neurons + i)->set(inv_g, rest, threshold, reset, noise);
    }
    fclose(fp);
    return 0;
}

int ilif_network::get_weight(const char *con)
{
    int i, j, weight, tau;
    FILE *fp;
    for(i = 0; i < _num_neurons; i++) {
        for(j = 0; j < _num_neurons; j++) {
            *(_scurrent + _num_neurons*i + j) = 0;
            *(_weight + _num_neurons*i + j) = 0;
            *(_tau + _num_neurons*i + j) = 1;
            *(_f + _num_neurons*i + j) = 0;
            *(_n + _num_neurons*i + j) = 0;
        }
        *(_biascurrent + i) = 0;
        *(_ncurrent + i) = 0;
    }
    fp = fopen(con, "r");
    if(fp == NULL) {
        printf("ILIF connection table file not opened\n");
        return 1;
    }

    while(fscanf(fp, "%d %d %d %d", &i, &j, &weight, &tau) == 4) {
        *(_weight + _num_neurons*i + j) = weight;
        *(_tau + _num_neurons*i + j) = tau;
        (_wlist + i)->push_front(j);
        if(tau >= 10) {
            *(_f + _num_neurons*i + j) = (int) (log10(0.875) / log10((tau-1)/(float) tau));
            //printf("synapse[%d][%d]: decays every %d steps\n", i, j, *(_f + _num_neurons*i + j));
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

int ilif_network::num_neurons()
{
    return _num_neurons;
}

void ilif_network::send_synapse()
{
    /* accumulating/decaying synapse current */
    if(_num_threads > 1) {
        #pragma omp parallel
        {
            int *ncurrent_private = new int[_num_neurons]();
            #pragma omp for
            for(int i = 0; i < _num_neurons; i++) {
                int *pts = _scurrent + _num_neurons*i;
                int *ptn = _n + _num_neurons*i;
                int *ptf = _f + _num_neurons*i;
                if((_neurons + i)->is_firing()) {
                    int *ptw = _weight + _num_neurons*i;
                    weight_index_node *j = (_wlist + i)->_first;
                    while(j != NULL) {
                        *(pts + j->_data) += *(ptw + j->_data);
                        ncurrent_private[j->_data] += *(pts + j->_data);
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
        for(int i = 0; i < _num_neurons; i++) {
            int *pts = _scurrent + _num_neurons*i;
            int *ptn = _n + _num_neurons*i;
            int *ptf = _f + _num_neurons*i;
            if((_neurons + i)->is_firing()) {
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
        (_neurons + i)->ilif(*(_ncurrent + i) + *(_biascurrent + i));
        *(_ncurrent + i) = 0;
    }
    return;
}

void ilif_network::printfile(FILE **fp)
{
    int i;

    for(i = 0; i < _num_neurons; i++) {
        fprintf(fp[i], "%d\n", (_neurons+i)->potential());
    }
    return;
}

int ilif_network::set_biascurrent(int neuron_index, int biascurrent)
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

int ilif_network::set_neuron(int neuron_index, int inv_g, int rest,
                             int threshold, int reset, int noise)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        (_neurons + neuron_index)->set(inv_g, rest, threshold, reset, noise);
        return 1;
    }
    else return 0;
}

int ilif_network::set_weight(int pre, int post, int weight, int tau)
{
    if(pre >= 0 && pre < _num_neurons && post >= 0 && post < _num_neurons && tau >= 10) {
        *(_weight + _num_neurons*pre + post) = weight;
        *(_tau + _num_neurons*pre + post) = tau;
        *(_f + _num_neurons*pre + post) = (int) (log10(0.875) / log10((tau-1)/(float) tau));
        return 1;
    }
    else {
        printf("Pre/post index out of range or tau not greater than 1.\n");
        printf("Please select index within 0 ~ %d.\n", _num_neurons-1);
        printf("Bad tau[%d][%d] = %d\n", pre, post, *(_tau + _num_neurons*pre + post));
        return 0;
    }
}

int ilif_network::set_vmax(int neuron_index, int vmax)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        (_neurons + neuron_index)->set_vmax(vmax);
        return 0;
    }
    else return 1;
}

int ilif_network::set_vmin(int neuron_index, int vmin)
{
    if(neuron_index >= 0 && neuron_index < _num_neurons) {
        (_neurons + neuron_index)->set_vmin(vmin);
        return 0;
    }
    else return 1;
}

int ilif_network::potential(int neuron_index)
{
    return (_neurons + neuron_index)->potential();
}

int ilif_network::spike_count(int neuron_index)
{
    return (_neurons + neuron_index)->spike_count();
}

float ilif_network::spike_rate(int neuron_index)
{
    return (_neurons + neuron_index)->spike_rate();
}

void ilif_network::set_num_threads(int num_threads)
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
    DLLEXPORTLIF int lif_network_set_vmax(lif_network* network, int neuron_index, float vmax) {return network->set_vmax(neuron_index, vmax);}
    DLLEXPORTLIF int lif_network_set_vmin(lif_network* network, int neuron_index, float vmin) {return network->set_vmin(neuron_index, vmin);}
    DLLEXPORTLIF float lif_network_potential(lif_network* network, int neuron_index) {return network->potential(neuron_index);}
    DLLEXPORTLIF int lif_network_spike_count(lif_network* network, int neuron_index) {return network->spike_count(neuron_index);}
    DLLEXPORTLIF float lif_network_spike_rate(lif_network* network, int neuron_index) {return network->spike_rate(neuron_index);}
    DLLEXPORTLIF void lif_network_set_num_threads(lif_network* network, int num_threads) {return network->set_num_threads(num_threads);}
    DLLEXPORTLIF ilif_network* ilif_network_new(const char *par, const char *con) {return new ilif_network(par, con);}
    DLLEXPORTLIF int ilif_network_num_neurons(ilif_network* network) {return network->num_neurons();}
    DLLEXPORTLIF void ilif_network_send_synapse(ilif_network* network) {return network->send_synapse();}
    DLLEXPORTLIF int ilif_network_set_biascurrent(ilif_network* network, int neuron_index, int biascurrent) {return network->set_biascurrent(neuron_index, biascurrent);}
    DLLEXPORTLIF int ilif_network_set_neuron(ilif_network* network, int neuron_index, int inv_g, int rest, int threshold, int reset, int noise) {return network->set_neuron(neuron_index, inv_g, rest, threshold, reset, noise);}
    DLLEXPORTLIF int ilif_network_set_weight(ilif_network* network, int pre, int post, int weight, int tau) {return network->set_weight(pre, post, weight, tau);}
    DLLEXPORTLIF int ilif_network_set_vmax(ilif_network* network, int neuron_index, int vmax) {return network->set_vmax(neuron_index, vmax);}
    DLLEXPORTLIF int ilif_network_set_vmin(ilif_network* network, int neuron_index, int vmin) {return network->set_vmin(neuron_index, vmin);}
    DLLEXPORTLIF int ilif_network_potential(ilif_network* network, int neuron_index) {return network->potential(neuron_index);}
    DLLEXPORTLIF int ilif_network_spike_count(ilif_network* network, int neuron_index) {return network->spike_count(neuron_index);}
    DLLEXPORTLIF float ilif_network_spike_rate(ilif_network* network, int neuron_index) {return network->spike_rate(neuron_index);}
    DLLEXPORTLIF void ilif_network_set_num_threads(ilif_network* network, int num_threads) {return network->set_num_threads(num_threads);}
}

