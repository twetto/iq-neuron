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
    _scurrent = new int[_num_neurons * _num_neurons]();
    _ncurrent = new int[_num_neurons]();
    _biascurrent = new int[_num_neurons]();

    get_weight();
    set_neurons();
    return;
}

iq_network::~iq_network()
{
    delete[] _weight;
    delete[] _scurrent;
    delete[] _ncurrent;
    delete[] _biascurrent;
    delete[] _tau;
    delete[] _f;
    delete[] _n;
    delete[] _neurons;
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
    int i, j, temp, temptwo;
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

    while(fscanf(fp, "%d %d %d %d", &i, &j, &temp, &temptwo) == 4) {
        *(_weight + _num_neurons*i + j) = temp;
        *(_tau + _num_neurons*i + j) = temptwo;
        if(temptwo >= 10) {
            *(_f + _num_neurons*i + j) = (int) (log10(0.9) / log10((temptwo-1)/(float) temptwo));
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

void iq_network::send_synapse(double &time_synapse, double &time_ode, double &time_decay)
{
    int i, j, temp;
    int *ptr, *pts, *ptt;
    clock_t start, end;

    /* accumulate individual synapse current */
    start = clock();
    for(i = 0; i < _num_neurons; i++) {
        if((_neurons + i)->is_firing()) {
            //printf("neuron %d has fired!\n", i);
            ptr = _scurrent + _num_neurons*i;
            pts = _weight + _num_neurons*i;
            for(j = 0; j < _num_neurons; j++) {
                //*(_scurrent + _num_neurons*i + j) += *(_weight + _num_neurons*i + j);
                *(ptr + j) += *(pts + j);
            }
        }
        /*
        ptr = _scurrent + _num_neurons*i;
        pts = _weight + _num_neurons*i;
        for(j = 0; j < _num_neurons; j++) {
            if((_neurons + i)->is_firing()) {
                *(ptr + j) += *(pts + j);
            }
            *(_ncurrent + j) += *(ptr + j);
        }
        */
    }
    end = clock();
    time_synapse += (double) (end - start) / CLOCKS_PER_SEC;

    /* accumulate and inject current into neurons, solving DE */
    start = clock();
    for(j = 0; j < _num_neurons; j++) {
        temp = 0;
        ptr = _scurrent + j;
        for(i = 0; i < _num_neurons; i++) {
            temp += *(ptr + _num_neurons*i);
        }
        (_neurons + j)->iq(temp + *(_biascurrent + j));
        /*
        (_neurons + j)->iq(*(_ncurrent + j) + *(_biascurrent + j));
        *(_ncurrent + j) = 0;
        */
    }
    end = clock();
    time_ode += (double) (end - start) / CLOCKS_PER_SEC;

    /* synapse exponential decay */
    start = clock();
    temp = _num_neurons * _num_neurons;
    for(i = 0; i < temp; i++) {
        if(*(_n + i) > *(_f + i)) {
            *(_n + i) = 0;
            (*(_scurrent + i)) *= 9;
            (*(_scurrent + i)) /= 10;
        }
        (*(_n + i))++;
    }
    end = clock();
    time_decay += (double) (end - start) / CLOCKS_PER_SEC;

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

extern "C"
{
    iq_network* iq_network_new() {return new iq_network();}
    int iq_network_set_neurons(iq_network* network) {return network->set_neurons();}
    int iq_network_get_weight(iq_network* network) {return network->get_weight();}
    int iq_network_num_neurons(iq_network* network) {return network->num_neurons();}
    void iq_network_send_synapse(iq_network* network, double time_synapse, double time_ode, double time_decay) {return network->send_synapse(time_synapse, time_ode, time_decay);}
    void iq_network_set_biascurrent(iq_network* network, int neuron_index, int biascurrent) {return network->set_biascurrent(neuron_index, biascurrent);}
    int iq_network_potential(iq_network* network, int neuron_index) {return network->potential(neuron_index);}
}

