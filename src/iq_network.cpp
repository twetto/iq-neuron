#include "iq_network.h"

using namespace std;

iq_network::iq_network()
{
    _num_neurons = linenum_neuronParameter();
    _neurons = new iq_neuron[_num_neurons];
    _weight = new int[_num_neurons * _num_neurons]();
    _current = new int[_num_neurons]();
    _biascurrent = new int[_num_neurons]();

    get_weight();
    set_neurons();
    return;
}

iq_network::~iq_network()
{
    delete[] _weight;
    delete[] _neurons;
    delete[] _current;
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

    if(_tau == 0) {
        printf("synapse decay time constant _tau = %d\n", _tau);
        printf("error: please get_weight() first to set _tau.\n");
        return 1;
    }
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
    int i, j, temp;
    FILE *fp;
    for(i = 0; i < _num_neurons; i++) {
        for(j = 0; j < _num_neurons; j++) {
            *(_weight + _num_neurons*i + j) = 0;
        }
    }
    fp = fopen("../inputs/Connection_Table_IQIF.txt", "r");
    if(fp == NULL) {
        printf("Connection_Table_IQIF.txt file not opened\n");
        return 1;
    }
    fscanf(fp, " %d", &_tau);
    if(_tau >= 10) {
        _f = (int) (log10(0.9) / log10((_tau-1)/(float) _tau));
    }
    else {
        printf("tau = %d\n", _tau);
        printf("error: synapse time constant cannot be less than 10!\n");
        return 1;
    }
    while(fscanf(fp, "%d %d %d", &i, &j, &temp) == 3) {
        *(_weight + _num_neurons*i + j) = temp;
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
    int i, j;
    static int n = 0;

    for(i = 0; i < _num_neurons; i++) {
        if((_neurons + i)->is_firing()) {
            //printf("neuron %d has fired!\n", i);
            for(j = 0; j < _num_neurons; j++) {
                *(_current + j) += *(_weight + _num_neurons*i + j);
            }
        }
    }
    for(i = 0; i < _num_neurons; i++) {
        (_neurons + i)->iq(*(_current + i) + *(_biascurrent + i));
        //printf("neuron %d current: %d\n", i, *(_current + i));
        //*(_current + i) = *(_current + i) * (tau-1) / tau;
    }
    if(n >= _f) {
        n = 0;
        for(i = 0; i < _num_neurons; i++) {
            *(_current + i) = *(_current + i) * 9 / 10;
        }
    }
    n++;

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
    _biascurrent[neuron_index] = biascurrent;
    return;
}


