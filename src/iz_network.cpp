#include "iz_network.h"

using namespace std;

iz_network::iz_network()
{
    _num_neurons = linenum_neuronParameter();
    _neurons = new iz_neuron[_num_neurons];
    _weight = new float[_num_neurons * _num_neurons]();
    _current = new float[_num_neurons]();
    _biascurrent = new float[_num_neurons]();

    get_weight();
    set_neurons();
    return;
}

iz_network::~iz_network()
{
    delete[] _weight;
    delete[] _neurons;
    delete[] _current;
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
    float a, b, c, d, k, rest, threshold, noise;
    FILE *fp;

    if(_tau == 0) {
        printf("synapse decay time constant _tau = %d\n", _tau);
        printf("error: please get_weight() first to set _tau.\n");
        return 1;
    }
    fp = fopen("../inputs/neuronParameter_Izhikevich.txt", "r");
    while(fscanf(fp, " %f %f %f %f %f %f %f", &i, &a, &b, &c, &d, &k, &rest, &threshold, &noise) == 9) {
        (_neurons + i)->set(a, b, c, d, k, rest, threshold, noise);
    }
    fclose(fp);
    return 0;
}

int iz_network::get_weight()
{
    int i, j;
    float temp;
    FILE *fp;
    for(i = 0; i < _num_neurons; i++) {
        for(j = 0; j < _num_neurons; j++) {
            *(_weight + _num_neurons*i + j) = 0;
        }
    }
    fp = fopen("../inputs/Connection_Table_Izhikevich.txt", "r");
    if(fp == NULL) {
        printf("Connection_Table_Izhikevich.txt file not opened\n");
        return 1;
    }
    fscanf(fp, " %d", &_tau);
    if(_tau < 1) {
        printf("tau = %d\n", _tau);
        printf("error: synapse time constant cannot be less than 1!\n");
        return 1;
    }
    while(fscanf(fp, "%d %d %f", &i, &j, &temp) == 3) {
        *(_weight + _num_neurons*i + j) = temp;
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
    int i, j;

    for(i = 0; i < _num_neurons; i++) {
        if((_neurons + i)->is_firing()) {
            //printf("neuron %d has fired!\n", i);
            for(j = 0; j < _num_neurons; j++) {
                *(_current + j) += *(_weight + _num_neurons*i + j);
            }
        }
    }
    for(i = 0; i < _num_neurons; i++) {
        (_neurons + i)->iz_rk4(*(_current + i) + *(_biascurrent + i));
        //printf("neuron %d current: %f\n", i, *(_current + i));
        *(_current + i) = *(_current + i) * (_tau-1) / _tau;
    }

    return;
}

void iz_network::printfile(FILE **fp)
{
    int i;

    for(i = 0; i < _num_neurons; i++) {
        fprintf(fp[i], "%f\n", (_neurons+i)->potential());
    }
    return;
}

void iz_network::set_biascurrent(int neuron_index, float biascurrent)
{
    _biascurrent[neuron_index] = biascurrent;
    return;
}


