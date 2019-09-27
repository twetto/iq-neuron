#include "iz_network.h"

using namespace std;

iz_network::iz_network()
{
    _num_neurons = linenum_neuronParameter();
    _neurons = new iz_neuron[_num_neurons];
    _tau = new int[_num_neurons * _num_neurons]();
    _weight = new float[_num_neurons * _num_neurons]();
    _current = new float[_num_neurons * _num_neurons]();
    _biascurrent = new float[_num_neurons]();

    get_weight();
    set_neurons();
    return;
}

iz_network::~iz_network()
{
    delete[] _weight;
    delete[] _current;
    delete[] _biascurrent;
    delete[] _tau;
    delete[] _neurons;
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
            *(_current + _num_neurons*i + j) = 0;
            *(_weight + _num_neurons*i + j) = 0;
            *(_tau + _num_neurons*i + j) = 1;
        }
        *(_biascurrent + i) = 0;
    }
    fp = fopen("../inputs/Connection_Table_Izhikevich.txt", "r");
    if(fp == NULL) {
        printf("Connection_Table_Izhikevich.txt file not opened\n");
        return 1;
    }

    while(fscanf(fp, "%d %d %f %d", &i, &j, &temp, &temptwo) == 4) {
        *(_weight + _num_neurons*i + j) = temp;
        *(_tau + _num_neurons*i + j) = temptwo;
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
    int i, j, temp;
    float total_current;

    /* accumulate individual synapse current */
    for(i = 0; i < _num_neurons; i++) {
        if((_neurons + i)->is_firing()) {
            //printf("neuron %d has fired!\n", i);
            for(j = 0; j < _num_neurons; j++) {
                *(_current + _num_neurons*i + j) += *(_weight + _num_neurons*i + j);
            }
        }
    }

    /* accumulate and inject current into neurons, solving DE */
    for(j = 0; j < _num_neurons; j++) {
        total_current = 0;
        for(i = 0; i < _num_neurons; i++) {
            total_current += *(_current + _num_neurons*i + j);
        }
        (_neurons + j)->iz_rk4(total_current + *(_biascurrent + j));
    }

    /* synapse exponential decay */
    for(i = 0; i < _num_neurons; i++) {
        for(j = 0; j < _num_neurons; j++) {
            temp = _num_neurons*i + j;
            *(_current + temp) = *(_current + temp) * (*(_tau + temp) - 1) / *(_tau + temp);
        }
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

extern "C"
{
    iz_network* iz_network_new() {return new iz_network();}
    int iz_network_set_neurons(iz_network* network) {return network->set_neurons();}
    int iz_network_get_weight(iz_network* network) {return network->get_weight();}
    int iz_network_num_neurons(iz_network* network) {return network->num_neurons();}
    void iz_network_send_synapse(iz_network* network) {return network->send_synapse();}
    void iz_network_set_biascurrent(iz_network* network, int neuron_index, int biascurrent) {return network->set_biascurrent(neuron_index, biascurrent);}
    float iz_network_potential(iz_network* network, int neuron_index) {return network->potential(neuron_index);}
    float iz_network_adaptive_term(iz_network* network, int neuron_index) {return network->adaptive_term(neuron_index);}
}

