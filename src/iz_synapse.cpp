#include "iz_synapse.h"

using namespace std;

int linenum_neuronParameter()
{
    int i[7], linenum = 0;
    FILE *fp = fopen("../inputs/neuronParameter.txt", "r");
    
    while(fscanf(fp, " %d %d %d %d %d %d %d", &i[0], &i[1], &i[2],
            &i[3], &i[4], &i[5], &i[6]) == 7) {
        linenum++;
    }
    fclose(fp);
    return linenum;
}

void set_neurons(int num_neurons, iz_neuron *neurons)
{
    int i;
    float a, b, c, d, k, rest, threshold, noise;
    FILE *fp;

    fp = fopen("../inputs/neuronParameter.txt", "r");
    while(fscanf(fp, " %d %f %f %f %f %f %f %f %f", &i, &a, &b, &c, &d, &k, &rest, &threshold, &noise) == 9) {
        (neurons + i)->set(a, b, c, d, k, rest, threshold, noise);
    }
    fclose(fp);
    return;
}

void get_weight(int num_neurons, float *weight)
{
    int i, j;
    float temp;
    FILE *fp;
    for(i = 0; i < num_neurons; i++) {
        for(j = 0; j < num_neurons; j++) {
            *(weight + num_neurons*i + j) = 0;
        }
    }
    fp = fopen("../inputs/Connection_Table.txt", "r");
    while(fscanf(fp, "%d %d %f", &i, &j, &temp) == 3) {
        *(weight + num_neurons*i + j) = temp;       // CAUTION: i/j relation
    }
    fclose(fp);
    return;
}

void send_synapse(int num_neurons, iz_neuron *neurons,
                  float *weight, int tau, float *current, float *biascurrent)
{
    int i, j;

    for(i = 0; i < num_neurons; i++) {
        if((neurons + i)->is_firing()) {
            //printf("neuron %d has fired!\n", i);
            for(j = 0; j < num_neurons; j++) {
                *(current + j) += *(weight + num_neurons*i + j);
            }
        }
    }
    for(i = 0; i < num_neurons; i++) {
        (neurons + i)->iz(*(current + i) + *(biascurrent + i));
        *(current + i) = *(current + i) * (tau-1) / tau;
    }
    return;
}

void delete_all(int *weight, int *current, iz_neuron *neurons)
{
    delete weight;
    delete current;
    delete neurons;
    return;
}

