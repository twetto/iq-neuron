#include "synapse.h"

using namespace std;

void get_weight(FILE *fp, int num_neurons, int *weight)
{
    int i, j, temp;
    for(i = 0; i < num_neurons; i++) {
        for(j = 0; j < num_neurons; j++) {
            *(weight + num_neurons*i + j) = 0;
        }
    }
    fp = fopen("../inputs/Connection_Table.txt", "r");
    while(fscanf(fp, "%d %d %d", &i, &j, &temp) == 3) {
        *(weight + num_neurons*i + j) = temp;       // CAUTION: i/j relation
    }
    fclose(fp);
    return;
}

void send_synapse(int num_neurons, iq_neuron *neurons,
                  int *weight, int tau, int *current)
{
    int i, j, temp;

    for(i = 0; i < num_neurons; i++) {
        if((neurons + i)->is_firing()) {
            printf("neuron %d has fired!\n", i);
            for(j = 0; j < num_neurons; j++) {
                *(current + j) += *(weight + num_neurons*i + j);
            }
        }
    }
    for(i = 0; i < num_neurons; i++) {
        //temp = abs(*(current + i)) / 10;
        //if(temp == 0) temp = 1;             // avoid floating point exception
        //*(current + i) += (rand() % temp) - (temp / 2);
        (neurons + i)->iq(*(current + i));
        *(current + i) = *(current + i) * (tau-1) / tau;
    }
    return;
}

void delete_all(int *weight, int *current, iq_neuron *neurons)
{
    delete weight;
    delete current;
    delete neurons;
    return;
}

